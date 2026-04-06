import json
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import os
import boto3

# Global caching
LOADED_SLICES = {}
S3_BUCKET = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3')

def download_from_s3(bucket, key, local_path):
    print(f"Downloading s3://{bucket}/{key} to {local_path}...")
    s3_client.download_file(bucket, key, local_path)

def upload_to_s3(bucket, local_path, key):
    print(f"Uploading {local_path} to s3://{bucket}/{key}...")
    s3_client.upload_file(local_path, bucket, key)

def get_slice(slice_target):
    # Slice target: 2, 3, 4, 5
    slice_name = f"slice_{slice_target}.pt"
    
    if slice_name in LOADED_SLICES:
        return LOADED_SLICES[slice_name]

    local_path = f"/tmp/{slice_name}"
    
    if not os.path.exists(local_path):
        download_from_s3(S3_BUCKET, slice_name, local_path)

    print(f"Loading {slice_name} from /tmp...")
    slice_module = torch.load(local_path, weights_only=False, map_location='cpu')
    
    if isinstance(slice_module, nn.ModuleDict):
        slice_module['features_end'].eval()
        slice_module['avgpool'].eval()
    else:
        slice_module.eval()
        
    LOADED_SLICES[slice_name] = slice_module
    return slice_module

def lambda_handler(event, context):
    try:
        # Step Function JSON payload expectations:
        # {
        #   "slice_target": 2, // 2, 3, 4, or 5
        #   "session_id": "uuid-...", 
        #   "bucket_name": "mobilenet-slices-...",
        #   "input_tensor_s3_key": "session-xyz/tensor_1.pt"
        # }
        
        slice_target = event.get('slice_target')
        session_id = event.get('session_id')
        bucket_name = event.get('bucket_name', S3_BUCKET)
        input_s3_key = event.get('input_tensor_s3_key')
        
        if not slice_target or not session_id or not input_s3_key:
            return {'error': 'Missing required fields: slice_target, session_id, input_tensor_s3_key'}

        # 1. Download intermediate tensor from previous slice
        local_input_tensor = f"/tmp/input_{session_id}_{slice_target}.pt"
        download_from_s3(bucket_name, input_s3_key, local_input_tensor)
        
        # Load tensor
        tensor = torch.load(local_input_tensor, map_location='cpu')

        # 2. Get active slice logic
        slice_module = get_slice(slice_target)
        
        # 3. Compute inference part
        with torch.no_grad():
            if slice_target == 4:
                # Custom handler for ModuleDict (features_end -> avgpool)
                inter_out = slice_module['features_end'](tensor)
                output_tensor = slice_module['avgpool'](inter_out)
            elif slice_target == 5:
                # Custom handler for Classifier (flatten -> classifier)
                flattened = torch.flatten(tensor, 1)
                outputs = slice_module(flattened)
                
                # We've reached the end of the pipeline
                _, predicted = torch.max(outputs, 1)
                class_idx = predicted.item()
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()
                
                return {
                    'class_idx': class_idx,
                    'confidence': confidence,
                    'architecture': f'n-slice-step-functions (5 slices)'
                }
            else:
                # Standard sequential blocks (Slices 2 & 3)
                output_tensor = slice_module(tensor)
        
        # 4. Save and Upload Intermediate Tensor for next step
        local_output_tensor = f"/tmp/output_{session_id}_{slice_target}.pt"
        torch.save(output_tensor, local_output_tensor)
        
        next_s3_key = f"{session_id}/tensor_{slice_target}.pt"
        upload_to_s3(bucket_name, local_output_tensor, next_s3_key)
        
        # 5. Return context for the Next State in AWS Step Functions
        return {
            'session_id': session_id,
            'bucket_name': bucket_name,
            'output_tensor_s3_key': next_s3_key
        }
        
    except Exception as e:
        print(f"Error resolving Step Function node slice={event.get('slice_target')}: {str(e)}")
        raise e  # Throw error to let Step Function retry state catch it

