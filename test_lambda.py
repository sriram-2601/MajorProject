import boto3
import json
import base64
from PIL import Image
import io
import sys

def main():
    print("=== TESTING LAMBDA FUNCTION ===")
    
    # 1. Create Dummy Image
    print("1. Creating dummy image...")
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # 2. Prepare Payload
    payload = {
        "image": img_b64
    }
    
    # 3. Invoke Lambda
    client = boto3.client('lambda', region_name='us-east-1')
    function_name = 'mobilenet-inference'
    
    print(f"2. Invoking {function_name}...")
    try:
        response = client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        payload_stream = response['Payload']
        response_data = json.loads(payload_stream.read())
        
        with open("lambda_test_result.txt", "w") as f:
            f.write(json.dumps(response_data, indent=2))
            
        print(f"3. Response Status: {response['StatusCode']}")
        print(f"4. Response Body written to lambda_test_result.txt")
        
        if 'errorMessage' in response_data:
            print("Lambda Execution Failed!")
            sys.exit(1)
            
        if 'body' in response_data:
             body = json.loads(response_data['body'])
             if 'error' in body:
                 print(f"Lambda Returned Error: {body['error']}")
                 sys.exit(1)
             print("SUCCESS! Inference returned result.")
        else:
            print("Unexpected response format.")
            
    except Exception as e:
        print(f"Invocation Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
