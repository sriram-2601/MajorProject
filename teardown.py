import subprocess
import json
import sys
import shutil

def run_command(command, ignore_error=False):
    print(f"Running: {command}", flush=True)
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        return result.decode('utf-8', errors='replace').strip()
    except subprocess.CalledProcessError as e:
        if not ignore_error:
            print(f"Failed: {e.output.decode('utf-8', errors='replace')}", flush=True)
        return None

def main():
    print("=== Tearing Down AWS Resources ===", flush=True)
    
    identity_json = run_command("aws sts get-caller-identity --output json --no-cli-pager")
    if not identity_json:
        print("Failed to get AWS identity. Make sure AWS CLI is configured.")
        return
        
    identity = json.loads(identity_json)
    account_id = identity['Account']
    
    region = run_command("aws configure get region", ignore_error=True)
    if not region:
        region = "us-east-1"
        
    bucket_name = f"mobilenet-slices-{account_id}-{region}"
    
    # 1. Delete Lambda
    print("\n--- Deleting Lambda Function ---")
    run_command("aws lambda delete-function --function-name mobilenet-inference --no-cli-pager", ignore_error=True)
    
    # 2. Delete ECR Repository (and all images)
    print("\n--- Deleting ECR Repository ---")
    run_command(f"aws ecr delete-repository --repository-name mobilenet-repo --force --region {region} --no-cli-pager", ignore_error=True)
    
    # 3. Empty and Delete S3 Bucket
    print("\n--- Deleting S3 Bucket ---")
    run_command(f"aws s3 rm s3://{bucket_name} --recursive --no-cli-pager", ignore_error=True)
    run_command(f"aws s3api delete-bucket --bucket {bucket_name} --region {region} --no-cli-pager", ignore_error=True)
    
    # 4. Detach Policies and Delete IAM Role
    print("\n--- Deleting IAM Role ---")
    role_name = "MobileNetLambdaRole_v2"
    run_command(f"aws iam detach-role-policy --role-name {role_name} --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole --no-cli-pager", ignore_error=True)
    run_command(f"aws iam detach-role-policy --role-name {role_name} --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess --no-cli-pager", ignore_error=True)
    run_command(f"aws iam delete-role --role-name {role_name} --no-cli-pager", ignore_error=True)
    
    print("\n=== Teardown Complete! All deployment costs stopped. ===", flush=True)

if __name__ == "__main__":
    main()
