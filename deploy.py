import subprocess
import json
import sys
import os
import io
import shutil

def run_command(command, capture_output=True, ignore_error=False):
    print(f"Running: {command}", flush=True)
    try:
        if capture_output:
            # Use errors='replace' to avoid crashing on non-utf8 output
            result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            return result.decode('utf-8', errors='replace').strip()
        else:
            subprocess.check_call(command, shell=True)
            return None
    except subprocess.CalledProcessError as e:
        if ignore_error:
            return None
        print(f"Error running command: {command}", flush=True)
        if e.output:
            print(e.output.decode('utf-8', errors='replace'), flush=True)
        sys.exit(1)

def main():
    print("=== Starting AWS Deployment ===", flush=True)
    
    # Locate executables
    aws_cmd = "aws"
    docker_cmd = "docker"
    
    if shutil.which("aws") is None:
        print("Error: 'aws' executable not found in PATH.", flush=True)
        sys.exit(1)
    if shutil.which("docker") is None:
        print("Error: 'docker' executable not found in PATH.", flush=True)
        sys.exit(1)
        
    print(f"Using aws: {shutil.which('aws')}", flush=True)
    print(f"Using docker: {shutil.which('docker')}", flush=True)
    
    # Check if Docker Daemon is running
    print("Checking Docker Daemon status...", flush=True)
    try:
        run_command("docker info", capture_output=True)
        print("   Docker Daemon is running.", flush=True)
    except Exception:
        print("Error: Docker Daemon is not running. Please start Docker Desktop and try again.", flush=True)
        sys.exit(1)
    
    # 1. Get AWS Identity
    print("1. Getting AWS Identity...", flush=True)
    identity_json = run_command("aws sts get-caller-identity --output json --no-cli-pager")
    identity = json.loads(identity_json)
    account_id = identity['Account']
    print(f"   Account ID: {account_id}", flush=True)
    
    # 2. Get Region
    print("2. Getting Region...", flush=True)
    region = run_command("aws configure get region")
    if not region:
        region = "us-east-1" # Fallback
    print(f"   Region: {region}", flush=True)
    
    repo_name = "mobilenet-repo"
    ecr_url = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    full_image_name = f"{ecr_url}/{repo_name}:latest"
    
    # 3. Create ECR Repo
    print("3. Creating ECR Repository...", flush=True)
    run_command(f"aws ecr create-repository --repository-name {repo_name} --region {region} --no-cli-pager", ignore_error=True)

    # 4. Login to ECR
    print("4. Logging in to ECR...", flush=True)
    try:
        password = run_command(f"aws ecr get-login-password --region {region} --no-cli-pager")
        
        # Run docker login safely using shell=False and list args
        login_cmd = [docker_cmd, "login", "--username", "AWS", "--password-stdin", ecr_url]
        print(f"Running: {' '.join(login_cmd)}", flush=True)
        
        p = subprocess.Popen(login_cmd, 
                             shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate(input=password.encode())
        if p.returncode != 0:
            print(f"Docker Login Failed: {stderr.decode('utf-8', errors='replace')}", flush=True)
            sys.exit(1)
        print("   Docker Login Successful.", flush=True)
    except Exception as e:
        print(f"   Login failed: {e}", flush=True)
        sys.exit(1)
    
    # 5. Build Docker Image
    print("5. Building Docker Image...", flush=True)
    # Ensure mobilenet_v3.pt exists
    if not os.path.exists("mobilenet_v3.pt"):
        print("Error: mobilenet_v3.pt not found!", flush=True)
        sys.exit(1)
        
    try:
        # redirect output to file
        with open("docker_build.log", "w", encoding="utf-8") as f:
            # Use shell=False for build too if possible, but command string is easier with shell=True for build context '.'
            # Let's try shell=False
            # Add --provenance=false to avoid OCI layout issues with Lambda
            build_cmd = [docker_cmd, "build", "--provenance=false", "-t", repo_name, "-f", "deployment/Dockerfile", "."]
            print(f"Running: {' '.join(build_cmd)}", flush=True)
            subprocess.check_call(build_cmd, shell=False, stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("Docker Build Failed! output:", flush=True)
        try:
            with open("docker_build.log", "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                print(content, flush=True)
        except Exception as read_err:
            print(f"Could not read docker_build.log: {read_err}", flush=True)
        sys.exit(1)
    
    # 6. Tag and Push
    print("6. Tagging and Pushing...", flush=True)
    run_command(f"docker tag {repo_name}:latest {full_image_name}", capture_output=False)
    run_command(f"docker push {full_image_name}", capture_output=False)
    
    print(f"=== Deployment Artifact Pushed Details ===", flush=True)
    print(f"Image URI: {full_image_name}", flush=True)
    
    # NEW STEP: Slice and Upload Model to S3
    print("=== Setting up Split Computing Storage (S3) ===", flush=True)
    
    # Run slicing script
    print("Running Model Slicing...", flush=True)
    run_command("python slice_model.py")
    
    bucket_name = f"mobilenet-slices-{account_id}-{region}"
    print(f"Target S3 Bucket: {bucket_name}", flush=True)
    
    # Check if bucket exists
    try:
        run_command(f"aws s3api head-bucket --bucket {bucket_name} --no-cli-pager")
        print("   Bucket exists.", flush=True)
    except SystemExit:
        # head-bucket returns 404/403 if not found/accessible, which run_command catches as error
        # We assume it doesn't exist or we can't access it. Try creating.
        print("   Bucket not found or not accessible. Creating...", flush=True)
        if region == "us-east-1":
            run_command(f"aws s3api create-bucket --bucket {bucket_name} --region {region} --no-cli-pager")
        else:
            run_command(f"aws s3api create-bucket --bucket {bucket_name} --region {region} --create-bucket-configuration LocationConstraint={region} --no-cli-pager")

    # Upload slices
    print(f"Uploading slices to s3://{bucket_name}...", flush=True)
    run_command(f"aws s3 cp slice_2.pt s3://{bucket_name}/slice_2.pt --no-cli-pager")
    run_command(f"aws s3 cp slice_3.pt s3://{bucket_name}/slice_3.pt --no-cli-pager")
    run_command(f"aws s3 cp slice_4.pt s3://{bucket_name}/slice_4.pt --no-cli-pager")
    run_command(f"aws s3 cp slice_5.pt s3://{bucket_name}/slice_5.pt --no-cli-pager")
    print("   Upload complete.", flush=True)

    
    # 7. Create/Get IAM Role
    print("7. Setting up IAM Role...", flush=True)
    role_name = "MobileNetLambdaRole_v2"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    # Check if role exists
    role_arn_json = run_command(f"aws iam get-role --role-name {role_name} --output json --no-cli-pager", capture_output=True, ignore_error=True)
    if role_arn_json:
        role_arn = json.loads(role_arn_json)['Role']['Arn']
        print(f"   Using existing role: {role_arn}", flush=True)
    else:
        print("   Creating new role...", flush=True)
        # Write trust policy to temp file
        with open("trust_policy.json", "w", encoding='utf-8') as f:
            json.dump(trust_policy, f)
            
        role_creation_json = run_command(f"aws iam create-role --role-name {role_name} --assume-role-policy-document file://trust_policy.json --output json --no-cli-pager")
        role_arn = json.loads(role_creation_json)['Role']['Arn']
        
        # Wait for propagation
        print("   Waiting 10s for role propagation...", flush=True)
        import time
        time.sleep(10)

    # Attach Policies (Idempotent, so safe to run every time)
    print("   Attaching policies...", flush=True)
    run_command(f"aws iam attach-role-policy --role-name {role_name} --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    run_command(f"aws iam attach-role-policy --role-name {role_name} --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess")
    print("   Policies attached (BasicExecution + S3FullAccess).", flush=True)

    # 8. Create/Update Lambda Function
    print("8. Creating/Updating Lambda Function...", flush=True)
    function_name = "mobilenet-inference"
    
    # Check if function exists
    func_json = run_command(f"aws lambda get-function --function-name {function_name} --no-cli-pager", capture_output=True, ignore_error=True)
    
    if func_json:
        print(f"   Updating existing function code for {function_name}...", flush=True)
        run_command(f"aws lambda update-function-code --function-name {function_name} --image-uri {full_image_name} --no-cli-pager")
        print("   Waiting 5s for update to settle...", flush=True)
        import time
        time.sleep(20)
        print(f"   Updating configuration (Timeout: 60s, Memory: 3008MB, EnvVars)...", flush=True)
        run_command(f"aws lambda update-function-configuration --function-name {function_name} --timeout 60 --memory-size 3008 --environment Variables={{BUCKET_NAME={bucket_name}}} --no-cli-pager")
    else:
        print(f"   Creating new function {function_name}...", flush=True)
        try:
            run_command(f"aws lambda create-function --function-name {function_name} --package-type Image --code ImageUri={full_image_name} --role {role_arn} --timeout 60 --memory-size 3008 --environment Variables={{BUCKET_NAME={bucket_name}}} --no-cli-pager")
        except Exception as e:
            print(f"   Failed to create function (might be role propagation issue, try running again): {e}", flush=True)
            sys.exit(1)
            
    print(f"\nSUCCESS! Lambda Function '{function_name}' is deployed.", flush=True)
    print("You can verify it in the AWS Console.", flush=True)

if __name__ == "__main__":
    main()
