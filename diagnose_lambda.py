import subprocess
import json
import sys

def run_command(command):
    print(f"Running: {command}")
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.decode('utf-8'))
        return output.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        print("COMMAND FAILED:")
        print(e.output.decode('utf-8'))
        return None

def main():
    print("=== DIAGNOSING LAMBDA DEPLOYMENT ===")
    
    # 1. Get Identity
    identity_json = run_command("aws sts get-caller-identity --output json --no-cli-pager")
    identity = json.loads(identity_json)
    account_id = identity['Account']
    print(f"Account ID: {account_id}")
    
    region = "us-east-1"
    repo_name = "mobilenet-repo"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"
    role_name = "MobileNetLambdaRole_v2"
    function_name = "MobileNetClassifier_v2"
    
    print(f"Image URI: {image_uri}")
    
    # 2. Get Role ARN
    role_json_str = run_command(f"aws iam get-role --role-name {role_name} --output json --no-cli-pager")
    role_arn = json.loads(role_json_str)['Role']['Arn']
    print(f"Role ARN: {role_arn}")
    
    # 3. Try to Create Function
    cmd = (
        f"aws lambda create-function "
        f"--function-name {function_name} "
        f"--package-type Image "
        f"--code ImageUri={image_uri} "
        f"--role {role_arn} "
        f"--timeout 30 "
        f"--memory-size 2048 "
        f"--no-cli-pager > lambda_error.txt 2>&1"
    )
    # Use shell=True to allow redirection
    subprocess.call(cmd, shell=True)
    
    with open("lambda_error.txt", "r") as f:
        print("CONTENTS OF lambda_error.txt:")
        print(f.read())

if __name__ == "__main__":
    main()
