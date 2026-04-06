import boto3
import json
import time

def deploy_step_function():
    print("=== Deploying AWS Step Function ===")
    sts = boto3.client('sts')
    iam = boto3.client('iam')
    account_id = sts.get_caller_identity()['Account']
    # Use standard region
    region = 'us-east-1'
    sf = boto3.client('stepfunctions', region_name=region)
    
    role_name = 'StepFunctionLambdaInvokeRole_v2'
    role_arn = ""
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "states.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    print("1. Setting up IAM Role for Step Functions...")
    try:
        role_resp = iam.get_role(RoleName=role_name)
        role_arn = role_resp['Role']['Arn']
        print(f"   Role {role_name} already exists.")
    except iam.exceptions.NoSuchEntityException:
        print(f"   Creating Role {role_name}...")
        role_resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy)
        )
        role_arn = role_resp['Role']['Arn']
        time.sleep(10) # Wait for propagation
        
    print("   Attaching AWSLambdaRole policy...")
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaRole'
    )
    time.sleep(5)
    
    print("2. Reading ASL Definition...")
    with open('step_function_state_machine.asl.json', 'r') as f:
        definition = f.read()
        
    state_machine_name = 'MobileNetInferenceStateMachine'
    state_machine_arn = f"arn:aws:states:{region}:{account_id}:stateMachine:{state_machine_name}"
    
    print("3. Creating/Updating Step Function...")
    try:
        sf.describe_state_machine(stateMachineArn=state_machine_arn)
        print("   Updating existing State Machine...")
        sf.update_state_machine(
            stateMachineArn=state_machine_arn,
            definition=definition,
            roleArn=role_arn
        )
    except sf.exceptions.StateMachineDoesNotExist:
        print("   Creating new State Machine...")
        sf.create_state_machine(
            name=state_machine_name,
            definition=definition,
            roleArn=role_arn
        )
    
    print("=== Step Function Deployment Complete ===")

if __name__ == "__main__":
    deploy_step_function()
