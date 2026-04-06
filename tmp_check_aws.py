import boto3

def get_resources():
    try:
        # Check Lambda
        lambda_client = boto3.client('lambda')
        try:
            lambda_client.get_function(FunctionName='mobilenet-inference')
            print('[ACTIVE] Lambda Function: mobilenet-inference')
        except lambda_client.exceptions.ResourceNotFoundException:
            print('[INACTIVE] Lambda Function: mobilenet-inference')

        # Check ECR
        ecr_client = boto3.client('ecr')
        try:
            ecr_client.describe_repositories(repositoryNames=['mobilenet-repo'])
            print('[ACTIVE] ECR Repository: mobilenet-repo')
        except ecr_client.exceptions.RepositoryNotFoundException:
            print('[INACTIVE] ECR Repository: mobilenet-repo')

        # Check Step Functions
        sf_client = boto3.client('stepfunctions')
        state_machines = sf_client.list_state_machines()
        found_sf = False
        for sm in state_machines['stateMachines']:
            if sm['name'] == 'MobileNetInferenceStateMachine':
                print('[ACTIVE] Step Function: MobileNetInferenceStateMachine')
                found_sf = True
                break
        if not found_sf:
             print('[INACTIVE] Step Function: MobileNetInferenceStateMachine')

        # Check S3 Buckets
        s3_client = boto3.client('s3')
        buckets = s3_client.list_buckets()
        found_bucket = False
        for b in buckets['Buckets']:
            if 'mobilenet-slices' in b['Name']:
                print(f"[ACTIVE] S3 Bucket: {b['Name']}")
                found_bucket = True
        if not found_bucket:
             print('[INACTIVE] S3 Bucket: mobilenet-slices-*')
             
    except Exception as e:
        print(f'Error querying AWS: {str(e)}')

if __name__ == "__main__":
    get_resources()
