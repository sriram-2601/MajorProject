FROM public.ecr.aws/lambda/python:3.9

# WORKDIR /app (Handled by base image)

# Install system dependencies
# RUN apt-get update && apt-get install -y ...

# Copy requirements
COPY requirements_deploy.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements_deploy.txt

# Copy model code and artifacts
# We verify the directory structure is flat for lambda or structured for container
COPY src/ ./src/ 
COPY deployment/lambda_function.py .
# COPY mobilenet_v3.pt . (Removed for Split Computing - Loaded from S3)

# Set entry point for Lambda (if using AWS Lambda Container Image)
CMD ["lambda_function.lambda_handler"]
