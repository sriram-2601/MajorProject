# AWS Deployment Guide for MobileNetV3

This guide explains how to deploy your trained MobileNetV3 model to AWS Lambda to achieve the "Cost Reduction" goal of your major project.

## Why AWS Lambda?
- **Serverless**: You don't manage servers.
- **Cost-Effective**: You only pay for the milliseconds the code runs (inference time).
- **Scalable**: Automatically handles 1 user or 1000 users.

## Prerequisites
1.  **AWS Account**: You need a free-tier account.
2.  **Trained Model**: You must run `python main.py --mode train` first to generate `mobilenet_v3.pth`.

## Deployment Steps

### Option A: Zip File Deployment (Simplest)
*Suitable if your total package size is < 250MB (unzipped).*

1.  **Prepare the Package**:
    Create a folder named `package`.
    Install dependencies into it:
    ```bash
    pip install --target ./package torch torchvision numpy pillow
    ```
    *Note: Standard PyTorch is large. For Lambda, use the CPU-only version to save space.*
    ```bash
    pip install --target ./package torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
    ```

2.  **Add Your Code**:
    - Copy `deployment/lambda_function.py` into `package/`.
    - Copy your `mobilenet_v3.pth` into `package/`.
    - Zip the contents of `package/` into `deploy.zip`.

3.  **Upload to AWS Lambda**:
    - Go to AWS Console -> Lambda -> Create Function.
    - Runtime: Python 3.9 (or newer).
    - Code Source: Upload `.zip` file.
    - **Handler**: Change from `lambda_function.lambda_handler` to `lambda_function.lambda_handler`.
    - **Configuration**: Increase Timeout to ~30 seconds and Memory to ~2048MB (MobileNet needs some RAM).

### Option B: Docker Container (Recommended for PyTorch)
*Since PyTorch is heavy, Docker is often easier.*

1.  **Build the Image**:
    (Make sure Docker Desktop is installed)
    ```bash
    docker build -t mobilenet-lambda -f deployment/Dockerfile .
    ```

2.  **Push to AWS ECR**:
    - Go to AWS Console -> ECR -> Create Repository.
    - Follow the "Push Commands" in the console to push your local docker image to AWS.

3.  **Create Lambda from Image**:
    - Go to Lambda -> Create Function -> Select "Container Image".
    - Browse ECR and select your image.

## Testing
Once deployed, click "Test" in the Lambda console and send a payload:
```json
{
  "body": "base64_string_of_your_image..."
}
```
Or if you implemented the direct image handler:
```json
{
  "image": "base64_string_of_your_image..."
}
```

## Cost Analysis (For your Report)
- **EC2 (Old Way)**: t2.medium = ~$0.0464/hour = ~$33/month (running 24/7).
- **Lambda (Your Way)**: $0.20 per 1M requests.
- **Savings**: Huge for low-traffic student projects.
