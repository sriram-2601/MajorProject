import sys
import os

print("=== DIAGNOSIS START ===")
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

try:
    print("Importing torch...")
    import torch
    print(f"Torch: {torch.__version__}")
    
    print("Importing torchvision...")
    import torchvision
    print(f"Torchvision: {torchvision.__version__}")
    
    print("Importing PIL...")
    import PIL
    print(f"PIL: {PIL.__version__}")
    
    print("Importing lambda_function...")
    import lambda_function
    print("Lambda Function imported successfully")
    
    print("Testing model load...")
    model = lambda_function.load_model()
    print("Model loaded successfully")
    
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=== DIAGNOSIS END ===")
