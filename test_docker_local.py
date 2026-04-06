import requests
import json
import base64
from PIL import Image
import io
import sys

def main():
    print("=== TESTING LOCAL DOCKER LAMBDA ===")
    
    # 1. Create Dummy Image
    print("1. Creating dummy image...")
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # 2. Prepare Payload
    payload = {
        "body": img_b64
    }
    
    # 3. Invoke Local Endpoint
    url = "http://localhost:9000/2015-03-31/functions/function/invocations"
    
    print(f"2. Invoking {url}...")
    try:
        response = requests.post(url, json=payload)
        
        print(f"3. Response Code: {response.status_code}")
        print(f"4. Response Text: {response.text}")
        
        if response.status_code == 200:
             print("✅ SUCCESS! Local Docker responded.")
        else:
            print("❌ Local Docker Failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
