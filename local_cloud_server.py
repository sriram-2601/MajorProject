import io
import torch
from fastapi import FastAPI, UploadFile, File
import uvicorn
import base64

app = FastAPI(title="Local Cloud Classification Server")

CLASSIFIER_MODEL = None

@app.on_event("startup")
def load_model():
    global CLASSIFIER_MODEL
    print("Loading Cloud Classifier Model...", flush=True)
    try:
        CLASSIFIER_MODEL = torch.load('mobilenet_v3_classifier.pt', map_location='cpu', weights_only=False)
        CLASSIFIER_MODEL['avgpool'].eval()
        CLASSIFIER_MODEL['classifier'].eval()
        print("Cloud Classifier Model Loaded Successfully!", flush=True)
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)

@app.post("/predict")
async def predict_features(file: UploadFile = File(...)):
    if CLASSIFIER_MODEL is None:
        return {"error": "Classifier model is not loaded"}
        
    try:
        # Read the serialized tensor
        tensor_bytes = await file.read()
        
        # Deserialize the PyTorch tensor (Features extracted by Edge)
        buffer = io.BytesIO(tensor_bytes)
        buffer.seek(0)
        features = torch.load(buffer, weights_only=False, map_location='cpu')
        
        # Cloud Simulation (Classification)
        with torch.no_grad():
            pooled = CLASSIFIER_MODEL['avgpool'](features)
            flattened = torch.flatten(pooled, 1)
            outputs = CLASSIFIER_MODEL['classifier'](flattened)
            
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()
            
        return {
            "class_idx": class_idx,
            "confidence": confidence,
            "architecture": "split-computing-local-cloud"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import threading
    print("Starting Uvicorn Server on Port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
