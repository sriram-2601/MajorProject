import torch
import torch.nn as nn
import os
import sys

# Import the model definition from src/model.py
sys.path.append('src')
try:
    from model import get_model
except ImportError:
    # Fallback if running from root
    from src.model import get_model

def slice_model():
    print("=== SLICING MODEL (N-SLICE PIPELINE) ===")
    
    # 1. Load the full model
    model_path = "mobilenet_v3.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train or provide the model first.")
        sys.exit(1)
        
    print(f"Loading {model_path}...")
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Infer num_classes from the last layer weight
    if 'classifier.3.weight' in state_dict:
        num_classes = state_dict['classifier.3.weight'].shape[0]
        print(f"Inferred num_classes: {num_classes}")
    else:
        print("Could not infer num_classes, defaulting to 2")
        num_classes = 2

    full_model = get_model(num_classes, pretrained=False)
    full_model.load_state_dict(state_dict)
    full_model.eval()
    
    # 2. Slice the model into 5 sequential parts
    # MobileNetV3 small `features` has 13 modules (0 to 12).
    # We will split it into:
    # Slice 1 (Edge): features[0:4]
    # Slice 2 (Lambda 1): features[4:8]
    # Slice 3 (Lambda 2): features[8:12]
    # Slice 4 (Lambda 3): features[12:] + avgpool (requires an nn.Sequential or ModuleDict)
    # Slice 5 (Lambda 4): classifier (flattening + classification)
    
    print("Extracting Slices...")
    features = full_model.features
    
    slice_1 = features[0:4]
    slice_2 = features[4:8]
    slice_3 = features[8:12]
    
    # Slice 4 consists of the last feature block and the avgpool
    # We will use a ModuleDict to pass them just like the old version
    slice_4 = nn.ModuleDict({
        'features_end': features[12:],
        'avgpool': full_model.avgpool
    })
    
    # Slice 5 is the classifier
    slice_5 = full_model.classifier
    
    # 3. Save Slices
    print("Saving slices...")
    torch.save(slice_1, "slice_1.pt")
    torch.save(slice_2, "slice_2.pt")
    torch.save(slice_3, "slice_3.pt")
    torch.save(slice_4, "slice_4.pt")
    torch.save(slice_5, "slice_5.pt")
    
    print("Slicing Complete.")
    print("Created: slice_1.pt (Edge)")
    print("Created: slice_2.pt (Step 1)")
    print("Created: slice_3.pt (Step 2)")
    print("Created: slice_4.pt (Step 3 - Wrapped in ModuleDict)")
    print("Created: slice_5.pt (Step 4 - Classifier)")

if __name__ == "__main__":
    slice_model()
