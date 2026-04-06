import torch
import torch.nn as nn
import os

# Wrapper classes are required because torch.onnx.export expects a top-level forward() method.
# By wrapping the custom logic in PyTorch before exporting, the resulting ONNX code natively computes
# flatten/avgpool operations directly on the serverless side without needing custom python handlers.

class Slice4Wrapper(nn.Module):
    def __init__(self, slice_dict):
        super().__init__()
        self.features_end = slice_dict['features_end']
        self.avgpool = slice_dict['avgpool']
        
    def forward(self, x):
        x = self.features_end(x)
        x = self.avgpool(x)
        return x

class Slice5Wrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)

def convert_to_onnx():
    print("=== Converting PyTorch Slices to ONNX ===")
    
    # We trace the graph execution using a dummy input exactly the shape of a general mobile image
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Slice 1
    if os.path.exists('slice_1.pt'):
        print("Exporting slice_1.pt -> slice_1.onnx")
        s1 = torch.load('slice_1.pt', map_location='cpu', weights_only=False)
        s1.eval()
        torch.onnx.export(s1, dummy_input, 'slice_1.onnx', input_names=['input'], output_names=['output'], opset_version=14)
        dummy_input = s1(dummy_input)
        
    # Slice 2
    if os.path.exists('slice_2.pt'):
        print("Exporting slice_2.pt -> slice_2.onnx")
        s2 = torch.load('slice_2.pt', map_location='cpu', weights_only=False)
        s2.eval()
        torch.onnx.export(s2, dummy_input, 'slice_2.onnx', input_names=['input'], output_names=['output'], opset_version=14)
        dummy_input = s2(dummy_input)

    # Slice 3
    if os.path.exists('slice_3.pt'):
        print("Exporting slice_3.pt -> slice_3.onnx")
        s3 = torch.load('slice_3.pt', map_location='cpu', weights_only=False)
        s3.eval()
        torch.onnx.export(s3, dummy_input, 'slice_3.onnx', input_names=['input'], output_names=['output'], opset_version=14)
        dummy_input = s3(dummy_input)

    # Slice 4
    if os.path.exists('slice_4.pt'):
        print("Exporting slice_4.pt (ModuleDict) -> slice_4.onnx")
        s4_dict = torch.load('slice_4.pt', map_location='cpu', weights_only=False)
        s4 = Slice4Wrapper(s4_dict)
        s4.eval()
        torch.onnx.export(s4, dummy_input, 'slice_4.onnx', input_names=['input'], output_names=['output'], opset_version=14)
        dummy_input = s4(dummy_input)

    # Slice 5
    if os.path.exists('slice_5.pt'):
        print("Exporting slice_5.pt (Classifier) -> slice_5.onnx")
        s5_module = torch.load('slice_5.pt', map_location='cpu', weights_only=False)
        s5 = Slice5Wrapper(s5_module)
        s5.eval()
        torch.onnx.export(s5, dummy_input, 'slice_5.onnx', input_names=['input'], output_names=['output'], opset_version=14)
        dummy_input = s5(dummy_input)
        
    print("=== ONNX Export Complete ===")

if __name__ == '__main__':
    convert_to_onnx()
