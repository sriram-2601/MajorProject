import torch.nn as nn
from torchvision import models

def get_model(num_classes, model_name='mobilenet_v3_small', pretrained=True):
    """
    Loads MobileNetV3 Small or Large and modifies the classifier head.
    """
    if model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
    else:
        raise ValueError("Invalid model name. Choose 'mobilenet_v3_small' or 'mobilenet_v3_large'.")

    # Modify the classifier to match the number of classes
    # The classifier structure in MobileNetV3 is:
    # (3): Linear(in_features=1024 (small) or 1280 (large), out_features=1000, bias=True)
    
    # We want to replace the last Linear layer.
    # The classifier is a Sequential block.
    
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model
