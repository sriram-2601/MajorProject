import argparse
import torch
import os
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import plot_history
import torch.nn as nn
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser(description='MobileNetV3 Training and Evaluation')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], help='Mode: train or evaluate')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='mobilenet_v3.pth', help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        print("Setting up data loaders...")
        train_loader, val_loader, class_names = get_dataloaders(args.data_dir, args.batch_size)
        num_classes = len(class_names)
        print(f"Classes: {class_names}")
        
        print("Initializing model...")
        model = get_model(num_classes, model_name='mobilenet_v3_small')
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        print("Starting training...")
        model, hist = train_model(model, {'train': train_loader, 'val': val_loader}, 
                                  criterion, optimizer, device, num_epochs=args.epochs)
        
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
        
        plot_history(hist)
        
    elif args.mode == 'evaluate':
        # Need to know num_classes to load model structure mostly, 
        # or we just assume folder structure exists to infer it.
        # For simplicity, we'll try to get it from data_dir again.
        _, val_loader, class_names = get_dataloaders(args.data_dir, args.batch_size)
        num_classes = len(class_names)
        
        model = get_model(num_classes, model_name='mobilenet_v3_small')
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        
        evaluate_model(model, val_loader, class_names, device)

if __name__ == '__main__':
    main()
