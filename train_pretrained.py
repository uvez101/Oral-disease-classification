import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights, VGG16_Weights
import os
import csv

# Import your DataLoaders from Step 1
from dataloader import train_loader, val_loader

def build_model(model_name, num_classes=6):
    if model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Freeze base layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    return model

def train_pretrained_models(epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} (Base layers frozen to speed up compute)")
    
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    models_to_train = ['resnet50', 'vgg16']
    
    # Initialize a CSV to store our comparison metrics
    with open('logs/model_comparison.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])
    
    for model_name in models_to_train:
        print(f"\n{'='*40}")
        print(f"Initializing {model_name.upper()}...")
        print(f"{'='*40}")
        
        model = build_model(model_name).to(device)
        criterion = nn.CrossEntropyLoss()
        # Only optimize the parameters that require gradients (our new classification head)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # --- TRAINING ---
            model.train()
            running_train_loss, correct_train, total_train = 0.0, 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
            train_loss = running_train_loss / total_train
            train_acc = correct_train / total_train
            
            # --- VALIDATION ---
            model.eval()
            running_val_loss, correct_val, total_val = 0.0, 0, 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    running_val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
            val_loss = running_val_loss / total_val
            val_acc = correct_val / total_val
            
            print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save metrics to CSV
            with open('logs/model_comparison.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([model_name, epoch+1, train_loss, train_acc, val_loss, val_acc])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"weights/best_{model_name}.pth")
                
        print(f">> Finished training {model_name}. Best weights saved.")

if __name__ == "__main__":
    # We will use 10 epochs again to keep the comparison fair against your custom CNN
    train_pretrained_models(epochs=10)