import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# Define the path to your unzipped dataset
DATA_DIR = r"D:\Instant\CV and CNN\dataset" # Update this to your actual path

# 1. Define Transforms (Augmentation + Normalization)
# We use RandomHorizontalFlip, RandomRotation, and ColorJitter to simulate 
# different camera angles and lighting conditions inside a mouth.
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Load the Dataset
full_dataset = datasets.ImageFolder(root=DATA_DIR)

# 3. Split into Training (80%) and Validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply respective transforms (PyTorch doesn't natively allow different transforms 
# for random_split subsets easily, so we map them here)
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

# 4. Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Classes found: {full_dataset.classes}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")