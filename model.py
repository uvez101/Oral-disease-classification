import torch
import torch.nn as nn

class OralDiseaseCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Block 1: Capturing low-level edges and color gradients (e.g., gum redness)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Capturing textures (e.g., plaque buildup, ulcer surfaces)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Capturing complex shapes (e.g., cavities, tooth shapes)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: High-level spatial features (e.g., gaps for hypodontia)
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling to handle dynamic input sizes, forcing output to 7x7 spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier: Fully connected layers mapping features to the 6 disease classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Feature Extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Pooling and Flattening
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        
        # Classification
        x = self.classifier(x)
        return x

# Quick test to verify tensor dimensions
if __name__ == "__main__":
    model = OralDiseaseCNN(num_classes=6)
    # Simulating a single batch with a 224x224 RGB image
    dummy_input = torch.randn(1, 3, 224, 224) 
    output = model(dummy_input)
    print(f"Output tensor shape: {output.shape}") # Expected: torch.Size([1, 6])