import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Import your custom architecture
from model import OralDiseaseCNN

# Define the exact same validation transforms used during training
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names must map alphabetically based on your folder structure
CLASS_NAMES = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

def predict_image(image_path, model_path="weights/best_oral_model.pth"):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    # Route to CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model and load the best weights
    model = OralDiseaseCNN(num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set model to evaluation mode (disables dropout, fixes batchnorm)

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = val_transforms(image).unsqueeze(0).to(device) # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    print(f"--- Inference Results ---")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence_score:.2f}%")

if __name__ == "__main__":
    # TASK: Change this path to point to a specific image you want to test
    test_image_path = r"C:\Users\bobbo\OneDrive\Desktop\489.jpg" 
    
    predict_image(test_image_path)