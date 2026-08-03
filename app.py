import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# 1. Device Setup & Class Mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

# 2. Rebuild the ResNet50 Architecture
# We do not need to download the default weights here, just the skeleton of the network
print("Initializing ResNet50 architecture...")
model = models.resnet50(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6) # Map the final layer to our 6 classes

# Load your saved weights from Step 3
try:
    model.load_state_dict(torch.load("weights/best_resnet50.pth", map_location=device, weights_only=True))
    print(f">> Model weights loaded successfully on {device}")
except Exception as e:
    print(f">> Error loading model weights: {e}")

model.to(device)
model.eval()

# 3. The Preprocessing Pipeline (Identical to Training)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. The Inference Function
def predict_disease(image):
    if image is None:
        return None
    
    # Gradio passes a PIL Image directly because we set type="pil"
    input_tensor = val_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0] # Extract the first batch item
        
    # Create a dictionary of {Class: Probability} for Gradio's Label UI
    result_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    return result_dict

# 5. Build the Web Interface
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Upload Dental Scan"),
    outputs=gr.Label(num_top_classes=3, label="Diagnostic Prediction"),
    title="Oral Disease Classification AI",
    description="Upload an image to detect signs of Calculus, Caries, Gingivitis, Hypodontia, Mouth Ulcers, or Tooth Discoloration. Powered by ResNet50."
)

if __name__ == "__main__":
    # Setting share=True will generate a public link you can send to anyone.
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)