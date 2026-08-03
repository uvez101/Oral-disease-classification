import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# ──────────────────────────────────────────────
# 1. Device Setup & Class Mapping
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

DISEASE_INFO = {
    'Calculus': {'emoji': '🪨', 'desc': 'Hardened mineral deposits (tarite) on tooth surfaces', 'severity': 'Moderate'},
    'Caries': {'emoji': '🦷', 'desc': 'Tooth decay caused by bacterial acid erosion', 'severity': 'High'},
    'Gingivitis': {'emoji': '🔴', 'desc': 'Inflammation and swelling of the gum tissue', 'severity': 'Moderate'},
    'Hypodontia': {'emoji': '⭕', 'desc': 'Congenital absence of one or more teeth', 'severity': 'Developmental'},
    'Mouth Ulcer': {'emoji': '💢', 'desc': 'Painful sores on the oral mucosa lining', 'severity': 'Low–Moderate'},
    'Tooth Discoloration': {'emoji': '🟡', 'desc': 'Abnormal staining or color changes of teeth', 'severity': 'Low'},
}

# ──────────────────────────────────────────────
# 2. Rebuild the ResNet50 Architecture
# ──────────────────────────────────────────────
print("Initializing ResNet50 architecture...")
model = models.resnet50(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

try:
    model.load_state_dict(torch.load("weights/best_resnet50.pth", map_location=device, weights_only=True))
    print(f">> Model weights loaded successfully on {device}")
except Exception as e:
    print(f">> Error loading model weights: {e}")

model.to(device)
model.eval()

# ──────────────────────────────────────────────
# 3. Preprocessing Pipeline (same as training)
# ──────────────────────────────────────────────
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────────
# 4. Inference Function
# ──────────────────────────────────────────────
def predict_disease(image):
    if image is None:
        return None, ""
    
    input_tensor = val_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    result_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    # Build a detailed result card for the top prediction
    top_class = max(result_dict, key=result_dict.get)
    top_prob = result_dict[top_class]
    info = DISEASE_INFO[top_class]
    
    confidence_level = "🟢 High" if top_prob > 0.75 else "🟡 Medium" if top_prob > 0.5 else "🔴 Low"
    
    detail_md = f"""
### {info['emoji']}  Detected: **{top_class}**

| | |
|---|---|
| **Confidence** | {confidence_level} ({top_prob:.1%}) |
| **Severity** | {info['severity']} |
| **Description** | {info['desc']} |

> ⚠️ *This is an AI screening tool, not a medical diagnosis. Please consult a dental professional for clinical evaluation.*
"""
    return result_dict, detail_md

# ──────────────────────────────────────────────
# 5. Custom CSS for Premium Dark Theme
# ──────────────────────────────────────────────
custom_css = """
/* ── Animated Gradient Background ── */
.gradio-container {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%) !important;
    min-height: 100vh;
}

/* ── Header Styling ── */
.hero-title {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
}
.hero-title h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00d2ff, #7b2ff7, #ff6fd8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.3rem !important;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    text-align: center;
    padding: 0 1rem 1rem;
}
.hero-subtitle p {
    color: #8b8fa3 !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
}

/* ── Glass Card Panels ── */
.glass-panel {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-panel:hover {
    border-color: rgba(123, 47, 247, 0.3) !important;
    box-shadow: 0 0 30px rgba(123, 47, 247, 0.08) !important;
}

/* ── Upload Area ── */
.upload-area .image-container,
.upload-area .upload-container {
    border: 2px dashed rgba(123, 47, 247, 0.3) !important;
    border-radius: 12px !important;
    background: rgba(123, 47, 247, 0.05) !important;
    transition: all 0.3s ease;
    min-height: 300px !important;
}
.upload-area .image-container:hover,
.upload-area .upload-container:hover {
    border-color: rgba(123, 47, 247, 0.6) !important;
    background: rgba(123, 47, 247, 0.08) !important;
    box-shadow: 0 0 20px rgba(123, 47, 247, 0.1) !important;
}

/* ── Prediction Labels ── */
.output-label .label-item {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
}

/* ── Analyze Button ── */
.analyze-btn {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 12px 32px !important;
    border: none !important;
    border-radius: 12px !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(123, 47, 247, 0.3) !important;
}
.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 30px rgba(123, 47, 247, 0.5) !important;
}

/* ── Detail Card ── */
.detail-card {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
.detail-card h3 {
    color: #e0e0ff !important;
    font-size: 1.3rem !important;
}
.detail-card table {
    width: 100%;
}
.detail-card td {
    padding: 6px 8px !important;
    color: #c0c4d6 !important;
    border: none !important;
}
.detail-card td:first-child {
    color: #8b8fa3 !important;
    font-weight: 600;
    white-space: nowrap;
}
.detail-card blockquote {
    border-left: 3px solid rgba(123, 47, 247, 0.5) !important;
    background: rgba(123, 47, 247, 0.05) !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 10px 14px !important;
    margin-top: 12px !important;
}
.detail-card p,
.detail-card span,
.detail-card em,
.detail-card li,
.detail-card blockquote p,
.detail-card blockquote em {
    color: #c0c4d6 !important;
}
.detail-card strong {
    color: #e0e2f0 !important;
}

/* ── Stats Badges ── */
.stat-badge {
    text-align: center;
}
.stat-badge p {
    margin: 0 !important;
    line-height: 1.4;
}
.stat-number {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00d2ff, #7b2ff7) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
.stat-label {
    color: #6b7094 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Section Headers ── */
.section-header p {
    color: #a0a4bc !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── Disease Grid ── */
.disease-tag {
    text-align: center;
    padding: 0.6rem 0.3rem !important;
}
.disease-tag p {
    margin: 0 !important;
    font-size: 0.85rem !important;
    color: #c0c4d6 !important;
    line-height: 1.5;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    padding-top: 1rem;
}
.footer-text p {
    color: #4a4e69 !important;
    font-size: 0.8rem !important;
}
"""

# ──────────────────────────────────────────────
# 6. Build the Premium UI with Gradio Blocks
# ──────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f3e8ff", c100="#e0cffc", c200="#c9a7fa",
        c300="#a87fdf", c400="#9061d4", c500="#7b2ff7",
        c600="#6a1fd6", c700="#5a17b5", c800="#4a1294",
        c900="#3a0d73", c950="#2a0852",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f8f9fc", c100="#e8e9f0", c200="#c0c4d6",
        c300="#a0a4bc", c400="#8b8fa3", c500="#6b7094",
        c600="#4a4e69", c700="#34374e", c800="#24243e",
        c900="#1a1a3e", c950="#0f0c29",
    ),
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_200",
    block_background_fill="transparent",
    block_border_width="0px",
    input_background_fill="rgba(255,255,255,0.05)",
    button_primary_background_fill="linear-gradient(135deg, #7b2ff7, #00d2ff)",
    button_primary_text_color="white",
)

with gr.Blocks(title="Oral Disease AI — Smart Dental Screening") as demo:
    
    # ── Hero Header ──
    gr.Markdown("# 🦷 Oral Disease AI", elem_classes=["hero-title"])
    gr.Markdown("Deep learning–powered dental screening · Upload an image to get an instant AI assessment", elem_classes=["hero-subtitle"])
    
    # ── Stats Bar ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=120):
            gr.Markdown('<p><span class="stat-number">6</span><br/><span class="stat-label">Conditions Detected</span></p>', elem_classes=["stat-badge"])
        with gr.Column(scale=1, min_width=120):
            gr.Markdown('<p><span class="stat-number">90%</span><br/><span class="stat-label">Validation Accuracy</span></p>', elem_classes=["stat-badge"])
        with gr.Column(scale=1, min_width=120):
            gr.Markdown('<p><span class="stat-number">ResNet50</span><br/><span class="stat-label">Model Architecture</span></p>', elem_classes=["stat-badge"])
        with gr.Column(scale=1, min_width=120):
            gr.Markdown('<p><span class="stat-number">&lt;2s</span><br/><span class="stat-label">Inference Time</span></p>', elem_classes=["stat-badge"])
    
    gr.HTML("<div style='height: 8px'></div>")
    
    # ── Main Content ──
    with gr.Row(equal_height=False):
        # Left Column — Upload
        with gr.Column(scale=1):
            gr.Markdown("📤  INPUT", elem_classes=["section-header"])
            with gr.Group(elem_classes=["glass-panel"]):
                input_image = gr.Image(
                    type="pil",
                    label="Upload Dental Image",
                    elem_classes=["upload-area"],
                    height=340,
                )
                analyze_btn = gr.Button(
                    "🔬  Analyze Image",
                    variant="primary",
                    elem_classes=["analyze-btn"],
                    size="lg",
                )
        
        # Right Column — Results
        with gr.Column(scale=1):
            gr.Markdown("📊  DIAGNOSIS", elem_classes=["section-header"])
            with gr.Group(elem_classes=["glass-panel"]):
                output_label = gr.Label(
                    num_top_classes=6,
                    label="Confidence Scores",
                    elem_classes=["output-label"],
                )
                detail_output = gr.Markdown(
                    value="*Upload an image and click **Analyze** to see results.*",
                    elem_classes=["detail-card"],
                )
    
    gr.HTML("<div style='height: 8px'></div>")
    
    # ── Disease Reference Grid ──
    gr.Markdown("🏷️  CONDITIONS DETECTED", elem_classes=["section-header"])
    with gr.Row(equal_height=True):
        for name, info in DISEASE_INFO.items():
            with gr.Column(scale=1, min_width=130):
                gr.Markdown(
                    f'<p>{info["emoji"]}<br/><strong>{name}</strong><br/><span style="font-size:0.75rem;color:#6b7094">{info["severity"]}</span></p>',
                    elem_classes=["disease-tag"],
                )
    
    # ── Footer ──
    gr.Markdown("Built with PyTorch & Gradio · ResNet50 Transfer Learning · Open Source", elem_classes=["footer-text"])
    
    # ── Event Wiring ──
    analyze_btn.click(
        fn=predict_disease,
        inputs=input_image,
        outputs=[output_label, detail_output],
    )
    input_image.change(
        fn=predict_disease,
        inputs=input_image,
        outputs=[output_label, detail_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=theme, css=custom_css)