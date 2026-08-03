# 🦷 Oral Disease Classification with CNN

A deep learning project for classifying oral diseases from dental images using both a **custom CNN** and **pretrained models** (ResNet50, VGG16, MobileNet V3). Includes a **Gradio web interface** for interactive diagnosis.

## 📋 Diseases Detected

| Class | Description |
|-------|-------------|
| Calculus | Hardened plaque deposits on teeth |
| Caries | Tooth decay / cavities |
| Gingivitis | Inflammation of the gums |
| Hypodontia | Congenital absence of teeth |
| Mouth Ulcer | Sores in the oral cavity |
| Tooth Discoloration | Abnormal tooth coloring |

## 🏗️ Project Structure

```
├── model.py              # Custom 4-block CNN architecture
├── dataloader.py         # Dataset loading, augmentation & splitting
├── train.py              # Training loop for the custom CNN
├── train_pretrained.py   # Transfer learning with ResNet50, VGG16
├── inference.py          # CLI inference on a single image
├── app.py                # Gradio web UI for interactive predictions
├── logs/
│   └── model_comparison.csv   # Training metrics across models
├── requirements.txt
└── README.md
```

## 📊 Model Performance (10 Epochs)

| Model | Best Val Accuracy | Best Val Loss |
|-------|:-----------------:|:-------------:|
| **ResNet50** | **90.0%** | **0.308** |
| VGG16 | 83.9% | 0.390 |
| Custom CNN | — | — |

> Results from transfer learning with frozen base layers and a fine-tuned classification head.

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/oral-disease-classification.git
cd oral-disease-classification
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the oral disease image dataset and place it in a `dataset/` folder with the following structure:

```
dataset/
├── Calculus/
├── Caries/
├── Gingivitis/
├── Hypodontia/
├── Mouth Ulcer/
└── Tooth Discoloration/
```

Update the `DATA_DIR` path in `dataloader.py` to point to your dataset location.

### 3. Train

```bash
# Train the custom CNN
python train.py

# Train pretrained models (ResNet50, VGG16)
python train_pretrained.py
```

Trained weights are saved to `weights/`.

### 4. Run Inference

```bash
# CLI inference on a single image
python inference.py

# Launch the Gradio web interface
python app.py
```

The web app will be available at `http://127.0.0.1:7860`.

## 🧠 Architecture

### Custom CNN (`model.py`)

A 4-block convolutional network designed for oral pathology:

- **Block 1** — Low-level features (edges, color gradients for gum redness)
- **Block 2** — Textures (plaque buildup, ulcer surfaces)
- **Block 3** — Complex shapes (cavities, tooth morphology)
- **Block 4** — High-level spatial features (gaps for hypodontia)
- **Classifier** — Adaptive pooling → FC layers with dropout

### Transfer Learning (`train_pretrained.py`)

Pretrained backbones (ImageNet) with frozen feature extractors and custom classification heads fine-tuned on the oral disease dataset.

## 🛠️ Tech Stack

- **PyTorch** — Model building & training
- **torchvision** — Pretrained models & image transforms
- **Gradio** — Interactive web UI
- **PIL/Pillow** — Image preprocessing

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
