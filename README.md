# Solar Filament Segmentation & Space Weather Intelligence System

> AI-powered solar filament detection and analysis using hybrid deep learning + classical computer vision.

## 🌟 Overview

This system automatically detects and segments solar filaments from H-alpha full-disk solar observations. It combines:

- **U-Net** deep learning segmentation model
- **Frangi vesselness filter** classical ridge detection
- **Hybrid fusion** of both approaches for robust detection

Solar filaments are elongated, dark absorption structures suspended above the Sun's surface in H-alpha imagery. They are important indicators of solar eruptions that can affect satellites, communication networks, and power infrastructure.

## 📁 Project Structure

```
solarf/
├── configs/                    # Configuration files
│   └── default_config.yaml     # Hyperparameters & settings
├── preprocessing/              # Image preprocessing
│   ├── solar_preprocessor.py   # Disk detection, normalization, CLAHE
│   └── dataset.py              # PyTorch Dataset, COCO parsing
├── classical/                  # Classical CV pipeline
│   ├── frangi.py               # Frangi vesselness filter
│   ├── hessian.py              # Hessian eigenvalue analysis
│   └── morphology.py           # Connected components, skeleton
├── models/                     # Deep learning models
│   └── unet.py                 # U-Net architecture
├── training/                   # Training pipeline
│   ├── train.py                # Training loop with AMP
│   ├── losses.py               # Dice + BCE combined loss
│   └── metrics.py              # Dice, IoU, Precision, Recall
├── inference/                  # Inference pipeline
│   └── predict.py              # Single-image prediction
├── hybrid/                     # Hybrid fusion
│   └── fusion.py               # U-Net + Frangi combination
├── analysis/                   # Filament analysis
│   └── filament_morphology.py  # Morphology measurements
├── visualization/              # Visualization utilities
│   └── viz.py                  # Overlays, heatmaps, plots
├── explainability/             # Model explainability
│   └── confidence.py           # Confidence & uncertainty maps
├── dashboard/                  # Web interface
│   └── app.py                  # Gradio dashboard
├── checkpoints/                # Saved model weights
├── experiments/                # Experiment results
├── images/                     # Dataset (MAGFiLO 1.0)
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 4050)
- NVIDIA Driver 525+

### Installation

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install scikit-image gradio pycocotools tqdm pyyaml albumentations
```

### Training

```bash
python training/train.py
```

### Inference Dashboard

```bash
python dashboard/app.py
```

Then open `http://localhost:7860` in your browser.

## 📊 Dataset

**MAGFiLO 1.0** (MLEcoFi 2024) — Solar filament dataset from GONG/NSO H-alpha observations.

| Property | Value |
|---|---|
| Training images | 707 |
| Test images | 180 |
| Image size | 2048 × 2048 |
| Annotations | 8,199 COCO polygon segmentations |
| Categories | Left, Right, Unidentifiable chirality |
| Avg filaments/image | 7.1 |

## 🧠 Methods

### Classical Pipeline (Frangi)
1. Solar disk detection & masking
2. Limb darkening correction
3. CLAHE contrast enhancement
4. Multi-scale Frangi vesselness filtering
5. Thresholding + morphological cleanup

### Deep Learning (U-Net)
- Encoder-decoder with skip connections
- Trained with Dice + BCE combined loss
- Automatic Mixed Precision (AMP)
- CosineAnnealingLR scheduler

### Hybrid Fusion
- Weighted combination: `final = α × UNet + (1-α) × Frangi`
- Optimal α found via validation set sweep

## 📏 Filament Morphology

The system measures (all in pixel units):
- Area, perimeter
- Length (skeleton-based)
- Average width
- Orientation
- Bounding box, centroid
- Segmentation confidence

## 🏗️ Tech Stack

- **PyTorch** — Deep learning framework
- **OpenCV** — Image processing
- **scikit-image** — Frangi filter, morphology
- **Gradio** — Web dashboard
- **NVIDIA CUDA** — GPU acceleration

## 📄 License

This project uses data from NSO/GONG (National Solar Observatory / Global Oscillation Network Group).

## 👥 Team

Built for GGSIPU Hackathon 2026 — Track 19: AI-Based Automated Solar Filament Segmentation & Space Weather Intelligence System
