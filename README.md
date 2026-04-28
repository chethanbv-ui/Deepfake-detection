# 🔍 DeepFake Detector — M-Tech Final Year Project

A research-grade deepfake image detection system that uses an **ensemble of four deep learning models** to classify images as real or AI-generated (GAN/Diffusion). Includes explainability via **Grad-CAM heatmaps** and **LIME superpixel explanations**, served through a **Flask web application** with a red-and-black UI.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Goal** | Detect AI-generated (deepfake) images using a multi-model ensemble |
| **Models** | HSFAN, EfficientNet-B0 (CNN), ResNet18, GAN Detector |
| **Explainability** | Grad-CAM + LIME |
| **Platform** | Kaggle (TPU v5e-8 or GPU T4x2) |
| **Interface** | Flask web app exposed via ngrok |
| **Year** | 2026 |

---

## 🧠 Model Architecture

### 1. HSFAN ⭐ (Primary Model — Hybrid Spatial-Frequency Attention Network)
- MobileNetV2 backbone for spatial features
- Custom frequency branch using `rfft2` (TPU-safe FFT)
- Attention gating over combined spatial + frequency features
- Binary classifier (`BCEWithLogitsLoss`)

### 2. CNN — EfficientNet-B0
- Pretrained EfficientNet-B0 backbone (timm)
- Fine-tuned blocks 6 & 7 + custom classification head
- Multi-class classifier (`CrossEntropyLoss`)

### 3. ResNet18 — Transfer Learning
- Fully unfrozen ResNet18 with differential learning rates
- Label smoothing (`CrossEntropyLoss(label_smoothing=0.1)`)
- CosineAnnealingWarmRestarts scheduler + gradient clipping
- Stronger head: Linear → BN → ReLU → Dropout → Linear

### 4. GAN Detector — 3-Branch Detector
- **Spectral Branch**: FFT frequency analysis (TPU-safe `rfft2`)
- **Checkerboard Branch**: Detects GAN upsampling artifacts
- **Spatial Branch**: Standard spatial CNN
- Attention-weighted fusion of all three branches

---

## 📂 Dataset Structure

The project uses three datasets from Kaggle (`chethan200321/deepfake-datasets`):

```
/kaggle/input/datasets/chethan200321/deepfake-datasets/
├── dataset_A/dataset_A/     # GAN-generated fakes + real faces
│   ├── real/
│   └── fake/
├── dataset_B/dataset_B/     # Cross-domain GAN fakes
│   ├── real/
│   └── fake/
└── dataset_c/dataset_c/     # CIFAKE (Stable Diffusion fakes)
    ├── real/
    └── fake/
```

Each split uses up to:
- Dataset A: 20,000 images
- Dataset B: 7,000 images
- Dataset C: 15,000 images

---

## 🚀 Setup & Running (Kaggle)

### Prerequisites
- Kaggle account with the dataset `chethan200321/deepfake-datasets` attached
- Accelerator set to **TPU v5e-8** (preferred) or **GPU T4x2**

### Step-by-step

**Cell 1** — Install packages:
```python
# Run Cell 1 to install: timm, lime, scikit-image, flask, pyngrok
```

**Cell 2** — Set up project directories and symlink datasets.

**Cell 3** — Detect accelerator (TPU or GPU) and configure `Config`.

**Cells 4–7** — Define dataset class, model architectures, Grad-CAM, and LIME.

**Cells 8–9** — Train all 4 models (~25–40 min on TPU, ~1.5–2 hrs on GPU T4x2).

**Cells 10–12** — Run 3 research evaluation tests.

**Cell 13** — Save all `.pth` weights and JSON results to `/kaggle/working`.

**Cells 14–15** — Write Flask app files and launch the web UI via ngrok.

---

## 🔁 Reusing Pre-trained Weights (2nd+ Run)

After your first successful training:

1. Click **Save Version** → **Save & Run All**
2. Go to your notebook → **Output** tab
3. Click ⋮ on the output dataset → **Add to notebook as input**
4. Note the path (e.g., `/kaggle/input/deepfake-project/`)
5. Update `LOAD_DIR` in **Cell 13B**
6. On future runs: execute **Cells 1–7 → Cell 13B → Cells 14–15** (skip training)

---

## 📊 Research Experiments

### Test 1 — Cross-Dataset Generalization
- **Train on**: Dataset A (GAN fakes + real faces)
- **Test on**: Dataset B (cross-domain GAN fakes)
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC

### Test 2 — JPEG Compression Robustness
- Simulates WhatsApp / social media sharing pipeline
- Tests HSFAN at compression levels: 100%, 90%, 70%, 50%, 30%, 10%

### Test 3 — Diffusion Fakes vs GAN Fakes
- All 4 models evaluated on Dataset C (CIFAKE — Stable Diffusion images)
- Tests generalization from GAN-trained models to diffusion-generated fakes

---

## 🌐 Web Application

The Flask app provides:
- **Image upload** (JPG, PNG, BMP, WEBP — up to 16 MB)
- **Ensemble prediction** with per-model breakdown
- **Frequency score** for additional signal
- **Grad-CAM heatmap** (vivid JET colormap)
- **LIME superpixel explanation**
- Exposed publicly via ngrok

### Ensemble Weights
| Model | Weight |
|---|---|
| HSFAN | 35% |
| GAN Detector | 25% |
| ResNet18 | 15% |
| CNN (EfficientNet-B0) | 5% |
| Frequency Score | 20% |

Dynamic threshold: `0.32–0.45` based on image frequency score.

---

## 🗂️ Project File Structure

```
/kaggle/working/deepfake/
├── app.py                   # Flask web application
├── gradcam.py               # Grad-CAM implementation
├── lime_explainer.py        # LIME explainability
├── model/
│   ├── __init__.py
│   ├── hsfan.py             # HSFAN model
│   ├── cnn_model.py         # EfficientNet-B0 CNN
│   ├── resnet_model.py      # ResNet18
│   └── gan_detector.py      # 3-branch GAN detector
├── templates/
│   └── index.html           # Red & black web UI
├── static/
│   ├── uploads/             # Uploaded images
│   ├── heatmaps/            # Grad-CAM outputs
│   └── lime/                # LIME outputs
├── outputs/
│   ├── hsfan_model.pth
│   ├── cnn_model.pth
│   ├── resnet_model.pth
│   └── gan_model.pth
├── results.json             # Test 1 results
├── compression_results.json # Test 2 results
└── diffusion_results.json   # Test 3 results
```

---

## ⚙️ Training Configuration

| Parameter | TPU v5e-8 | GPU T4x2 |
|---|---|---|
| Batch Size | 128 | 64 |
| Epochs | 30 | 30 |
| Learning Rate | 1e-4 | 1e-4 |
| Image Size | 128×128 | 128×128 |
| Optimizer | Adam / AdamW | Adam / AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |

---

## 📝 Notes

- **TPU Safety**: All FFT operations use `torch.fft.rfft2().abs().float()` + bilinear interpolation instead of `fft2()`, which keeps complex dtype on XLA and breaks training.
- **pin_memory**: Disabled on TPU (only enabled for GPU dataloaders).
- **ngrok token**: Replace the `NGROK_TOKEN` in Cell 15 with your own token from [ngrok.com](https://ngrok.com).

---

## 👨‍💻 Author

M-Tech Final Year Project — Computer Science, 2026
