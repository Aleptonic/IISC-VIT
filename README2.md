![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# 🧑‍🔬 Vision Transformer (ViT) on CIFAR-10 & Text-Driven Segmentation with SAM 2

This repository contains implementations for the IISc internship task:

- **Q1 — Vision Transformer (ViT) on CIFAR-10** (PyTorch, Colab)  
- **Q2 — Text-Driven Image Segmentation with SAM 2**

Both tasks are fully reproducible in **Google Colab** with GPU runtime enabled.

---

## 🚀 Q1 — Vision Transformer on CIFAR-10

> A from-scratch PyTorch implementation of the Vision Transformer (ViT), trained on CIFAR-10 using modern training recipes (inspired by *Dosovitskiy et al., 2021* and *Touvron et al., 2021*).

### 📊 Final Results

| Model  | Epochs | Test Accuracy (%) |
| ------ | :----: | :----------------: |
| ViT    | 100    | 79.3               |
| ViT    | 252/300 (Colab timeout) | **90.9** |

<p align="center">
  <img src="https://github.com/user-attachments/assets/8285d148-a496-46d1-a13c-8b2ae5cd9080" alt="Confusion Matrix" width="400" height="400" />
</p>

---

### ▶️ How to Run
1. Open `q1.ipynb` in **Google Colab**.  
2. Set runtime to **GPU** (`Runtime > Change runtime type`).  
3. Run all cells sequentially:  
   - Dataset download & preprocessing  
   - ViT model construction  
   - Training & checkpoint saving  
   - Final evaluation + plots  

---

### ⚙️ Best Model Configuration (DeiT-Ti Inspired)

| Category          | Setting |
| ----------------- | ------- |
| **Architecture**  | ViT (Pre-Norm) |
| Patch Size        | 4×4 |
| Embedding Dim     | 192 |
| Transformer Depth | 12 |
| Attention Heads   | 3 |
| **Training**      | 300 epochs, Batch size 1024 |
| Optimizer         | AdamW |
| LR Scheduler      | OneCycleLR (warmup + cosine) |
| Weight Decay      | 0.05 |
| Augmentations     | RandAugment, CutMix, Mixup, RandomCrop, RandomFlip |
| Dropout           | 0.1 (MLP & Embedding) |
| Parameters        | ~5M |

---

### 🏗️ Methodology

- **Patch Embedding**: CIFAR-10 images (3×32×32) → 64 patches (4×4) → linear projection → embeddings.  
- **[CLS] Token + Positional Embeddings**: Learnable tokens providing spatial awareness.  
- **Transformer Encoder**: 12 stacked encoder blocks (MHSA + MLP + residuals, pre-norm).  

<p align="center">
  <img src="https://github.com/user-attachments/assets/541d1a71-4a60-408c-ae6c-d742866bd833" alt="ViT Architecture" width="500" />
</p>
<p align="center"><sub><i>Vision Transformer (ViT) architecture, adapted from Dosovitskiy et al., 2021.</i></sub></p>

---

### 🔍 Analysis & Insights

1. **Architecture scaling**: Using a small DeiT-Ti (~5M params) prevented overfitting and fit Colab runtime constraints.  
2. **Aggressive regularization**: RandAugment + CutMix + Mixup boosted generalization significantly.  
3. **Training vs Validation accuracy gap**: Training accuracy < Validation accuracy, caused by strong augmentation. This indicates regularization worked correctly.  
4. **Schedulers**: OneCycleLR was critical for stable convergence.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/7eff5f2c-27e8-4b53-a686-6285c5f05ab6" alt="Training Curve" width="600" height="600" />
</p>

---

### 🧪 Ablation Study (Bonus)

| Variant                  | Test Accuracy (%) |
| ------------------------- | :---------------: |
| ViT (no Mixup/CutMix)    | 83.2 |
| ViT + Mixup only         | 86.5 |
| ViT + CutMix only        | 87.4 |
| ViT + Mixup + CutMix     | **90.9** |

**Observation**: The synergy of Mixup + CutMix yields the strongest improvement, validating the need for aggressive regularization in training ViTs on CIFAR-10.  

---

## 🖼️ Q2 — Text-Driven Image Segmentation with SAM 2

> Composed **GroundingDINO** (open-set detector) with **SAM 2** (segmentation) in a *Finder–Painter* pipeline.

### 🔗 Pipeline Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/c09eb155-8308-493b-b60b-5f704eb7731d" alt="Pipeline Diagram" width="400" height="400" />
</p>

1. **Finder (GroundingDINO)**: Detects bounding boxes for text-prompted objects.  
2. **Painter (SAM 2)**: Refines bounding boxes into pixel-perfect segmentation masks.  
3. **Visualization**: Overlays results on the input image.  

---

### 📌 Example

Prompt: `"skateboard, cap"`

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/11dbfb4f-364d-4903-8c25-e7fe905544a0" width="350" height="350" /><br>
        <sub><b>Input</b></sub>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/b612004e-cae7-4e8a-8fce-a87f68e90866" width="350" height="350" /><br>
        <sub><b>Output</b></sub>
      </td>
    </tr>
  </table>
</p>

---

### ▶️ How to Run
1. Open `q2.ipynb` in **Google Colab**.  
2. Enable GPU runtime.  
3. Run all cells (first cells install dependencies).  
4. Use the interactive UI: upload an image, enter a prompt, run segmentation.  

---

### ⚠️ Limitations
- Dependent on **GroundingDINO** detection accuracy.  
- Ambiguous prompts may yield multiple/incorrect detections.  
- SAM 2 itself has no semantic grounding: it segments whatever is inside the given box.  

---

## 📚 References

### Papers
1. Dosovitskiy et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*  
2. Touvron et al. (2021). *Training Data-Efficient Image Transformers (DeiT)*  
3. Steiner et al. (2022). *How to Train Your ViT?*  
4. Chen et al. (2022). *Better Plain ViT Baselines*  

### Repositories
- [google-research/vision_transformer](https://github.com/google-research/vision_transformer)  
- [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)  
- [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)  
- [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)  
