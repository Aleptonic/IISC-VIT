# Vision Transformer (ViT) on CIFAR-10

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> This project presents a from-scratch implementation of the Vision Transformer (ViT) architecture in PyTorch, trained on the CIFAR-10 dataset. The primary objective was to achieve the highest possible test accuracy by leveraging state-of-the-art training and regularization techniques from recent research, notably the two papers "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE" and  "Training data-efficient image transformers" were the backbone for this implementation 

## Final Results

The model was trained for 300 epochs, and the best-performing checkpoint was evaluated on the held-out test set.

| Model                                | Epochs | Test Accuracy |
| ------------------------------------ |               :--------:                           |:--------:|
| ViT (DeiT-Ti Inspired Configuration) | 100                                      | 79.3 |
| ViT (DeiT-Ti Inspired Configuration) | 252/300 (training time expired on colab) | 90.9 |

Below is the confusion matrix from the final evaluation on the 10,000 test images.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/8285d148-a496-46d1-a13c-8b2ae5cd9080" />


## How to Run

This project is designed to be run in a Google Colab environment.

1.  Open the `q1.ipynb` notebook in Google Colab.
2.  Ensure the runtime is set to a GPU instance (e.g., T4) via `Runtime` > `Change runtime type`.
3.  Run all cells from top to bottom. The notebook will automatically:
    * Download and prepare the CIFAR-10 dataset.
    * Build the ViT model and the training harness.
    * Train the model for the specified number of epochs, saving the best checkpoint.
    * Load the best checkpoint and run a final evaluation, printing metrics and generating plots.

## Best Model Configuration

The best results were achieved using a model architecture and training recipe inspired by the **DeiT-Ti (Tiny)** variant.

| Parameter            | Value                                         |
| -------------------- | --------------------------------------------- |
| **Architecture** | Vision Transformer (Pre-Norm)                 |
| Patch Size           | 4x4                                           |
| Embedding Dimension  | 192                                           |
| Transformer Depth    | 12 Layers                                     |
| Attention Heads      | 3                                             |
| **Optimizer** | AdamW                                         |
| Learning Rate        | 0.001 (Linearly scaled: `5e-4 * batch_size/512`)     |
| LR Scheduler         | OneCycleLR (Warmup + Cosine Annealing)        |
| **Training** |                                               |
| Epochs               | 300                                           |
| Batch Size           | 1024                                          |
| Weight Decay         | 0.05                                          |
| Label Smoothing      | N/A (CrossEntropyLoss)                        |
| **Regularization** |                                               |
| Augmentations        | RandAugment, RandomHorizontalFlip, RandomCrop |
| Batch-Level Augs     | Mixup & CutMix (`combine_fn`)                   |
| Dropout Rates        | MLP: 0.1, Embedding: 0.1, Attention: 0.0      |

## Methodology & Implementation

### 1. Architecture

The model is a standard Vision Transformer as described in "An Image is Worth 16x16 Words", with a Pre-Norm configuration (LayerNorm applied *before* the attention/MLP blocks) for improved training stability.

* **PatchEmbedding:** Images of size `(3, 32, 32)` are converted into a sequence of 64 flattened patches (`4x4`), which are then linearly projected into a 192-dimensional embedding space.
* **CLS Token & Positional Embeddings:** A learnable `[CLS]` token is prepended to the sequence, and learnable positional embeddings are added to provide the model with spatial information.
* **Transformer Encoder:** The core of the model is a stack of 12 standard Transformer Encoder blocks, each containing Multi-Head Self-Attention and an MLP sub-layer.

*Diagram showing the ViT architecture (from Dosovitskiy et al., 2021).*
![ViT Architecture](https://i.imgur.com/g21B5zW.png)
*(Source: "An Image is Worth 16x16 Words" paper)*

### 2. Training Strategy

The key challenge with ViTs is their data-hungriness. To overcome this on a small dataset like CIFAR-10, a sophisticated training recipe inspired by the DeiT paper was adopted. The core of this strategy is aggressive regularization to prevent overfitting.
* **Heavy Data Augmentation:** The training pipeline uses `RandAugment` in conjunction with `Mixup` and `CutMix` (applied at the batch level via a custom `collate_fn`). This forces the model to learn robust and generalizable features.
* **Optimizer & Scheduler:** The `AdamW` optimizer was used with a `OneCycleLR` scheduler, which automatically handles a learning rate warmup phase followed by a cosine decay. This disciplined LR schedule is crucial for stable and effective training.

## Analysis & Key Learnings (Bonus)

This project was a practical exploration of the challenges and solutions for training Vision Transformers on smaller datasets.

1.  **The Power of Regularization:** The most significant finding was the profound impact of the DeiT-style training recipe. A key indicator of its effectiveness can be seen in the training logs, where the **training accuracy was consistently lower than the validation accuracy**. This counter-intuitive result demonstrates that the strong augmentations (Mixup/CutMix) made the training task exceptionally difficult, which in turn acted as a powerful regularizer, forcing the model to generalize better to the clean validation data.

    [TODO: INSERT YOUR TRAINING/VALIDATION CURVE PLOT HERE. It will visually show the validation accuracy above the training accuracy.]

2.  **Architecture Matters:** An initial consideration was the model size. By choosing a smaller, more efficient **DeiT-Ti** architecture (192-dim, 12 layers) instead of a larger ViT-Base, the model had fewer parameters (~5M), making it more suitable for the limited size of the CIFAR-10 dataset and reducing the risk of overfitting.

3.  **Implementation Journey:** The process involved building each component (Patching, Attention, MLP, Encoder) from scratch. A key correction during development was fixing the **positional embedding** implementation to ensure it had the correct shape (`1, num_patches + 1, D`) and was added correctly to the token sequence, a crucial step for the model to learn spatial relationships.

> replace the above two sections methods and analysis witht the below two
## The Development Journey: From Baseline to High-Performance ViT

Training a Vision Transformer from scratch on a small dataset like CIFAR-10 is a known challenge. The original ViT paper demonstrated that these models require massive datasets to outperform CNNs. This project's core challenge was to overcome this limitation by building a robust pipeline that incorporates a suite of modern, data-efficient training techniques.

The following sections detail the step-by-step implementation, highlighting the key decisions and optimizations made at each stage.

### Stage 1: Data & Augmentation Pipeline

The foundation of any deep learning project is a robust data pipeline. For a ViT, this is even more critical, as aggressive data augmentation serves as the primary regularizer.

* **Initial Approach:** A standard data loader with basic transformations (`ToTensor`, `Normalize`).
* **Key Improvement:** To make the model generalize better, a sophisticated augmentation strategy inspired by the DeiT paper was implemented.
    * **In-Transform Augmentations:** `RandomHorizontalFlip` and `RandomResizedCrop` were added to provide basic geometric variance. The key addition was **`RandAugment`**, a powerful technique that automatically applies a random set of strong augmentations, significantly increasing the diversity of the training data.
    * **Batch-Level Augmentations:** Standard transforms are applied per-image. To further regularize the model, **`Mixup` and `CutMix`** were introduced. These methods combine multiple images and their labels within a batch, forcing the model to learn from interpolated and partially occluded examples. This was efficiently implemented using a custom `collate_fn` in the `DataLoader`, which randomly applies one of the two techniques to each training batch.
    * **Validation Integrity:** A crucial detail was ensuring that these aggressive augmentations were **only** applied to the training set. The validation and test sets use only basic normalization to provide a consistent and unbiased measure of the model's true performance.

### Stage 2: Model Architecture & Implementation

The goal was to build a standard ViT from scratch, with a focus on clean, modular code and stability.

* **Core Components:** The architecture was broken down into logical `nn.Module` classes: `PatchEmbedding`, `MultiHeadAttention`, `MLP`, and `TransformerEncoderBlock`.
* **Efficient Patching:** Instead of manually slicing and embedding image patches, a more efficient method was used: a single `nn.Conv2d` layer. With a `kernel_size` and `stride` equal to the `patch_size`, this layer performs both the patching and the linear projection into the embedding dimension in one highly optimized operation.
* **Stable Architecture (Pre-Norm):** While the original Transformer used a "Post-Norm" structure (LayerNorm *after* the residual connection), this project adopted the more modern and stable **"Pre-Norm"** configuration. Applying `LayerNorm` *before* the Multi-Head Attention and MLP blocks leads to smoother gradients and more stable training, which is particularly beneficial when training from scratch.
* **Model Sizing:** Recognizing the risk of overfitting on CIFAR-10, a smaller model variant was chosen. The architecture follows the **DeiT-Ti (Tiny)** specification (`dim=192`, `depth=12`, `heads=3`), providing a better capacity-to-data ratio.
* **Positional Information:** A critical debugging step involved correcting the shape and application of the learnable **positional embeddings**. Ensuring the embedding tensor had the correct length (`num_patches + 1`) and was correctly added to the sequence of patch and CLS tokens was essential for the model to learn spatial relationships.

### Stage 3: Training & Optimization Strategy

With the data and model in place, the final stage was to design a training strategy that would allow the model to converge effectively.

* **Optimizer:** The **`AdamW`** optimizer was chosen over vanilla `Adam`. AdamW decouples the weight decay from the gradient updates, providing more effective regularization, which is a standard practice for training Transformers.
* **Learning Rate Scheduling:** A fixed learning rate is suboptimal for complex models. The **`OneCycleLR`** scheduler was implemented. This powerful scheduler automatically manages a two-phase cycle:
    1.  A **warmup phase**, where the learning rate gradually increases. This prevents large, destabilizing updates at the start of training when the model weights are random.
    2.  An **annealing phase**, where the learning rate smoothly decays. This allows the model to settle into a deep and stable minimum in the loss landscape.
    This disciplined approach to the learning rate was a key factor in the model's successful convergence.
* **Efficient Training:** To accelerate the process, training was performed using **Automatic Mixed Precision (AMP)** via `torch.cuda.amp.autocast` and `GradScaler`. This allows the GPU to use faster FP16 computations for certain operations without sacrificing model accuracy. This optimization significantly reduced the training time per epoch from an estimated ~11 hours to just **1.5 hours for 100 epochs**.

### Analysis: Key to High Performance without Pre-training

The final test accuracy of **[TODO: INSERT FINAL ACCURACY]** demonstrates that Vision Transformers can indeed be trained effectively on smaller datasets if the right strategy is employed. The key takeaways are:

1.  **Regularization is Paramount:** The success of this project hinges on the aggressive regularization strategy borrowed from DeiT. The combination of `RandAugment`, `Mixup`, `CutMix`, and `AdamW`'s weight decay successfully prevented the model from overfitting, a primary risk for ViTs.
2.  **Training Dynamics as a Signal:** A key observation was that the **training accuracy was consistently lower than the validation accuracy**. This counter-intuitive result is a direct consequence of the regularization. The training task is made artificially difficult by the augmentations, which forces the model to learn more robust features that generalize better to the "clean" validation data. This is a powerful indicator that the regularization is working as intended.
3.  **Modern Schedulers are Non-Negotiable:** The stability and performance of the training run were heavily reliant on the `OneCycleLR` scheduler. Without its intelligent management of the learning rate, the model would likely have converged much slower or to a less optimal result.

This project validates that with a thoughtful, modern approach to data augmentation and training dynamics, the performance gap for Vision Transformers on smaller datasets can be significantly closed, reducing the dependency on massive pre-training corpora.




## Acknowledgements

This implementation was made possible by studying the following seminal papers and high-quality open-source repositories.

### Papers
1.  Dosovitskiy et al. (2021). AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE.
2.  Touvron et al. (2021). Training data-efficient image transformers & distillation through attention (DeiT).
3.  Steiner et al. (2022). How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers.
4.  Chen et al. (2022). Better plain ViT baselines for ImageNet-1k.

### Repositories
1.  Official Google Research ViT: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
2.  PyTorch Image Models (timm) by Ross Wightman: [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
3.  Phil Wang's (lucidrains) ViT implementation: [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
4.  Jeon's ViT implementation: [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
