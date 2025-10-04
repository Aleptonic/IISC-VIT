![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Vision Transformer (ViT) on CIFAR-10



> This project presents a from-scratch implementation of the Vision Transformer (ViT) architecture in PyTorch, trained on the CIFAR-10 dataset. The primary objective was to achieve the highest possible test accuracy by leveraging state-of-the-art training and regularization techniques from recent research, notably the two papers "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE" and  "Training data-efficient image transformers" were the backbone for this implementation 

## Final Results

The model was trained for 300 epochs, and the best-performing checkpoint was evaluated on the held-out test set.

| Model                                | Epochs | Test Accuracy |
| ------------------------------------ |               :--------:                           |:--------:|
| ViT  | 100                                      | 79.3 |
| ViT  | 252/300 (training time expired on colab) | 90.9 |

Below is the confusion matrix from the final evaluation on the 10,000 test images.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8285d148-a496-46d1-a13c-8b2ae5cd9080" alt="image" width="400" height="400" />
</p>


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
| Trainable Parameters        | ~5M |

## Methodology & Implementation

### 1. Architecture

The model is a standard Vision Transformer as described in "An Image is Worth 16x16 Words", with a Pre-Norm configuration (LayerNorm applied *before* the attention/MLP blocks) for improved training stability.

* **PatchEmbedding:** Images of size `(3, 32, 32)` are converted into a sequence of 64 flattened patches (`4x4`), which are then linearly projected into a 192-dimensional embedding space.
* **CLS Token & Positional Embeddings:** A learnable `[CLS]` token is prepended to the sequence, and learnable positional embeddings are added to provide the model with spatial information.

* **Transformer Encoder:** The core of the model is a stack of 12 standard Transformer Encoder blocks, each containing Multi-Head Self-Attention and an MLP sub-layer.

<p align="center">
  <img src="https://github.com/user-attachments/assets/541d1a71-4a60-408c-ae6c-d742866bd833" alt="ViT Architecture" width="500" />
</p>

<p align="center">
  <sub><i>Diagram of the Vision Transformer (ViT) architecture, adapted from Dosovitskiy et al., 2021.</i></sub><br>
  <sub><i>Source: <b>"An Image is Worth 16x16 Words"</b> paper</i></sub>
</p>

### 2. Training Strategy

The key challenge with ViTs is their data-hungriness. To overcome this on a small dataset like CIFAR-10, a sophisticated training recipe inspired by the DeiT paper was adopted. The core of this strategy is aggressive regularization to prevent overfitting.
* **Heavy Data Augmentation:** The training pipeline uses `RandAugment` in conjunction with `Mixup` and `CutMix` (applied at the batch level via a custom `collate_fn`). This forces the model to learn robust and generalizable features.
* **Optimizer & Scheduler:** The `AdamW` optimizer was used with a `OneCycleLR` scheduler, which automatically handles a learning rate warmup phase followed by a cosine decay. This disciplined LR schedule is crucial for stable and effective training.


### Analysis: Key to High Performance without Pre-training

The final test accuracy of **90.9%** demonstrates that Vision Transformers can indeed be trained effectively on smaller datasets if the right strategy is employed. The key takeaways are:

1. **Architecture Matters:** An initial consideration was the model size. By choosing a smaller, more efficient **DeiT-Ti** architecture `(192-dim, 12 layers)` instead of a larger ViT-Base, the model had fewer parameters `(~5M)`, making it more suitable for the limited size of the CIFAR-10 dataset and reducing the risk of overfitting.
2.  **Regularization is Paramount:** The success of this project hinges on the aggressive regularization strategy borrowed from DeiT. The combination of `RandAugment`, `Mixup`, `CutMix`, and `AdamW`'s weight decay successfully prevented the model from overfitting, a primary risk for ViTs.
3.  **Training Dynamics as a Signal:** A key observation was that the **training accuracy was consistently lower than the validation accuracy**. This counter-intuitive result is a direct consequence of the regularization. The training task is made artificially difficult by the augmentations, which forces the model to learn more robust features that generalize better to the "clean" validation data. This is a powerful indicator that the regularization is working as intended.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7eff5f2c-27e8-4b53-a686-6285c5f05ab6" alt="image" width="600" height="600" />
</p>

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

# Text-Driven Image Segmentation with SAM 2

> This project implements a pipeline to perform segmentation on an image using an arbitrary text prompt. It demonstrates a powerful modern AI technique of composing two different state-of-the-art models, GroundingDINO and the Segment Anything Model (SAM 2), to achieve a task that neither can perform alone.

#### Pipeline Overview


<p align="center">
  <img src="https://github.com/user-attachments/assets/c09eb155-8308-493b-b60b-5f704eb7731d" alt="image" width="400" height="400" />
</p>

The system operates on a "Finder-Painter" principle:

1.  **Input:** The user provides an image and a free-form text prompt (e.g., "a person on a surfboard").
2.  **The "Finder" (GroundingDINO):** The text prompt is fed to the GroundingDINO model, an open-set object detector. It identifies and draws bounding boxes around the object(s) described in the text.
3.  **The "Painter" (SAM 2):** The original image and the bounding boxes from GroundingDINO are then passed to the Segment Anything Model. The boxes act as spatial prompts, guiding SAM to generate precise, pixel-perfect segmentation masks for the detected objects.
4.  **Visualization:** The final output overlays the generated masks and bounding boxes onto the original image, clearly highlighting the segmented object.



#### Example Result

Here is an example of the pipeline segmenting **"a skateboard"** and **"a cap"** from an image.

- **Text Prompt:**  
  `"skateboard, cap"`

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/11dbfb4f-364d-4903-8c25-e7fe905544a0" alt="Input image" width="400" height="400" style="border-radius:10px;" /><br>
        <sub><b>Input Image</b></sub>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/b612004e-cae7-4e8a-8fce-a87f68e90866" alt="Final output" width="400" height="400" style="border-radius:10px;" /><br>
        <sub><b>Final Output</b></sub>
      </td>
    </tr>
  </table>
</p>


#### How to Run

1.  Open the `q2_dino_sam.ipynb` notebook in Google Colab.
2.  Ensure the runtime is set to a **GPU** instance (`Runtime` > `Change runtime type`).
3.  Run all cells from top to bottom. The initial cells will handle all installations and model downloads.
4.  The final cell launches an interactive UI. Use the "Upload Image" button to select a local image, enter your desired object in the "Prompt" text box, and click "Run Segmentation".

#### Limitations

* **Dependency on Detection:** The final mask quality is entirely dependent on the initial bounding box from GroundingDINO. If the "Finder" fails to locate the object, the "Painter" has nothing to segment.
* **Ambiguous Prompts:** Vague text prompts (e.g., "the person") in a crowded scene can lead to multiple or incorrect detections, resulting in poor segmentation.
* **No Semantic Link in SAM:** SAM segments whatever is in the prompt box, it does not understand the text itself. If GroundingDINO incorrectly boxes a car when prompted for a "dog", SAM will diligently segment the car.
