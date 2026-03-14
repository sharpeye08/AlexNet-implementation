# AlexNet — Paper Notes

> **Tags:** Deep Learning · CNN · Research Paper  
> **Date:** 2026-03-14

AlexNet is the landmark architecture that sparked the modern deep learning era for computer vision, directly inspiring later networks like VGG16. It was trained on the ImageNet dataset and introduced several ideas that became standard practice.

---

## Architecture Overview

AlexNet contains **8 learned layers** — 5 convolutional and 3 fully connected.

| Layer | Details |
|-------|---------|
| Conv 1 | 96 kernels of size `11×11×3`, stride 4, input `224×224×3` |
| Conv 2 | 256 kernels of size `5×5×48` |
| Conv 3 | 384 kernels of size `3×3×256` |
| Conv 4 | 384 kernels of size `3×3×192` |
| Conv 5 | 256 kernels of size `3×3×192` |
| FC 1–2 | 4096 neurons each |
| FC 3 | 1000-way softmax output |

**Key structural notes:**
- Response normalization layers follow Conv 1 and Conv 2.
- Max pooling follows both normalization layers and Conv 5.
- Conv 3, 4, and 5 are connected directly with no intervening pooling or normalization.
- ReLU is applied after every convolutional and fully connected layer.
- Kernels in Conv 2, 4, and 5 connect only to kernel maps on the same GPU (due to the dual-GPU training setup).

---

## Core Design Choices

### ReLU Activation
ReLU was chosen over sigmoid and tanh because it trains several times faster on large models and large datasets. It also does not require input normalization to avoid saturation — as long as some training examples produce a positive input, the neuron will learn.

### Overlapping Pooling
Traditional pooling uses non-overlapping windows (`s = z`). AlexNet uses **overlapping pooling** with stride `s = 2` and window size `z = 3` (`s < z`). This makes the model slightly harder to overfit during training.

---

## Reducing Overfitting

### 1. Data Augmentation
- Random `224×224` patches (and horizontal reflections) are extracted from `256×256` images, enlarging the training set by a factor of **2048**.
- At test time, predictions are averaged over 10 patches: the four corners + center, plus their horizontal reflections.
- A second augmentation method applies PCA-based perturbations to the RGB channel intensities of training images.

### 2. Dropout
- With probability **0.5**, each hidden neuron's output is set to zero during training.
- Dropped neurons do not contribute to the forward pass or backpropagation.
- At test time, all neurons are used but their outputs are multiplied by 0.5.

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | SGD |
| Batch size | 128 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Initial learning rate | 0.01 (reduced 3× before termination) |
| Training duration | ~90 epochs over 1.2M images |

**Weight initialization:**
- Weights: zero-mean Gaussian with σ = 0.01
- Biases in Conv 2, 4, 5 and all FC layers: initialized to **1**
- Biases in remaining layers: initialized to **0**

> Note: Weight decay serves as more than a regularizer — it also reduces training error directly.

---

## References
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS.
