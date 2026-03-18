# AlexNet  Paper Implementation

> Personal implementation of **"ImageNet Classification with Deep Convolutional Neural Networks"** (Krizhevsky et al., 2012) in PyTorch, with notes.

---

##  Paper

**Title:** ImageNet Classification with Deep Convolutional Neural Networks  
**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
**Link:** [proceedings.neurips.cc/paper/2012](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

---

##  Repo Structure

```
AlexNet-implementation/
├── alexnet.py      # PyTorch implementation
└── AlexNet.md      # Personal notes (paper summary, architecture breakdown, insights)
```

---

##  What's in the Notes (`AlexNet.md`)

- **Paper Summary** — Why AlexNet was a landmark: scale, GPU training, and top-5 error on ImageNet
- **Architecture Breakdown** — 5 conv layers + 3 FC layers, ReLU, LRN, overlapping max-pooling, dropout
- **My Insights** — Personal observations while reading and implementing the paper

---

##  Key Ideas

AlexNet introduced several techniques that became standard in deep learning:

| Technique | Significance |
|---|---|
| ReLU activations | Faster training vs. tanh/sigmoid |
| Dropout | Regularization in FC layers |
| Data augmentation | Random crops, horizontal flips |
| GPU training | Trained on 2× GTX 580s in parallel |

---

## Implementation

**Framework:** PyTorch

```bash
# Clone the repo
git clone https://github.com/sharpeye08/AlexNet-implementation.git
cd AlexNet-implementation

# Run
python alexnet.py
```

---

## Notes

This repo is primarily for learning purposes — the code follows the paper closely and is meant to be readable, not optimized for production.
