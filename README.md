# CustomPetNet: Fine-Grained 37-Breed Pet Classification

This repository contains the implementation of a custom neural network designed to tackle a fine-grained image classification challenge on the Oxford-IIIT Pet dataset. The model identifies 37 distinct categories of cat and dog breeds, built entirely from scratch under strict computational constraints. 

Developed as part of the MEng Computer Science coursework for the Intelligent Systems: Machine Learning and Optimisation module at the University of York.

## 🎯 The Challenge & Constraints
Standard vision pipelines rely heavily on transfer learning and extensive training epochs. This project was developed under two absolute constraints:
* **No Pre-training:** Pre-trained weights (like ImageNet) were strictly forbidden; all weights were randomly initialized and learned purely from the target dataset.
* **Strict 30-Epoch Budget:** Training was capped at a maximum of 30 epochs, requiring highly optimized architectural and scheduling choices to ensure rapid convergence.

## 🧠 Architecture: Custom ResNet-9
To balance model expressiveness against hardware limits and the 30-epoch ceiling, this project implements a streamlined, custom ResNet-9 architecture (`model.py`). 
* **Lean Parameter Count:** Operates with just ~6.59 million parameters, avoiding the sluggish gradient flow and underfitting typical of 50-layer networks trained from scratch in short windows.
* **Front-End Prep Block:** Immediately downsamples inputs from 128x128 to 64x64, reducing the memory footprint and computational load by 75% at the entrance.
* **Residual Connections:** Features skip connections across targeted convolutional stages to maintain direct gradient flow and extract hierarchical visual representations.
* **Classifier Head:** Utilizes `AdaptiveMaxPool2d` to collapse spatial features, followed by dropout (0.2) for regularization, mapping directly to 37 class logits.

## ⚡ Training Pipeline & Optimization
Every aspect of the training script (`train.py`) was tuned for early stability and rapid feature acquisition:
* **Super-Convergence:** Utilizes Leslie Smith's OneCycleLR schedule paired with an AdamW optimizer (learning rate of 5e-3, weight decay of 1e-4). This allows the model to explore wide loss basins early before smoothly annealing to an optimal local minimum.
* **Early Stability:** All convolutional layers are initialized with Kaiming (He) Normal initialization, and gradient clipping (max norm 1.0) is applied to prevent exploding activations during the high learning rate phase.
* **Data Augmentation:** The input pipeline (`dataset.py`) mitigates overfitting on the small subset (~90 images per class) using `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation(10)`, and `ColorJitter(0.1)`. 
* **Strict Isolation:** Validation and test splits are kept strictly clean from augmentations and data leakage, isolated via a fixed random seed (seed=42).

## 📊 Results
Despite the complexity of distinguishing fine-grained morphological cues between highly similar breeds without a pre-existing visual feature bank, the model achieved significant learning in just 30 epochs.

* **Total Parameters:** 6,586,981
* **Training Accuracy:** 95.92%
* **Best Validation Accuracy:** 59.24%
* **Final Test Accuracy:** 56.85%

> **Note on Benchmark Performance:** Random guessing across 37 classes yields a baseline accuracy of 2.70%. Reaching 56.85% on completely unseen test images constitutes a **21x improvement** over random chance.

## 🚀 Future Roadmap
If unconstrained by compute and time limits, planned extensions for this architecture include:
* Integrating advanced regularization techniques like CutMix, MixUp, and Label Smoothing (0.1) to reduce memorization.
* Implementing progressive resizing—starting training at 128x128 and finishing at 224x224 to capture micro-level coat textures and facial geometries.

## 👨‍💻 Author
**Arnav Jamidar**
* [GitHub Profile](https://github.com/ajamidar)
