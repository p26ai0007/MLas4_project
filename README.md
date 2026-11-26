\# MedLiT-seed on PathMNIST: Adversarial Robustness \& Efficiency



\## Setup \& Installation



Install the required dependencies:



```bash

pip install torch torchvision

For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install medmnist scikit-learn matplotlib seaborn tqdm pillow numpy fvcore

```



\## How to Run



Each version folder contains a `run\_notebook.sh` script. First, make sure each folder has the `pathmnist.npz` file in it, and then run `run\_notebook.sh`:



```bash

bash run\_notebook.sh

```



\---



\## Dataset



\- \*\*Source:\*\* \[PathMNIST Colon Pathology Dataset (Kaggle)](https://www.kaggle.com/datasets/dongquan/pathmnist-colon-pathology-dataset)

\- \*\*Task:\*\* Multi-Class Classification (9 distinct tissue types).

\- \*\*Classes:\*\* 0 - Adipose, 1 - Background, 2 - Debris, 3 - Lymphocytes, 4 - Mucus, 5 - Smooth muscle, 6 - Normal colon mucosa, 7 - Cancer-associated stroma, 8 - Colorectal adenocarcinoma epithelium.

\- \*\*Format:\*\* The `.npz` file contains RGB images standardized to 28x28 pixels (resized to 224x224 in the training pipeline).

\- \*\*Volume:\*\* 107,180 total samples split into Training (89,996), Validation (10,004), and Test (7,180).



\---



\## Version Overview



Evaluation of \*\*MedLiT-seed\*\* (Sparse Top-2 MoE + Grouped Query Attention + SwiGLU + Hard-Masked Focal Loss) under \*\*Standard (Clean)\*\* training and \*\*FGSM Adversarial Training\*\* ($\\epsilon = 0.05$, 50% clean + 50% adversarial loss, 100 epochs).



| Version | Description | Key Changes |

| :--- | :--- | :--- |

| \*\*v1\*\* | \*\*Baseline adversarial training\*\* | Raw training from scratch (`lr=1e-3`, `MultiStepLR`, batch size 512, no data augmentation, zero-init ViT embeddings). |

| \*\*v2\*\* | \*\*Stabilized training\*\* | Added ViT weight initialization (`trunc\_normal\_`/`xavier\_uniform\_`), histology augmentations (flips, rotation, color jitter), `AdamW` (`lr=5e-4`, `wd=1e-2`), `CosineAnnealingLR`, batch size 256, and fixed per-layer MoE loss scaling. |

| \*\*v3\*\* | \*\*Mixture-of-Depth routing\*\* | Built on \*\*v2\*\* with an `MoDRouter` (`capacity\_factor=0.5`) in each block, routing only the top 50% of tokens through Attention + MoE while bypassing the rest via residual connections. |



\---



\## 1. Clean Test Set Performance (7,180 samples)



| Version | Training Mode | Val Acc | Test Acc (OA) | AUC | Precision (Macro) | Recall (Macro) | Specificity (Macro) | F1 (Macro) |

| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| \*\*v1\*\* | Standard | 71.48% | 68.19% | 0.9311 | 61.23% | 60.04% | 96.03% | 59.86% |

| \*\*v1\*\* | Adversarial (AT) | 65.56% | 65.65% | 0.9069 | 59.51% | 58.47% | 95.68% | 58.41% |

| \*\*v2\*\* | Standard | 89.85% | \*\*83.25%\*\* | \*\*0.9747\*\* | \*\*78.26%\*\* | \*\*78.13%\*\* | \*\*97.93%\*\* | \*\*77.85%\*\* |

| \*\*v2\*\* | Adversarial (AT) | 80.65% | 78.68% | 0.9696 | 74.47% | 74.67% | 97.34% | 73.77% |

| \*\*v3\*\* | Standard (MoD) | \*\*93.14%\*\* | 79.57% | 0.9719 | 75.61% | 76.01% | 97.46% | 75.12% |

| \*\*v3\*\* | Adversarial (MoD + AT) | 79.01% | 80.63% | 0.9688 | 74.32% | 74.38% | 97.60% | 73.94% |



\---



\## 2. FGSM Adversarial Robustness (Test Accuracy % vs. $\\epsilon$)



| Version | Mode | $\\epsilon=0.0$ | $\\epsilon=0.01$ | $\\epsilon=0.03$ | $\\epsilon=0.05$ | $\\epsilon=0.10$ | $\\epsilon=0.15$ | $\\epsilon=0.20$ |

| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| \*\*v1\*\* | Standard | 68.19 | 43.12 | 33.66 | 29.86 | 25.49 | 22.66 | 22.02 |

| \*\*v1\*\* | AT | 65.65 | 58.73 | 57.53 | 56.36 | 45.21 | 37.10 | 27.97 |

| \*\*v2\*\* | Standard | \*\*83.25\*\* | 44.72 | 39.82 | 40.24 | 41.77 | 41.50 | 40.57 |

| \*\*v2\*\* | AT | 78.68 | \*\*61.13\*\* | \*\*61.14\*\* | \*\*71.03\*\* | \*\*61.43\*\* | \*\*53.15\*\* | \*\*47.45\*\* |

| \*\*v3\*\* | Standard (MoD) | 79.57 | 46.52 | 46.06 | 46.70 | 47.08 | 46.56 | 46.96 |

| \*\*v3\*\* | AT (MoD) | 80.63 | 59.04 | 59.37 | 68.22 | 57.99 | 48.72 | 37.33 |



\---



\## 3. Computational Cost (Inference FLOPs \& MACs)



| Version | Architecture | FLOPs / Image | MACs / Image | Total Test FLOPs (7,180 imgs) | Total Test MACs |

| :--- | :--- | :---: | :---: | :---: | :---: |

| \*\*v1 / v2\*\* | MedLiT-seed (Full Tokens) | 0.5610 GFLOPs | 0.2805 GMACs | 4028.0727 GFLOPs | 2014.0364 GMACs |

| \*\*v3\*\* | MedLiT-seed + MoD ($k=0.5$) | \*\*0.2581 GFLOPs\*\* | \*\*0.1290 GMACs\*\* | \*\*1853.0519 GFLOPs\*\* | \*\*926.5260 GMACs\*\* |



\---



\## Key Takeaways



\- \*\*Training Stabilization (v1 -> v2):\*\* Proper ViT initialization, histology augmentations, and `CosineAnnealingLR` boosted clean test accuracy by \*\*+15.06%\*\* (`68.19%` -> `83.25%`) and AT test accuracy under $\\epsilon=0.05$ attack by \*\*+14.67%\*\* (`56.36%` -> `71.03%`).

\- \*\*Mixture-of-Depths Efficiency (v2 -> v3):\*\* MoD token routing cut inference compute by \*\*\~54%\*\* (`0.5610` -> `0.2581` GFLOPs/image) while maintaining competitive accuracy (`80.63%` clean / `68.22%` at $\\epsilon=0.05$ for AT) and acting as a regularizer against overfitting in adversarial training.

