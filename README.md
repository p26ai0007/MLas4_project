\# MedLiT-seed on PathMNIST: Adversarial Robustness \& Efficiency



\## Setup and Installation



Install the required dependencies.



\### CPU Installation



```bash

pip install torch torchvision

pip install medmnist scikit-learn matplotlib seaborn tqdm pillow numpy fvcore

```



\### GPU Installation (CUDA 12.8)



```bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install medmnist scikit-learn matplotlib seaborn tqdm pillow numpy fvcore

```



\## How to Run



Each version folder contains a `run\_notebook.sh` script. Make sure the `pathmnist.npz` dataset file is present in the respective folder before running the script.



```bash

bash run\_notebook.sh

```



\## Dataset



\- \*\*Source:\*\* \[PathMNIST Colon Pathology Dataset (Kaggle)](https://www.kaggle.com/datasets/dongquan/pathmnist-colon-pathology-dataset)

\- \*\*Task:\*\* Multi-class classification (9 tissue classes).

\- \*\*Number of Classes:\*\* 9

\- \*\*Image Format:\*\* RGB images, originally 28 × 28 pixels, resized to 224 × 224 pixels in the training pipeline.

\- \*\*Total Samples:\*\* 107,180



\### Dataset Split



| Split | Number of Samples |

|---|---:|

| Training | 89,996 |

| Validation | 10,004 |

| Test | 7,180 |

| \*\*Total\*\* | \*\*107,180\*\* |



\### Class Labels



| Label | Tissue Type |

|---|---|

| 0 | Adipose |

| 1 | Background |

| 2 | Debris |

| 3 | Lymphocytes |

| 4 | Mucus |

| 5 | Smooth muscle |

| 6 | Normal colon mucosa |

| 7 | Cancer-associated stroma |

| 8 | Colorectal adenocarcinoma epithelium |



\## Version Overview



MedLiT-seed combines Sparse Top-2 Mixture-of-Experts (MoE), Grouped Query Attention (GQA), SwiGLU, and Hard-Masked Focal Loss.



The models were evaluated under standard training and FGSM adversarial training. Adversarial training used an epsilon of 0.05 and a 50% clean / 50% adversarial loss mixture for 100 epochs.



| Version | Description | Key Changes |

|---|---|---|

| \*\*v1\*\* | Baseline adversarial training | Training from scratch, learning rate = 1e-3, MultiStepLR, batch size = 512, no augmentation, zero-initialized ViT embeddings. |

| \*\*v2\*\* | Stabilized training | ViT initialization (trunc\_normal\_ / xavier\_uniform\_), histology augmentations (flips, rotation, color jitter), AdamW (lr = 5e-4, weight decay = 1e-2), CosineAnnealingLR, batch size = 256, fixed MoE loss scaling. |

| \*\*v3\*\* | Mixture-of-Depth routing | Based on v2, with an MoD router (capacity factor = 0.5) in each block. Only the top 50% of tokens are routed through Attention and MoE; remaining tokens bypass these operations through residual connections. |



\## 1. Clean Test Set Performance



Evaluation was conducted on the 7,180-image test set.



| Version | Training Mode | Val Acc | Test Acc (OA) | AUC | Precision (Macro) | Recall (Macro) | Specificity (Macro) | F1 (Macro) |

|---|---|---:|---:|---:|---:|---:|---:|

| \*\*v1\*\* | Standard | 71.48% | 68.19% | 0.9311 | 61.23% | 60.04% | 96.03% | 59.86% |

| \*\*v1\*\* | Adversarial (AT) | 65.56% | 65.65% | 0.9069 | 59.51% | 58.47% | 95.68% | 58.41% |

| \*\*v2\*\* | Standard | 89.85% | \*\*83.25%\*\* | \*\*0.9747\*\* | \*\*78.26%\*\* | \*\*78.13%\*\* | \*\*97.93%\*\* | \*\*77.85%\*\* |

| \*\*v2\*\* | Adversarial (AT) | 80.65% | 78.68% | 0.9696 | 74.47% | 74.67% | 97.34% | 73.77% |

| \*\*v3\*\* | Standard (MoD) | \*\*93.14%\*\* | 79.57% | 0.9719 | 75.61% | 76.01% | 97.46% | 75.12% |

| \*\*v3\*\* | Adversarial (MoD + AT) | 79.01% | 80.63% | 0.9688 | 74.32% | 74.38% | 97.60% | 73.94% |



\## 2. FGSM Adversarial Robustness



Test accuracy (%) under FGSM attacks with varying epsilon values.



| Version | Mode | ε = 0.00 | ε = 0.01 | ε = 0.03 | ε = 0.05 | ε = 0.10 | ε = 0.15 | ε = 0.20 |

|---|---|---:|---:|---:|---:|---:|---:|---:|

| \*\*v1\*\* | Standard | 68.19 | 43.12 | 33.66 | 29.86 | 25.49 | 22.66 | 22.02 |

| \*\*v1\*\* | AT | 65.65 | 58.73 | 57.53 | 56.36 | 45.21 | 37.10 | 27.97 |

| \*\*v2\*\* | Standard | \*\*83.25\*\* | 44.72 | 39.82 | 40.24 | 41.77 | 41.50 | 40.57 |

| \*\*v2\*\* | AT | 78.68 | \*\*61.13\*\* | \*\*61.14\*\* | \*\*71.03\*\* | \*\*61.43\*\* | \*\*53.15\*\* | \*\*47.45\*\* |

| \*\*v3\*\* | Standard (MoD) | 79.57 | 46.52 | 46.06 | 46.70 | 47.08 | 46.56 | 46.96 |

| \*\*v3\*\* | AT (MoD) | 80.63 | 59.04 | 59.37 | 68.22 | 57.99 | 48.72 | 37.33 |



\## 3. Computational Cost



Inference computation measured per image.



| Version | Architecture | FLOPs / Image | MACs / Image | Total Test FLOPs (7,180 Images) | Total Test MACs |

|---|---|---:|---:|---:|

| \*\*v1 / v2\*\* | MedLiT-seed (Full Tokens) | 0.5610 GFLOPs | 0.2805 GMACs | 4028.0727 GFLOPs | 2014.0364 GMACs |

| \*\*v3\*\* | MedLiT-seed + MoD (k = 0.5) | \*\*0.2581 GFLOPs\*\* | \*\*0.1290 GMACs\*\* | \*\*1853.0519 GFLOPs\*\* | \*\*926.5260 GMACs\*\* |



\## Key Takeaways



\- \*\*Training stabilization (v1 → v2):\*\* ViT initialization, histology augmentations, and CosineAnnealingLR improved clean test accuracy from 68.19% to 83.25% (+15.06 percentage points). Under adversarial training, test accuracy at ε = 0.05 increased from 56.36% to 71.03% (+14.67 percentage points).

\- \*\*Mixture-of-Depth efficiency (v2 → v3):\*\* MoD token routing reduced inference computation by approximately 54% (0.5610 to 0.2581 GFLOPs/image). The v3 adversarially trained model achieved 80.63% clean test accuracy and 68.22% accuracy at ε = 0.05.



