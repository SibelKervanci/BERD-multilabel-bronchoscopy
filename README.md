# Multi-Label Bronchoscopic Finding Classification

Code repository for the paper:

> **"Multi-Label Bronchoscopic Finding Classification Using CNN, Transformer, and Hybrid Architectures: A Systematic Evaluation on the BERD Dataset"**
> İlkay Sibel Kervancı, Gaziantep University
> *BMC Medical Imaging*, 2026

---

## Overview

This repository provides the complete implementation for systematic evaluation of five deep learning architectures — ResNet-50, EfficientNet-B3, ViT-B/16, Swin-T, and the proposed Hybrid-EffSwin — for 23-class multi-label bronchoscopic finding classification using the publicly available BERD dataset.

### Key Contributions

- First independent application of the BERD dataset for automated multi-label bronchoscopic image classification
- Systematic comparison of CNN, Transformer, and hybrid CNN-Transformer architectures
- Proposed Hybrid-EffSwin: EfficientNet-B3 + Swin-T with average pooling fusion
- Ablation studies: loss function (BCE vs ASL) and fusion strategy (avg pooling vs cross-attention vs concatenation)
- Per-class threshold optimization with consistent F1 improvement across all architectures
- Patient-level train/validation split to prevent data leakage

---

## Dataset

The BERD (Bronchoscopy Examination Report Dataset) is publicly available at:

**DOI:** https://doi.org/10.57760/sciencedb.28018

The dataset contains 6,328 bronchoscopic images with 23 pathological finding categories, split at the patient level into training (6,012 images) and test (316 images) sets.

**Expected directory structure:**
```
dataset/
├── images/
│   ├── image_001.jpg
│   └── ...
└── annotations/
    ├── dataset_train.json
    └── dataset_test.json
```

---

## Requirements

```bash
pip install torch torchvision
pip install optuna scikit-learn
pip install matplotlib seaborn pandas scipy statsmodels
pip install umap-learn
```

**Python:** 3.8+
**PyTorch:** 1.12+
**CUDA:** Recommended (tested on NVIDIA RTX 4080)

---

## Repository Structure

```
├── berd_tpe_search.py     # Step 1: Bayesian hyperparameter search (BOHB)
├── berd_main.py           # Step 2: Training, ablation, evaluation, visualization
└── README.md
```

---

## Usage

### Step 1 — Hyperparameter Search (Optional)

Runs Tree-structured Parzen Estimator (TPE) with HyperbandPruner to identify architecture-specific hyperparameters. Results are cached in `results/`.

```bash
python berd_tpe_search.py
```

**Expected runtime:** ~24 hours on RTX 4080
**Output:** `results/<model>_tpe_results.json`

> **Note:** This step is optional. If skipped, `berd_main.py` uses fixed hyperparameters (lr = 1e-4, max_weight = 50.0), which were found to be competitive with BOHB-identified configurations.

---

### Step 2 — Training, Ablation and Evaluation

Runs all experiments and produces results, statistical tests, and figures.

```bash
python berd_main.py
```

**Expected runtime:** ~8-10 hours on RTX 4080

**Configurable flags at the top of `berd_main.py`:**

```python
RUN_TPE_TRAIN   = True   # Train with BOHB-optimized hyperparameters
RUN_FIXED_TRAIN = True   # Train with fixed hyperparameters
RUN_ASL         = True   # Loss function ablation (BCE vs ASL)
RUN_CA_ABLATION = True   # Fusion strategy ablation
FORCE_RETRAIN   = False  # Set True to retrain from scratch
```

**Output directories:**
```
checkpoints_fixed/     <- Model checkpoints (fixed hyperparameters)
checkpoints_tpe/       <- Model checkpoints (BOHB hyperparameters)
checkpoints_asl/       <- ASL ablation checkpoints
checkpoints_ca/        <- Fusion strategy ablation checkpoints
results/               <- CSV summaries, JSON logs, statistical tests
figures/               <- All figures (300 DPI PNG)
```

---

## Architectures

| Model | Parameters | Paradigm |
|---|---|---|
| ResNet-50 | 24.6M | CNN |
| EfficientNet-B3 | 11.5M | CNN |
| ViT-B/16 | 86.2M | Pure Transformer |
| Swin-T | 27.9M | Hierarchical Transformer |
| **Hybrid-EffSwin** | **40.4M** | **CNN + Transformer (proposed)** |

**Hybrid-EffSwin** combines EfficientNet-B3 (local texture extraction) and Swin-T (global contextual reasoning) via element-wise average pooling of their projected 512-dimensional feature vectors.

---

## Experimental Design

### Train/Validation/Test Split

The BERD training set is partitioned at the **patient level** into:

| Split | Patients | Images | Purpose |
|---|---|---|---|
| Train | ~2,810 (80%) | ~4,810 | Gradient updates |
| Validation | ~702 (20%) | ~1,202 | Early stopping only |
| Test | 180 (fixed) | 316 | Final evaluation only |

Patient-level partitioning ensures no subject-level data leakage between splits.

### Fixed Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Max epochs | 70 |
| Early stopping patience | 15 |
| pos_weight cap (max_weight) | 50.0 |
| Classification threshold | 0.5 |

### Ablation Studies

**1. Loss function ablation:**
- Weighted BCE with pos_weight capping
- Asymmetric Loss (ASL): gamma_neg=4.0, gamma_pos=0.0, clip=0.05

**2. Fusion strategy ablation (Hybrid-EffSwin variants):**
- Average pooling (proposed)
- Cross-attention (8 heads)
- Concatenation + linear projection

---

## Results

**Primary results (fixed hyperparameters, theta=0.5):**

| Model | F1 macro | MCC | Accuracy | 95% CI |
|---|---|---|---|---|
| ResNet-50 | 0.291 | 0.245 | 0.218 | [0.243, 0.326] |
| EfficientNet-B3 | 0.299 | 0.256 | 0.209 | [0.261, 0.333] |
| ViT-B/16 | 0.295 | 0.248 | 0.218 | [0.245, 0.337] |
| Swin-T | 0.308 | 0.268 | 0.247 | [0.263, 0.342] |
| **Hybrid-EffSwin** | **0.354** | **0.315** | **0.269** | **[0.303, 0.390]** |

**After per-class threshold optimization:**

| Model | F1 (theta=0.5) | F1 (theta*) | Delta F1 |
|---|---|---|---|
| ResNet-50 | 0.291 | 0.352 | +0.061 |
| EfficientNet-B3 | 0.299 | 0.361 | +0.062 |
| ViT-B/16 | 0.295 | 0.347 | +0.051 |
| Swin-T | 0.308 | 0.349 | +0.041 |
| **Hybrid-EffSwin** | **0.354** | **0.378** | +0.025 |

---

## Citation

If you use this code or the BERD dataset in your research, please cite:

```bibtex
@article{kervanci2026berd,
  title   = {Multi-Label Bronchoscopic Finding Classification Using CNN,
             Transformer, and Hybrid Architectures: A Systematic Evaluation
             on the BERD Dataset},
  author  = {Kervanc{\i}, {\.I}lkay Sibel},
  journal = {BMC Medical Imaging},
  year    = {2026},
  doi     = {to be added upon publication}
}
```

Also cite the original BERD dataset paper:

```bibtex
@article{luo2026berd,
  title   = {BERD: A Bronchoscopy Examination Report Dataset},
  author  = {Luo, et al.},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.57760/sciencedb.28018}
}
```

---

## Funding

This research was funded by the Scientific Research Projects Coordination Unit of Gaziantep University (Grant No. BNG-2501).

---

## Acknowledgements

During the preparation of this manuscript, the author used Claude (Anthropic) for assistance with code development, data analysis pipeline construction, and language editing. The author has reviewed and edited the output and takes full responsibility for the content of this publication.

---

## License

This code is released under the MIT License. See LICENSE for details.
