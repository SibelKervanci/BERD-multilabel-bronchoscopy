# Multi-label bronchoscopic finding classification on BERD

Code for: Kervancı IS. Multi-Label Bronchoscopic Finding Classification Using CNN,
Transformer, and Hybrid Architectures: A Systematic Evaluation on the BERD Dataset.

## Data
BERD is publicly available at https://doi.org/10.57760/sciencedb.28018.
Place images in `dataset/images/` and annotation files in `dataset/annotations/`.

## Pipeline (run in order)
| Step | Script | Purpose | Uses test set? |
|---|---|---|---|
| 1 | `1_berd_tpe_search.py` | TPE + Hyperband search on the patient-level validation split | No |
| 2 | `2_berd_main.py` | Trains all models and saves checkpoints; early stopping on validation | No |
| 3 | `3_berd_posthoc_eval.py` | Data audit, inference, all reported metrics, statistics, and figures | Evaluation only |
| 4 | `4_berd_duplicate_sensitivity.py` | Sensitivity analysis excluding pixel-identical test images (Table S3) | Evaluation only |
| 5 | `5_berd_gradcam.py` | Grad-CAM visualizations (Figure 10) | Evaluation only |

All numbers reported in the manuscript are produced by step 3 (and step 4 for Table S3).

## Evaluation protocol
- Patient-level train/validation split (80/20, seed 42) of the official BERD training set;
  the official test set (316 images, 180 patients) is used only for final evaluation.
- Macro-averaged metrics are computed over the 21 classes with at least one positive
  test instance; 23-class values are also saved for reference.
- Per-class decision thresholds are selected on the validation subset and applied
  unchanged to the test set.
- Uncertainty: patient-level (cluster) bootstrap, 2,000 resamples. Pairwise tests:
  Wilcoxon (example-based F1) and McNemar (exact match), Holm-corrected.

## Environment
Python 3.x, PyTorch 2.x, torchvision, scikit-learn, statsmodels, optuna.
See `requirements.txt`. Experiments were run on an NVIDIA RTX 4080 (Windows 11).

## Reproducibility note
Seeds are fixed (42), but cuDNN autotuning was enabled, so bitwise reproducibility
across GPU runs is not guaranteed.
