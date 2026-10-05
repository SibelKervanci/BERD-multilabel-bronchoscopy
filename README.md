# Benchmarking CNN, Transformer, Hybrid, and Foundation Models for Multi-Label Bronchoscopic Finding Classification on BERD

Code for: **Kervancı IS.** *Benchmarking CNN, Transformer, Hybrid, and Foundation Models for
Multi-Label Bronchoscopic Finding Classification on the BERD Dataset.*

Five architectures (ResNet-50, EfficientNet-B3, ViT-B/16, Swin-T and the hybrid
Hybrid-EffSwin = EfficientNet-B3 + Swin-T with element-wise average-pooling fusion) are trained for
23-class multi-label classification of bronchoscopic findings. ConvNeXt-T, DINOv2 ViT-B/14 and
BiomedCLIP ViT-B/16 are evaluated as additional (secondary) baselines, and the five primary
models are replicated with three training seeds. A late-fusion ensemble of separately trained
EfficientNet-B3 and Swin-T is evaluated to separate joint training from backbone combination.

## Data
BERD is publicly available at https://doi.org/10.57760/sciencedb.28018.
Place the files as follows (paths are relative to this repository):

```
dataset/
├── images/
└── annotations/
    ├── dataset_train.json
    └── dataset_test.json
```

## Pipeline (run in order)

| Step | Script | Purpose | Test set used? |
|---|---|---|---|
| 1 | `1_berd_tpe_search.py` | Label encoding, class weights, TPE + Hyperband search on the validation subset | No |
| 2 | `2_berd_main.py` | Trains all configurations and saves checkpoints (early stopping on validation): fixed, TPE, ASL, fusion ablation, additional baselines, seed replicates | No |
| 3 | `3_berd_posthoc_eval.py` | Data audit, inference, all reported metrics, statistics and figures, including validation-based design selection, late-fusion ensemble, patient-level tests, calibration, computational cost and near-duplicate checks | Evaluation only |
| 4 | `4_berd_duplicate_sensitivity.py` | Sensitivity analysis excluding pixel-identical test images (Table S3) | Evaluation only |
| 5 | `5_berd_gradcam.py` | Grad-CAM visualizations (Figure 10) | Evaluation only |

```bash
pip install -r requirements.txt
python 1_berd_tpe_search.py
python 2_berd_main.py
python 3_berd_posthoc_eval.py
python 4_berd_duplicate_sensitivity.py
python 5_berd_gradcam.py
```

All numbers reported in the manuscript are produced by step 3 (and step 4 for Table S3).
Checkpoints and predictions are cached, so steps 3–5 can be re-run without retraining.

## Evaluation protocol
- The official BERD training set is split 80/20 **at the patient level** (seed 42) into
  training (2,809 patients, 4,826 images) and validation (702 patients, 1,186 images)
  subsets. The official test set (180 patients, 316 images) is used only for final evaluation.
- The label *diverticulum* (2 training images) is excluded, giving 23 classes.
- Macro-averaged metrics are computed over the 21 classes with at least one positive test
  instance (*mass* and *widened* have none); 23-class values are saved for reference.
- Per-class decision thresholds are selected on the validation subset and applied
  unchanged to the test set.
- Seed replicates: the five primary models are trained with seeds 42, 7 and 123 on the same
  patient-level split (only training stochasticity varies).
- Additional baselines: ConvNeXt-T (lr 1e-4); DINOv2 and BiomedCLIP with backbone lr 1e-5 and
  head lr 1e-4. They are compared with Hybrid-EffSwin and Swin-T outside the Holm-corrected family.
- Fusion ablation: average pooling (proposed), residual projection and concatenation. The
  residual-projection variant is implemented as multi-head attention (`hybrid_cross_attn`), but
  because both inputs are single pooled vectors its attention weight is identically 1 and it
  reduces to a learned residual projection of the Transformer feature.
- Design choices (fusion strategy, loss function, foundation-model learning rate) were made on
  the validation subset.
- Primary endpoint: macro F1 over the 21 evaluable classes.
- Uncertainty: patient-level (cluster) bootstrap with 2,000 resamples. Pairwise tests:
  Wilcoxon signed-rank on per-patient mean example-based F1 and per-patient exact-match rate,
  Holm-corrected; image-level Wilcoxon and McNemar tests are reported as supplementary results.

## Outputs and manuscript items

| Manuscript item | File |
|---|---|
| Table 1 | `results/posthoc_summary.csv` |
| Table 2 (patient-level tests) | `results/posthoc_patient_level_wilcoxon.csv` |
| Table 3 (bootstrap differences) | `results/posthoc_paired_differences.csv` |
| Table 4 (seed replicates) | `results/posthoc_seed_summary.csv`, `posthoc_seed_paired.csv` |
| Table 5 (threshold calibration) | `results/posthoc_summary.csv` (`f1_valthr`, `dF1_valthr`) |
| Tables 6–7 (fusion and loss ablations) | `results/posthoc_summary.csv`, `posthoc_paired_differences.csv`, `posthoc_validation_selection.csv` |
| Table 8 (additional baselines, ensemble) | `results/posthoc_extra_baselines.csv`, `posthoc_late_fusion_ensemble.csv` |
| Table S1 | `results/posthoc_table_S1.csv` |
| Table S2 | `results/posthoc_table_S2.csv` |
| Table S3 | `results/duplicate_sensitivity.csv` (step 4) |
| Table S4 (image-level tests) | `results/posthoc_pairwise_tests.csv` |
| Figure 1 | `figures/class_distribution.png` |
| Figure 2 | `figures/model_comparison_fixed.png` |
| Figures 3–4 | `figures/roc_curves.png`, `figures/pr_curves.png` |
| Figure 5 | `figures/per_class_f1_heatmap.png` |
| Figure 6 | `figures/confusion_matrix_hybrid_eff_swin.png` |
| Figure 7 | `figures/threshold_optimization.png` |
| Figure 8 | `figures/ablation_fusion.png` |
| Figure 9 | `figures/ablation_loss.png` |
| Figure 10 | `figures/gradcam/gradcam_comparison_congested.png` (step 5) |
| Figure S1 | `figures/training_curves_fixed.png` |

Further analyses written by step 3: `posthoc_per_class_auc.csv`, `posthoc_auc_by_support.csv`,
`posthoc_support_sensitivity.csv`, `posthoc_calibration.csv` (with `figures/reliability_diagram.png`),
`posthoc_computational_cost.csv`, `posthoc_near_duplicates.csv`,
`posthoc_near_duplicate_sensitivity.csv`, `posthoc_label_consistency.csv`, `posthoc_fixed_vs_tpe_table.csv`.

## Environment
Experiments were run on an NVIDIA GeForce RTX 4080 (Windows 11) with PyTorch 2.5.1.
Seeds are fixed (42), but cuDNN autotuning is enabled, so bitwise reproducibility across
GPU runs is not guaranteed.

## Citation
If you use this code, please cite the article above (citation details will be added upon publication).
