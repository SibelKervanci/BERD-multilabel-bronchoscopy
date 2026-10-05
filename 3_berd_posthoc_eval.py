"""
3_berd_posthoc_eval.py — Evaluation of trained checkpoints: all metrics, statistics and
figures reported in the manuscript. No training is performed.

Steps
  A) Data audit: image/patient counts after label filtering, patient overlap between
     train/validation/test, pixel-identical images across splits (MD5), class counts.
  B) Inference on the validation and test sets for every checkpoint (cached in
     results/preds_cache/, so later runs do not need a GPU).
  C) Metrics: macro F1, MCC, AUPRC, AUC-ROC and Brier score over the classes with at
     least one positive test instance (21 of 23); 23-class values are saved for reference.
  D) Per-class thresholds selected on the VALIDATION subset and applied to the test set.
  E) Statistics: patient-level (cluster) bootstrap 95% CIs (2,000 resamples) for each model
     and for paired differences; Wilcoxon signed-rank (example-based F1) and McNemar
     (exact match) tests with Holm and Bonferroni correction.
  F) Figures: model comparison, ROC/PR (micro-average), per-class heatmaps, confusion
     matrices, threshold calibration, fusion and loss ablations, fixed vs TPE, training
     curves, class distribution, reliability diagram.
  G) Extended analyses: design selection on the validation subset (fusion, loss); late-fusion
     ensemble of EfficientNet-B3 and Swin-T; patient-level Wilcoxon tests; per-class AUC-ROC;
     sensitivity restricted to classes with >= MIN_SUPPORT test positives; calibration (ECE);
     fixed vs TPE table; computational cost (parameters, GFLOPs, latency); perceptual-hash
     near-duplicate check with sensitivity analysis; label consistency of "normal".

Outputs (results/):  posthoc_summary.csv, posthoc_pairwise_tests.csv,
  posthoc_paired_differences.csv, posthoc_fixed_vs_tpe.csv, posthoc_extra_baselines.csv,
  audit_class_counts.csv,
  thresholds_valselected_*.json      Figures: figures/
Run:  python 3_berd_posthoc_eval.py
"""

import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ── Load 2_berd_main.py as a module (file name starts with a digit) ──
MAIN_PATH = Path(__file__).resolve().parent / "2_berd_main.py"

RUN_DATA_AUDIT   = True
RUN_MD5_CHECK    = True     # hashes ~6.3k images; takes a few minutes
RUN_INFERENCE    = True     # cached predictions are reused
FORCE_INFERENCE  = False
N_BOOT           = 2000
BOOT_SEED        = 2026


RUN_FIGURES      = True
RUN_EXTENDED     = True     # section G (see docstring)
# Sections of the extended analyses (G); H loads all models on the GPU, I hashes all images
EXT = dict(A=True, B=True, C=True, D=True, E=True, F=True, G=True, H=True, I=True, J=True)
MIN_SUPPORT = 5                # minimum number of positive test instances per class (sensitivity)
PHASH_THRESHOLDS = [0, 4, 8]   # Hamming distance (bits of 64) counted as near-duplicate
PHASH_EXCLUDE_AT = 4           # threshold used for the near-duplicate sensitivity analysis


def load_main():
    spec = importlib.util.spec_from_file_location("berd_main", MAIN_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)      # its __main__ block is not executed
    return m


# ════════════════════════════════════════════════════════════════
#  A) DATA AUDIT
# ════════════════════════════════════════════════════════════════

def data_audit(bm, label2idx):
    valid = set(label2idx.keys())
    raw_tr = bm.load_json(bm.TRAIN_JSON)
    raw_te = bm.load_json(bm.TEST_JSON)

    def summarize(records, tag):
        exists = [r for r in records
                  if bm.resolve_path(r.get("image_path", ""), bm.IMAGE_DIR).exists()]
        with_lab = [r for r in exists if bm.parse_labels(r.get("label", ""), valid)]
        dropped = [r for r in exists if not bm.parse_labels(r.get("label", ""), valid)]
        no_pid = [r for r in with_lab if not r.get("patient_id", "")]
        pids_all = {r.get("patient_id") for r in records if r.get("patient_id")}
        pids_ok = {r.get("patient_id") for r in with_lab if r.get("patient_id")}
        lost = pids_all - pids_ok
        print(f"\n[{tag}] records: {len(records)} | image found: {len(exists)} | "
              f"with >=1 valid label: {len(with_lab)}")
        print(f"[{tag}] images dropped (no valid label left): {len(dropped)}")
        for r in dropped[:10]:
            print(f"      - {r.get('image_path')} | label='{r.get('label')}' | "
                  f"patient={r.get('patient_id')}")
        print(f"[{tag}] valid records without patient_id: {len(no_pid)}")
        print(f"[{tag}] patients: {len(pids_all)} before filtering -> {len(pids_ok)} after "
              f"(fully removed: {len(lost)})")
        return with_lab, pids_ok

    tr_ok, _ = summarize(raw_tr, "TRAIN")
    te_ok, te_pids = summarize(raw_te, "TEST")

    train_recs, val_recs = bm.split_patients_train_val(label2idx)
    trp = {r["patient_id"] for r in train_recs}
    vap = {r["patient_id"] for r in val_recs}
    print(f"\n[SPLIT] train: {len(trp)} patients / {len(train_recs)} images")
    print(f"[SPLIT] val  : {len(vap)} patients / {len(val_recs)} images")
    print(f"[SPLIT] test : {len(te_pids)} patients / {len(te_ok)} images")
    print(f"[OVERLAP] train&val={len(trp & vap)} | train&test={len(trp & te_pids)} | "
          f"val&test={len(vap & te_pids)}")

    def label_stats(recs, tag):
        cnt = Counter(); nlab = []
        for r in recs:
            ls = bm.parse_labels(r.get("label", ""), valid)
            cnt.update(ls); nlab.append(len(ls))
        nlab = np.array(nlab)
        print(f"[{tag}] images={len(recs)} | label occurrences={sum(cnt.values())} | "
              f"mean labels/image={nlab.mean():.2f} | >=2 labels: {100 * (nlab >= 2).mean():.1f}%")
        return cnt

    print()
    c_all = label_stats(tr_ok, "TRAIN pool")
    c_tr = label_stats(train_recs, "TRAIN subset")
    c_va = label_stats(val_recs, "VAL subset")
    c_te = label_stats(te_ok, "TEST")

    df = pd.DataFrame({"label": list(label2idx.keys())})
    for tag, c in [("train_pool", c_all), ("train_subset", c_tr), ("val", c_va), ("test", c_te)]:
        df[tag] = [c.get(l, 0) for l in df["label"]]
    df.to_csv(bm.RESULTS_DIR / "audit_class_counts.csv", index=False)
    print(df.sort_values("train_pool", ascending=False).to_string(index=False))

    if RUN_MD5_CHECK:
        def md5s(recs):
            h = defaultdict(list)
            for r in recs:
                p = bm.resolve_path(r["image_path"], bm.IMAGE_DIR)
                h[hashlib.md5(p.read_bytes()).hexdigest()].append(r.get("patient_id"))
            return h
        htr, hva, hte = md5s(train_recs), md5s(val_recs), md5s(te_ok)
        print(f"\n[MD5] pixel-identical images: train&test={len(set(htr) & set(hte))} | "
              f"val&test={len(set(hva) & set(hte))} | train&val={len(set(htr) & set(hva))}")


# ════════════════════════════════════════════════════════════════
#  B) INFERENCE (cached)
# ════════════════════════════════════════════════════════════════

def checkpoint_list(bm):
    L = []
    for n in bm.MODELS:          # fixed hyperparameters (Hybrid = avg-pool fusion run)
        p = bm.CKPT_CA / "hybrid_eff_swin_ca_best.pt" if n == "hybrid_eff_swin" \
            else bm.CKPT_FIXED / f"{n}_fixed_best.pt"
        L.append((f"{n}|fixed", n, p))
    for n in bm.MODELS:          # TPE-Hyperband hyperparameters
        L.append((f"{n}|tpe", n, bm.CKPT_TPE / f"{n}_tpe_best.pt"))
    for n in bm.MODELS:          # Asymmetric Loss
        if n != "hybrid_eff_swin":
            L.append((f"{n}|asl", n, bm.CKPT_ASL / f"{n}_asl_best.pt"))
    for n in ["hybrid_cross_attn", "hybrid_concat"]:   # fusion ablation
        L.append((f"{n}|ca", n, bm.CKPT_CA / f"{n}_ca_best.pt"))
    for n in getattr(bm, "EXTRA_MODELS", []):           # additional baselines (secondary)
        L.append((f"{n}|extra", n, bm.CKPT_EXTRA / f"{n}_fixed_best.pt"))
    for sd in getattr(bm, "SEED_RUNS", []):             # seed replicates of the fixed configuration
        for n in getattr(bm, "SEED_MODELS", []):
            L.append((f"{n}|seed{sd}", n, bm.CKPT_SEEDS / f"{n}_seed{sd}_best.pt"))
    return L


def run_inference(bm, label2idx):
    import torch
    from torch.utils.data import DataLoader
    cache = bm.RESULTS_DIR / "preds_cache"; cache.mkdir(exist_ok=True)
    nc = len(label2idx)

    _, val_recs = bm.split_patients_train_val(label2idx)
    va = bm.BERDDataset(bm.TRAIN_JSON, bm.IMAGE_DIR, label2idx, bm.get_tf("test"), "val")
    va.data = val_recs
    te = bm.BERDDataset(bm.TEST_JSON, bm.IMAGE_DIR, label2idx, bm.get_tf("test"), "test")
    vl = DataLoader(va, bm.BATCH_SIZE, shuffle=False, num_workers=0)
    tl = DataLoader(te, bm.BATCH_SIZE, shuffle=False, num_workers=0)
    np.save(cache / "test_patient_ids.npy",
            np.array([r.get("patient_id", f"img{i}") for i, r in enumerate(te.data)]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, name, ck in checkpoint_list(bm):
        f = cache / (key.replace("|", "__") + ".npz")
        if f.exists() and not FORCE_INFERENCE:
            continue
        if not ck.exists():
            print(f"  [missing checkpoint] {ck}"); continue
        print(f"  [inference] {key}")
        state = torch.load(ck, map_location=device)
        dr = state.get("params", {}).get(
            "dropout", bm.FIXED_DROPOUT_HYB if "hybrid" in name else bm.FIXED_DROPOUT_CNN)
        model = bm.build_model(name, nc, dr).to(device)
        model.load_state_dict(state["model_state"])
        pv, _, yv = bm.get_preds(model, vl, device)
        pt, _, yt = bm.get_preds(model, tl, device)
        np.savez_compressed(f, val_probs=pv, val_labels=yv, test_probs=pt, test_labels=yt)
        del model; torch.cuda.empty_cache()


def load_preds(bm):
    cache = bm.RESULTS_DIR / "preds_cache"
    P = {}
    for f in sorted(cache.glob("*.npz")):
        d = np.load(f); P[f.stem.replace("__", "|")] = {k: d[k] for k in d.files}
    return P, np.load(cache / "test_patient_ids.npy", allow_pickle=True)


# ════════════════════════════════════════════════════════════════
#  C) METRICS (vectorised)
# ════════════════════════════════════════════════════════════════

def per_class_counts(y, p):
    tp = (y * p).sum(0); fp = ((1 - y) * p).sum(0)
    fn = (y * (1 - p)).sum(0); tn = ((1 - y) * (1 - p)).sum(0)
    return tp, fp, fn, tn


def f1_mcc_per_class(y, p):
    tp, fp, fn, tn = per_class_counts(y, p)
    d = 2 * tp + fp + fn
    f1 = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
    return f1, mcc


def auprc_per_class(y, s, mask):
    from sklearn.metrics import average_precision_score
    return np.array([average_precision_score(y[:, c], s[:, c]) if mask[c] else np.nan
                     for c in range(y.shape[1])])


def auroc_per_class(y, s, mask):
    from sklearn.metrics import roc_auc_score
    out = []
    for c in range(y.shape[1]):
        ok = mask[c] and 0 < y[:, c].sum() < len(y)
        out.append(roc_auc_score(y[:, c], s[:, c]) if ok else np.nan)
    return np.array(out)


def sample_f1(y, p):
    """Example-based F1 per image: 2|y & p| / (|y| + |p|)."""
    inter = (y * p).sum(1); den = y.sum(1) + p.sum(1)
    return np.where(den > 0, 2 * inter / np.maximum(den, 1e-12), 1.0)


def full_metrics(y, s, thr, mask):
    p = (s >= thr).astype(float)
    f1, mcc = f1_mcc_per_class(y, p)
    ap = auprc_per_class(y, s, mask); au = auroc_per_class(y, s, mask)
    brier = ((s - y) ** 2).mean(0)
    return {
        "f1_macro": f1[mask].mean(), "mcc_macro": mcc[mask].mean(),
        "auprc_macro": np.nanmean(ap), "auc_roc_macro": np.nanmean(au),
        "brier_macro": brier[mask].mean(),
        "f1_macro_23cls": f1.mean(), "mcc_macro_23cls": mcc.mean(),
        "exact_match": (p == y).all(1).mean(), "hamming": (p != y).mean(),
        "f1_samples": sample_f1(y, p).mean(),
    }


# ════════════════════════════════════════════════════════════════
#  D) THRESHOLDS — selected on validation, applied to test
# ════════════════════════════════════════════════════════════════

def thresholds_from_val(yv, sv):
    from sklearn.metrics import f1_score
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    th = np.full(yv.shape[1], 0.5)
    for c in range(yv.shape[1]):
        if yv[:, c].sum() == 0:
            continue
        best, bt = f1_score(yv[:, c], sv[:, c] >= 0.5, zero_division=0), 0.5
        for t in grid:
            f = f1_score(yv[:, c], sv[:, c] >= t, zero_division=0)
            if f > best + 1e-12:
                best, bt = f, t
        th[c] = bt
    return th


# ════════════════════════════════════════════════════════════════
#  E) STATISTICS
# ════════════════════════════════════════════════════════════════

def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def cluster_indices(pids):
    groups = defaultdict(list)
    for i, g in enumerate(pids):
        groups[g].append(i)
    return [np.array(v) for v in groups.values()]


def boot_samples(groups, B, seed):
    rng = np.random.default_rng(seed); G = len(groups)
    for _ in range(B):
        pick = rng.integers(0, G, G)
        yield np.concatenate([groups[g] for g in pick])


def boot_single(y, s, thr, groups, B=N_BOOT, seed=BOOT_SEED, with_ap=True):
    p = (s >= thr).astype(float); rows = []
    for idx in boot_samples(groups, B, seed):
        yy, pp = y[idx], p[idx]; m = yy.sum(0) > 0
        f1, mcc = f1_mcc_per_class(yy, pp)
        r = [f1[m].mean(), mcc[m].mean()]
        if with_ap:
            r.append(np.nanmean(auprc_per_class(yy, s[idx], m)))
        rows.append(r)
    a = np.array(rows)
    return {k: (np.percentile(a[:, j], 2.5), np.percentile(a[:, j], 97.5))
            for j, k in enumerate(["f1", "mcc", "auprc"][:a.shape[1]])}


def boot_paired(y, s1, t1, s2, t2, groups, B=N_BOOT, seed=BOOT_SEED):
    """Patient-level paired bootstrap of dF1 and dMCC (model 1 minus model 2)."""
    p1 = (s1 >= t1).astype(float); p2 = (s2 >= t2).astype(float)
    full = y.sum(0) > 0
    f1a, ma = f1_mcc_per_class(y, p1); f1b, mb = f1_mcc_per_class(y, p2)
    obs = (f1a[full].mean() - f1b[full].mean(), ma[full].mean() - mb[full].mean())
    d = []
    for idx in boot_samples(groups, B, seed):
        yy = y[idx]; m = yy.sum(0) > 0
        fa, xa = f1_mcc_per_class(yy, p1[idx]); fb, xb = f1_mcc_per_class(yy, p2[idx])
        d.append((fa[m].mean() - fb[m].mean(), xa[m].mean() - xb[m].mean()))
    d = np.array(d); res = {}
    for j, k in enumerate(["dF1", "dMCC"]):
        lo, hi = np.percentile(d[:, j], [2.5, 97.5])
        pb = min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean()))
        res[k] = obs[j]; res[k + "_lo"] = lo; res[k + "_hi"] = hi; res[k + "_p"] = pb
    return res


def pairwise_tests(names, P, y, thr=0.5):
    from scipy.stats import wilcoxon
    from statsmodels.stats.contingency_tables import mcnemar
    rows = []
    for a, b in combinations(names, 2):
        pa = (P[a]["test_probs"] >= thr).astype(float); pb = (P[b]["test_probs"] >= thr).astype(float)
        sa, sb = sample_f1(y, pa), sample_f1(y, pb)
        try:
            W, pw = wilcoxon(sa, sb, alternative="two-sided")
        except ValueError:
            W, pw = np.nan, 1.0
        ca = (pa == y).all(1); cb = (pb == y).all(1)
        tab = [[(~ca & ~cb).sum(), (~ca & cb).sum()], [(ca & ~cb).sum(), (ca & cb).sum()]]
        r = mcnemar(tab, exact=False, correction=True)
        rows.append({"m1": a.split("|")[0], "m2": b.split("|")[0],
                     "wilcoxon_W": W, "wilcoxon_p": pw,
                     "mcnemar_chi2": r.statistic, "mcnemar_p": r.pvalue})
    df = pd.DataFrame(rows)
    k = len(df)
    for col in ["wilcoxon_p", "mcnemar_p"]:
        df[col + "_bonf"] = np.minimum(df[col] * k, 1.0)
        df[col + "_holm"] = holm(df[col].values)
    return df


# ════════════════════════════════════════════════════════════════
#  F) FIGURES (plotting functions live in 2_berd_main.py)
# ════════════════════════════════════════════════════════════════

def metrics_frame(bm, M, suffix):
    """Rows keyed by bare model name, columns named as expected by the plotting functions."""
    rows = []
    for n in bm.MODELS:
        k = f"{n}|{suffix}"
        if k in M:
            m = M[k]
            rows.append({"model": n, "f1_macro": m["f1_macro"], "mcc_macro": m["mcc_macro"],
                         "auprc_macro": m["auprc_macro"], "auc_roc_macro": m["auc_roc_macro"],
                         "accuracy": m["exact_match"], "hamming_loss": m["hamming"],
                         "brier_macro": m["brier_macro"]})
    return pd.DataFrame(rows)


def make_figures(bm, P, y, M, TH, idx2label):
    bm.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fixed = metrics_frame(bm, M, "fixed")
    if len(fixed):
        bm.plot_model_comparison(fixed, "_fixed")                          # Figure 2
    tpe = metrics_frame(bm, M, "tpe")
    if len(tpe) and len(fixed):
        bm.plot_tpe_vs_fixed(tpe, fixed)
    asl = metrics_frame(bm, M, "asl")
    if len(asl) and len(fixed):
        bm.plot_ablation_loss(fixed, asl)                                  # Figure 9
    fusion = {}
    for name, key in [("hybrid_eff_swin", "hybrid_eff_swin|fixed"),
                      ("hybrid_cross_attn", "hybrid_cross_attn|ca"),
                      ("hybrid_concat", "hybrid_concat|ca")]:
        if key in M:
            m = M[key]
            fusion[name] = {"f1_macro": m["f1_macro"], "mcc_macro": m["mcc_macro"],
                            "accuracy": m["exact_match"], "hamming_loss": m["hamming"]}
    if fusion:
        bm.plot_ca_ablation(fusion)                                        # Figure 8
    thr = {n: {"f1_base": M[f"{n}|fixed"]["f1_macro"],
               "f1_opt": M[f"{n}|fixed"]["f1_valthr"]}
           for n in bm.MODELS if f"{n}|fixed" in M}
    if thr:
        bm.plot_threshold(thr)                                             # Figure 7
    probs = {n: P[f"{n}|fixed"]["test_probs"] for n in bm.MODELS if f"{n}|fixed" in P}
    preds = {n: (v >= 0.5).astype(float) for n, v in probs.items()}
    if probs:
        bm.plot_roc_curves(probs, y)                                       # Figure 3
        bm.plot_pr_curves(probs, y)                                        # Figure 4
        bm.plot_heatmaps(probs, preds, y, idx2label)                       # Figure 5
        best = max(probs, key=lambda n: M[f"{n}|fixed"]["f1_macro"])
        bm.plot_confusion_best(preds, y, idx2label, best)                  # Figure 6
    for exp in ["fixed", "tpe", "asl"]:
        bm.plot_training_curves(exp)                                       # Figure S1
    bm.plot_class_distribution(bm.load_label_encoder()[0])                 # Figure 1



def supplementary_tables(bm, P, idx2label):
    """Tables S1 (supplementary metrics) and S2 (per-class F1 and MCC) for the five primary models."""
    from sklearn.metrics import f1_score, label_ranking_average_precision_score
    R = bm.RESULTS_DIR
    y = next(iter(P.values()))["test_labels"]; m = y.sum(0) > 0
    names = [idx2label[c] for c in range(y.shape[1])]
    s1, s2 = [], {"class": names, "n_pos": y.sum(0).astype(int)}
    for n in bm.MODELS:
        k = f"{n}|fixed"
        if k not in P:
            continue
        s = P[k]["test_probs"]; p = (s >= 0.5).astype(float)
        tp, fp, fn_, tn = per_class_counts(y, p)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0)
        rec = tp / np.maximum(tp + fn_, 1); spec = tn / np.maximum(tn + fp, 1)
        f1, mcc = f1_mcc_per_class(y, p)
        s1.append({"model": n, "f1_micro": f1_score(y, p, average="micro", zero_division=0),
                   "f1_weighted": f1_score(y, p, average="weighted", zero_division=0),
                   "precision_macro": prec[m].mean(), "recall_macro": rec[m].mean(),
                   "specificity_macro": spec[m].mean(),
                   "lrap": label_ranking_average_precision_score(y, s),
                   "brier_macro": ((s - y) ** 2).mean(0)[m].mean(),
                   "f1_macro_23cls": f1.mean(), "mcc_macro_23cls": mcc.mean()})
        s2[f"{n}_F1"] = np.where(m, f1, np.nan); s2[f"{n}_MCC"] = np.where(m, mcc, np.nan)
    pd.DataFrame(s1).round(3).to_csv(R / "posthoc_table_S1.csv", index=False)
    pd.DataFrame(s2).round(2).to_csv(R / "posthoc_table_S2.csv", index=False)
    print("\n=== Table S1 ===\n" + pd.DataFrame(s1).round(3).to_string(index=False))

# ════════════════════════════════════════════════════════════════
#  G) EXTENDED ANALYSES
# ════════════════════════════════════════════════════════════════

def _nice(k):
    n, tag = k.split("|")
    return _NAMES.get(n, n) + ("" if tag in ("fixed", "extra", "ca") else f" [{tag}]")


def _macro(y, s, thr=0.5, mask=None):
    mask = (y.sum(0) > 0) if mask is None else mask
    return full_metrics(y, s, thr, mask)


def _ece(y, s, bins=15):
    y = y.ravel(); s = s.ravel(); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi) if hi < 1 else (s >= lo) & (s <= hi)
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return e


def _phash(path):
    from PIL import Image
    from scipy.fft import dctn
    a = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dctn(a, norm="ortho")[:8, :8].ravel()
    bits = d[1:] > np.median(d[1:])
    return np.packbits(np.concatenate([[d[0] > 0], bits]))   # 64-bit hash


_NAMES = {"resnet50": "ResNet-50", "efficientnet_b3": "EfficientNet-B3", "vit_b16": "ViT-B/16",
          "swin_t": "Swin-T", "hybrid_eff_swin": "Hybrid-EffSwin", "hybrid_cross_attn": "Residual projection",
          "hybrid_concat": "Concatenation", "convnext_t": "ConvNeXt-T", "dinov2_vitb14": "DINOv2 ViT-B/14",
          "biomedclip_vitb16": "BiomedCLIP ViT-B/16", "ensemble_eff_swin": "Ensemble (EffNet-B3 + Swin-T)"}

def extended_analyses(bm, P, PIDS, idx2label):
    """Design selection on validation, late-fusion ensemble, patient-level tests, per-class AUC,
    support-restricted sensitivity, calibration, hyperparameter-search table, computational
    cost, perceptual-hash near-duplicates and label consistency."""
    import time
    from collections import Counter
    OUT = bm.RESULTS_DIR
    P = dict(P)
    Y = next(iter(P.values()))["test_labels"]
    MASK = Y.sum(0) > 0
    GROUPS = cluster_indices(PIDS)
    label2idx = {l: i for i, l in idx2label.items()} if isinstance(idx2label, dict) else {l: i for i, l in enumerate(idx2label)}
    PRIMARY = [f"{n}|fixed" for n in bm.MODELS]
    EXTRA = [k for k in P if k.endswith("|extra")]
    N_BOOT_EXT = N_BOOT

    def _save(df, name):
        df.to_csv(OUT / f"posthoc_{name}", index=False)
        print(f"\n=== posthoc_{name} ===\n{df.to_string(index=False)}")

    # ── A) validation-based design selection ─────────────────────────────────────
    if EXT["A"]:
        rows = []
        for grp, keys in [("fusion", ["hybrid_eff_swin|fixed", "hybrid_cross_attn|ca", "hybrid_concat|ca"]),
                          ("loss", [f"{n}|{t}" for n in bm.MODELS if n != "hybrid_eff_swin" for t in ("fixed", "asl")])]:
            for k in keys:
                if k not in P:
                    continue
                v = _macro(P[k]["val_labels"], P[k]["val_probs"])
                t = _macro(Y, P[k]["test_probs"], mask=MASK)
                rows.append({"ablation": grp, "model": _nice(k), "variant": k.split("|")[1],
                             "val_f1": round(v["f1_macro"], 4), "val_mcc": round(v["mcc_macro"], 4),
                             "test_f1": round(t["f1_macro"], 4), "test_mcc": round(t["mcc_macro"], 4)})
        _save(pd.DataFrame(rows), "validation_selection.csv")

    # ── B) late-fusion ensemble ──────────────────────────────────────────────────
    if EXT["B"]:
        rows = []
        for tag in ["fixed"] + [f"seed{s}" for s in getattr(bm, "SEED_RUNS", [])]:
            e, s, h = f"efficientnet_b3|{tag}", f"swin_t|{tag}", f"hybrid_eff_swin|{tag}"
            if not all(k in P for k in (e, s, h)):
                continue
            ens_t = (P[e]["test_probs"] + P[s]["test_probs"]) / 2
            ens_v = (P[e]["val_probs"] + P[s]["val_probs"]) / 2
            m = _macro(Y, ens_t, mask=MASK); mv = _macro(P[e]["val_labels"], ens_v)
            r_h = boot_paired(Y, P[h]["test_probs"], 0.5, ens_t, 0.5, GROUPS, B=N_BOOT_EXT)
            r_s = boot_paired(Y, ens_t, 0.5, P[s]["test_probs"], 0.5, GROUPS, B=N_BOOT_EXT)
            rows.append({"seed": 42 if tag == "fixed" else int(tag[4:]),
                         "ens_f1": round(m["f1_macro"], 4), "ens_mcc": round(m["mcc_macro"], 4),
                         "ens_auprc": round(m["auprc_macro"], 4), "ens_exact": round(m["exact_match"], 4),
                         "ens_val_f1": round(mv["f1_macro"], 4),
                         "hybrid_f1": round(_macro(Y, P[h]["test_probs"], mask=MASK)["f1_macro"], 4),
                         "dF1_hybrid_minus_ens": round(r_h["dF1"], 4),
                         "CI_hybrid_minus_ens": f"[{r_h['dF1_lo']:.3f}, {r_h['dF1_hi']:.3f}]",
                         "p_hybrid_minus_ens": round(r_h["dF1_p"], 3),
                         "dF1_ens_minus_swin": round(r_s["dF1"], 4),
                         "CI_ens_minus_swin": f"[{r_s['dF1_lo']:.3f}, {r_s['dF1_hi']:.3f}]"})
            if tag == "fixed":
                P["ensemble_eff_swin|fixed"] = {"test_probs": ens_t, "val_probs": ens_v,
                                                "test_labels": Y, "val_labels": P[e]["val_labels"]}
        _save(pd.DataFrame(rows), "late_fusion_ensemble.csv")

    # ── C) patient-level non-parametric tests ────────────────────────────────────
    if EXT["C"]:
        from scipy.stats import wilcoxon
        pid_list = sorted(set(PIDS)); pos = {p: i for i, p in enumerate(pid_list)}
        gi = np.array([pos[p] for p in PIDS]); n_pat = len(pid_list)

        def per_patient(v):
            s = np.bincount(gi, weights=v, minlength=n_pat); c = np.bincount(gi, minlength=n_pat)
            return s / c

        feats = {}
        for k in PRIMARY:
            p = (P[k]["test_probs"] >= 0.5).astype(float)
            feats[k] = (per_patient(sample_f1(Y, p)), per_patient((p == Y).all(1).astype(float)))
        rows = []
        for a, b in combinations(PRIMARY, 2):
            r = {"m1": _nice(a), "m2": _nice(b)}
            for j, lab in enumerate(["exF1", "exact"]):
                try:
                    W, pv = wilcoxon(feats[a][j], feats[b][j])
                except ValueError:
                    W, pv = np.nan, 1.0
                r[f"{lab}_W"] = W; r[f"{lab}_p"] = pv
            rows.append(r)
        df = pd.DataFrame(rows)
        for lab in ["exF1", "exact"]:
            df[f"{lab}_p_holm"] = holm(df[f"{lab}_p"].values)
        _save(df.round(4), "patient_level_wilcoxon.csv")

    # ── D) per-class AUC-ROC ─────────────────────────────────────────────────────
    if EXT["D"]:
        keys = PRIMARY + EXTRA
        rows = []
        for c in range(Y.shape[1]):
            r = {"class": idx2label[c], "n_pos": int(Y[:, c].sum())}
            for k in keys:
                au = auroc_per_class(Y, P[k]["test_probs"], MASK)[c]
                r[_nice(k)] = round(au, 3) if not np.isnan(au) else np.nan
            rows.append(r)
        df = pd.DataFrame(rows); _save(df, "per_class_auc.csv")
        summ = []
        for k in keys:
            au = auroc_per_class(Y, P[k]["test_probs"], MASK)
            sup = Y.sum(0)
            summ.append({"model": _nice(k), "macro_auc_all21": round(np.nanmean(au), 4),
                         f"macro_auc_support>={MIN_SUPPORT}": round(np.nanmean(au[sup >= MIN_SUPPORT]), 4),
                         "macro_auc_support<5": round(np.nanmean(au[(sup > 0) & (sup < 5)]), 4)})
        _save(pd.DataFrame(summ), "auc_by_support.csv")

    # ── E) sensitivity analysis on well-supported classes ────────────────────────
    if EXT["E"]:
        m5 = Y.sum(0) >= MIN_SUPPORT
        print(f"\n[E] classes with >= {MIN_SUPPORT} test positives: {int(m5.sum())}")
        keys = PRIMARY + EXTRA + (["ensemble_eff_swin|fixed"] if "ensemble_eff_swin|fixed" in P else [])
        rows = []
        for k in keys:
            m = full_metrics(Y, P[k]["test_probs"], 0.5, m5)
            r = {"model": _nice(k), "n_classes": int(m5.sum()), "f1_macro": round(m["f1_macro"], 4),
                 "mcc_macro": round(m["mcc_macro"], 4), "auprc_macro": round(m["auprc_macro"], 4),
                 "auc_roc_macro": round(m["auc_roc_macro"], 4)}
            if k != "hybrid_eff_swin|fixed":
                # restrict the paired bootstrap to the same class subset
                y5 = Y[:, m5]; s1 = P["hybrid_eff_swin|fixed"]["test_probs"][:, m5]; s2 = P[k]["test_probs"][:, m5]
                b = boot_paired(y5, s1, 0.5, s2, 0.5, GROUPS, B=N_BOOT_EXT)
                r["dF1_hybrid_minus_model"] = round(b["dF1"], 4)
                r["CI"] = f"[{b['dF1_lo']:.3f}, {b['dF1_hi']:.3f}]"
            rows.append(r)
        _save(pd.DataFrame(rows), "support_sensitivity.csv")

    # ── F) calibration ───────────────────────────────────────────────────────────
    if EXT["F"]:
        keys = PRIMARY + EXTRA + (["ensemble_eff_swin|fixed"] if "ensemble_eff_swin|fixed" in P else [])
        rows = []
        for k in keys:
            s = P[k]["test_probs"][:, MASK]; yy = Y[:, MASK]
            cw = np.mean([_ece(yy[:, c], s[:, c]) for c in range(yy.shape[1])])
            rows.append({"model": _nice(k), "ECE_pooled": round(_ece(yy, s), 4), "ECE_classwise": round(cw, 4),
                         "brier_macro": round(((s - yy) ** 2).mean(0).mean(), 4)})
        _save(pd.DataFrame(rows), "calibration.csv")
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
            edges = np.linspace(0, 1, 16)
            for k in ["hybrid_eff_swin|fixed", "swin_t|fixed", "efficientnet_b3|fixed", "dinov2_vitb14|extra"]:
                if k not in P:
                    continue
                s = P[k]["test_probs"][:, MASK].ravel(); yy = Y[:, MASK].ravel(); xs, ys = [], []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (s >= lo) & (s <= hi)
                    if m.sum() >= 20:
                        xs.append(s[m].mean()); ys.append(yy[m].mean())
                ax.plot(xs, ys, "o-", ms=4, label=_nice(k))
            ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
            ax.set_title("Reliability diagram (test set, 21 evaluable classes)"); ax.legend(fontsize=8)
            ax.grid(alpha=.3); plt.tight_layout()
            bm.FIGURES_DIR.mkdir(exist_ok=True); plt.savefig(bm.FIGURES_DIR / "reliability_diagram.png", dpi=300)
            plt.close(); print("✓ figures/reliability_diagram.png")
        except Exception as ex:
            print("[F] reliability diagram skipped:", ex)

    # ── G) hyperparameter-search table ───────────────────────────────────────────
    if EXT["G"]:
        rows = []
        for tag in ["fixed", "tpe"]:
            sc = {n: _macro(Y, P[f"{n}|{tag}"]["test_probs"], mask=MASK) for n in bm.MODELS if f"{n}|{tag}" in P}
            rank = {n: r for r, n in enumerate(sorted(sc, key=lambda n: -sc[n]["f1_macro"]), 1)}
            for n, m in sc.items():
                rows.append({"config": tag, "model": _NAMES[n], "f1_macro": round(m["f1_macro"], 4),
                             "mcc_macro": round(m["mcc_macro"], 4), "auprc_macro": round(m["auprc_macro"], 4),
                             "exact_match": round(m["exact_match"], 4), "rank_f1": rank[n]})
        _save(pd.DataFrame(rows), "fixed_vs_tpe_table.csv")

    # ── H) computational cost ────────────────────────────────────────────────────
    if EXT["H"]:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = []
        for n in bm.MODELS + list(getattr(bm, "EXTRA_MODELS", [])):
            try:
                m = bm.build_model(n, len(label2idx)).to(dev).eval()
                params = sum(p.numel() for p in m.parameters()) / 1e6
                x1 = torch.randn(1, 3, 224, 224, device=dev)
                with torch.no_grad():
                    try:
                        with FlopCounterMode(display=False) as fc:
                            m(x1)
                        gflops = fc.get_total_flops() / 1e9
                    except Exception:
                        gflops = np.nan
                    for _ in range(10):
                        m(x1)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(50):
                        m(x1)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) / 50 * 1000
                rows.append({"model": _NAMES.get(n, n), "params_M": round(params, 1),
                             "GFLOPs": round(gflops, 2), "latency_ms_bs1": round(ms, 2), "device": str(dev)})
                del m; torch.cuda.empty_cache()
            except Exception as ex:
                print(f"[H] {n} skipped: {ex}")
        _save(pd.DataFrame(rows), "computational_cost.csv")

    # ── I) near-duplicates by perceptual hash ────────────────────────────────────
    if EXT["I"]:
        valid = set(label2idx.keys())
        pool = [r for r in bm.load_json(bm.TRAIN_JSON)
                if bm.parse_labels(r.get("label", ""), valid)
                and bm.resolve_path(r["image_path"], bm.IMAGE_DIR).exists()]
        te = bm.BERDDataset(bm.TEST_JSON, bm.IMAGE_DIR, label2idx, None, "test").data   # inference order
        print(f"\n[I] hashing {len(pool)} training-pool and {len(te)} test images ...")
        hp = np.stack([_phash(bm.resolve_path(r["image_path"], bm.IMAGE_DIR)) for r in pool])
        ht = np.stack([_phash(bm.resolve_path(r["image_path"], bm.IMAGE_DIR)) for r in te])
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        dist = np.zeros((len(te), len(pool)), dtype=np.uint8)
        for b in range(8):
            dist += lut[np.bitwise_xor.outer(ht[:, b], hp[:, b])]
        dmin = dist.min(1)
        rows = [{"max_hamming_bits": t, "n_test_images_with_match": int((dmin <= t).sum()),
                 "n_test_patients_affected": len({te[i].get("patient_id") for i in np.where(dmin <= t)[0]})}
                for t in PHASH_THRESHOLDS]
        _save(pd.DataFrame(rows), "near_duplicates.csv")
        near = np.where(dmin <= PHASH_EXCLUDE_AT)[0]
        pd.DataFrame([{"test_image": te[i]["image_path"], "test_patient": te[i].get("patient_id"),
                       "train_image": pool[int(dist[i].argmin())]["image_path"],
                       "train_patient": pool[int(dist[i].argmin())].get("patient_id"),
                       "hamming": int(dmin[i])} for i in near]).to_csv(OUT / "posthoc_near_duplicate_pairs.csv", index=False)
        keep = np.setdiff1d(np.arange(len(Y)), near); mk = Y[keep].sum(0) > 0
        rows = []
        for k in PRIMARY + EXTRA:
            a = _macro(Y, P[k]["test_probs"], mask=MASK); b = full_metrics(Y[keep], P[k]["test_probs"][keep], 0.5, mk)
            rows.append({"model": _nice(k), "n_test_kept": len(keep), "f1_all": round(a["f1_macro"], 4),
                         "f1_excl": round(b["f1_macro"], 4), "dF1": round(b["f1_macro"] - a["f1_macro"], 4)})
        _save(pd.DataFrame(rows), "near_duplicate_sensitivity.csv")

    # ── J) label consistency ─────────────────────────────────────────────────────
    if EXT["J"]:
        valid = set(label2idx.keys())
        out = []
        for tag, recs in [("training pool", bm.load_json(bm.TRAIN_JSON)), ("test", bm.load_json(bm.TEST_JSON))]:
            labs = [bm.parse_labels(r.get("label", ""), valid) for r in recs]
            labs = [l for l in labs if l]
            nn_ = sum("normal" in l for l in labs); co = sum("normal" in l and len(l) > 1 for l in labs)
            pids = Counter(r.get("patient_id") for r in recs if bm.parse_labels(r.get("label", ""), valid))
            out.append({"set": tag, "images": len(labs), "normal_images": nn_,
                        "normal_with_pathology": co, "pct_of_normal": round(100 * co / max(nn_, 1), 1),
                        "patients": len(pids), "images_per_patient_median": float(np.median(list(pids.values()))),
                        "images_per_patient_max": max(pids.values())})
        ni = label2idx["normal"]
        for k in PRIMARY + EXTRA:
            p = (P[k]["test_probs"] >= 0.5)
            out.append({"set": f"predictions: {_nice(k)}",
                        "normal_with_pathology": int((p[:, ni] & (p.sum(1) > 1)).sum()),
                        "pct_of_normal": round(100 * (p[:, ni] & (p.sum(1) > 1)).sum() / max(p[:, ni].sum(), 1), 1)})
        _save(pd.DataFrame(out), "label_consistency.csv")



# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def evaluate(bm, idx2label):
    P, pids = load_preds(bm)
    R = bm.RESULTS_DIR
    y = next(iter(P.values()))["test_labels"]
    for k in P:
        assert np.array_equal(P[k]["test_labels"], y), f"test label order differs: {k}"
    mask = y.sum(0) > 0
    print(f"\nEvaluable classes (>=1 positive test instance): {mask.sum()}/{len(mask)} "
          f"| excluded: {[idx2label[c] for c in np.where(~mask)[0]]}")
    groups = cluster_indices(pids)
    print(f"Test set: {len(y)} images, {len(groups)} patients (bootstrap unit = patient)")

    # 1) all models at theta = 0.5 and at validation-selected thresholds
    rows, M, TH = [], {}, {}
    for k, d in P.items():
        th = thresholds_from_val(d["val_labels"], d["val_probs"]); TH[k] = th
        m05 = full_metrics(y, d["test_probs"], 0.5, mask)
        mth = full_metrics(y, d["test_probs"], th, mask)
        m05["f1_valthr"] = mth["f1_macro"]; M[k] = m05
        ci = boot_single(y, d["test_probs"], 0.5, groups)
        rows.append({"model": k, **{kk: round(v, 4) for kk, v in m05.items() if kk != "f1_valthr"},
                     "f1_ci": f"[{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]",
                     "mcc_ci": f"[{ci['mcc'][0]:.3f}, {ci['mcc'][1]:.3f}]",
                     "auprc_ci": f"[{ci['auprc'][0]:.3f}, {ci['auprc'][1]:.3f}]",
                     "f1_valthr": round(mth["f1_macro"], 4),
                     "mcc_valthr": round(mth["mcc_macro"], 4),
                     "dF1_valthr": round(mth["f1_macro"] - m05["f1_macro"], 4)})
        json.dump({idx2label[c]: float(th[c]) for c in range(len(th))},
                  open(R / f"thresholds_valselected_{k.replace('|', '__')}.json", "w"), indent=2)
    summ = pd.DataFrame(rows); summ.to_csv(R / "posthoc_summary.csv", index=False)
    print("\n=== SUMMARY (test set, macro over evaluable classes, patient-level 95% CI) ===")
    print(summ[["model", "f1_macro", "f1_ci", "mcc_macro", "mcc_ci", "auprc_macro",
                "auc_roc_macro", "exact_match", "f1_valthr", "dF1_valthr"]].to_string(index=False))

    supplementary_tables(bm, P, idx2label)

    # 2) fixed vs TPE
    ft = [{"model": n, "f1_fixed": round(M[f"{n}|fixed"]["f1_macro"], 4),
           "f1_tpe": round(M[f"{n}|tpe"]["f1_macro"], 4),
           "dF1_tpe_minus_fixed": round(M[f"{n}|tpe"]["f1_macro"] - M[f"{n}|fixed"]["f1_macro"], 4)}
          for n in bm.MODELS if f"{n}|fixed" in M and f"{n}|tpe" in M]
    if ft:
        ftd = pd.DataFrame(ft); ftd.to_csv(R / "posthoc_fixed_vs_tpe.csv", index=False)
        print("\n=== Fixed vs TPE-Hyperband (macro F1, theta = 0.5) ===")
        print(ftd.to_string(index=False))

    # 3) pairwise tests between the five fixed models
    main = [f"{n}|fixed" for n in bm.MODELS if f"{n}|fixed" in P]
    pw = pairwise_tests(main, P, y); pw.to_csv(R / "posthoc_pairwise_tests.csv", index=False)
    print("\n=== Wilcoxon (example-based F1) + McNemar (exact match), Holm/Bonferroni ===")
    print(pw.round(4).to_string(index=False))

    # 4) paired bootstrap differences
    comps = list(combinations(main, 2))
    comps += [("hybrid_eff_swin|fixed", "hybrid_cross_attn|ca"),
              ("hybrid_eff_swin|fixed", "hybrid_concat|ca")]
    comps += [(f"{n}|fixed", f"{n}|asl") for n in bm.MODELS if n != "hybrid_eff_swin"]
    drows = []
    for a, b in comps:
        if a in P and b in P:
            r = boot_paired(y, P[a]["test_probs"], 0.5, P[b]["test_probs"], 0.5, groups)
            drows.append({"A": a, "B": b, **{k: round(v, 4) for k, v in r.items()}})
    dd = pd.DataFrame(drows); dd.to_csv(R / "posthoc_paired_differences.csv", index=False)
    print("\n=== Patient-level paired bootstrap: A - B (theta = 0.5) ===")
    print(dd.to_string(index=False))

    # 5) additional baselines (secondary analysis): each vs Hybrid-EffSwin and Swin-T
    extra = [k for k in P if k.endswith("|extra")]
    if extra:
        erows = []
        for k in extra:
            m = M[k]
            ci = boot_single(y, P[k]["test_probs"], 0.5, groups)
            row = {"model": k.split("|")[0], "f1_macro": round(m["f1_macro"], 4),
                   "f1_ci": f"[{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]",
                   "mcc_macro": round(m["mcc_macro"], 4), "auprc_macro": round(m["auprc_macro"], 4),
                   "auc_roc_macro": round(m["auc_roc_macro"], 4), "exact_match": round(m["exact_match"], 4),
                   "hamming": round(m["hamming"], 4), "brier_macro": round(m["brier_macro"], 4)}
            for ref in ["hybrid_eff_swin|fixed", "swin_t|fixed"]:
                if ref in P:
                    r = boot_paired(y, P[ref]["test_probs"], 0.5, P[k]["test_probs"], 0.5, groups)
                    tag = ref.split("|")[0]
                    row[f"dF1_{tag}_minus_model"] = round(r["dF1"], 4)
                    row[f"dF1_{tag}_CI"] = f"[{r['dF1_lo']:.3f}, {r['dF1_hi']:.3f}]"
                    row[f"dF1_{tag}_p"] = round(r["dF1_p"], 3)
                    row[f"dMCC_{tag}_minus_model"] = round(r["dMCC"], 4)
                    row[f"dMCC_{tag}_CI"] = f"[{r['dMCC_lo']:.3f}, {r['dMCC_hi']:.3f}]"
            erows.append(row)
        ed = pd.DataFrame(erows); ed.to_csv(R / "posthoc_extra_baselines.csv", index=False)
        print("\n=== Additional baselines (secondary analysis; theta = 0.5) ===")
        print(ed.T.to_string())

    # 6) seed replicates: seed 42 (= the "fixed" run) plus SEED_RUNS
    seeds = [42] + list(getattr(bm, "SEED_RUNS", []))
    runs = {n: {42: f"{n}|fixed"} for n in bm.MODELS if f"{n}|fixed" in P}
    for n in runs:
        for sd in seeds[1:]:
            if f"{n}|seed{sd}" in P:
                runs[n][sd] = f"{n}|seed{sd}"
    if any(len(v) > 1 for v in runs.values()):
        srows = []
        for n, rr in runs.items():
            for sd, k in rr.items():
                m = M[k]
                srows.append({"model": n, "seed": sd, "f1_macro": m["f1_macro"], "mcc_macro": m["mcc_macro"],
                              "auprc_macro": m["auprc_macro"], "auc_roc_macro": m["auc_roc_macro"],
                              "exact_match": m["exact_match"]})
        sr = pd.DataFrame(srows)
        sr["rank_f1"] = sr.groupby("seed")["f1_macro"].rank(ascending=False, method="min")
        sr.round(4).to_csv(R / "posthoc_seed_runs.csv", index=False)
        agg = sr.groupby("model").agg(n_runs=("seed", "count"),
                                      f1_mean=("f1_macro", "mean"), f1_sd=("f1_macro", "std"),
                                      mcc_mean=("mcc_macro", "mean"), mcc_sd=("mcc_macro", "std"),
                                      auprc_mean=("auprc_macro", "mean"), auprc_sd=("auprc_macro", "std"),
                                      auc_mean=("auc_roc_macro", "mean"), auc_sd=("auc_roc_macro", "std"),
                                      acc_mean=("exact_match", "mean"), acc_sd=("exact_match", "std"),
                                      mean_rank_f1=("rank_f1", "mean")).reset_index()
        agg = agg.set_index("model").loc[[n for n in bm.MODELS if n in runs]].reset_index()
        agg.round(4).to_csv(R / "posthoc_seed_summary.csv", index=False)
        print("\n=== Seed replicates: mean +/- SD across training seeds (split fixed, seed 42) ===")
        print(agg.round(4).to_string(index=False))
        # per-seed paired differences Hybrid-EffSwin minus each baseline
        ref = "hybrid_eff_swin"
        if ref in runs:
            drows = []
            for n in runs:
                if n == ref:
                    continue
                for sd in seeds:
                    if sd in runs[ref] and sd in runs[n]:
                        r = boot_paired(y, P[runs[ref][sd]]["test_probs"], 0.5,
                                        P[runs[n][sd]]["test_probs"], 0.5, groups)
                        drows.append({"comparison": f"{ref} - {n}", "seed": sd,
                                      "dF1": round(r["dF1"], 4),
                                      "dF1_CI": f"[{r['dF1_lo']:.3f}, {r['dF1_hi']:.3f}]",
                                      "dF1_p": round(r["dF1_p"], 3), "dMCC": round(r["dMCC"], 4)})
            dd2 = pd.DataFrame(drows); dd2.to_csv(R / "posthoc_seed_paired.csv", index=False)
            print("\n=== Hybrid-EffSwin minus baseline, per training seed (patient-level bootstrap) ===")
            print(dd2.to_string(index=False))

    if RUN_EXTENDED:
        print("\n=== Extended analyses ===")
        extended_analyses(bm, P, pids, idx2label)

    if RUN_FIGURES:
        print("\n=== Figures ===")
        make_figures(bm, P, y, M, TH, idx2label)
    print(f"\nDone. Tables in {R}, figures in {bm.FIGURES_DIR}")


if __name__ == "__main__":
    bm = load_main()
    label2idx, idx2label = bm.load_label_encoder()
    if RUN_DATA_AUDIT:
        data_audit(bm, label2idx)
    if RUN_INFERENCE:
        run_inference(bm, label2idx)
    evaluate(bm, idx2label)
