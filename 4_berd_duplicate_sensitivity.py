"""
4_berd_duplicate_sensitivity.py — Sensitivity analysis for pixel-identical images (Table S3)

Identifies test images whose pixels are identical (MD5) to an image in the training or
validation subset and recomputes the test metrics with these images excluded.
Uses the cached predictions from 3_berd_posthoc_eval.py; no inference or training.

Output:  results/duplicate_sensitivity.csv
Run:     python 4_berd_duplicate_sensitivity.py   (after 3_berd_posthoc_eval.py)
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


bm = load(BASE_DIR / "2_berd_main.py", "berd_main")
pe = load(BASE_DIR / "3_berd_posthoc_eval.py", "berd_posthoc")

label2idx, idx2label = bm.load_label_encoder()
train_recs, val_recs = bm.split_patients_train_val(label2idx)
te = bm.BERDDataset(bm.TEST_JSON, bm.IMAGE_DIR, label2idx, None, "test")  # same order as inference

md5 = lambda r: hashlib.md5(bm.resolve_path(r["image_path"], bm.IMAGE_DIR).read_bytes()).hexdigest()
seen = {md5(r): ("train", r) for r in train_recs}
seen.update({md5(r): ("val", r) for r in val_recs})

dup_idx = []
print("\nTest images pixel-identical to a training/validation image:")
for i, r in enumerate(te.data):
    h = md5(r)
    if h in seen:
        src, tr = seen[h]
        dup_idx.append(i)
        print(f"  test[{i}] {r['image_path']} (patient {r.get('patient_id')}, labels '{r.get('label')}')"
              f"  ==  {src}: {tr['image_path']} (patient {tr.get('patient_id')}, labels '{tr.get('label')}')")

P, _ = pe.load_preds(bm)
y = next(iter(P.values()))["test_labels"]
keep = np.setdiff1d(np.arange(len(y)), dup_idx)
mask_full, mask_keep = y.sum(0) > 0, y[keep].sum(0) > 0
print(f"\nTest images: {len(y)} -> {len(keep)}; evaluable classes {mask_full.sum()} -> {mask_keep.sum()}")

rows = []
for k in [f"{n}|fixed" for n in bm.MODELS]:
    s = P[k]["test_probs"]
    a = pe.full_metrics(y, s, 0.5, mask_full)
    b = pe.full_metrics(y[keep], s[keep], 0.5, mask_keep)
    rows.append({"model": k.split("|")[0],
                 "f1_all": round(a["f1_macro"], 4), "f1_excl": round(b["f1_macro"], 4),
                 "mcc_all": round(a["mcc_macro"], 4), "mcc_excl": round(b["mcc_macro"], 4),
                 "auprc_all": round(a["auprc_macro"], 4), "auprc_excl": round(b["auprc_macro"], 4),
                 "acc_all": round(a["exact_match"], 4), "acc_excl": round(b["exact_match"], 4)})
df = pd.DataFrame(rows)
df["dF1"] = (df.f1_excl - df.f1_all).round(4)
print(df.to_string(index=False))
df.to_csv(bm.RESULTS_DIR / "duplicate_sensitivity.csv", index=False)
print(f"\nSaved: {bm.RESULTS_DIR / 'duplicate_sensitivity.csv'}")
