"""
berd_duplicate_sensitivity.py
Eğitim/validation görüntüleriyle piksel-özdeş (MD5) olan test görüntülerini
listeler ve bunlar ÇIKARILDIĞINDA metriklerin nasıl değiştiğini hesaplar.
Yeniden eğitim/inference YOK — results/preds_cache/ kullanılır.
Spyder: runfile('C:/sibel/diagnostics/berd_duplicate_sensitivity.py', wdir='C:/sibel/diagnostics')
"""
import hashlib, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path("C:/sibel/diagnostics")

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

bm = load(BASE / "Berd_main08-11.py", "berd_main")
pe = load(BASE / "berd_posthoc_eval.py", "berd_posthoc")

label2idx, idx2label = bm.load_label_encoder()
train_recs, val_recs = bm.split_patients_train_val(label2idx)
te = bm.BERDDataset(bm.TEST_JSON, bm.IMAGE_DIR, label2idx, None, "test")   # inference ile AYNI sıra

md5 = lambda r: hashlib.md5(bm.resolve_path(r["image_path"], bm.IMAGE_DIR).read_bytes()).hexdigest()
train_hash = {md5(r): ("train", r) for r in train_recs}
train_hash.update({md5(r): ("val", r) for r in val_recs})

dup_idx = []
print("\nEğitim/val görüntüsüyle piksel-özdeş TEST görüntüleri:")
for i, r in enumerate(te.data):
    h = md5(r)
    if h in train_hash:
        src, tr = train_hash[h]
        dup_idx.append(i)
        print(f"  test[{i}] {r['image_path']} (hasta {r.get('patient_id')}, etiket '{r.get('label')}')"
              f"  ==  {src}: {tr['image_path']} (hasta {tr.get('patient_id')}, etiket '{tr.get('label')}')")

P, pids = pe.load_preds(bm)
y = next(iter(P.values()))["test_labels"]
keep = np.setdiff1d(np.arange(len(y)), dup_idx)
mask_full = y.sum(0) > 0
mask_keep = y[keep].sum(0) > 0
print(f"\nTest: {len(y)} → {len(keep)} görüntü; değerlendirilebilir sınıf {mask_full.sum()} → {mask_keep.sum()}")

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
print(f"\n✓ {bm.RESULTS_DIR / 'duplicate_sensitivity.csv'}")