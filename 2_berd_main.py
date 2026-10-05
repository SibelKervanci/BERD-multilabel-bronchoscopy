"""
2_berd_main.py — Model training for multi-label bronchoscopic finding classification (BERD)

Trains all configurations reported in the manuscript and saves the best checkpoint of
each run (selected by validation macro F1). The test set is NOT used here; all test-set
evaluation, statistics, and figures are produced by 3_berd_posthoc_eval.py.

Experiments
  1. TPE     : 5 architectures with hyperparameters from 1_berd_tpe_search.py
  2. FIXED   : ResNet-50, EfficientNet-B3, ViT-B/16, Swin-T with fixed hyperparameters
  3. ASL     : same 4 architectures trained with Asymmetric Loss
  4. FUSION  : Hybrid-EffSwin (average pooling), residual-projection and concatenation variants
               (the average-pooling run is the "fixed" Hybrid-EffSwin model)

Protocol
  - Official BERD training set split 80/20 at the patient level (seed 42) into training
    and validation subsets; class weights are computed on the training subset only.
  - Early stopping and learning-rate scheduling monitor validation macro F1.
  - Existing checkpoints are skipped unless FORCE_RETRAIN = True.

Run:  python 2_berd_main.py
"""

import json, pickle, time, warnings
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss, matthews_corrcoef,
    average_precision_score, roc_auc_score, precision_score,
    recall_score, label_ranking_average_precision_score,
    brier_score_loss, roc_curve, precision_recall_curve,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR      = Path(__file__).resolve().parent
DATA_ROOT     = BASE_DIR / "dataset"
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR   = BASE_DIR / "results"
FIGURES_DIR   = BASE_DIR / "figures"

CKPT_TPE   = BASE_DIR / "checkpoints_tpe"
CKPT_FIXED = BASE_DIR / "checkpoints_fixed"
CKPT_ASL   = BASE_DIR / "checkpoints_asl"
CKPT_CA    = BASE_DIR / "checkpoints_ca"
CKPT_EXTRA = BASE_DIR / "checkpoints_extra"   # additional baselines (secondary analysis)
CKPT_SEEDS = BASE_DIR / "checkpoints_seeds"   # seed replicates of the fixed configuration

TRAIN_JSON    = DATA_ROOT / "annotations" / "dataset_train.json"
TEST_JSON     = DATA_ROOT / "annotations" / "dataset_test.json"
IMAGE_DIR     = DATA_ROOT / "images"
LABEL_ENCODER = PROCESSED_DIR / "label_encoder.pkl"
CLASS_WEIGHTS = PROCESSED_DIR / "class_weights.pkl"

KNOWN_LABELS = [
    "blood","clot","congested","edematous","external pressure",
    "fistula","granulation","infiltration changes","mass","narrow",
    "necrotic","neoplasm","new organism","nodules","normal",
    "pigmentation","postoperative change","rough","sputum",
    "surgical stump","tube","ulcer","widened",
]

# ── Sabit hiperparametreler ───────────────────────────────────────
FIXED_LR         = 1e-4
FIXED_WD         = 1e-4
FIXED_MAX_WEIGHT = 50.0
FIXED_DROPOUT_CNN= 0.4
FIXED_DROPOUT_HYB= 0.3

# ── Eğitim ayarları ───────────────────────────────────────────────
BATCH_SIZE   = 32
# ── PERFORMANS DÜZELTMESİ (berd_tpe_search.py ile aynı gerekçe) ──
# NUM_WORKERS=0, tüm augmentasyonların CPU'da tek işlemde seri
# yapılmasına yol açıyordu. Kendi çekirdek sayınıza göre ayarlayın
# (import os; print(os.cpu_count())). 20 çekirdekli makinenizde 14
# civarı makul bir başlangıç.
NUM_WORKERS  = 0
PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR    = 2 if NUM_WORKERS > 0 else None
SEED         = 42
VAL_FRACTION = 0.20   # berd_tpe_search.py ile AYNI oran — split örtüşsün
FINAL_EPOCHS = 70
LR_PATIENCE  = 5
EARLY_STOP   = 15
THRESHOLD    = 0.5
BOOTSTRAP_N  = 1000

# ── ASL parametreleri ─────────────────────────────────────────────
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 0.0
ASL_CLIP      = 0.05

# ── Hangi deneyleri çalıştır ─────────────────────────────────────
RUN_TPE_TRAIN   = True   # TPE parametreleriyle 5 model (artık hibrit dahil)
RUN_FIXED_TRAIN = True   # Sabit parametrelerle 4 model (hibrit CA'da eğitiliyor)
RUN_ASL         = True   # ASL ablation (sabit params, 4 model)
RUN_CA_ABLATION = True
RUN_EXTRA_BASELINES = True   # ConvNeXt-T, DINOv2 ViT-B/14, BiomedCLIP ViT-B/16 (secondary analysis)
RUN_SEED_REPLICATES = True   # re-train the fixed configuration with additional training seeds
# The patient-level split always uses SEED = 42; only the training seed changes
# (weight initialisation of the heads, sampler order, dropout masks, MixUp).
SEED_RUNS   = [7, 123]                      # together with seed 42 -> 3 runs per model
SEED_MODELS = ["resnet50","efficientnet_b3","vit_b16","swin_t","hybrid_eff_swin"]
# fastest useful option: SEED_MODELS = ["swin_t","hybrid_eff_swin"]   # Cross-attention ablation (3 hibrit varyantı)
FORCE_RETRAIN   = False  # True: checkpoint olsa bile yeniden eğit

MODELS = ["resnet50","efficientnet_b3","vit_b16","swin_t","hybrid_eff_swin"]
# Additional baselines, trained with the same fixed hyperparameters but analysed separately
# so that the pre-specified primary comparison (MODELS, Holm over 10 pairs) is unchanged.
EXTRA_MODELS = ["convnext_t","dinov2_vitb14","biomedclip_vitb16"]
# Foundation models are fine-tuned with a reduced backbone learning rate (head keeps FIXED_LR)
FOUNDATION_MODELS = ["dinov2_vitb14","biomedclip_vitb16"]
FOUNDATION_BACKBONE_LR_MULT = 0.1
CA_VARIANTS = ["hybrid_eff_swin","hybrid_concat","hybrid_cross_attn"]  # ← artık üçü de

for d in [CKPT_TPE, CKPT_FIXED, CKPT_ASL, CKPT_CA, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  YARDIMCI                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_labels(lf, valid=None):
    if lf is None: return []
    raw = [l.strip().lower() for l in
           (lf if isinstance(lf,list) else lf.split(",")) if str(l).strip()]
    return [l for l in raw if l in valid] if valid else raw

def resolve_path(ip, image_dir):
    p = Path(ip); c = image_dir/p.name
    return c if c.exists() else DATA_ROOT/ip

def load_json(path):
    with open(path, encoding="utf-8") as f: d = json.load(f)
    return d if isinstance(d,list) else d.get("annotations",[])

def load_label_encoder():
    with open(LABEL_ENCODER,"rb") as f: enc = pickle.load(f)
    filtered = [l for l in enc["idx2label"] if l in KNOWN_LABELS]
    return {l:i for i,l in enumerate(filtered)}, filtered

def load_tpe_params(model_name):
    """TPE JSON'unu oku. Yoksa sabit parametreleri döndür."""
    path = RESULTS_DIR / f"{model_name}_tpe_results.json"
    if not path.exists():
        print(f"  [UYARI] TPE sonucu yok: {path}")
        print(f"  Sabit parametreler kullanılacak.")
        dr = FIXED_DROPOUT_HYB if "hybrid" in model_name else FIXED_DROPOUT_CNN
        return {"lr":FIXED_LR,"wd":FIXED_WD,"dropout":dr,"max_weight":FIXED_MAX_WEIGHT}
    with open(path) as f: data = json.load(f)
    p = data["best_params"]
    if "weight_decay" not in p and "wd" in p:
        p["weight_decay"] = p.pop("wd")
    return p


# ╔══════════════════════════════════════════════════════════════╗
# ║  DATASET + DATALOADER                                         ║
# ╚══════════════════════════════════════════════════════════════╝

class BERDDataset(Dataset):
    def __init__(self, json_path, image_dir, label2idx, transform=None, split="train"):
        self.image_dir=Path(image_dir); self.label2idx=label2idx
        self.valid=set(label2idx.keys()); self.nc=len(label2idx); self.transform=transform
        records=load_json(json_path)
        self.data=[r for r in records
                   if resolve_path(r.get("image_path",""),self.image_dir).exists()
                   and parse_labels(r.get("label",""),self.valid)]
        print(f"[Dataset] {split}: {len(self.data)} geçerli")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        rec=self.data[idx]
        img=Image.open(resolve_path(rec["image_path"],self.image_dir)).convert("RGB")
        if self.transform: img=self.transform(img)
        vec=torch.zeros(self.nc)
        for l in parse_labels(rec.get("label",""),self.valid): vec[self.label2idx[l]]=1.0
        return img, vec, {"label": rec.get("label",""), "id": rec.get("image_id",str(idx))}

def get_tf(split):
    mean,std=[0.485,0.456,0.406],[0.229,0.224,0.225]
    if split=="train":
        return transforms.Compose([
            transforms.Resize((256,256)), transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(15), transforms.ColorJitter(0.25,0.25,0.1,0.05),
            transforms.RandomAutocontrast(0.4), transforms.ToTensor(),
            transforms.Normalize(mean,std),
            transforms.RandomApply([transforms.GaussianBlur(3,(0.1,1.5))],0.2)])
    return transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(), transforms.Normalize(mean,std)])

class MixUp:
    def __call__(self, batch):
        imgs,labels,metas=zip(*batch); imgs,labels=torch.stack(imgs),torch.stack(labels)
        if np.random.random()<0.3:
            lam=float(np.random.beta(0.4,0.4)); idx=torch.randperm(imgs.size(0))
            imgs=lam*imgs+(1-lam)*imgs[idx]; labels=lam*labels+(1-lam)*labels[idx]
        return imgs,labels,list(metas)


def split_patients_train_val(label2idx):
    """
    Hasta bazlı, DETERMİNİSTİK train/val ayrımı — SADECE TRAIN_JSON
    üzerinden. TEST_JSON bu fonksiyona hiç girmez.

    berd_tpe_search.py'deki split_patients_train_val() ile BİREBİR
    AYNI mantık (aynı SEED, aynı VAL_FRACTION, aynı sorted()+shuffle
    sırası) — iki dosya da aynı hasta grubunu train/val'e ayırır.
    """
    all_records = load_json(TRAIN_JSON)
    valid_set   = set(label2idx.keys())

    valid_records = [
        r for r in all_records
        if resolve_path(r.get("image_path",""), Path(IMAGE_DIR)).exists()
        and parse_labels(r.get("label",""), valid_set)
    ]

    # ── DETERMİNİZM DÜZELTMESİ (berd_tpe_search.py ile aynı) ─────
    patient_ids = sorted({r.get("patient_id","") for r in valid_records
                          if r.get("patient_id","")})
    rng = np.random.default_rng(SEED)
    rng.shuffle(patient_ids)

    val_size      = int(len(patient_ids) * VAL_FRACTION)
    val_patients  = set(patient_ids[:val_size])
    train_patients= set(patient_ids[val_size:])

    train_records = [r for r in valid_records if r.get("patient_id","") in train_patients]
    val_records   = [r for r in valid_records if r.get("patient_id","") in val_patients]

    print(f"[Split] Train: {len(train_patients)} hasta, {len(train_records)} görüntü")
    print(f"[Split] Val  : {len(val_patients)} hasta, {len(val_records)} görüntü")

    overlap = train_patients & val_patients
    assert not overlap, f"[HATA] train/val arasında {len(overlap)} ortak hasta var!"

    # ── SIZINTI/TUTARLILIK DÜZELTMESİ: class_weights.pkl artık
    # SADECE bu train_records'tan, split sonrası hesaplanıyor —
    # dışarıdaki hiçbir dosyaya (örn. tpe_search'ün kendi run'ı) bağlı
    # değil. Her build_loaders() çağrısında güncel split'e göre
    # yeniden yazılır.
    train_label_counts = Counter()
    for r in train_records:
        train_label_counts.update(parse_labels(r.get("label",""), valid_set))
    n_train = len(train_records)
    class_weights = {l: (n_train - c) / max(c, 1) for l, c in train_label_counts.items()}
    with open(CLASS_WEIGHTS, "wb") as f:
        pickle.dump(class_weights, f)
    print(f"[Split] class_weights.pkl SADECE train split'inden yeniden hesaplandı "
          f"({len(train_label_counts)} sınıf).")

    return train_records, val_records


def build_loaders(label2idx):
    """
    Hasta bazlı train/val/test split:
      - BERD test seti (180 hasta) → final test, dokunulmaz
      - BERD train seti → hasta bazlı %80/%20 split (split_patients_train_val)
    Bu yapı tam sızıntı-sız: aynı hastanın görüntüleri asla farklı setlerde değil.
    """
    train_records, val_records = split_patients_train_val(label2idx)

    tr = BERDDataset(TRAIN_JSON, IMAGE_DIR, label2idx, get_tf("train"), "train")
    tr.data = train_records

    va = BERDDataset(TRAIN_JSON, IMAGE_DIR, label2idx, get_tf("test"), "val")
    va.data = val_records

    te = BERDDataset(TEST_JSON, IMAGE_DIR, label2idx, get_tf("test"), "test")

    print(f"[Loaders] Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")

    # Weighted sampler — sadece train
    cnt = Counter()
    for r in tr.data: cnt.update(parse_labels(r.get("label",""), tr.valid))
    tot = sum(cnt.values())
    lw  = {l: tot/max(c,1) for l,c in cnt.items()}
    sw  = [max((lw.get(l,1) for l in parse_labels(r.get("label",""),tr.valid)),default=1)
           for r in tr.data]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(sw), replacement=True)

    dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True)
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        dl_kwargs["prefetch_factor"]    = PREFETCH_FACTOR

    trl = DataLoader(tr, BATCH_SIZE, sampler=sampler,
                     collate_fn=MixUp(), drop_last=True, **dl_kwargs)
    val_loader = DataLoader(va, BATCH_SIZE, shuffle=False, **dl_kwargs)
    tel = DataLoader(te, BATCH_SIZE, shuffle=False, **dl_kwargs)
    return trl, val_loader, tel


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODELLER                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

from torchvision import models as M
from torchvision.models import (ResNet50_Weights, EfficientNet_B3_Weights,
                                 Swin_T_Weights, ViT_B_16_Weights)

def _head(f,nc,dr):
    return nn.Sequential(nn.Dropout(dr),nn.Linear(f,512),nn.GELU(),
                          nn.Dropout(dr/2),nn.Linear(512,nc))

class ResNet50(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        f=b.fc.in_features; b.fc=nn.Identity(); self.b=b; self.h=_head(f,nc,dr)
    def forward(self,x): return self.h(self.b(x))
    def features(self,x): return self.b(x)

class EffB3(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        f=b.classifier[1].in_features; b.classifier=nn.Identity(); self.b=b; self.h=_head(f,nc,dr)
    def forward(self,x): return self.h(self.b(x))
    def features(self,x): return self.b(x)

class ViTB16(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        f=b.heads.head.in_features; b.heads.head=nn.Identity(); self.b=b
        self.h=nn.Sequential(nn.LayerNorm(f),nn.Dropout(dr),nn.Linear(f,512),
                              nn.GELU(),nn.Dropout(dr/2),nn.Linear(512,nc))
    def forward(self,x): return self.h(self.b(x))
    def features(self,x): return self.b(x)

class SwinT(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        f=b.head.in_features; b.head=nn.Identity(); self.b=b
        self.h=nn.Sequential(nn.LayerNorm(f),nn.Dropout(dr),nn.Linear(f,512),
                              nn.GELU(),nn.Dropout(dr/2),nn.Linear(512,nc))
    def forward(self,x): return self.h(self.b(x))
    def features(self,x): return self.b(x)

class CrossAttn(nn.Module):
    """Multi-head attention with the CNN feature as query and the Transformer feature as key/value.
    Both inputs are single pooled vectors (sequence length 1), so the softmax weight is identically 1
    and the query has no effect: the block computes LayerNorm(q + W_o W_v k), i.e. a learned residual
    projection of the Transformer feature. It is reported as 'residual-projection fusion'."""
    def __init__(self,d,h=8):
        super().__init__(); self.a=nn.MultiheadAttention(d,h,batch_first=True); self.n=nn.LayerNorm(d)
    def forward(self,q,k):
        o,_=self.a(q.unsqueeze(1),k.unsqueeze(1),k.unsqueeze(1)); return self.n(o.squeeze(1)+q)

class HybridCrossAttn(nn.Module):
    def __init__(self,nc,dr):
        super().__init__()
        eff=M.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        cd=eff.classifier[1].in_features; eff.classifier=nn.Identity(); self.cnn=eff
        sw=M.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        td=sw.head.in_features; sw.head=nn.Identity(); self.tr=sw; H=512
        self.cp=nn.Sequential(nn.Linear(cd,H),nn.LayerNorm(H),nn.GELU())
        self.tp=nn.Sequential(nn.Linear(td,H),nn.LayerNorm(H),nn.GELU())
        self.ca=CrossAttn(H); self.h=nn.Sequential(nn.Dropout(dr/2),nn.Linear(H,nc))
    def forward(self,x): return self.h(self.ca(self.cp(self.cnn(x)),self.tp(self.tr(x))))
    def features(self,x): return self.ca(self.cp(self.cnn(x)),self.tp(self.tr(x)))

class HybridEffSwin(nn.Module):
    """CA Ablation A: average pooling — ASIL ÖNERİLEN MODEL"""
    def __init__(self,nc,dr):
        super().__init__()
        eff=M.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        cd=eff.classifier[1].in_features; eff.classifier=nn.Identity(); self.cnn=eff
        sw=M.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        td=sw.head.in_features; sw.head=nn.Identity(); self.tr=sw; H=512
        self.cp=nn.Sequential(nn.Linear(cd,H),nn.LayerNorm(H),nn.GELU())
        self.tp=nn.Sequential(nn.Linear(td,H),nn.LayerNorm(H),nn.GELU())
        self.h=nn.Sequential(nn.Dropout(dr/2),nn.Linear(H,nc))
    def forward(self,x):
        return self.h((self.cp(self.cnn(x))+self.tp(self.tr(x)))/2.0)

class HybridConcat(nn.Module):
    """CA Ablation B: concatenation + linear"""
    def __init__(self,nc,dr):
        super().__init__()
        eff=M.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        cd=eff.classifier[1].in_features; eff.classifier=nn.Identity(); self.cnn=eff
        sw=M.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        td=sw.head.in_features; sw.head=nn.Identity(); self.tr=sw; H=512
        self.cp=nn.Sequential(nn.Linear(cd,H),nn.LayerNorm(H),nn.GELU())
        self.tp=nn.Sequential(nn.Linear(td,H),nn.LayerNorm(H),nn.GELU())
        self.fusion=nn.Sequential(nn.Linear(H*2,H),nn.LayerNorm(H),nn.GELU())
        self.h=nn.Sequential(nn.Dropout(dr/2),nn.Linear(H,nc))
    def forward(self,x):
        return self.h(self.fusion(torch.cat([self.cp(self.cnn(x)),
                                             self.tp(self.tr(x))],dim=1)))


def _feat_dim(backbone):
    """Output dimension of a backbone, determined with a dummy forward pass."""
    with torch.no_grad():
        was=backbone.training; backbone.eval()
        d=backbone(torch.zeros(1,3,224,224)).shape[-1]
        backbone.train(was)
    return d

def _tr_head(f,nc,dr):
    return nn.Sequential(nn.LayerNorm(f),nn.Dropout(dr),nn.Linear(f,512),
                         nn.GELU(),nn.Dropout(dr/2),nn.Linear(512,nc))

class ConvNeXtT(nn.Module):
    """ConvNeXt-Tiny (ImageNet-1k) -- modern CNN baseline."""
    def __init__(self,nc,dr):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights
        b=M.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        b.classifier[2]=nn.Identity()          # keep LayerNorm2d + Flatten -> 768-d
        self.b=b; self.h=_head(_feat_dim(b),nc,dr)
    def forward(self,x): return self.h(self.b(x))

class DINOv2ViTB14(nn.Module):
    """DINOv2 ViT-B/14 (self-supervised foundation model, LVD-142M), full fine-tuning.
    Requires internet access on first use (torch.hub download)."""
    def __init__(self,nc,dr):
        super().__init__()
        b=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14")   # forward -> CLS embedding
        self.b=b; self.h=_tr_head(_feat_dim(b),nc,dr)
    def forward(self,x): return self.h(self.b(x))

class BiomedCLIPViTB16(nn.Module):
    """BiomedCLIP image encoder (ViT-B/16, PMC-15M biomedical image-text pretraining), full fine-tuning.
    Requires: pip install open_clip_torch  (weights downloaded from Hugging Face on first use)."""
    _IN_MEAN=(0.485,0.456,0.406); _IN_STD=(0.229,0.224,0.225)
    _CLIP_MEAN=(0.48145466,0.4578275,0.40821073); _CLIP_STD=(0.26862954,0.26130258,0.27577711)
    def __init__(self,nc,dr):
        super().__init__()
        import open_clip
        clip,_=open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.b=clip.visual; del clip                 # image tower only
        # inputs arrive ImageNet-normalized; re-normalize to CLIP statistics inside the model
        t=lambda v: torch.tensor(v).view(1,3,1,1)
        self.register_buffer("in_mean",t(self._IN_MEAN)); self.register_buffer("in_std",t(self._IN_STD))
        self.register_buffer("cl_mean",t(self._CLIP_MEAN)); self.register_buffer("cl_std",t(self._CLIP_STD))
        self.h=_tr_head(_feat_dim(self.b),nc,dr)
    def forward(self,x):
        x=(x*self.in_std+self.in_mean-self.cl_mean)/self.cl_std
        return self.h(self.b(x))

def build_model(name, nc, dropout=None):
    if dropout is None:
        dropout = FIXED_DROPOUT_HYB if "hybrid" in name else FIXED_DROPOUT_CNN
    catalogue = {
        "resnet50":           lambda: ResNet50(nc,dropout),
        "efficientnet_b3":    lambda: EffB3(nc,dropout),
        "vit_b16":            lambda: ViTB16(nc,dropout),
        "swin_t":             lambda: SwinT(nc,dropout),
        "hybrid_eff_swin":    lambda: HybridEffSwin(nc,dropout),   # avg pool — ana model
        "hybrid_cross_attn":  lambda: HybridCrossAttn(nc,dropout), # ablation: residual projection
        "hybrid_concat":      lambda: HybridConcat(nc,dropout),    # ablation: concatenation
        "convnext_t":         lambda: ConvNeXtT(nc,dropout),       # additional baseline
        "dinov2_vitb14":      lambda: DINOv2ViTB14(nc,dropout),    # additional baseline (foundation)
        "biomedclip_vitb16":  lambda: BiomedCLIPViTB16(nc,dropout),# additional baseline (biomedical)
    }
    model = catalogue[name]()
    n = sum(p.numel() for p in model.parameters())/1e6
    print(f"[Model] {name}: {n:.1f}M params, dropout={dropout:.3f}")
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║  LOSS                                                         ║
# ╚══════════════════════════════════════════════════════════════╝

def build_bce(label2idx, device, max_weight=FIXED_MAX_WEIGHT):
    with open(CLASS_WEIGHTS,"rb") as f: wd=pickle.load(f)
    w=torch.tensor([min(wd.get(l,1.0),max_weight) for l in label2idx],
                   dtype=torch.float32).to(device)
    print(f"[Loss] BCE (max_weight={max_weight})")
    return nn.BCEWithLogitsLoss(pos_weight=w)

class AsymmetricLoss(nn.Module):
    """Ridnik et al. ICCV 2021. arXiv:2009.14119"""
    def __init__(self,gn=4.0,gp=0.0,clip=0.05,eps=1e-8):
        super().__init__()
        self.gn,self.gp,self.clip,self.eps=gn,gp,clip,eps
    def forward(self,logits,targets):
        p=torch.sigmoid(logits)
        pn=(p+self.clip).clamp(max=1.0) if self.clip>0 else p
        lp=targets*torch.log(p.clamp(min=self.eps))
        ln=(1-targets)*torch.log((1-pn).clamp(min=self.eps))
        if self.gn>0 or self.gp>0:
            w=targets*(1-p)**self.gp+(1-targets)*(1-(1-pn))**self.gn
            return -(w*(lp+ln)).mean()
        return -(lp+ln).mean()

def build_asl():
    print(f"[Loss] ASL (γ_neg={ASL_GAMMA_NEG}, γ_pos={ASL_GAMMA_POS}, clip={ASL_CLIP})")
    return AsymmetricLoss(ASL_GAMMA_NEG,ASL_GAMMA_POS,ASL_CLIP)


# ╔══════════════════════════════════════════════════════════════╗
# ║  EĞİTİM                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def train_epoch(model, loader, opt, criterion, scaler, device):
    model.train(); total=0; n=len(loader)
    for i,(imgs,labels,_) in enumerate(loader):
        imgs,labels=imgs.to(device,non_blocking=True),labels.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(): loss=criterion(model(imgs),labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(opt); scaler.update()
        total+=loss.item()
        if (i+1)%max(1,n//4)==0:
            print(f"  [{i+1}/{n}] loss={loss.item():.4f}")
    return total/n

@torch.no_grad()
def quick_f1(model, loader, device):
    model.eval(); probs,labels=[],[]
    for imgs,lbs,_ in loader:
        with autocast():
            p=torch.sigmoid(model(imgs.to(device))).cpu().float().numpy()
        probs.append(p); labels.append(lbs.numpy())
    probs,labels=np.vstack(probs),np.vstack(labels)
    return f1_score(labels,(probs>=THRESHOLD).astype(float),
                   average="macro",zero_division=0)

def train_model(name, nc, label2idx, train_loader, val_loader, test_loader,
                device, criterion, ckpt_dir, experiment, params, train_seed=None):
    ckpt_path = ckpt_dir / f"{name}_{experiment}_best.pt"
    log_path  = RESULTS_DIR / f"{name}_{experiment}_log.json"

    if ckpt_path.exists() and not FORCE_RETRAIN:
        print(f"  [{experiment}] {name}: cache bulundu → atlanıyor.")
        return

    # ── DETERMİNİZM: her modelin ağırlık başlatması/dropout'u run'lar
    # arası tam tekrarlanabilir olsun diye burada sıfırlanıyor.
    if train_seed is None:
        torch.manual_seed(SEED)                     # original runs: unchanged behaviour
    else:                                           # seed replicates
        import random
        random.seed(train_seed); np.random.seed(train_seed)
        torch.manual_seed(train_seed); torch.cuda.manual_seed_all(train_seed)

    lr  = params.get("lr", FIXED_LR)
    wd  = params.get("weight_decay", params.get("wd", FIXED_WD))
    dr  = params.get("dropout", FIXED_DROPOUT_HYB if "hybrid" in name else FIXED_DROPOUT_CNN)
    mw  = params.get("max_weight", FIXED_MAX_WEIGHT)

    print(f"\n{'='*55}")
    print(f"[{experiment.upper()}] {name} | lr={lr:.2e} | max_w={mw:.1f} | dr={dr:.3f}")
    print(f"{'='*55}")

    model=build_model(name,nc,dr).to(device)
    if name in FOUNDATION_MODELS:
        # Foundation backbones: 10x lower learning rate for the pretrained backbone, full rate for
        # the new head (standard fine-tuning practice; a uniform 1e-4 degraded DINOv2 features).
        opt=AdamW([{"params":model.b.parameters(),"lr":lr*FOUNDATION_BACKBONE_LR_MULT},
                   {"params":model.h.parameters(),"lr":lr}],weight_decay=wd)
        print(f"  [LR] backbone={lr*FOUNDATION_BACKBONE_LR_MULT:.1e} | head={lr:.1e}")
    else:
        opt=AdamW(model.parameters(),lr=lr,weight_decay=wd)
    sched=ReduceLROnPlateau(opt,mode="max",patience=LR_PATIENCE,factor=0.5)
    scaler=GradScaler()
    best_f1,patience_cnt,history=0.0,0,[]

    try:
        for epoch in range(FINAL_EPOCHS):
            t0=time.time()
            print(f"\nEpoch {epoch+1}/{FINAL_EPOCHS}")
            tl=train_epoch(model,train_loader,opt,criterion,scaler,device)
            vf=quick_f1(model,val_loader,device)  # ✓ val set, not test set
            sched.step(vf); elapsed=time.time()-t0
            history.append({"epoch":epoch+1,"train_loss":round(tl,4),
                            "val_f1":round(vf,4),"lr":opt.param_groups[0]["lr"],
                            "time_s":round(elapsed,1)})
            print(f"  train_loss={tl:.4f} | val_f1={vf:.4f} | {elapsed:.0f}s")
            if vf>best_f1:
                best_f1,patience_cnt=vf,0
                torch.save({"model_state":model.state_dict(),"best_f1":best_f1,
                            "model_name":name,"num_classes":nc,"label2idx":label2idx,
                            "experiment":experiment,"params":params},ckpt_path)
                print(f"  ✓ En iyi F1: {best_f1:.4f}")
            else:
                patience_cnt+=1
            if patience_cnt>=EARLY_STOP:
                print(f"  Erken durdurma."); break
    finally:
        # GPU bellek temizliği — art arda 15+ model eğitiminde birikmeyi önler
        del model, opt, sched, scaler
        torch.cuda.empty_cache()

    log_path.write_text(json.dumps(history,indent=2))
    print(f"\n  ✓ {name} ({experiment}) tamamlandı. En iyi F1: {best_f1:.4f}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  DEĞERLENDİRME                                                ║
# ╚══════════════════════════════════════════════════════════════╝

@torch.no_grad()
def get_preds(model, loader, device):
    model.eval(); probs,labels=[],[]
    for imgs,lbs,_ in loader:
        with autocast():
            p=torch.sigmoid(model(imgs.to(device))).cpu().float().numpy()
        probs.append(p); labels.append(lbs.numpy())
    probs,labels=np.vstack(probs),np.vstack(labels)
    return probs,(probs>=THRESHOLD).astype(float),labels

# ╔══════════════════════════════════════════════════════════════╗
# ║  PLOTTING UTILITIES (called by 3_berd_posthoc_eval.py)        ║
# ╚══════════════════════════════════════════════════════════════╝

COLORS=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
MODEL_LABELS=["ResNet-50","EfficientNet-B3","ViT-B/16","Swin-T","Hybrid-EffSwin"]

def plot_model_comparison(summary_df, tag=""):
    metrics=[("f1_macro","F1 Macro ↑"),("mcc_macro","MCC ↑"),
             ("auprc_macro","AUPRC ↑"),("accuracy","Accuracy ↑"),
             ("hamming_loss","Hamming ↓"),("auc_roc_macro","AUC-ROC ↑")]
    available=[(m,l) for m,l in metrics if m in summary_df.columns]
    fig,axes=plt.subplots(2,3,figsize=(16,9)); axes=axes.flatten()
    lower_better={"hamming_loss","brier_macro"}
    for i,(metric,label) in enumerate(available[:6]):
        ax=axes[i]
        vals=[float(summary_df[summary_df["model"]==m][metric].values[0])
              if m in summary_df["model"].values else 0 for m in MODELS]
        best=min(vals) if metric in lower_better else max(vals)
        colors=["#DD8452" if abs(v-best)<1e-6 else "#4C72B0" for v in vals]
        bars=ax.bar(MODEL_LABELS,vals,color=colors,alpha=0.85)
        ax.set_title(label,fontsize=11)
        ax.tick_params(axis="x",rotation=30,labelsize=9)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,
                    f"{v:.3f}",ha="center",fontsize=8)
    for j in range(i+1,6): axes[j].set_visible(False)
    plt.suptitle("Model Performance Comparison (fixed hyperparameters)",fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/f"model_comparison{tag}.png",dpi=300,bbox_inches="tight")
    plt.close(); print(f"✓ model_comparison{tag}.png")

def plot_tpe_vs_fixed(tpe_df, fixed_df):
    """TPE parametreleri vs sabit parametreler karşılaştırması."""
    metrics=[("f1_macro","F1 Macro"),("mcc_macro","MCC"),
             ("auprc_macro","AUPRC"),("accuracy","Accuracy")]
    fig,axes=plt.subplots(1,4,figsize=(16,5))
    x=np.arange(len(MODELS)); w=0.35
    for i,(metric,label) in enumerate(metrics):
        ax=axes[i]
        tv=[float(tpe_df[tpe_df["model"]==m][metric].values[0])
            if m in tpe_df["model"].values else 0 for m in MODELS]
        fv=[float(fixed_df[fixed_df["model"]==m][metric].values[0])
            if m in fixed_df["model"].values else 0 for m in MODELS]
        ax.bar(x-w/2,fv,w,label="Fixed",color="#4C72B0",alpha=0.85)
        ax.bar(x+w/2,tv,w,label="TPE",color="#DD8452",alpha=0.85)
        ax.set_title(label,fontsize=11); ax.set_xticks(x)
        short=["ResNet","Eff-B3","ViT","Swin-T","Hybrid"]
        ax.set_xticklabels(short,rotation=25,fontsize=9)
        for j,(f,t) in enumerate(zip(fv,tv)):
            ax.text(j-w/2,f+0.002,f"{f:.3f}",ha="center",fontsize=7)
            ax.text(j+w/2,t+0.002,f"{t:.3f}",ha="center",fontsize=7)
        if i==0: ax.legend(fontsize=9)
    plt.suptitle("Fixed Hyperparameters vs TPE–Hyperband (70 Epochs)",fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"tpe_vs_fixed.png",dpi=300,bbox_inches="tight")
    plt.close(); print("✓ tpe_vs_fixed.png")

def plot_ablation_loss(bce_df, asl_df):
    """BCE vs ASL ablation."""
    metrics=[("f1_macro","F1 Macro"),("mcc_macro","MCC"),
             ("auprc_macro","AUPRC"),("accuracy","Accuracy"),
             ("hamming_loss","Hamming ↓"),("brier_macro","Brier ↓")]
    fig,axes=plt.subplots(2,3,figsize=(16,9)); axes=axes.flatten()
    # only architectures trained under BOTH losses (Hybrid-EffSwin was trained under BCE only)
    names={"resnet50":"ResNet-50","efficientnet_b3":"EfficientNet-B3","vit_b16":"ViT-B/16",
           "swin_t":"Swin-T","hybrid_eff_swin":"Hybrid-EffSwin"}
    models=[m for m in MODELS if m in asl_df["model"].values and m in bce_df["model"].values]
    x=np.arange(len(models)); w=0.35
    short=[names[m] for m in models]
    for i,(metric,label) in enumerate(metrics):
        ax=axes[i]
        bv=[float(bce_df[bce_df["model"]==m][metric].values[0]) for m in models]
        av=[float(asl_df[asl_df["model"]==m][metric].values[0]) for m in models]
        ax.bar(x-w/2,bv,w,label="BCE",color="#4C72B0",alpha=0.85)
        ax.bar(x+w/2,av,w,label="ASL",color="#DD8452",alpha=0.85)
        ax.set_title(label,fontsize=11); ax.set_xticks(x)
        ax.set_xticklabels(short,rotation=25,fontsize=9)
        for j,(b,a) in enumerate(zip(bv,av)):
            ax.text(j-w/2,b+0.002,f"{b:.3f}",ha="center",fontsize=7)
            ax.text(j+w/2,a+0.002,f"{a:.3f}",ha="center",fontsize=7)
        if i==0: ax.legend(fontsize=9)
    plt.suptitle("Loss Function Ablation: BCE vs ASL (Fixed Hyperparameters)",fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"ablation_loss.png",dpi=300,bbox_inches="tight")
    plt.close(); print("✓ ablation_loss.png")

def plot_ca_ablation(results):
    """Figure 8: fusion strategy ablation (average pooling vs residual projection vs concatenation).
    The 'hybrid_cross_attn' variant applies multi-head attention to single pooled vectors
    (sequence length 1); the attention weight is therefore identically 1 and the block reduces to a
    learned residual projection of the Transformer feature, hence the label 'Residual Proj.'."""
    variants=["hybrid_eff_swin","hybrid_cross_attn","hybrid_concat"]
    labels=["Avg Pool\n(Proposed)","Residual\nProj.","Concat"]
    colors=["#E1956B","#6A88BF","#6FB57F"]
    metrics=[("f1_macro","F1 Macro \u2191",True),("mcc_macro","MCC \u2191",True),
             ("accuracy","Accuracy \u2191",True),("hamming_loss","Hamming \u2193",False)]
    fig,axes=plt.subplots(1,4,figsize=(16,4.2))
    for ax,(metric,title,hib) in zip(axes,metrics):
        vals=[results.get(v,{}).get(metric,0) for v in variants]
        bars=ax.bar(labels,vals,color=colors,edgecolor="black",linewidth=0.6)
        best=max(vals) if hib else min(vals)
        for b_,v in zip(bars,vals):
            ax.text(b_.get_x()+b_.get_width()/2,v+max(vals)*0.015,f"{v:.3f}",ha="center",va="bottom",
                    fontsize=10,fontweight="bold" if v==best else "normal")
        ax.set_title(title,fontsize=12); ax.set_ylim(0,max(vals)*1.15)
        ax.tick_params(axis="x",labelsize=10); ax.tick_params(axis="y",labelsize=9)
        ax.grid(axis="y",alpha=0.3); ax.set_axisbelow(True)
    fig.suptitle("Fusion Strategy Ablation \u2014 Hybrid-EffSwin "
                 "(Avg Pool vs Residual Projection vs Concat)",fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.94])
    plt.savefig(FIGURES_DIR/"ablation_fusion.png",dpi=300,bbox_inches="tight")
    plt.close(); print("✓ ablation_fusion.png")

def plot_threshold(results):
    """Threshold optimization — F1 öncesi/sonrası."""
    models=list(results.keys())
    f1_base=[results[m]["f1_base"] for m in models]
    f1_opt =[results[m]["f1_opt"]  for m in models]
    x=np.arange(len(models)); w=0.35
    fig,ax=plt.subplots(figsize=(11,6))
    b1=ax.bar(x-w/2,f1_base,w,label="θ=0.5",color="#4C72B0",alpha=0.85)
    b2=ax.bar(x+w/2,f1_opt, w,label="θ_val (validation-selected)",color="#DD8452",alpha=0.85)
    for bar,v in zip(b1,f1_base):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f"{v:.3f}",ha="center",fontsize=9)
    for bar,v in zip(b2,f1_opt):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f"{v:.3f}",ha="center",fontsize=9)
    short=[m.replace("efficientnet_b3","Eff-B3").replace("hybrid_eff_swin","Hybrid")
            .replace("resnet50","ResNet-50").replace("vit_b16","ViT-B/16")
            .replace("swin_t","Swin-T") for m in models]
    ax.set_xticks(x); ax.set_xticklabels(short,fontsize=11)
    ax.set_ylabel("F1 Macro (21 evaluable classes)"); ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.3)
    ax.set_title("Per-Class Threshold Calibration (thresholds selected on validation)",fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"threshold_optimization.png",dpi=300)
    plt.close(); print("✓ threshold_optimization.png")

def plot_roc_curves(all_probs, labels_ref):
    fig,ax=plt.subplots(figsize=(8,7))
    for (name,label,color) in zip(MODELS,MODEL_LABELS,COLORS):
        if name not in all_probs: continue
        fpr,tpr,_=roc_curve(labels_ref.ravel(),all_probs[name].ravel())
        auc_val=roc_auc_score(labels_ref,all_probs[name],average="micro")
        ax.plot(fpr,tpr,label=f"{label} (AUC={auc_val:.3f})",color=color,linewidth=2)
    ax.plot([0,1],[0,1],"k--",alpha=0.4)
    ax.set_xlabel("FPR",fontsize=12); ax.set_ylabel("TPR",fontsize=12)
    ax.set_title("ROC Curves — Micro-Average",fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"roc_curves.png",dpi=300)
    plt.close(); print("✓ roc_curves.png")

def plot_pr_curves(all_probs, labels_ref):
    fig,ax=plt.subplots(figsize=(8,7))
    for (name,label,color) in zip(MODELS,MODEL_LABELS,COLORS):
        if name not in all_probs: continue
        prec,rec,_=precision_recall_curve(labels_ref.ravel(),all_probs[name].ravel())
        ap=average_precision_score(labels_ref.ravel(),all_probs[name].ravel())
        ax.plot(rec,prec,label=f"{label} (AP={ap:.3f})",color=color,linewidth=2)
    ax.set_xlabel("Recall",fontsize=12); ax.set_ylabel("Precision",fontsize=12)
    ax.set_title("Precision-Recall Curves — Micro-Average",fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"pr_curves.png",dpi=300)
    plt.close(); print("✓ pr_curves.png")

def plot_heatmaps(all_probs, all_preds, labels_ref, idx2label):
    nc=labels_ref.shape[1]
    for metric_name, fn in [
        ("f1", lambda l,p: f1_score(l,p,zero_division=0)),
        ("mcc",lambda l,p: matthews_corrcoef(l,p) if l.sum()>0 and p.sum()>0 else 0),
    ]:
        matrix=np.zeros((len(MODELS),nc))
        for i,name in enumerate(MODELS):
            if name not in all_preds: continue
            for c in range(nc):
                matrix[i,c]=fn(labels_ref[:,c],all_preds[name][:,c])
        fig,ax=plt.subplots(figsize=(max(14,nc*0.7),5))
        sns.heatmap(matrix,annot=True,fmt=".2f",xticklabels=idx2label,
                   yticklabels=MODEL_LABELS,cmap="RdYlGn",ax=ax,
                   vmin=0,vmax=1,linewidths=0.3,annot_kws={"size":7})
        ax.set_title(f"Per-Class {metric_name.upper()} — Model Comparison",fontsize=13)
        plt.xticks(rotation=45,ha="right",fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR/f"per_class_{metric_name}_heatmap.png",dpi=300)
        plt.close(); print(f"✓ per_class_{metric_name}_heatmap.png")

def plot_training_curves(experiment):
    ckpt_dir=CKPT_FIXED if experiment=="fixed" else CKPT_TPE if experiment=="tpe" else CKPT_ASL
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for i,(name,label,color) in enumerate(zip(MODELS,MODEL_LABELS,COLORS)):
        log_path=RESULTS_DIR/f"{name}_{experiment}_log.json"
        if not log_path.exists() and experiment=="fixed" and name=="hybrid_eff_swin":
            log_path=RESULTS_DIR/"hybrid_eff_swin_ca_log.json"   # fixed Hybrid = avg-pool fusion run
        if not log_path.exists(): continue
        with open(log_path) as f: log=json.load(f)
        epochs=[e["epoch"] for e in log]
        tl=[e["train_loss"] for e in log]; vf=[e["val_f1"] for e in log]
        axes[0].plot(epochs,tl,label=label,color=color,linewidth=1.8)
        axes[1].plot(epochs,vf,label=label,color=color,linewidth=1.8,
                     marker="o",markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss")
    axes[0].set_title(f"Training Loss — {experiment.upper()}")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val F1 (macro)")
    axes[1].set_title(f"Validation F1 — {experiment.upper()}")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/f"training_curves_{experiment}.png",dpi=300)
    plt.close(); print(f"✓ training_curves_{experiment}.png")

def plot_confusion_best(all_preds, labels_ref, idx2label, best_name):
    preds=all_preds[best_name]; nc=labels_ref.shape[1]
    top6=np.argsort(labels_ref.sum(axis=0))[::-1][:6]
    fig,axes=plt.subplots(2,3,figsize=(14,9)); axes=axes.flatten()
    for i,c in enumerate(top6):
        cm=confusion_matrix(labels_ref[:,c],preds[:,c])
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=axes[i],
                   xticklabels=["Pred 0","Pred 1"],yticklabels=["True 0","True 1"])
        axes[i].set_title(f"{idx2label[c]} (n={int(labels_ref[:,c].sum())})",fontsize=10)
    plt.suptitle(f"Confusion Matrices — {best_name}",fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/f"confusion_matrix_{best_name}.png",dpi=300)
    plt.close(); print(f"✓ confusion_matrix_{best_name}.png")

def plot_class_distribution(label2idx):
    from collections import Counter
    train_cnt=Counter(); test_cnt=Counter()
    valid=set(label2idx.keys())
    for r in load_json(TRAIN_JSON): train_cnt.update(parse_labels(r.get("label",""),valid))
    for r in load_json(TEST_JSON):  test_cnt.update(parse_labels(r.get("label",""),valid))
    labels_sorted=sorted(valid,key=lambda l:-train_cnt.get(l,0))
    tv=[train_cnt.get(l,0) for l in labels_sorted]
    te=[test_cnt.get(l,0)  for l in labels_sorted]
    fig,ax=plt.subplots(figsize=(12,7))
    x=np.arange(len(labels_sorted)); w=0.4
    ax.barh(x+0.2,tv,w,label="Training pool (train + val)",color="#4C72B0",alpha=0.85)
    ax.barh(x-0.2,te,w,label="Test", color="#DD8452",alpha=0.85)
    ax.set_yticks(x); ax.set_yticklabels(labels_sorted,fontsize=9)
    ax.set_xlabel("Number of images")
    ax.set_title("BERD Class Distribution — Train vs Test",fontsize=13)
    ax.legend(); ax.grid(axis="x",alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"class_distribution.png",dpi=300)
    plt.close(); print("✓ class_distribution.png")


# ╔══════════════════════════════════════════════════════════════╗
# ║  ANA ÇALIŞTIRMA                                               ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__=="__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if device.type=="cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    label2idx,idx2label=load_label_encoder()
    nc=len(label2idx)
    print(f"Sınıf sayısı: {nc}")
    train_loader,val_loader,test_loader=build_loaders(label2idx)

    # ── DENEY 1: TPE parametreleriyle eğitim (artık hibrit DAHİL) ──
    if RUN_TPE_TRAIN:
        print(f"\n{'='*55}\nDENEY 1: TPE PARAMETRELERİYLE EĞİTİM\n{'='*55}")
        for name in MODELS:
            params=load_tpe_params(name)
            mw=params.get("max_weight",FIXED_MAX_WEIGHT)
            crit=build_bce(label2idx,device,mw)
            train_model(name,nc,label2idx,train_loader,val_loader,test_loader,
                       device,crit,CKPT_TPE,"tpe",params)

    # ── DENEY 2: Sabit parametrelerle eğitim (hibrit CA'da eğitiliyor) ──
    if RUN_FIXED_TRAIN:
        print(f"\n{'='*55}\nDENEY 2: SABİT PARAMETRELERİYLE EĞİTİM\n{'='*55}")
        bce_crit=build_bce(label2idx,device)
        for name in MODELS:
            if name == "hybrid_eff_swin":
                print(f"  [FIXED] {name}: CA ablation (DENEY 4) içinde aynı "
                      f"parametrelerle eğitiliyor, burada tekrar edilmiyor "
                      f"(gereksiz GPU zamanı israfını önlemek için).")
                continue
            dr=FIXED_DROPOUT_HYB if "hybrid" in name else FIXED_DROPOUT_CNN
            params={"lr":FIXED_LR,"weight_decay":FIXED_WD,
                    "dropout":dr,"max_weight":FIXED_MAX_WEIGHT}
            train_model(name,nc,label2idx,train_loader,val_loader,test_loader,
                       device,bce_crit,CKPT_FIXED,"fixed",params)

    # ── DENEY 3: ASL ablation (sabit params) ──────────────────────
    if RUN_ASL:
        print(f"\n{'='*55}\nDENEY 3: ASL ABLATION\n{'='*55}")
        asl_crit=build_asl()
        for name in MODELS:
            # hybrid_eff_swin için ASL eğitimi yok — atla
            if name == "hybrid_eff_swin":
                print(f"  [ASL] {name}: ASL eğitimi uygulanmadı, atlanıyor.")
                continue
            dr=FIXED_DROPOUT_HYB if "hybrid" in name else FIXED_DROPOUT_CNN
            params={"lr":FIXED_LR,"weight_decay":FIXED_WD,
                    "dropout":dr,"max_weight":FIXED_MAX_WEIGHT}
            train_model(name,nc,label2idx,train_loader,val_loader,test_loader,
                       device,asl_crit,CKPT_ASL,"asl",params)

    # ── DENEY 4: Cross-attention/fusion ablation (ÜÇ varyant) ──────
    if RUN_CA_ABLATION:
        print(f"\n{'='*55}\nDENEY 4: FUSION STRATEGY ABLATION (Avg Pool / Residual Projection / Concat)\n{'='*55}")
        bce_crit=build_bce(label2idx,device)
        for name in CA_VARIANTS:
            params={"lr":FIXED_LR,"weight_decay":FIXED_WD,
                    "dropout":FIXED_DROPOUT_HYB,"max_weight":FIXED_MAX_WEIGHT}
            train_model(name,nc,label2idx,train_loader,val_loader,test_loader,
                       device,bce_crit,CKPT_CA,"ca",params)

    # ── EXPERIMENT 5: additional baselines (secondary analysis, fixed hyperparameters) ──
    if RUN_EXTRA_BASELINES:
        print(f"\n{'='*55}\nEXPERIMENT 5: ADDITIONAL BASELINES (ConvNeXt-T, DINOv2, BiomedCLIP)\n{'='*55}")
        CKPT_EXTRA.mkdir(exist_ok=True)
        bce_crit=build_bce(label2idx,device)
        for name in EXTRA_MODELS:
            params={"lr":FIXED_LR,"weight_decay":FIXED_WD,
                    "dropout":FIXED_DROPOUT_CNN,"max_weight":FIXED_MAX_WEIGHT}
            try:
                train_model(name,nc,label2idx,train_loader,val_loader,test_loader,
                            device,bce_crit,CKPT_EXTRA,"fixed",params)
            except Exception as e:      # e.g. missing open_clip or no internet for torch.hub
                print(f"  [EXTRA] {name} skipped: {type(e).__name__}: {e}")

    # ── EXPERIMENT 6: seed replicates of the fixed configuration ──
    if RUN_SEED_REPLICATES:
        print(f"\n{'='*55}\nEXPERIMENT 6: SEED REPLICATES {SEED_RUNS}\n{'='*55}")
        CKPT_SEEDS.mkdir(exist_ok=True)
        bce_crit=build_bce(label2idx,device)
        for sd in SEED_RUNS:
            for name in SEED_MODELS:
                dr=FIXED_DROPOUT_HYB if "hybrid" in name else FIXED_DROPOUT_CNN
                params={"lr":FIXED_LR,"weight_decay":FIXED_WD,
                        "dropout":dr,"max_weight":FIXED_MAX_WEIGHT}
                train_model(name,nc,label2idx,train_loader,val_loader,test_loader,
                            device,bce_crit,CKPT_SEEDS,f"seed{sd}",params,train_seed=sd)

    print("\n✓ Training finished. Checkpoints saved in checkpoints_*/.")
    print("  Next step: python 3_berd_posthoc_eval.py  (all reported metrics, statistics, figures)")
