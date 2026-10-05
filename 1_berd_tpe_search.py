"""
1_berd_tpe_search.py — Hyperparameter search (TPE sampler + Hyperband pruning, Optuna)

For each architecture, runs N_TRIALS Optuna trials (TPE sampler, HyperbandPruner) of up to
SEARCH_EPOCHS epochs. The objective is macro F1 on the patient-level VALIDATION subset of
the official BERD training set; the test set is never used for optimization, pruning, or
early stopping. The first trial is seeded with the fixed configuration used in the paper.

Also performs the one-time preprocessing used by all later steps:
  processed/label_encoder.pkl   (23 classes; "diverticulum" excluded, n = 2)
  processed/class_weights.pkl   (computed on the training subset only)

Outputs:  results/<model>_tpe_results.json   (best parameters and all trials)
Run:      python 1_berd_tpe_search.py
Next:     python 2_berd_main.py
"""

import json, pickle, time, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR      = Path(__file__).resolve().parent
DATA_ROOT     = BASE_DIR / "dataset"
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR   = BASE_DIR / "results"

TRAIN_JSON    = DATA_ROOT / "annotations" / "dataset_train.json"
TEST_JSON     = DATA_ROOT / "annotations" / "dataset_test.json"   # sadece EDA'da okunur, bkz. yukarıdaki not (4)
IMAGE_DIR     = DATA_ROOT / "images"
LABEL_ENCODER = PROCESSED_DIR / "label_encoder.pkl"
CLASS_WEIGHTS = PROCESSED_DIR / "class_weights.pkl"
EDA_DONE_FLAG = PROCESSED_DIR / ".eda_done"

KNOWN_LABELS = [
    "blood","clot","congested","edematous","external pressure",
    "fistula","granulation","infiltration changes","mass","narrow",
    "necrotic","neoplasm","new organism","nodules","normal",
    "pigmentation","postoperative change","rough","sputum",
    "surgical stump","tube","ulcer","widened",
]

BATCH_SIZE   = 32
# ── PERFORMANS DÜZELTMESİ ─────────────────────────────────────────
# Eski NUM_WORKERS=0, tüm augmentasyonların (RandomRotation,
# ColorJitter, RandomAutocontrast, GaussianBlur vb.) CPU'da TEK
# işlemde, GPU'yu bekleterek seri yapılmasına yol açıyordu. RTX 4080
# gibi bir GPU için ResNet-50/EffNet-B3 ileri+geri geçişi milisaniyeler
# sürer; trial başına ~20+ dakika sürmesi neredeyse kesin veri
# yükleme darboğazından kaynaklanıyor. Kendi makinenizde
#   import os; print(os.cpu_count())
# ile çekirdek sayınıza bakıp NUM_WORKERS'ı buna göre ayarlayın
# (genelde çekirdek_sayısı-2 civarı iyi bir başlangıçtır; 4-8 arası
# çoğu masaüstü için makuldür). 0 asla bırakmayın.
NUM_WORKERS  = 14  # 20 çekirdekli makineniz için: birkaç çekirdek OS/ana sürece kalsın diye tamamı değil
PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR    = 2 if NUM_WORKERS > 0 else None
SEED         = 42
VAL_FRACTION = 0.20   # berd_main.py ile AYNI oran — tutarlılık için

# ── TPE / BOHB ayarları ──────────────────────────────────────────
# Eski bütçe (25 trial × max 50 epoch × 5 model) pratikte 1-2 gün
# sürüyordu. Aşağıdaki değerler bütçeyi küçültüyor; BOHB/Hyperband
# zaten kötü giden trial'ları erken buduyor, bu yüzden daha küçük
# bir bütçeyle de makul kalitede sonuç alınır. İsterseniz N_TRIALS'ı
# tekrar artırabilirsiniz — önemli olan SEARCH_EPOCHS'un final
# eğitimdeki (FINAL_EPOCHS=70, EARLY_STOP=15) davranışla orantılı
# kalması, birebir aynı olması gerekmiyor.
N_TRIALS         = 15    # model başına trial sayısı (25 → 15)
SEARCH_EPOCHS    = 30    # HyperbandPruner max_resource ile aynı (50 → 30)
HYPERBAND_MIN    = 5     # minimum epoch/trial. NOT: ReduceLROnPlateau patience=5
                          # olduğu için bu 5'in altına düşürülmemeli — aksi halde
                          # scheduler LR'ı hiç düşüremeden trial budanabilir.
HYPERBAND_MAX    = 30    # maksimum epoch/trial (50 → 30)
HYPERBAND_ETA    = 3     # reduction factor (her turda 1/3 elenir)
EARLY_STOP_SEARCH_DEFAULT = 8   # objective() içindeki patience (15 → 8)
FORCE_SEARCH     = False # True → cache olsa bile yeniden ara

# ── DEVAM EDİLEBİLİRLİK (resume) ──────────────────────────────────
# Eski kod optuna.create_study(...) çağrısında storage vermiyordu,
# yani study TAMAMEN BELLEKTE tutuluyordu — süreç kesilirse (Ctrl+C,
# uyku modu, çökme) o modeldeki TÜM trial ilerlemesi kaybolur ve
# baştan başlanır. Artık her model için bir SQLite dosyasına
# yazılıyor; süreci kesip tekrar çalıştırdığınızda kaldığı trial'dan
# devam eder.
USE_PERSISTENT_STUDY = True

MODELS = ["resnet50","efficientnet_b3","vit_b16","swin_t","hybrid_eff_swin"]

# ── Mimari-özgü arama uzayları ───────────────────────────────────
# CNN: daha yüksek lr tolere eder
# Transformer: daha düşük lr, daha yüksek wd gerektirir
# Hybrid: iki backbone, orta aralık
SEARCH_SPACES = {
    "resnet50": {
        "lr":         (5e-5, 1e-3),
        "wd":         (1e-5, 1e-3),
        "dropout":    (0.2, 0.6),
        "max_weight": (30.0, 60.0),
    },
    "efficientnet_b3": {
        "lr":         (1e-4, 2e-3),
        "wd":         (1e-5, 1e-3),
        "dropout":    (0.2, 0.6),
        "max_weight": (30.0, 60.0),
    },
    "vit_b16": {
        "lr":         (1e-6, 2e-4),
        # wd alt sınırı literatür pratiğiyle uyumlu olsun diye yükseltildi
        # (ViT/AdamW için genelde 1e-2–1e-1 aralığı verimli; 1e-5 gibi çok
        # küçük değerler TPE bütçesini gereksiz yere israf ediyordu).
        "wd":         (1e-3, 5e-2),
        "dropout":    (0.1, 0.5),
        "max_weight": (20.0, 60.0),
    },
    "swin_t": {
        "lr":         (5e-5, 3e-4),
        "wd":         (1e-3, 5e-2),
        "dropout":    (0.1, 0.5),
        "max_weight": (15.0, 50.0),
    },
    "hybrid_eff_swin": {
        "lr":         (5e-5, 5e-4),
        "wd":         (1e-5, 1e-2),
        "dropout":    (0.2, 0.4),
        "max_weight": (40.0, 60.0),
    },
}

for d in [PROCESSED_DIR, RESULTS_DIR]:
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


# ╔══════════════════════════════════════════════════════════════╗
# ║  EDA (cache'li)                                               ║
# ╚══════════════════════════════════════════════════════════════╝
#
# NOT: Aşağıda TEST_JSON iki amaçla okunuyor — (a) global etiket
# kümesini keşfetmek, (b) hasta bazlı örtüşme olmadığını doğrulamak.
# Bu adım hiçbir modelin ağırlığını, hiperparametresini veya erken
# durdurma kararını ETKİLEMEZ; sadece split'in sızıntısız olduğunu
# teyit eden bir sağlık kontrolüdür. Eğitim/arama döngüsünün TAMAMI
# yalnızca TRAIN_JSON'dan türetilen train/val split'i kullanır.

def run_eda(force=False):
    if EDA_DONE_FLAG.exists() and not force:
        print("[EDA] Cache bulundu — atlanıyor.")
        return

    print("="*55)
    print("[EDA] Başlıyor...")
    print("="*55)

    train_records = load_json(TRAIN_JSON)
    test_records  = load_json(TEST_JSON)   # sadece keşif + sızıntı kontrolü için
    print(f"[EDA] Train: {len(train_records)}, Test: {len(test_records)}")

    all_labels = sorted({l for r in train_records+test_records
                         for l in parse_labels(r.get("label",""))})
    print(f"[EDA] Keşfedilen {len(all_labels)} etiket")

    label2idx = {l:i for i,l in enumerate(all_labels)}
    with open(LABEL_ENCODER,"wb") as f:
        pickle.dump({"label2idx":label2idx,"idx2label":all_labels}, f)

    # ── SIZINTI DÜZELTMESİ ────────────────────────────────────────
    # ESKİ KOD: CLASS_WEIGHTS burada, split'ten ÖNCE, TÜM train
    # havuzu (gelecekteki %20'lik val hastaları DAHİL) üzerinden
    # hesaplanıyordu. Bu ağırlıklar doğrudan BCEWithLogitsLoss'un
    # pos_weight'ine gidip eğitimde kullanıldığından, val kümesinin
    # etiket istatistikleri dolaylı olarak eğitime sızmış oluyordu.
    # YENİ KOD: class_weights.pkl artık burada YAZILMIYOR. Gerçek
    # hesaplama split_patients_train_val() içinde, split YAPILDIKTAN
    # SONRA, SADECE %80'lik train_records üzerinden yapılıyor (aşağı
    # bakın). Bu fonksiyon (run_eda) artık sadece etiket keşfi ve
    # train/test hasta örtüşme kontrolü yapıyor.

    train_pids = {r.get("patient_id") for r in train_records if r.get("patient_id")}
    test_pids  = {r.get("patient_id") for r in test_records  if r.get("patient_id")}
    leak = train_pids & test_pids
    if leak: print(f"[EDA] ⚠️ DATA LEAK: {len(leak)} hasta train/test arasında ortak!")
    else:    print(f"[EDA] ✓ Patient-level (train/test) leak yok.")

    EDA_DONE_FLAG.touch()
    print("[EDA] ✓ Tamamlandı.\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  DATASET                                                      ║
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
        return img, vec, {}

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
        imgs,labels,_=zip(*batch); imgs,labels=torch.stack(imgs),torch.stack(labels)
        if np.random.random()<0.3:
            lam=float(np.random.beta(0.4,0.4)); idx=torch.randperm(imgs.size(0))
            imgs=lam*imgs+(1-lam)*imgs[idx]; labels=lam*labels+(1-lam)*labels[idx]
        return imgs,labels,[]


def split_patients_train_val(label2idx):
    """
    Hasta bazlı, DETERMİNİSTİK train/val ayrımı — SADECE TRAIN_JSON
    üzerinden. TEST_JSON bu fonksiyona hiç girmez.

    ⚠️ berd_main.py'deki split fonksiyonuyla BİREBİR AYNI olmalı
    (aynı SEED, aynı oran, aynı sıralama mantığı) ki TPE aramasında
    kullanılan validation seti ile final eğitimdeki validation seti
    örtüşsün. berd_main.py'de de patient_ids listesini `sorted()`
    ile deterministik hale getirdiğinizden emin olun.
    """
    all_records = load_json(TRAIN_JSON)
    valid_set   = set(label2idx.keys())

    valid_records = [
        r for r in all_records
        if resolve_path(r.get("image_path",""), Path(IMAGE_DIR)).exists()
        and parse_labels(r.get("label",""), valid_set)
    ]

    # ── DETERMİNİZM DÜZELTMESİ ──────────────────────────────────
    # Eski kod: list({...})  → set() iterasyon sırası process başına
    # rastgele (hash randomization), bu yüzden shuffle'a giren liste
    # her çalıştırmada farklı sırada olabiliyordu → seed'li shuffle
    # bile farklı sonuç veriyordu.
    # Yeni kod: önce sorted() ile sabit bir sıra garanti edilir,
    # SONRA seed'li rng ile shuffle edilir → her çalıştırmada
    # birebir aynı train/val ayrımı.
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

    # Sağlık kontrolü: train/val hasta kümeleri arasında örtüşme yok
    overlap = train_patients & val_patients
    assert not overlap, f"[HATA] train/val arasında {len(overlap)} ortak hasta var!"

    # ── SIZINTI DÜZELTMESİ: CLASS_WEIGHTS burada, SPLIT SONRASI,
    # SADECE train_records üzerinden hesaplanıyor. Val kümesi
    # istatistikleri artık loss ağırlığına hiç karışmıyor.
    # (Cache'lenmiyor — her build_loaders() çağrısında güncel split'e
    # göre yeniden hesaplanır, ekstra maliyeti ihmal edilebilir.)
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
    SADECE train_loader ve val_loader döndürür. TEST_JSON bu
    fonksiyonda YOK — hiperparametre aramasının hiçbir noktasında
    test setine erişim mümkün değil.
    """
    train_records, val_records = split_patients_train_val(label2idx)

    tr = BERDDataset(TRAIN_JSON, IMAGE_DIR, label2idx, get_tf("train"), "train")
    tr.data = train_records

    va = BERDDataset(TRAIN_JSON, IMAGE_DIR, label2idx, get_tf("test"), "val")
    va.data = val_records

    print(f"[Loaders] Train: {len(tr)} | Val: {len(va)}")

    cnt=Counter()
    for r in tr.data: cnt.update(parse_labels(r.get("label",""),tr.valid))
    tot=sum(cnt.values()); lw={l:tot/max(c,1) for l,c in cnt.items()}
    sw=[max((lw.get(l,1) for l in parse_labels(r.get("label",""),tr.valid)),default=1)
        for r in tr.data]
    sampler=WeightedRandomSampler(torch.DoubleTensor(sw),len(sw),replacement=True)

    dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True)
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        dl_kwargs["prefetch_factor"]    = PREFETCH_FACTOR

    trl=DataLoader(tr,BATCH_SIZE,sampler=sampler,
                   collate_fn=MixUp(),drop_last=True,**dl_kwargs)
    val_loader=DataLoader(va,BATCH_SIZE,shuffle=False,**dl_kwargs)
    return trl, val_loader


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

class EffB3(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        f=b.classifier[1].in_features; b.classifier=nn.Identity(); self.b=b; self.h=_head(f,nc,dr)
    def forward(self,x): return self.h(self.b(x))

class ViTB16(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        f=b.heads.head.in_features; b.heads.head=nn.Identity(); self.b=b
        self.h=nn.Sequential(nn.LayerNorm(f),nn.Dropout(dr),nn.Linear(f,512),
                              nn.GELU(),nn.Dropout(dr/2),nn.Linear(512,nc))
    def forward(self,x): return self.h(self.b(x))

class SwinT(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        f=b.head.in_features; b.head=nn.Identity(); self.b=b
        self.h=nn.Sequential(nn.LayerNorm(f),nn.Dropout(dr),nn.Linear(f,512),
                              nn.GELU(),nn.Dropout(dr/2),nn.Linear(512,nc))
    def forward(self,x): return self.h(self.b(x))

class HybridEffSwin(nn.Module):
    """
    ⚠️ MİMARİ TUTARLILIĞI: berd_main.py'de "hybrid_eff_swin" adı
    AVERAGE POOLING füzyonlu modele karşılık geliyor (asıl önerilen
    model); cross-attention'lı versiyon orada "hybrid_cross_attn"
    olarak ayrı bir sınıf. Bu dosyanın ESKİ sürümünde ise
    "hybrid_eff_swin" YANLIŞLIKLA cross-attention kullanıyordu — yani
    TPE araması, main.py'de hiç var olmayan bir mimari için
    hiperparametre buluyordu (main.py zaten "TPE parametresi yok,
    atlanıyor" diyerek bu sonucu kullanmıyordu, o yüzden yanlış
    sonuç raporlanmadı ama GPU zamanı boşa harcanıyordu).

    Bu sürümde mimari, main.py'deki gerçek "hybrid_eff_swin"
    (average pooling) ile BİREBİR aynı yapıldı — böylece TPE'nin
    bulacağı hiperparametreler gerçekten kullanılacak modele ait
    olur ve main.py'de artık bu model için de TPE sonucu
    uygulanabilir (main.py'deki "atlanıyor" satırını kaldırmanız
    gerekir — ayrı bir adımda ele alınabilir).
    """
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

def build_model(name, nc, dropout):
    return {"resnet50":        lambda: ResNet50(nc,dropout),
            "efficientnet_b3": lambda: EffB3(nc,dropout),
            "vit_b16":         lambda: ViTB16(nc,dropout),
            "swin_t":          lambda: SwinT(nc,dropout),
            "hybrid_eff_swin": lambda: HybridEffSwin(nc,dropout)}[name]()

def build_loss(label2idx, device, max_weight):
    with open(CLASS_WEIGHTS,"rb") as f: wd=pickle.load(f)
    w=torch.tensor([min(wd.get(l,1.0),max_weight) for l in label2idx],
                   dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=w)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TPE ARAMA (BOHB - HyperbandPruner)                          ║
# ╚══════════════════════════════════════════════════════════════╝

def make_objective(name, label2idx, nc, train_loader, val_loader, device, space):
    """
    ⚠️ SIZINTI DÜZELTMESİ: parametre adı `test_loader` → `val_loader`
    olarak değiştirildi ve fonksiyon artık SADECE val_loader (train
    setinden ayrılmış, hasta bazlı validation) üzerinde skor
    hesaplıyor. TEST_JSON bu fonksiyona hiçbir şekilde girmiyor.
    """
    def objective(trial):
        lr         = trial.suggest_float("lr",         *space["lr"],         log=True)
        wd         = trial.suggest_float("wd",         *space["wd"],         log=True)
        dropout    = trial.suggest_float("dropout",    *space["dropout"])
        max_weight = trial.suggest_float("max_weight", *space["max_weight"])

        torch.manual_seed(SEED)
        model     = build_model(name, nc, dropout).to(device)
        criterion = build_loss(label2idx, device, max_weight)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
        scaler    = GradScaler()

        best_f1 = 0.0
        f1_history = []
        patience_counter = 0
        EARLY_STOP_SEARCH = EARLY_STOP_SEARCH_DEFAULT  # arama için kısaltılmış patience

        # ── GPU BELLEK TEMİZLİĞİ (try/finally) ────────────────────
        # NOT: raise optuna.TrialPruned() aşağıda her epoch'un train+val
        # DataLoader döngüleri TAMAMEN bittikten SONRA çağrılıyor —
        # yani hiçbir DataLoader iterator'ı yarım kesilmiyor, bu yüzden
        # "worker sızıntısı" bu mekanizmadan kaynaklanmıyor. Asıl risk:
        # 15 trial × 5 model boyunca her trial'da yeni model/optimizer/
        # scheduler GPU'da oluşturuluyor; bunlar açıkça silinip CUDA
        # cache boşaltılmazsa bellek fragmantasyonu birikip özellikle
        # büyük modellerde (ViT-B/16 86M, Hybrid 39M) birkaç trial
        # sonra OOM'a yol açabilir. Bu yüzden try/finally ile her
        # trial sonunda (başarılı, pruned ya da hatalı fark etmeksizin)
        # temizlik garanti ediliyor.
        try:
            for epoch in range(SEARCH_EPOCHS):
                # Eğitim
                model.train()
                for imgs, labels, _ in train_loader:
                    imgs,labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(): loss = criterion(model(imgs), labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()

                # Değerlendirme — SADECE validation (test DEĞİL)
                model.eval()
                probs, lbls = [], []
                with torch.no_grad():
                    for imgs, labels, _ in val_loader:
                        with autocast():
                            p = torch.sigmoid(model(imgs.to(device))).cpu().float().numpy()
                        probs.append(p); lbls.append(labels.numpy())
                probs = np.vstack(probs); lbls = np.vstack(lbls)
                val_f1 = f1_score(lbls, (probs>=0.5).astype(float),
                                 average="macro", zero_division=0)

                # Scheduler — final eğitimle aynı davranış
                scheduler.step(val_f1)
                f1_history.append(val_f1)

                # Early stopping takibi — final eğitimle aynı
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1

                # HyperbandPruner'a raporla — val_f1 (test DEĞİL)
                trial.report(val_f1, epoch)
                if trial.should_prune() or patience_counter >= EARLY_STOP_SEARCH:
                    raise optuna.TrialPruned()

            # Son 5 epoch ortalaması — sadece stabil ve yakınsayan parametreler
            # yüksek skor alır, peak noise etkisi önlenir
            return float(np.mean(f1_history[-5:])) if len(f1_history)>=5 else best_f1
        finally:
            del model, optimizer, scheduler, scaler
            torch.cuda.empty_cache()

    return objective


def search_model(name, label2idx, nc, train_loader, val_loader, device):
    out_path = RESULTS_DIR / f"{name}_tpe_results.json"

    # Cache kontrolü
    if out_path.exists() and not FORCE_SEARCH:
        with open(out_path) as f: cached = json.load(f)
        print(f"\n[TPE] {name}: cache bulundu → F1(val)={cached['best_f1']:.4f}")
        print(f"  {cached['best_params']}")
        return cached["best_params"], cached["best_f1"]

    space = SEARCH_SPACES[name]
    print(f"\n{'='*55}")
    print(f"[TPE/BOHB] {name} | {N_TRIALS} trial | max {SEARCH_EPOCHS} epoch")
    print(f"  (skor: TRAIN-içi validation split — TEST setine erişim yok)")
    print(f"  lr: {space['lr']}  max_weight: {space['max_weight']}")
    print(f"{'='*55}")

    sampler = TPESampler(seed=SEED, n_startup_trials=5)
    pruner  = HyperbandPruner(
        min_resource=HYPERBAND_MIN,
        max_resource=HYPERBAND_MAX,
        reduction_factor=HYPERBAND_ETA,
    )

    # ── Devam edilebilirlik: SQLite'a yazan study ────────────────
    # Süreç kesilirse (Ctrl+C, uyku modu, çökme), bir sonraki
    # çalıştırmada `load_if_exists=True` sayesinde kaldığı trial'dan
    # devam eder; tamamlanmış trial'lar tekrar çalıştırılmaz.
    study_kwargs = dict(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{name}_bohb",
    )
    if USE_PERSISTENT_STUDY:
        db_path = RESULTS_DIR / f"{name}_optuna.db"
        study_kwargs["storage"] = f"sqlite:///{db_path}"
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)

    n_done = len([t for t in study.trials
                  if t.state in (optuna.trial.TrialState.COMPLETE,
                                 optuna.trial.TrialState.PRUNED)])
    n_remaining = max(0, N_TRIALS - n_done)
    if n_done > 0:
        print(f"  [Resume] {n_done} trial zaten tamamlanmış (DB'den), "
              f"{n_remaining} trial kaldı.")

    # Sabit parametreleri başlangıç noktası olarak ver (enqueue_trial)
    # TPE buradan başlar, daha iyisini arar. Sadece study boşsa ekle
    # (aksi halde her resume'da tekrar tekrar enqueue edilir).
    if n_done == 0:
        dropout_default = 0.3 if "hybrid" in name else 0.4
        study.enqueue_trial({
            "lr":         1e-4,
            "wd":         1e-4,
            "dropout":    dropout_default,
            "max_weight": 50.0,
        })

    obj = make_objective(name, label2idx, nc, train_loader, val_loader, device, space)
    t0 = time.time()
    if n_remaining > 0:
        study.optimize(obj, n_trials=n_remaining, show_progress_bar=True)
    else:
        print("  [Resume] Bu model için hedeflenen trial sayısına zaten ulaşılmış.")
    elapsed = time.time()-t0

    pruned = sum(1 for t in study.trials if t.state==optuna.trial.TrialState.PRUNED)
    print(f"\n[TPE/BOHB] {name} TAMAMLANDI ({elapsed/60:.1f} dk)")
    print(f"  En iyi F1 (val) : {study.best_value:.4f}")
    print(f"  Parametreler    : {study.best_params}")
    print(f"  Pruned          : {pruned}/{len(study.trials)}")

    out = {
        "model":         name,
        "best_f1":       study.best_value,       # ← bu artık VALIDATION F1, test F1 değil
        "best_f1_metric":"validation_macro_f1 (patient-level split, test set NEVER used)",
        "best_params":   study.best_params,
        "n_trials":      len(study.trials),
        "pruned":        pruned,
        "elapsed_min":   round(elapsed/60,1),
        "method":        "TPE+HyperbandPruner (BOHB)",
        "hyperband":     {"min":HYPERBAND_MIN,"max":HYPERBAND_MAX,"eta":HYPERBAND_ETA},
        "all_trials":    [{"number":t.number,"value":t.value,
                           "params":t.params,"state":str(t.state)}
                          for t in study.trials],
    }
    with open(out_path,"w") as f: json.dump(out,f,indent=2)
    print(f"  Kaydedildi  : {out_path}")
    return study.best_params, study.best_value


# ╔══════════════════════════════════════════════════════════════╗
# ║  ANA ÇALIŞTIRMA                                               ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if device.type=="cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1) EDA (test setine sadece keşif/leak-kontrolü için bakar, eğitime dokunmaz)
    run_eda(force=False)

    # 2) DataLoaders — SADECE train_loader ve val_loader; test_loader YOK
    label2idx, idx2label = load_label_encoder()
    nc = len(label2idx)
    print(f"Sınıf sayısı: {nc}")
    train_loader, val_loader = build_loaders(label2idx)

    # 3) TPE arama — tamamen train-içi validation üzerinde
    all_results = {}
    for name in MODELS:
        params, f1 = search_model(name, label2idx, nc,
                                   train_loader, val_loader, device)
        all_results[name] = {"best_params": params, "best_f1_val": f1}

    # Özet
    print(f"\n{'='*55}")
    print("TPE/BOHB ARAMA ÖZETI (validation F1 — test seti kullanılmadı)")
    print(f"{'='*55}")
    print(f"\n{'Model':<20} {'F1 (val)':>12} {'lr':>12} {'max_w':>8}")
    print("-"*56)
    for name, res in all_results.items():
        p = res["best_params"]
        print(f"{name:<20} {res['best_f1_val']:>12.4f} "
              f"{p.get('lr',0):>12.2e} {p.get('max_weight',0):>8.1f}")

    with open(RESULTS_DIR/"all_tpe_summary.json","w") as f:
        json.dump(all_results,f,indent=2)
    print(f"\n✓ Özet kaydedildi → {RESULTS_DIR/'all_tpe_summary.json'}")
    print("\n⚠️  ÖNEMLİ: Bu dosyanın ürettiği *_tpe_results.json içindeki")
    print("   'best_f1' değeri VALIDATION F1'dir, TEST F1 DEĞİLDİR.")
    print("   Makalede raporlanacak nihai TEST metrikleri sadece")
    print("   berd_main.py'nin test seti üzerindeki değerlendirmesinden")
    print("   alınmalıdır — bu dosyanın çıktısı asla doğrudan makaleye")
    print("   'test sonucu' olarak yazılmamalıdır.")
    print("\nSonraki adım: berd_main.py")