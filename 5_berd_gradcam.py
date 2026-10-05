"""
5_berd_gradcam.py — Grad-CAM visualizations (Figure 10)

Loads the fixed-hyperparameter checkpoints produced by 2_berd_main.py and generates
Grad-CAM heatmaps for test images.

Target layers
  ResNet-50        : layer4 (last residual stage)
  EfficientNet-B3  : features[-1] (last convolutional block)
  Swin-T           : features[-1] (last stage, 7x7x768 token grid reshaped to a spatial map)
  Hybrid-EffSwin   : EfficientNet-B3 branch, features[-1]
  ViT-B/16 is not included: its [CLS]-token classifier does not natively expose a
  spatial feature map.

Example selection is not curated: for each target class, the first N_EXAMPLES_PER_CLASS
test images (in dataset order) containing that label are used.
Grad-CAM maps are qualitative only; no ground-truth localization is available in BERD.

Run:  python 5_berd_gradcam.py
"""

import json, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms
from torchvision import models as M
from torchvision.models import (ResNet50_Weights, EfficientNet_B3_Weights,
                                 Swin_T_Weights)

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG — berd_main.py ile AYNI yollar                        ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR      = Path(__file__).resolve().parent
DATA_ROOT     = BASE_DIR / "dataset"
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR   = BASE_DIR / "results"
FIGURES_DIR   = BASE_DIR / "figures"

CKPT_CA    = BASE_DIR / "checkpoints_ca"
CKPT_FIXED = BASE_DIR / "checkpoints_fixed"

TEST_JSON  = DATA_ROOT / "annotations" / "dataset_test.json"
IMAGE_DIR  = DATA_ROOT / "images"
LABEL_ENCODER = PROCESSED_DIR / "label_encoder.pkl"

GRADCAM_DIR = FIGURES_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_LABELS = [
    "blood","clot","congested","edematous","external pressure",
    "fistula","granulation","infiltration changes","mass","narrow",
    "necrotic","neoplasm","new organism","nodules","normal",
    "pigmentation","postoperative change","rough","sputum",
    "surgical stump","tube","ulcer","widened",
]

FIXED_DROPOUT_CNN = 0.4
FIXED_DROPOUT_HYB = 0.3
THRESHOLD = 0.5

# Her sınıf için kaç örnek görüntü gösterilecek (Confusion matrix
# figürünüzdeki gibi en sık 6 sınıfla tutarlılık için varsayılan bunlar)
TARGET_CLASSES_FOR_FIGURE = ["congested", "edematous", "sputum",
                             "normal", "narrow", "neoplasm"]
N_EXAMPLES_PER_CLASS = 2


# ╔══════════════════════════════════════════════════════════════╗
# ║  YARDIMCI — berd_main.py ile aynı                              ║
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

def get_tf_test():
    mean,std=[0.485,0.456,0.406],[0.229,0.224,0.225]
    return transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(), transforms.Normalize(mean,std)])


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODELLER — berd_main.py ile BİREBİR AYNI mimari               ║
# ║  (checkpoint'lerin state_dict'i ile uyuşması ZORUNLU)          ║
# ╚══════════════════════════════════════════════════════════════╝

def _head(f,nc,dr):
    return nn.Sequential(nn.Dropout(dr),nn.Linear(f,512),nn.GELU(),
                          nn.Dropout(dr/2),nn.Linear(512,nc))

class ResNet50(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        f=b.fc.in_features; b.fc=nn.Identity(); self.b=b; self.h=_head(f,nc,dr)
    def forward(self,x): return self.h(self.b(x))
    def gradcam_target_layer(self): return self.b.layer4[-1]

class EffB3(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        f=b.classifier[1].in_features; b.classifier=nn.Identity(); self.b=b; self.h=_head(f,nc,dr)
    def forward(self,x): return self.h(self.b(x))
    def gradcam_target_layer(self): return self.b.features[-1]

class SwinT(nn.Module):
    def __init__(self,nc,dr):
        super().__init__(); b=M.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        f=b.head.in_features; b.head=nn.Identity(); self.b=b
        self.h=nn.Sequential(nn.LayerNorm(f),nn.Dropout(dr),nn.Linear(f,512),
                              nn.GELU(),nn.Dropout(dr/2),nn.Linear(512,nc))
    def forward(self,x): return self.h(self.b(x))
    def gradcam_target_layer(self): return self.b.features[-1]  # [B,H,W,C] — CAM hesabında permute edilecek
    def gradcam_channel_last(self): return True

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
    def gradcam_target_layer(self):
        # Grad-CAM SADECE CNN (EfficientNet-B3) dalına uygulanıyor —
        # bkz. dosya başındaki gerekçe. Swin dalı, füzyon ve sınıflandırma
        # başlığı forward/backward'da olduğu gibi kalır.
        return self.cnn.features[-1]

def build_model(name, nc, dropout=None):
    if dropout is None:
        dropout = FIXED_DROPOUT_HYB if "hybrid" in name else FIXED_DROPOUT_CNN
    catalogue = {
        "resnet50":        lambda: ResNet50(nc,dropout),
        "efficientnet_b3": lambda: EffB3(nc,dropout),
        "swin_t":          lambda: SwinT(nc,dropout),
        "hybrid_eff_swin": lambda: HybridEffSwin(nc,dropout),
    }
    return catalogue[name]()


# ╔══════════════════════════════════════════════════════════════╗
# ║  GRAD-CAM ÇEKİRDEĞİ                                            ║
# ╚══════════════════════════════════════════════════════════════╝

class GradCAM:
    """
    Hook tabanlı, mimariden bağımsız Grad-CAM.
    target_layer: uzamsal [B,C,H,W] (veya channel_last=True ise [B,H,W,C])
                  harita üreten modül.
    """
    def __init__(self, model, target_layer, channel_last=False):
        self.model = model
        self.target_layer = target_layer
        self.channel_last = channel_last
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x, class_idx):
        """
        x: [1,3,H,W] giriş tensörü (normalize edilmiş)
        class_idx: ısı haritası üretilecek sınıfın indeksi
        Döndürür: [H,W] aralığı [0,1]'e normalize edilmiş CAM haritası (numpy)
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()

        act = self.activations   # [B,C,H,W] ya da [B,H,W,C]
        grad = self.gradients

        if self.channel_last:
            # Swin çıktısı [B,H,W,C] — [B,C,H,W]'ye çevir
            act  = act.permute(0, 3, 1, 2)
            grad = grad.permute(0, 3, 1, 2)

        weights = grad.mean(dim=(2, 3), keepdim=True)      # global average pooling — [B,C,1,1]
        cam = (weights * act).sum(dim=1, keepdim=True)      # [B,1,H,W]
        cam = F.relu(cam)
        cam = cam[0, 0].cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam


def overlay_cam_on_image(pil_img, cam, alpha=0.45):
    """cam [H,W] (0-1) ısı haritasını orijinal PIL görüntünün üzerine bindirir."""
    img = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return img, heatmap, overlay


# ╔══════════════════════════════════════════════════════════════╗
# ║  ÖRNEK SEÇİMİ                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def pick_examples(label2idx, target_classes, n_per_class):
    """Her hedef sınıf için test setinden n_per_class pozitif örnek seçer."""
    records = load_json(TEST_JSON)
    valid = set(label2idx.keys())
    picked = {c: [] for c in target_classes}
    for r in records:
        img_path = resolve_path(r.get("image_path",""), IMAGE_DIR)
        if not img_path.exists(): continue
        labels = parse_labels(r.get("label",""), valid)
        if not labels: continue
        for c in target_classes:
            if c in labels and len(picked[c]) < n_per_class:
                picked[c].append((img_path, labels))
    return picked


# ╔══════════════════════════════════════════════════════════════╗
# ║  ANA ÇALIŞTIRMA                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def load_checkpoint_for(name, nc, device):
    """
    hybrid_eff_swin için CA ablation checkpoint'i (berd_main.py'deki
    DENEY 2 mantığıyla tutarlı — hibrit orada da CA'dan okunuyordu).
    Diğer modeller için CKPT_FIXED.
    """
    if name == "hybrid_eff_swin":
        ckpt_path = CKPT_CA / f"{name}_ca_best.pt"
    else:
        ckpt_path = CKPT_FIXED / f"{name}_fixed_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt.get("params", {})
    dr = params.get("dropout", FIXED_DROPOUT_HYB if "hybrid" in name else FIXED_DROPOUT_CNN)
    model = build_model(name, nc, dr).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_gradcam_for_model(name, label2idx, idx2label, examples, device):
    nc = len(label2idx)
    print(f"\n[Grad-CAM] {name} yükleniyor...")
    model = load_checkpoint_for(name, nc, device)

    target_layer = model.gradcam_target_layer()
    channel_last = getattr(model, "gradcam_channel_last", lambda: False)()
    cam_engine = GradCAM(model, target_layer, channel_last=channel_last)

    tf = get_tf_test()

    for class_name, items in examples.items():
        class_idx = label2idx[class_name]
        for i, (img_path, gt_labels) in enumerate(items):
            pil_img = Image.open(img_path).convert("RGB")
            x = tf(pil_img).unsqueeze(0).to(device)
            x.requires_grad_(False)

            cam = cam_engine(x, class_idx)
            img, heatmap, overlay = overlay_cam_on_image(pil_img, cam)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(heatmap); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
            axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
            plt.suptitle(f"{name} — class: {class_name} — GT: {', '.join(gt_labels)}",
                        fontsize=10)
            plt.tight_layout()
            out_path = GRADCAM_DIR / f"gradcam_{name}_{class_name}_{i}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  ✓ {out_path.name}")

    cam_engine.remove_hooks()
    del model
    torch.cuda.empty_cache()


def build_comparison_figure(model_names, label2idx, class_name, example_idx,
                            examples, device):
    """
    Tek bir görüntü için BİRDEN FAZLA modelin Grad-CAM'ini yan yana
    gösteren karşılaştırma figürü — makalede tek bir "temsili" figür
    olarak kullanılabilir (örn. Figure 8: Grad-CAM comparison).
    """
    nc = len(label2idx)
    img_path, gt_labels = examples[class_name][example_idx]
    pil_img = Image.open(img_path).convert("RGB")
    tf = get_tf_test()
    x = tf(pil_img).unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, len(model_names) + 1, figsize=(4 * (len(model_names) + 1), 4))
    axes[0].imshow(np.array(pil_img.resize((224, 224))))
    axes[0].set_title("Original"); axes[0].axis("off")

    for i, name in enumerate(model_names):
        model = load_checkpoint_for(name, nc, device)
        target_layer = model.gradcam_target_layer()
        channel_last = getattr(model, "gradcam_channel_last", lambda: False)()
        cam_engine = GradCAM(model, target_layer, channel_last=channel_last)

        cam = cam_engine(x, label2idx[class_name])
        _, _, overlay = overlay_cam_on_image(pil_img, cam)
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(name)
        axes[i+1].axis("off")

        cam_engine.remove_hooks()
        del model
        torch.cuda.empty_cache()

    plt.suptitle(f"Grad-CAM Comparison — class: {class_name}", fontsize=12)
    plt.tight_layout()
    out_path = GRADCAM_DIR / f"gradcam_comparison_{class_name}.png"
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✓ {out_path.name}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    label2idx, idx2label = load_label_encoder()
    print(f"Sınıf sayısı: {len(label2idx)}")

    examples = pick_examples(label2idx, TARGET_CLASSES_FOR_FIGURE, N_EXAMPLES_PER_CLASS)
    for c, items in examples.items():
        print(f"  {c}: {len(items)} örnek bulundu")

    # ── Her model için ayrı ayrı Grad-CAM üret ──────────────────────
    # NOT: ViT-B/16 kasıtlı olarak dışlanmıştır (bkz. dosya başındaki
    # gerekçe — attention rollout gerektirir, Grad-CAM değil).
    MODELS_FOR_GRADCAM = ["resnet50", "efficientnet_b3", "swin_t", "hybrid_eff_swin"]
    for name in MODELS_FOR_GRADCAM:
        try:
            run_gradcam_for_model(name, label2idx, idx2label, examples, device)
        except FileNotFoundError as e:
            print(f"  [ATLANDI] {name}: {e}")

    # ── Makale için tek karşılaştırma figürü (örn. "congested" sınıfı) ──
    print("\n[Karşılaştırma figürü]")
    try:
        build_comparison_figure(MODELS_FOR_GRADCAM, label2idx,
                                class_name="congested", example_idx=0,
                                examples=examples, device=device)
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"  [ATLANDI] Karşılaştırma figürü: {e}")

    print(f"\n✓ Tüm Grad-CAM görselleri: {GRADCAM_DIR}")
    print("\n⚠️  MAKALEYE EKLEME NOTU:")
    print("   3.7 Grad-CAM bölümüne şunu ekleyin:")
    print("   - Hangi katmana hook'landığı (CNN son evrişim bloğu; Hybrid için")
    print("     sadece EfficientNet-B3 dalı — Swin dalı Grad-CAM kapsamında değil)")
    print("   - ViT-B/16'nın bu analize DAHİL OLMADIĞI (mimari uyumsuzluğu)")
    print("   - Isı haritalarının klinik olarak anlamlı bölgelere (lezyon/darlık")
    print("     alanı vb.) odaklanıp odaklanmadığına dair NİTEL bir değerlendirme")
    print("     (bir uzman/pulmonoloğun görsel değerlendirmesi olmadan, sadece")
    print("     'model X bölgesine bakıyor' demek yeterli değildir — mümkünse")
    print("     bir klinisyenin kısa bir doğrulama yorumu ekleyin)")