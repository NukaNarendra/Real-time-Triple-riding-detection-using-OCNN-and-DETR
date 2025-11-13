# training/eval_rider_counter.py
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np

from models.rider_counter.rider_cnn import RiderCNN
from utils.logger import get_logger

logger = get_logger("eval_rider_counter")

# Model-class index -> count value used for MAE metric
# (same mapping as during training)
CLASS_VALUES = {0: 1, 1: 2, 2: 3}


# ----------------------- Dataset -----------------------
class SimpleImageFolder(Dataset):
    """Expects a directory with subfolders '1', '2', '3' containing images."""
    VALID_CLASSES = ["1", "2", "3"]
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.classes: List[str] = self.VALID_CLASSES
        self.class_to_idx = {"1": 0, "2": 1, "3": 2}
        self.files: List[Tuple[str, int]] = []

        # Collect files class-wise (ignore hidden dirs like ".ipynb_checkpoints")
        for c in self.classes:
            p = self.root / c
            if not p.exists() or not p.is_dir():
                continue
            for f in p.iterdir():
                if f.is_file() and f.suffix.lower() in self.VALID_EXTS and not f.name.startswith("."):
                    self.files.append((str(f), self.class_to_idx[c]))

        # Default transform mirrors training eval pipeline (Resize->CenterCrop->Normalize)
        self.transform = transform or T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Log summary
        counts = {c: 0 for c in self.classes}
        for _, idx in self.files:
            counts[self.classes[idx]] += 1
        logger.info(f"Loaded {len(self.files)} images from {self.root}")
        logger.info("Class mapping: " + ", ".join([f"{i}:{c}" for c, i in self.class_to_idx.items()]))
        logger.info("Class counts: " + ", ".join([f"{c}={counts[c]}" for c in self.classes]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp, label = self.files[idx]
        img = Image.open(fp).convert("RGB")
        img = self.transform(img)
        return img, label


# ----------------------- Metrics -----------------------
def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1)
        acc = np.nan_to_num(acc)
    return acc


# ----------------------- Eval Loop -----------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, want_confusion: bool = False):
    model.eval()
    correct = 0
    total = 0
    mae_sum = 0.0
    cm = np.zeros((3, 3), dtype=np.int64) if want_confusion else None

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # MAE on the mapped class values (1,2,3)
        pred_vals = np.vectorize(CLASS_VALUES.get)(preds.cpu().numpy())
        true_vals = np.vectorize(CLASS_VALUES.get)(labels.cpu().numpy())
        mae_sum += np.mean(np.abs(pred_vals - true_vals))

        if want_confusion:
            cm += confusion_matrix(preds.cpu().numpy(), labels.cpu().numpy(), num_classes=3)

    acc = correct / total if total > 0 else 0.0
    mae = mae_sum / len(loader) if len(loader) > 0 else 0.0

    out = {"acc": acc, "mae": mae}
    if want_confusion:
        out["cm"] = cm
        out["per_class_acc"] = per_class_accuracy(cm)
    return out


# ----------------------- Load Model -----------------------
def build_mbv3_same_as_training(dropout: float = 0.30) -> nn.Module:
    """
    Build MobileNetV3-Small with the SAME classifier head used in training:
    Linear(576->256) -> Hardswish -> Dropout(p=dropout) -> Linear(256->3)
    """
    import torchvision.models as tvm
    backbone = tvm.mobilenet_v3_small(weights=None)  # keep default features
    in_features = backbone.classifier[0].in_features  # should be 576
    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(256, 3),
    )
    return backbone


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: str) -> None:
    """Load EMA state if present; else 'model'; else assume raw state_dict."""
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        # prefer EMA if available
        if "model_ema" in state and isinstance(state["model_ema"], dict):
            model.load_state_dict(state["model_ema"], strict=True)
            logger.info("Loaded EMA weights from checkpoint.")
            return
        if "model" in state and isinstance(state["model"], dict):
            model.load_state_dict(state["model"], strict=True)
            logger.info("Loaded model weights from checkpoint.")
            return
    # fall back: assume it's a bare state_dict
    model.load_state_dict(state, strict=True)
    logger.info("Loaded bare state_dict from checkpoint.")


# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--val_dir", required=True, help="Validation set root with subfolders 1/2/3")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--arch", default="mbv3", choices=["mbv3", "cnn"])
    parser.add_argument("--dropout", type=float, default=0.30, help="(Only for mbv3) classifier dropout used in training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confusion", action="store_true", help="Print confusion matrix and per-class accuracy")
    args = parser.parse_args()

    device = args.device

    # Data
    val_ds = SimpleImageFolder(args.val_dir)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0)
    )

    # Model
    if args.arch == "cnn":
        model = RiderCNN(num_classes=3).to(device)
    else:
        model = build_mbv3_same_as_training(dropout=args.dropout).to(device)

    # Load weights
    load_checkpoint_into_model(model, args.model, device)

    # Eval
    metrics = evaluate(model, val_loader, device, want_confusion=args.confusion)
    logger.info(f"Eval acc={metrics['acc']:.4f} mae={metrics['mae']:.4f}")

    if args.confusion:
        cm = metrics["cm"]
        per_cls = metrics["per_class_acc"]
        cm_str = "; ".join([f"row{ri}:" + ",".join(map(str, row)) for ri, row in enumerate(cm.tolist())])
        per_cls_str = ", ".join([f"class{ci}={acc:.3f}" for ci, acc in enumerate(per_cls)])
        logger.info(f"Confusion matrix [{cm_str}]")
        logger.info(f"Per-class accuracy [{per_cls_str}]")


if __name__ == "__main__":
    main()
