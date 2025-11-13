# training/train_rider_counter.py
import os
import argparse
import time
from pathlib import Path
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image
import numpy as np

from models.rider_counter.rider_cnn import RiderCNN  # still supported if you want
from utils.logger import get_logger

logger = get_logger("train_rider_counter")

# class index -> human rider count used for MAE metric
CLASS_VALUES = {0: 1, 1: 2, 2: 3}


# ----------------------- Transforms -----------------------
def _try_randaugment():
    try:
        from torchvision.transforms import RandAugment
        return RandAugment(num_ops=2, magnitude=7)
    except Exception:
        class Identity:
            def __call__(self, x): return x
        logger.info("RandAugment not available; proceeding without it.")
        return Identity()

def build_train_transform(img_size):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        _try_randaugment(),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
    ])

def build_eval_transform(img_size):
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ----------------------- Dataset -----------------------
class SimpleImageFolder(Dataset):
    """
    Strictly accepts class dirs named '1', '2', '3' and ignores hidden folders (like .ipynb_checkpoints).
    Maps labels: '1'->0, '2'->1, '3'->2
    """
    VALID_CLASSES = ["1", "2", "3"]

    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)

        # discover subdirs
        all_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        hidden = [d.name for d in all_dirs if d.name.startswith(".")]
        extra = [d.name for d in all_dirs if (d.name not in self.VALID_CLASSES and not d.name.startswith("."))]

        if hidden:
            logger.info(f"Ignoring hidden folders: {hidden}")
        if extra:
            logger.warning(f"Ignoring unexpected class folders: {extra}")

        self.classes = self.VALID_CLASSES
        self.class_to_idx = {"1": 0, "2": 1, "3": 2}

        self.files = []
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for cname in self.classes:
            p = self.root / cname
            if not p.exists():
                logger.warning(f"Class folder missing: {p}")
                continue
            for f in p.iterdir():
                if f.is_file() and f.suffix.lower() in img_exts:
                    self.files.append((str(f), self.class_to_idx[cname]))

        self.transform = transform

        # logs
        counts = {c: 0 for c in self.classes}
        for _, idx in self.files:
            counts[self.classes[idx]] += 1
        logger.info(f"Loaded {len(self.files)} images from {root_dir}")
        logger.info("Class mapping: " + ", ".join([f"{self.class_to_idx[c]}:{c}" for c in self.classes]))
        logger.info("Class counts: " + ", ".join([f"{c}={counts[c]}" for c in self.classes]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp, label = self.files[idx]
        img = Image.open(fp).convert("RGB")
        img = self.transform(img) if self.transform is not None else img
        # safety
        if not (0 <= label <= 2):
            raise RuntimeError(f"Label out of range for file {fp}: {label}")
        return img, label


# ----------------------- Metrics helpers -----------------------
def confusion_matrix(preds, labels, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm

def per_class_accuracy(cm):
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(cm) / cm.sum(axis=1)
        acc = np.nan_to_num(acc)
    return acc


# ----------------------- MixUp helpers -----------------------
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0.0:
        return x, y, 1.0, None
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_b = y[index]
    return mixed_x, (y, y_b), lam, index

def mixup_ce_loss(criterion, pred, targets, lam):
    y_a, y_b = targets
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ----------------------- EMA (safe) -----------------------
class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        msd, esd = model.state_dict(), self.ema.state_dict()
        d = self.decay
        for k in esd.keys():
            m = msd[k]
            e = esd[k]
            if torch.is_floating_point(e) and torch.is_floating_point(m):
                esd[k].mul_(d).add_(m, alpha=1.0 - d)
            else:
                esd[k].copy_(m)


# ----------------------- Train / Eval loops -----------------------
def train_epoch(model, loader, criterion, optimizer, device, ema=None, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0
    total_seen = 0
    start = time.time()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_alpha > 0.0:
            imgs, targets, lam, _ = mixup_data(imgs, labels, alpha=mixup_alpha)
        else:
            targets, lam = labels, 1.0

        # defensive check
        if torch.any((targets if isinstance(targets, torch.Tensor) else targets[0]) > 2):
            raise RuntimeError("Found label >= 3 in batch. Check dataset folders are only 1/2/3.")

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = mixup_ce_loss(criterion, outputs, targets, lam) if mixup_alpha > 0.0 else criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_seen += bs

    dur = time.time() - start
    avg_loss = total_loss / max(total_seen, 1)
    ips = total_seen / max(dur, 1e-6)
    return avg_loss, dur, ips


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    mae_sum = 0.0
    cm = np.zeros((3, 3), dtype=np.int64)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        cm += confusion_matrix(preds.cpu().numpy(), labels.cpu().numpy(), num_classes=3)

        pred_vals = np.vectorize(CLASS_VALUES.get)(preds.cpu().numpy())
        true_vals = np.vectorize(CLASS_VALUES.get)(labels.cpu().numpy())
        mae_sum += np.mean(np.abs(pred_vals - true_vals))

    acc = correct / total if total > 0 else 0.0
    mae = mae_sum / len(loader) if len(loader) > 0 else 0.0
    per_cls = per_class_accuracy(cm)
    return {"acc": acc, "mae": mae, "cm": cm, "per_class_acc": per_cls}


def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg.get("lr", None)


# ----------------------- Checkpoint IO -----------------------
def save_checkpoint(path, model, optimizer, epoch, best_acc, args, ema=None):
    pkg = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "args": vars(args),
    }
    if ema is not None:
        pkg["model_ema"] = ema.ema.state_dict()
    torch.save(pkg, path)
    logger.info(f"Saved checkpoint to {path}")

def load_checkpoint(path, model, optimizer=None, device="cpu", ema=None):
    ckpt = torch.load(path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ema is not None and "model_ema" in ckpt:
            ema.ema.load_state_dict(ckpt["model_ema"])
        epoch = int(ckpt.get("epoch", 0))
        best_acc = float(ckpt.get("best_acc", 0.0))
    else:
        # bare state_dict
        model.load_state_dict(ckpt)
        epoch, best_acc = 0, 0.0
    logger.info(f"Loaded checkpoint from {path} (epoch={epoch}, best_acc={best_acc:.4f})")
    return epoch, best_acc


# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--out", default="checkpoints/rider_counter.pth")
    parser.add_argument("--resume", default="", help="path to checkpoint to resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--dropout2d", type=float, default=0.10)
    parser.add_argument("--arch", default="cnn", choices=["cnn", "mbv3"])
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha; 0 disables")
    parser.add_argument("--imbalance_strategy", default="class_weights",
                        choices=["sampler", "class_weights", "none"],
                        help="Use either 'sampler' or 'class_weights' (not both), or 'none'")
    args = parser.parse_args()

    # Transforms / datasets
    train_tf = build_train_transform(args.img_size)
    val_tf = build_eval_transform(args.img_size)
    train_ds = SimpleImageFolder(args.train_dir, transform=train_tf)
    val_ds = SimpleImageFolder(args.val_dir, transform=val_tf)

    # Class stats (force length 3)
    train_labels = [idx for _, idx in train_ds.files]
    class_sample_count = np.bincount(train_labels, minlength=3)[:3]

    # DataLoaders (imbalance handling)
    pin = (args.device == "cuda")
    if args.imbalance_strategy == "sampler":
        weights_per_class = 1.0 / (class_sample_count + 1e-6)
        sample_weights = [weights_per_class[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler,
            num_workers=args.num_workers, pin_memory=pin,
            persistent_workers=(args.num_workers > 0),
        )
        use_class_weights = False
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, pin_memory=pin,
            persistent_workers=(args.num_workers > 0),
        )
        use_class_weights = (args.imbalance_strategy == "class_weights")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )

    # Model selection
    device = args.device
    if args.arch == "cnn":
        # NOTE: original RiderCNN expects 64x64; if you use other sizes, ensure RiderCNN uses Adaptive pooling.
        model = RiderCNN(num_classes=3, dropout=args.dropout, dropout2d=args.dropout2d).to(device)
    else:
        import torchvision.models as tvm
        backbone = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        if args.freeze_backbone:
            for p in backbone.features.parameters():
                p.requires_grad = False
        # Correct in-features for MBv3-small = classifier[0].in_features (=576)
        in_features = backbone.classifier[0].in_features  # 576
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, 3),
        )
        model = backbone.to(device)

    # Loss
    if use_class_weights:
        class_weights = torch.tensor(1.0 / (class_sample_count + 1e-6), dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine with warmup (per-epoch stepping)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    total_epochs = args.epochs
    warmup_epochs = min(args.warmup_epochs, total_epochs - 1) if total_epochs > 1 else 0
    base_lr = args.lr
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=base_lr * 0.05)

    # EMA
    ema = ModelEMA(model, decay=0.999, device=device)

    # Log model size
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model params: {n_params/1e6:.3f}M | Train images: {len(train_ds)} | Val images: {len(val_ds)} | "
        f"Batch: {args.batch} | LR: {base_lr} | Imbalance: {args.imbalance_strategy}"
    )

    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, device=device, ema=ema)

    os.makedirs(Path(args.out).parent, exist_ok=True)

    # Train
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Warmup then cosine
        if epoch <= warmup_epochs:
            warm_lr = base_lr * epoch / max(1, warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr
        else:
            cosine.step()

        train_loss, train_time, ips = train_epoch(
            model, train_loader, criterion, optimizer, device, ema=ema, mixup_alpha=args.mixup
        )
        metrics = evaluate(ema.ema, val_loader, device)
        cm = metrics["cm"]
        per_cls = metrics["per_class_acc"]
        cm_str = "; ".join([f"row{ri}:" + ",".join(map(str, row)) for ri, row in enumerate(cm.tolist())])
        per_cls_str = ", ".join([f"class{ci}={acc:.3f}" for ci, acc in enumerate(per_cls)])

        logger.info(
            f"Epoch {epoch:>3} | train_loss {train_loss:.4f} | val_acc {metrics['acc']:.4f} | "
            f"val_mae {metrics['mae']:.4f} | lr {get_lr(optimizer):.6f} | "
            f"time {train_time:.1f}s | throughput {ips:.1f} img/s"
        )
        logger.info(f"Epoch {epoch:>3} | per_class_acc [{per_cls_str}]")
        logger.info(f"Epoch {epoch:>3} | confusion_matrix [{cm_str}]")

        # Save rolling "last"
        save_checkpoint(args.out.replace(".pth", "_last.pth"), model, optimizer, epoch, best_acc, args, ema=ema)

        # Save "best"
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            save_checkpoint(args.out, model, optimizer, epoch, best_acc, args, ema=ema)
            logger.info(f"New best model with acc={best_acc:.4f}")

    logger.info("Rider counter training complete")


if __name__ == "__main__":
    main()
