import os
import yaml
import argparse
import time
from pathlib import Path
import random
import math

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR

from db.session import SessionLocal
from utils.logger import get_logger

logger = get_logger("train_detector")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------
# Box helpers
# ---------------------------

def _clip_boxes(boxes, w, h):
    if boxes.numel() == 0:
        return boxes
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)
    # ensure x1 < x2, y1 < y2 after clipping
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1, x2 = torch.min(x1, x2), torch.max(x1, x2)
    y1, y2 = torch.min(y1, y2), torch.max(y1, y2)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes


def _hflip_boxes(boxes, w):
    if boxes.numel() == 0:
        return boxes
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    boxes[:, 0] = w - x2
    boxes[:, 2] = w - x1
    return boxes


def _rotate90_boxes_ccw(boxes, w, h, k):
    """
    Rotate boxes by k * 90 degrees counter-clockwise.
    boxes: (N,4) in [x1,y1,x2,y2] with 0<=x<=w, 0<=y<=h
    Returns updated boxes and new (w,h)
    """
    if k % 4 == 0 or boxes.numel() == 0:
        return boxes, (w, h)

    x1, y1, x2, y2 = boxes.unbind(-1)

    if k % 4 == 1:  # 90° CCW
        nx1 = y1
        ny1 = w - x2
        nx2 = y2
        ny2 = w - x1
        nw, nh = h, w
    elif k % 4 == 2:  # 180°
        nx1 = w - x2
        ny1 = h - y2
        nx2 = w - x1
        ny2 = h - y1
        nw, nh = w, h
    else:  # 270° CCW
        nx1 = h - y2
        ny1 = x1
        nx2 = h - y1
        ny2 = x2
        nw, nh = h, w

    nboxes = torch.stack([nx1, ny1, nx2, ny2], dim=-1)
    nboxes = _clip_boxes(nboxes, nw, nh)
    return nboxes, (nw, nh)


# ---------------------------
# Dataset with safer/leaner aug
# ---------------------------
class YoloDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None, classes=None,
                 augment=False, rot90_p=0.15, hflip_p=0.4):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.classes = classes or []
        self.augment = augment
        self.rot90_p = rot90_p
        self.hflip_p = hflip_p
        self.items = [p for p in sorted(self.images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]

        # Photometric-only (gentler than before)
        self.photometric = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)], p=0.5),
            T.RandomAutocontrast(p=0.2),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        label_file = self.labels_dir / (p.stem + ".txt")
        boxes = []
        labels = []
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid, xc, yc, bw, bh = map(float, parts)
                    cx = xc * w
                    cy = yc * h
                    bw_px = bw * w
                    bh_px = bh * h
                    x1 = cx - bw_px / 2.0
                    y1 = cy - bh_px / 2.0
                    x2 = cx + bw_px / 2.0
                    y2 = cy + bh_px / 2.0
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cid) + 1)
        boxes = torch.as_tensor(np.array(boxes, dtype=np.float32)) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels, dtype=np.int64)) if labels else torch.zeros((0,), dtype=torch.int64)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.augment:
            orig_w, orig_h = image.size
            # HFlip
            if random.random() < self.hflip_p:
                image = F.hflip(image)
                boxes = _hflip_boxes(boxes.clone(), orig_w)
            # Rot90 ("data rototary")
            if random.random() < self.rot90_p:
                k = random.choice([1, 2, 3])  # 90/180/270
                boxes, (new_w, new_h) = _rotate90_boxes_ccw(boxes.clone(), orig_w, orig_h, k)
                image = F.rotate(image, angle=90 * k, expand=True)
                orig_w, orig_h = new_w, new_h
            # Photometric
            image = self.photometric(image)

        image = self.transforms(image) if self.transforms else T.ToTensor()(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        return image, target


# ---------------------------
# Data utils
# ---------------------------

def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------
# Model
# ---------------------------

def get_model(num_classes, device):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model.to(device)


# ---------------------------
# Train / Evaluate
# ---------------------------

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, use_amp=True, max_grad_norm=1.0, scheduler=None):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        # unscale and clip for stability
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if i % print_freq == 0:
            logger.info(f"Epoch {epoch} Iter {i}/{len(data_loader)} Loss {losses.item():.4f}")

    if scheduler is not None:
        scheduler.step()


def evaluate_simple(model, data_loader, device, score_threshold=0.5, iou_threshold=0.5):
    """Simple P/R eval with score filtering to cut false positives."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].cpu().numpy()
                gt_labels = tgt["labels"].cpu().numpy() if "labels" in tgt else np.array([])

                # filter predictions by confidence
                keep = out["scores"] >= score_threshold
                pred_boxes = out["boxes"][keep].cpu().numpy()
                pred_labels = out["labels"][keep].cpu().numpy()

                matched = set()
                for pb, pl in zip(pred_boxes, pred_labels):
                    best_iou = 0.0
                    best_j = -1
                    for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if j in matched:
                            continue
                        ix1 = max(pb[0], gb[0]); iy1 = max(pb[1], gb[1])
                        ix2 = min(pb[2], gb[2]); iy2 = min(pb[3], gb[3])
                        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                        inter = iw * ih
                        area_p = max(0, (pb[2]-pb[0])*(pb[3]-pb[1]))
                        area_g = max(0, (gb[2]-gb[0])*(gb[3]-gb[1]))
                        union = area_p + area_g - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_iou >= iou_threshold:
                        tp += 1
                        matched.add(best_j)
                    else:
                        fp += 1
                fn += max(0, len(gt_boxes) - len(matched))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------
# Checkpoints
# ---------------------------

def save_checkpoint(state, path):
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, device, path):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))  # back-compat if a bare state_dict was saved
    if "optimizer" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_prec = ckpt.get("best_prec", 0.0)
    no_improve_epochs = ckpt.get("no_improve_epochs", 0)
    logger.info(f"Resumed from {path} at epoch {start_epoch} (best_prec={best_prec:.4f})")
    return start_epoch, best_prec, no_improve_epochs


# ---------------------------
# Schedulers
# ---------------------------

def build_warmup_cosine(optimizer, total_epochs, warmup_epochs=2, min_lr=1e-6):
    # linear warmup from 0 -> base lr
    def warmup_lambda(current_epoch):
        if warmup_epochs == 0:
            return 1.0
        return float(current_epoch + 1) / float(max(1, warmup_epochs))

    warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr)
    if warmup_epochs > 0:
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        return cosine


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--classes", nargs="+", default=["__background__","motorcycle","person","helmet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/detector_best.pth")
    parser.add_argument("--last", default="checkpoints/detector_last.pth")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-workers", type=int, default=2)
    parser.add_argument("--rot90-p", type=float, default=0.15, help="Probability of 90° rotation augmentation")
    parser.add_argument("--hflip-p", type=float, default=0.4, help="Probability of horizontal flip")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold for eval")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    imgs_train = Path(args.dataset) / "train" / "images"
    lbls_train = Path(args.dataset) / "train" / "labels"
    imgs_val = Path(args.dataset) / "valid" / "images"
    lbls_val = Path(args.dataset) / "valid" / "labels"

    train_ds = YoloDetectionDataset(
        imgs_train,
        lbls_train,
        transforms=T.Compose([T.ToTensor()]),
        classes=args.classes,
        augment=True,
        rot90_p=args.rot90_p,
        hflip_p=args.hflip_p,
    )
    val_ds = YoloDetectionDataset(
        imgs_val,
        lbls_val,
        transforms=T.Compose([T.ToTensor()]),
        classes=args.classes,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=args.val_workers, collate_fn=collate_fn)

    num_classes = len(args.classes)
    model = get_model(num_classes, args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    scheduler = build_warmup_cosine(optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs, min_lr=args.min_lr)
    use_amp = (args.device.startswith("cuda") and (not args.no_amp))

    best_prec = 0.0
    no_improve_epochs = 0
    patience = 5  # Stop if no improvement for 5 epochs

    os.makedirs(Path(args.out).parent, exist_ok=True)
    os.makedirs(Path(args.last).parent, exist_ok=True)

    start_epoch = 1
    if args.resume is not None and Path(args.resume).exists():
        start_epoch, best_prec, no_improve_epochs = load_checkpoint(model, optimizer, args.device, args.resume)
        # Rebuild scheduler to continue from correct epoch index
        for _ in range(start_epoch - 1):
            scheduler.step()

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"Epoch {epoch} starting | lr={[group['lr'] for group in optimizer.param_groups]}")
        train_one_epoch(
            model, optimizer, train_loader, args.device, epoch,
            print_freq=50, use_amp=use_amp, max_grad_norm=args.max_grad_norm, scheduler=None
        )
        # step scheduler per-epoch here
        scheduler.step()

        metrics = evaluate_simple(model, val_loader, args.device, score_threshold=args.score_threshold)
        logger.info(f"Epoch {epoch} eval precision {metrics['precision']:.4f} recall {metrics['recall']:.4f}")

        # Always save a "+last" checkpoint for resuming
        save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_prec": best_prec,
            "no_improve_epochs": no_improve_epochs,
            "classes": args.classes,
            "score_threshold": args.score_threshold,
        }, args.last)

        # Save best by precision
        if metrics["precision"] > best_prec:
            best_prec = metrics["precision"]
            no_improve_epochs = 0
            save_checkpoint({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_prec": best_prec,
                "no_improve_epochs": no_improve_epochs,
                "classes": args.classes,
                "score_threshold": args.score_threshold,
            }, args.out)
        else:
            no_improve_epochs += 1
            logger.info(f"No improvement for {no_improve_epochs} epoch(s)")

        if no_improve_epochs >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info("Training complete")


if __name__ == "__main__":
    main()
