import argparse
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
from utils.logger import get_logger

logger = get_logger("eval_detector")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class YoloDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.items = [p for p in sorted(self.images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]

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
        boxes = torch.as_tensor(np.array(boxes, dtype=np.float32)) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels, dtype=np.int64)) if labels else torch.zeros((0,), dtype=torch.int64)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transforms(image) if self.transforms else T.ToTensor()(image)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(checkpoint, device, num_classes):
    # Build the same architecture as training
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    ckpt = torch.load(checkpoint, map_location=device)
    # training saved a dict with key "model"; also accept bare state_dict for safety
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate(model, data_loader, device, score_threshold=0.5, iou_threshold=0.5, match_labels=False):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].cpu().numpy()
                gt_labels = tgt["labels"].cpu().numpy()

                # filter predictions by confidence
                scores = out["scores"].cpu().numpy() if "scores" in out else None
                keep = np.ones(len(out["boxes"]), dtype=bool) if scores is None else (scores >= score_threshold)
                pred_boxes = out["boxes"][keep].cpu().numpy()
                pred_labels = out["labels"][keep].cpu().numpy()

                matched = set()
                for pb, pl in zip(pred_boxes, pred_labels):
                    best_iou = 0.0
                    best_j = -1
                    for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if j in matched:
                            continue
                        # if label matching is required, skip mismatched classes
                        if match_labels and (pl != gl):
                            continue
                        ix1 = max(pb[0], gb[0]); iy1 = max(pb[1], gb[1])
                        ix2 = min(pb[2], gb[2]); iy2 = min(pb[3], gb[3])
                        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
                        inter = iw * ih
                        area_p = max(0.0, (pb[2]-pb[0])*(pb[3]-pb[1]))
                        area_g = max(0.0, (gb[2]-gb[0])*(gb[3]-gb[1]))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", nargs="+", default=["__background__", "motorcycle", "person", "helmet"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--match-labels", action="store_true", help="Require class label match when counting TP")
    args = parser.parse_args()

    imgs_test = Path(args.dataset) / "test" / "images"
    lbls_test = Path(args.dataset) / "test" / "labels"

    ds = YoloDetectionDataset(imgs_test, lbls_test, transforms=T.Compose([T.ToTensor()]))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    num_classes = len(args.classes)
    model = load_model(args.checkpoint, args.device, num_classes)

    metrics = evaluate(model, dl, args.device, score_threshold=args.score_threshold,
                       iou_threshold=args.iou_threshold, match_labels=args.match_labels)

    logger.info(
        f"Eval results: precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
        f"(score_th={args.score_threshold}, iou_th={args.iou_threshold}, match_labels={args.match_labels})"
    )


if __name__ == "__main__":
    main()
