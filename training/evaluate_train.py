#!/usr/bin/env python3
"""
Evaluate a trained license-plate detector on a split (val/test) and optionally export predictions.

Examples:
    python training/evaluate_train.py --weights checkpoints/plate/best.pt --split val
    python training/evaluate_train.py --weights checkpoints/plate/best.pt --split test --save-plots
"""

import argparse
import json
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained YOLO plate detector.")
    p.add_argument("--weights", type=str, default="checkpoints/plate/best.pt",
                   help="Path to trained weights .pt")
    p.add_argument("--data", type=str,
                   default="data/indian_license_plates_dataset/data.yaml",
                   help="Path to data.yaml")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--project", type=str, default="checkpoints/plate",
                   help="Where to put eval artifacts")
    p.add_argument("--name", type=str, default="eval",
                   help="Eval run name (under project)")
    p.add_argument("--save-plots", action="store_true", help="Save PR/Confusion plots")
    p.add_argument("--save-json", action="store_true", help="Write a metrics.json summary")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(
            "ERROR: Ultralytics not found. Ensure it's installed.\nOriginal error: ",
            e,
            file=sys.stderr
        )
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}", file=sys.stderr)
        sys.exit(2)

    model = YOLO(str(weights_path))

    val_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=args.save_plots,  # produces PR, F1, confusion, etc.
    )
    if args.device is not None:
        val_kwargs["device"] = args.device

    print("Running evaluation with args:", val_kwargs)
    metrics = model.val(**val_kwargs)

    # Print a concise summary
    try:
        box = metrics.box
        print(
            f"Results ({args.split}): "
            f"mAP50-95={box.map:.4f}, mAP50={box.map50:.4f}, "
            f"P={box.mp:.4f}, R={box.mr:.4f}"
        )
    except Exception:
        print("Evaluation complete.")

    # Optionally save a JSON summary
    if args.save_json:
        out_dir = Path(args.project) / args.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "metrics.json"
        try:
            payload = {
                "split": args.split,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "metrics": {
                    "map50_95": float(metrics.box.map),
                    "map50": float(metrics.box.map50),
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "fitness": float(getattr(metrics, "fitness", 0.0)),
                },
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote summary: {out_json}")
        except Exception as e:
            print(f"[WARN] Could not write metrics.json: {e}")

if __name__ == "__main__":
    main()
