#!/usr/bin/env python3
"""
Train a license-plate detector using Ultralytics YOLO on the dataset in /data.

Typical usage:
    python training/train_plate.py \
        --data data/indian_license_plates_dataset/data.yaml \
        --model yolov8n.pt \
        --epochs 50 --imgsz 640 --batch 16

Outputs:
- Runs/logs under checkpoints/plate/<run_name>/
- Best weights saved by Ultralytics as best.pt (we also copy them to checkpoints/plate/best.pt)
"""

import argparse
import shutil
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train license-plate detector (YOLO).")
    p.add_argument("--data", type=str,
                   default="data/indian_license_plates_dataset/data.yaml",
                   help="Path to data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="YOLO base model or path to .pt to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default=None,
                   help="CUDA device string, e.g. '0' or '0,1' or 'cpu'")
    p.add_argument("--name", type=str, default="plate_det",
                   help="Run name (appears under checkpoints/plate/)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr0", type=float, default=None, help="Override initial LR")
    p.add_argument("--resume", action="store_true", help="Resume the last run")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(
            "ERROR: Ultralytics not found. Ensure 'ultralytics' is in requirements.txt "
            "and installed in your environment.\nOriginal error: ", e,
            file=sys.stderr
        )
        sys.exit(1)

    project_dir = Path("checkpoints/plate")
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create/Load model
    model = YOLO(args.model)

    # Train
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.resume:
        train_kwargs["resume"] = True

    print("Starting training with args:", train_kwargs)
    results = model.train(**train_kwargs)

    # Try to copy best weights to a stable path for convenience
    run_dir = project_dir / args.name
    best_in_run = run_dir / "weights" / "best.pt"
    stable_best = project_dir / "best.pt"
    if best_in_run.exists():
        try:
            shutil.copy2(best_in_run, stable_best)
            print(f"[OK] Copied best weights to: {stable_best}")
        except Exception as e:
            print(f"[WARN] Could not copy best weights: {e}")
    else:
        print("[WARN] best.pt not found under run directory yet.")

    # Print a short metrics summary if available
    try:
        # Ultralytics returns a Results object; metrics accessed via results.metrics
        m = getattr(results, "metrics", None)
        if m is not None and hasattr(m, "box"):
            print(
                f"mAP50: {m.box.map50:.4f}  mAP50-95: {m.box.map:.4f}  "
                f"Precision: {m.box.mp:.4f}  Recall: {m.box.mr:.4f}"
            )
    except Exception:
        pass

if __name__ == "__main__":
    main()
