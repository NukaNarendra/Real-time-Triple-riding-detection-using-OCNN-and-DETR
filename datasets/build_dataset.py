"""
datasets/build_dataset.py

Utility script for constructing, validating, and preparing datasets for training and evaluation.
This includes verifying that every image has a corresponding label file, creating train/val/test
splits, and optionally merging datasets.

Usage:
    python build_dataset.py --dataset motorcycle_rider_dataset --split 0.8 0.1 0.1
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from typing import Tuple, List

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def list_images(path: Path) -> List[Path]:
    """List all image files in a directory recursively."""
    return [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def has_label(image_path: Path, label_dir: Path) -> bool:
    """Check if corresponding YOLO label exists."""
    label_path = label_dir / (image_path.stem + ".txt")
    return label_path.exists()


def split_dataset(images: List[Path], ratios: Tuple[float, float, float]) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split list of images into train, val, test based on ratios."""
    random.shuffle(images)
    total = len(images)
    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)
    return images[:train_end], images[train_end:val_end], images[val_end:]


def copy_files(image_paths: List[Path], base_dst: Path, label_dir: Path):
    """Copy images and corresponding labels to the new split directories."""
    for img in image_paths:
        rel_name = img.name
        lbl = label_dir / (img.stem + ".txt")
        dst_img = base_dst / "images" / rel_name
        dst_lbl = base_dst / "labels" / (img.stem + ".txt")

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(str(img), str(dst_img))
        if lbl.exists():
            shutil.copy(str(lbl), str(dst_lbl))


def build_dataset(root_dir: str, split: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Verify and (re)build dataset into train/valid/test structure.

    Args:
        root_dir: path to dataset folder (e.g., datasets/motorcycle_rider_dataset)
        split: train, val, test ratio
    """
    root = Path(root_dir)
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"No 'images/' found under {root_dir}")

    all_images = [p for p in list_images(images_dir) if has_label(p, labels_dir)]
    train_imgs, val_imgs, test_imgs = split_dataset(all_images, split)

    for name, imgset in [("train", train_imgs), ("valid", val_imgs), ("test", test_imgs)]:
        dst = root / name
        if dst.exists():
            shutil.rmtree(dst)
        copy_files(imgset, dst, labels_dir)

    print(f"✅ Dataset rebuilt successfully: {root_dir}")
    print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Train/Val/Test split ratios")
    args = parser.parse_args()

    build_dataset(args.dataset, tuple(args.split))
