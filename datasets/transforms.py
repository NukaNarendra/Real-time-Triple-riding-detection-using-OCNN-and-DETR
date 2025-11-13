"""
datasets/transforms.py

Defines augmentation and preprocessing pipelines using Albumentations.
Includes a fallback to OpenCV-only transformations if Albumentations is not installed.
"""

import numpy as np
import cv2

try:
    import albumentations as A
    ALB_AVAILABLE = True
except ImportError:
    ALB_AVAILABLE = False


def get_train_transforms(img_size: int = 640):
    """Return a training augmentation pipeline."""
    if ALB_AVAILABLE:
        return A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.4),
            A.Resize(img_size, img_size)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:
        # Fallback: only resize
        def _simple_resize(image, bboxes=None, class_labels=None):
            image = cv2.resize(image, (img_size, img_size))
            return {"image": image, "bboxes": bboxes or [], "class_labels": class_labels or []}
        return _simple_resize


def get_valid_transforms(img_size: int = 640):
    """Validation transforms (minimal)."""
    if ALB_AVAILABLE:
        return A.Compose([
            A.Resize(img_size, img_size)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:
        def _simple_resize(image, bboxes=None, class_labels=None):
            image = cv2.resize(image, (img_size, img_size))
            return {"image": image, "bboxes": bboxes or [], "class_labels": class_labels or []}
        return _simple_resize
