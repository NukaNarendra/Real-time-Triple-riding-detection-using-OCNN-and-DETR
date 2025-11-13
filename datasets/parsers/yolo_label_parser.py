"""
datasets/parsers/yolo_label_parser.py

Parses YOLO-format label files and returns structured bounding boxes.

Format:
    class_id x_center y_center width height (normalized 0–1)
"""

from typing import List, Tuple
import numpy as np


def parse_yolo_label(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a YOLO label file.

    Returns:
        List of (class_id, x_center, y_center, width, height)
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid, xc, yc, w, h = map(float, parts)
            boxes.append((int(cid), xc, yc, w, h))
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    """Convert YOLO normalized box to (x1, y1, x2, y2)."""
    cid, xc, yc, w, h = box
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return cid, x1, y1, x2, y2
