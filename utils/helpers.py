import os
import cv2
import random
import string
import time
from pathlib import Path

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def unique_id(prefix="violation"):
    ts = int(time.time() * 1000)
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{prefix}_{ts}_{rand}"

def save_image(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def list_images(folder, exts=(".jpg", ".jpeg", ".png")):
    p = Path(folder)
    files = []
    for ext in exts:
        files.extend(p.rglob(f"*{ext}"))
    return [str(f) for f in files]

def timestamp_str():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def normalize_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
