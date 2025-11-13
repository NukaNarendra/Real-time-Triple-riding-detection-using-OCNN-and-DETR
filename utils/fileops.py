"""
utils/fileops.py
Small file and I/O helpers: ensure directories, save images, create unique names, timestamp helpers.
"""

import os
import uuid
import time
from datetime import datetime
from typing import Tuple, Optional
from PIL import Image
import io

# NOTE: Pillow is used for safe image saving; avoids OpenCV dependency here for basic fileops.


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def timestamp_now() -> str:
    """Return ISO-like timestamp safe for filenames."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def make_unique_filename(prefix: str = "img", ext: str = ".jpg") -> str:
    """Create a unique filename with uuid and timestamp."""
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp_now()}_{uid}{ext}"


def save_image_pil(image, save_path: str, quality: int = 85) -> None:
    """
    Save an image-like object to disk using PIL.
    Accepts: PIL.Image.Image or a numpy array convertible to PIL (but keep numpy import out of this file).
    """
    ensure_dir(os.path.dirname(save_path))
    if hasattr(image, "save"):
        # Assume PIL image
        image.save(save_path, format="JPEG", quality=quality)
    else:
        # Try converting through bytes if image is bytes-like
        if isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
            img.convert("RGB").save(save_path, format="JPEG", quality=quality)
        else:
            # As fallback try to create PIL from numpy array (import locally to avoid dependency for callers not using numpy)
            try:
                from PIL import Image as PILImage
                import numpy as _np
                arr = _np.asarray(image)
                pil_img = PILImage.fromarray(arr)
                pil_img.convert("RGB").save(save_path, format="JPEG", quality=quality)
            except Exception as e:
                raise ValueError(f"Cannot save image, unsupported type: {type(image)}. Err: {e}")


def safe_join(root: str, filename: str) -> str:
    """Safe join ensuring path is inside root (prevents path traversal)."""
    final_path = os.path.abspath(os.path.join(root, filename))
    root_abs = os.path.abspath(root)
    if not final_path.startswith(root_abs):
        raise ValueError("Attempt to write outside target directory")
    return final_path
