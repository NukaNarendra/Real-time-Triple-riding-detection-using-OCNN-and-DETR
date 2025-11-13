# inference/ocr.py
from typing import Tuple, Optional
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

class OCRReader:
    def __init__(self, langs=['en']):
        self.reader = None
        if EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(langs, gpu=False)
        elif PYTESSERACT_AVAILABLE:
            self.reader = None
        else:
            self.reader = None

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        if plate_crop is None:
            return None, None
        if EASYOCR_AVAILABLE and self.reader is not None:
            results = self.reader.readtext(plate_crop)
            if not results:
                return None, None
            best = max(results, key=lambda r: r[2])  # text, bbox, conf
            text = best[1]
            conf = float(best[2])
            return text, conf
        if PYTESSERACT_AVAILABLE:
            import cv2
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            text = pytesseract.image_to_string(gray, config=config)
            text = "".join(ch for ch in text if ch.isalnum())
            conf = None
            return (text if text else None, conf)
        return None, None
