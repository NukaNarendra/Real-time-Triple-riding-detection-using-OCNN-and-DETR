# inference/privacy.py
import cv2
import numpy as np
from typing import List, Tuple, Optional

def blur_region(img: np.ndarray, bbox: Tuple[int,int,int,int], ksize: int = 23) -> None:
    x1,y1,x2,y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(img.shape[1]-1, int(x2)); y2 = min(img.shape[0]-1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = ksize if ksize % 2 == 1 else ksize+1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = blurred

def blur_faces_and_plates(image: np.ndarray, face_bboxes: Optional[List[Tuple[int,int,int,int]]] = None,
                          plate_bboxes: Optional[List[Tuple[int,int,int,int]]] = None) -> np.ndarray:
    out = image.copy()
    if face_bboxes:
        for b in face_bboxes:
            blur_region(out, b, ksize=31)
    if plate_bboxes:
        for p in plate_bboxes:
            blur_region(out, p, ksize=31)
    return out
