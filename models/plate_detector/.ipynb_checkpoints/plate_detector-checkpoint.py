"""
models/plate_detector/plate_detector.py

A lightweight license-plate localization module based on simple image processing heuristics.
This is *not* a state-of-the-art plate detector; it is a practical starting point useful
for prototyping and offline batch use. For production you should replace this with a
trained plate-localization model (YOLO/RetinaNet/DETR variant trained for plates).

API:
    pd = PlateDetector()
    plate_bbox = pd.detect_plate(image, vehicle_bbox=None)
    # returns (x1,y1,x2,y2) or None
"""

from typing import Optional, Tuple
import numpy as np
import cv2


class PlateDetector:
    def __init__(self, min_area: int = 500, aspect_ratio_range: Tuple[float, float] = (2.0, 7.0)):
        """
        Args:
            min_area: ignore contours smaller than this area
            aspect_ratio_range: expected plate width/height ratio range
        """
        self.min_area = min_area
        self.aspect_ratio_range = aspect_ratio_range

    def detect_plate(self, image: np.ndarray, vehicle_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Attempt to localize a rectangular license plate region in the image.

        Args:
            image: HxWxC (BGR)
            vehicle_bbox: optional bbox around vehicle to restrict search; format x1,y1,x2,y2

        Returns:
            bbox (x1,y1,x2,y2) if found, else None
        """
        # restrict search region to vehicle bbox if provided
        img_h, img_w = image.shape[:2]
        if vehicle_bbox is not None:
            x1, y1, x2, y2 = map(int, vehicle_bbox)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_w - 1, x2); y2 = min(img_h - 1, y2)
            search_img = image[y1:y2, x1:x2]
            base_x, base_y = x1, y1
        else:
            search_img = image
            base_x, base_y = 0, 0

        gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        # Edge detection
        edged = cv2.Canny(gray, 30, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < self.min_area:
                continue
            aspect = (w / float(h)) if h > 0 else 0
            if aspect < self.aspect_ratio_range[0] or aspect > self.aspect_ratio_range[1]:
                continue
            # approx polygon to check rectangularity
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) >= 4:
                candidates.append((x, y, w, h, area))

        if not candidates:
            return None

        # choose largest area candidate
        best = max(candidates, key=lambda t: t[4])
        x, y, w, h, _ = best
        x1, y1 = base_x + x, base_y + y
        x2, y2 = base_x + x + w, base_y + y + h

        # ensure bbox within image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w - 1, x2); y2 = min(img_h - 1, y2)

        return (int(x1), int(y1), int(x2), int(y2))
