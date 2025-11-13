"""
models/rider_counter/person_association.py

Simple utilities to associate detected person bounding boxes with a vehicle bounding box.
This module does not perform person detection by itself; it expects person boxes from
the detector (e.g., DetectorWrapper).

Primary function:
    count = count_riders(person_boxes, vehicle_box, iou_thresh=0.2)

Where:
    person_boxes: list of (x1,y1,x2,y2)
    vehicle_box: (x1,y1,x2,y2)
"""

from typing import List, Tuple
import math


def _iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union between two boxes in (x1,y1,x2,y2) format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def count_riders(person_boxes: List[Tuple[float, float, float, float]],
                 vehicle_box: Tuple[float, float, float, float],
                 iou_thresh: float = 0.2) -> int:
    """
    Count the number of person boxes that overlap the vehicle bounding box by IoU > iou_thresh.

    Args:
        person_boxes: list of (x1,y1,x2,y2)
        vehicle_box: (x1,y1,x2,y2)
        iou_thresh: IoU threshold to consider a person as 'on' the vehicle

    Returns:
        int: number of associated riders (0,1,2,...)
    """
    count = 0
    for p in person_boxes:
        if _iou(p, vehicle_box) > iou_thresh:
            count += 1
    return count
