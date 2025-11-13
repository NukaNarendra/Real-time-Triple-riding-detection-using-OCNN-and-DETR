# inference/tracker.py
from typing import Tuple, Dict, List
import time
import numpy as np

class Track:
    def __init__(self, tid: int, bbox: Tuple[float, float, float, float], score: float = 1.0):
        self.id = tid
        self.bbox = bbox  # x1,y1,x2,y2
        self.score = score
        self.last_seen = time.time()
        self.hits = 1
        self.missed = 0

    def update(self, bbox: Tuple[float, float, float, float], score: float = 1.0):
        self.bbox = bbox
        self.score = score
        self.last_seen = time.time()
        self.hits += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

class SimpleTracker:
    def __init__(self, max_missed: int = 15, iou_threshold: float = 0.3):
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: List[Dict]) -> List[Track]:
        assigned_tracks = set()
        assigned_dets = set()
        det_boxes = [tuple(d["bbox"]) for d in detections]
        det_scores = [d.get("score", 1.0) for d in detections]

        # If no tracks, initialize tracks for all detections
        if not self.tracks:
            for i, (b, s) in enumerate(zip(det_boxes, det_scores)):
                t = Track(self._next_id, b, s)
                self.tracks[self._next_id] = t
                self._next_id += 1
            return list(self.tracks.values())

        # Build IoU matrix
        track_ids = list(self.tracks.keys())
        iou_mat = np.zeros((len(track_ids), len(det_boxes)), dtype=float)
        for i, tid in enumerate(track_ids):
            for j, db in enumerate(det_boxes):
                iou_mat[i, j] = self._iou(self.tracks[tid].bbox, db)

        # Greedy assignment
        for _ in range(min(iou_mat.shape[0], iou_mat.shape[1])):
            i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            if iou_mat[i, j] < self.iou_threshold:
                break
            tid = track_ids[i]
            self.tracks[tid].update(det_boxes[j], det_scores[j])
            assigned_tracks.add(tid)
            assigned_dets.add(j)
            iou_mat[i, :] = -1
            iou_mat[:, j] = -1

        # unmatched detections -> new tracks
        for j, (b, s) in enumerate(zip(det_boxes, det_scores)):
            if j in assigned_dets:
                continue
            t = Track(self._next_id, b, s)
            self.tracks[self._next_id] = t
            self._next_id += 1

        # unmatched tracks -> mark missed and remove if too old
        for tid in list(self.tracks.keys()):
            if tid in assigned_tracks:
                continue
            self.tracks[tid].mark_missed()
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

        return list(self.tracks.values())
