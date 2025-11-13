# inference/annotate.py
import cv2
from typing import List, Tuple, Dict, Optional

def draw_bbox(img, bbox, label=None, color=(0,255,0), thickness=2):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label:
        text = str(label)
        ((w,h), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 18), (x1 + w + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

def annotate_frame(img, detections, person_boxes=None, track_map=None, violation_tracks=None):
    out = img.copy()
    for d in detections:
        bbox = d["bbox"]
        label = d.get("label", "")
        score = d.get("score", 0)
        txt = f"{label} {score:.2f}"
        draw_bbox(out, bbox, txt, color=(0,200,0))
    if person_boxes:
        for pb in person_boxes:
            draw_bbox(out, pb, label="person", color=(200,200,0))
    if track_map:
        for tid, tb in track_map.items():
            x1,y1,x2,y2 = map(int, tb["bbox"])
            txt = f"ID:{tid}"
            draw_bbox(out, (x1,y1,x2,y2), label=txt, color=(255,100,0))
    if violation_tracks:
        for tid in violation_tracks:
            tb = track_map.get(tid)
            if tb:
                draw_bbox(out, tb["bbox"], label=f"VIOLATION ID:{tid}", color=(0,0,255))
    return out
