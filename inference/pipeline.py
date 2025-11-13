# inference/pipeline.py
import cv2
import time
from typing import Optional, Generator, Tuple, List, Dict
import numpy as np
import os
from utils.logger import get_logger
from utils.fileops import ensure_dir, make_unique_filename, save_image_pil, safe_join
from db.session import SessionLocal
from db.models import Violation, RegisteredVehicle
from models.detector.detr_wrapper import DetectorWrapper
from models.rider_counter.person_association import count_riders
from models.plate_detector.plate_detector import PlateDetector
from inference.tracker import SimpleTracker
from inference.temporal_validator import TemporalValidator
from inference.ocr import OCRReader
from inference.privacy import blur_faces_and_plates
from inference.annotate import annotate_frame

# optional SMS adapter
try:
    from sms.twilio_adapter import send_sms_twilio
except Exception:
    send_sms_twilio = None

logger = get_logger("inference.pipeline")

EVIDENCE_DIR = "evidence_store/raw"
BLURRED_DIR = "evidence_store/blurred"
THUMB_DIR = "evidence_store/thumbnails"
ensure_dir(EVIDENCE_DIR)
ensure_dir(BLURRED_DIR)
ensure_dir(THUMB_DIR)


def notify_on_violation(violation_id: int) -> bool:
    """
    Lookup violation by id, pick recipient (owner if registered else admin),
    craft message and send via Twilio adapter (if available). Returns True on send success.
    This function creates its own DB session.
    """
    try:
        session = SessionLocal()
        v = session.query(Violation).filter(Violation.id == int(violation_id)).first()
        if not v:
            logger.warning("notify_on_violation: violation id not found: %s", violation_id)
            session.close()
            return False

        # Try to resolve owner phone by plate
        plate = getattr(v, "vehicle_plate", None) or getattr(v, "plate_text", None)
        phone = None
        if plate:
            reg = session.query(RegisteredVehicle).filter(RegisteredVehicle.plate_text == plate).first()
            if reg and getattr(reg, "phone_number", None):
                phone = reg.phone_number

        admin_num = os.getenv("NOTIFY_ADMIN_NUMBER")
        recipient = phone or admin_num
        if not recipient:
            logger.warning("No recipient configured for violation id=%s plate=%s", v.id, plate)
            session.close()
            return False

        # build evidence link or path
        evidence = getattr(v, "evidence_path_raw", None) or getattr(v, "evidence_path", None) or "no-evidence"
        base = os.getenv("PUBLIC_BASE_URL")
        if base and evidence and os.path.exists(evidence):
            rel = os.path.relpath(evidence, start=os.getcwd()).replace("\\", "/")
            evidence_link = f"{base}/download?file={rel}"
        else:
            evidence_link = evidence

        riders = getattr(v, "num_riders", getattr(v, "riders", "N/A"))
        timestamp = getattr(v, "timestamp", getattr(v, "created_at", None))

        msg = (
            f"ALERT: Triple riding detected at {timestamp}. "
            f"Riders: {riders}. "
            f"Plate: {plate or 'unknown'}. Evidence: {evidence_link}"
        )

        if send_sms_twilio is None:
            logger.info("SMS adapter not available; mock-send: to=%s msg=%s", recipient, msg)
            session.close()
            return True

        ok = send_sms_twilio(recipient, msg, mark_violation_id=v.id)
        session.close()
        return bool(ok)
    except Exception as e:
        logger.exception("notify_on_violation failed: %s", e)
        return False


def mark_unidentified_and_queue(violation_id: int) -> bool:
    """
    If a violation has no plate, mark it for manual review.
    Uses field `needs_review` if present on model, otherwise appends to `notes` if present.
    Also triggers admin notification (so humans get alerted).
    """
    try:
        session = SessionLocal()
        v = session.query(Violation).filter(Violation.id == int(violation_id)).first()
        if not v:
            logger.warning("mark_unidentified_and_queue: violation id not found: %s", violation_id)
            session.close()
            return False

        plate = getattr(v, "vehicle_plate", None) or getattr(v, "plate_text", None)
        if plate:
            session.close()
            return False  # plate exists -> no need to queue

        changed = False
        if hasattr(v, "needs_review"):
            v.needs_review = True
            changed = True
        elif hasattr(v, "notes"):
            v.notes = (v.notes or "") + " | needs_review:plate_unknown"
            changed = True

        if changed:
            session.add(v)
            session.commit()

        # notify admin so a human can review
        try:
            notify_on_violation(v.id)
        except Exception:
            logger.exception("Failed to notify admin for review of violation %s", v.id)

        session.close()
        return True
    except Exception as e:
        logger.exception("mark_unidentified_and_queue failed: %s", e)
        return False


class InferencePipeline:
    def __init__(
        self,
        detector_device: str = "cpu",
        detector_name: str = "fasterrcnn",
        min_frames_for_violation: int = 5,
        rider_threshold: int = 3,
    ):
        self.detector = DetectorWrapper(device=detector_device, model_name=detector_name)
        self.plate_detector = PlateDetector()
        self.ocr = OCRReader()
        self.tracker = SimpleTracker(max_missed=15, iou_threshold=0.3)
        self.validator = TemporalValidator(min_consecutive=min_frames_for_violation, cooldown_seconds=5)
        self.rider_threshold = rider_threshold
        self.session_maker = SessionLocal

    def _detect(self, frame: np.ndarray):
        dets = self.detector.predict(frame, conf_th=0.4)
        return dets

    def _get_person_boxes(self, detections: List[Dict]) -> List[Tuple[float, float, float, float]]:
        persons = []
        for d in detections:
            if d.get("label") == "person":
                persons.append(tuple(d["bbox"]))
        return persons

    def _save_evidence(self, frame, track_id, plate_text=None):
        fname_raw = make_unique_filename(prefix=f"evidence_tid{track_id}")
        p_raw = safe_join(EVIDENCE_DIR, fname_raw)
        save_image_pil(frame, p_raw)
        blurred = blur_faces_and_plates(frame, face_bboxes=None, plate_bboxes=None)
        fname_blur = fname_raw.replace(".jpg", "_blurred.jpg")
        p_blur = safe_join(BLURRED_DIR, fname_blur)
        save_image_pil(blurred, p_blur)
        return p_raw, p_blur

    def process_video(self, video_source: str) -> Generator[bytes, None, None]:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error("Unable to open video source: %s", video_source)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            detections = self._detect(frame)
            person_boxes = self._get_person_boxes(detections)

            # Identify vehicle detections: include common motorcycle labels or label_id fallback
            vehicles = [d for d in detections if (d.get("label") in ("motorcycle", "motorbike") or d.get("label_id") == 4)]

            # update tracker with vehicle detections
            tracks = self.tracker.update(vehicles)

            # build a quick track_map for annotator
            track_map = {t.id: {"bbox": t.bbox, "score": t.score} for t in tracks}
            violation_ids = []

            for t in tracks:
                # count riders by person association
                riders = count_riders(person_boxes, t.bbox, iou_thresh=0.15)
                is_violation_frame = riders >= self.rider_threshold
                violated_now = self.validator.update(t.id, is_violation_frame)
                if violated_now:
                    # detect plate near vehicle bbox
                    plate_box = self.plate_detector.detect_plate(frame, t.bbox)
                    plate_text, ocr_conf = None, None
                    if plate_box is not None:
                        x1, y1, x2, y2 = map(int, plate_box)
                        # guard for crop boundaries
                        h, w = frame.shape[:2]
                        x1 = max(0, min(w - 1, x1))
                        x2 = max(0, min(w, x2))
                        y1 = max(0, min(h - 1, y1))
                        y2 = max(0, min(h, y2))
                        if x2 > x1 and y2 > y1:
                            plate_crop = frame[y1:y2, x1:x2]
                            plate_text, ocr_conf = self.ocr.read_plate(plate_crop)

                    raw_path, blur_path = self._save_evidence(frame, t.id, plate_text)

                    # write to DB (create violation record)
                    session = self.session_maker()
                    try:
                        violation = Violation(
                            vehicle_plate=plate_text,
                            track_id=str(t.id),
                            video_source=video_source,
                            num_riders=riders,
                            ocr_confidence=str(ocr_conf) if ocr_conf is not None else None,
                            evidence_path_raw=raw_path,
                            evidence_path_blurred=blur_path,
                        )
                        session.add(violation)
                        session.commit()

                        # after commit we have violation.id assigned; trigger notifications & review queue
                        try:
                            notify_on_violation(violation.id)
                        except Exception:
                            logger.exception("notify_on_violation failed for id=%s", getattr(violation, "id", None))

                        try:
                            mark_unidentified_and_queue(violation.id)
                        except Exception:
                            logger.exception("mark_unidentified_and_queue failed for id=%s", getattr(violation, "id", None))

                    except Exception as e:
                        logger.exception("Failed to write violation to DB: %s", e)
                        try:
                            session.rollback()
                        except Exception:
                            pass
                    finally:
                        session.close()

                    violation_ids.append(t.id)

            annotated = annotate_frame(frame, detections, person_boxes=person_boxes, track_map=track_map, violation_tracks=violation_ids)
            _, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield jpg.tobytes()

        cap.release()
