"""
models/detector/detr_wrapper.py

A simple DetectorWrapper with a practical default implementation using torchvision's
Faster R-CNN (if torchvision is available). The wrapper provides a consistent interface
so you can later replace the internals with a true DETR/DINO/RT-DETR inference implementation.

API:
    det = DetectorWrapper(device='cpu', model_name='fasterrcnn')
    detections = det.predict(image, conf_th=0.5)
    # detections: list of dicts with keys: bbox (x1,y1,x2,y2), score, label, label_id
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

# Try to import torch + torchvision; else provide a clear fallback
try:
    import torch
    import torchvision
    from torchvision.transforms import functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Minimal COCO class mapping for common classes we care about
COCO_LABELS = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    # ... (full COCO list can be added if needed)
}


class DetectorWrapper:
    """
    Detector wrapper with a simple default using torchvision's Faster R-CNN.
    Replace internals with DETR/DINO inference as needed.
    """

    def __init__(self, device: str = "cpu", model_name: str = "fasterrcnn"):
        """
        Args:
            device: "cpu" or "cuda"
            model_name: 'fasterrcnn' (default) or 'detr' (not implemented here)
        """
        self.device = device
        self.model_name = model_name.lower()
        self.model = None
        if TORCH_AVAILABLE and self.model_name == "fasterrcnn":
            self._load_fasterrcnn()
        else:
            # If torch not available, we still allow the wrapper to exist;
            # predict() will return an empty list or a deterministic mock result.
            self.model = None

    def _load_fasterrcnn(self):
        """Load a torchvision pretrained Faster R-CNN model (COCO pretrained)."""
        # Use pretrained weights - good for initial prototyping
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(torch.device(self.device))

    def predict(self, image: np.ndarray, conf_th: float = 0.5) -> List[Dict]:
        """
        Run detection on a single image.

        Args:
            image: numpy array HxWxC (BGR or RGB — we convert to RGB)
            conf_th: confidence threshold for returning detections

        Returns:
            List of detections: each is {'bbox': (x1,y1,x2,y2), 'score': float, 'label': str, 'label_id': int}
        """
        if self.model is None:
            # Fallback deterministic (no torch): return empty list
            return []

        if not TORCH_AVAILABLE:
            return []

        # Convert numpy -> tensor
        # Accept BGR (OpenCV) or RGB: assume BGR if dtype==uint8 and color channels typical for cv2 usage.
        img = image
        if img.ndim != 3:
            raise ValueError("Input image must be HxWxC numpy array")

        # Convert BGR to RGB if necessary (common CV pipeline)
        # Heuristic: if color is uint8 and mean of first channel differs from last
        if img.shape[2] == 3:
            # naive check: if mean of channel 0 > mean of channel 2 by > 1, assume BGR
            if float(img[:, :, 0].mean()) - float(img[:, :, 2].mean()) > 1.0:
                img_rgb = img[:, :, ::-1].copy()
            else:
                img_rgb = img.copy()
        else:
            img_rgb = img.copy()

        import torch
        tensor = F.to_tensor(img_rgb).to(torch.device(self.device))
        with torch.no_grad():
            outputs = self.model([tensor])
        outputs = outputs[0]

        boxes = outputs["boxes"].cpu().numpy()  # Nx4
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        result = []
        for bbox, score, label_id in zip(boxes, scores, labels):
            if score < conf_th:
                continue
            x1, y1, x2, y2 = [float(x) for x in bbox]
            label_name = COCO_LABELS.get(int(label_id), f"cls_{int(label_id)}")
            result.append({
                "bbox": (x1, y1, x2, y2),
                "score": float(score),
                "label": label_name,
                "label_id": int(label_id)
            })
        return result

    def warmup(self):
        """Optional warmup call to load model onto device; useful for first-call latency mitigation."""
        if TORCH_AVAILABLE and self.model is not None:
            import torch
            dummy = torch.zeros((3, 224, 224)).to(torch.device(self.device))
            with torch.no_grad():
                _ = self.model([dummy])


# If you plan to plug in a DETR/DINO wrapper later, create a small adapter class matching this interface.
# See dino_wrapper.py and rtdetr_wrapper.py for placeholders.
