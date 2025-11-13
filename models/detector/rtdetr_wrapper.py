"""
models/detector/rtdetr_wrapper.py

Placeholder for real-time DETR (RT-DETR) adapter. Implement this if you decide to
use RT-DETR for low-latency inference. For now this is a placeholder skeleton that
can be filled in with the RT-DETR model loading and predict logic.
"""

from typing import List, Dict
import numpy as np


class RTDETRWrapper:
    def __init__(self, weights: str = None, device: str = "cpu"):
        self.device = device
        self.weights = weights
        self.model = None
        raise NotImplementedError("RTDETRWrapper is a placeholder. Implement loading/inference here.")

    def predict(self, image: np.ndarray, conf_th: float = 0.5) -> List[Dict]:
        raise NotImplementedError("Implement RT-DETR inference and return detections.")
