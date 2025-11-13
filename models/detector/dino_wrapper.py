"""
models/detector/dino_wrapper.py

Placeholder adapter for DINO (DETR-like) detectors. This file provides the
class definition and a clear place to implement your DINO inference code.

Right now this class raises informative errors reminding you to load DINO weights.
When you fine-tune DINO or acquire pre-trained weights, replace 'NotImplementedError'
with code that loads the model and performs inference returning the same output format
as DetectorWrapper.predict().

The expected output format:
    [
      {"bbox": (x1,y1,x2,y2), "score": float, "label": str, "label_id": int},
      ...
    ]
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class DinoWrapper:
    def __init__(self, weights_path: str = None, device: str = "cpu"):
        """
        Args:
            weights_path: path to DINO checkpoint (optional)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.weights_path = weights_path
        self.model = None
        # user must implement model loading here
        raise NotImplementedError(
            "DinoWrapper is a placeholder. Replace this with code to load your "
            "DINO/DETR model (weights_path) and set self.model."
        )

    def predict(self, image: np.ndarray, conf_th: float = 0.5) -> List[Dict]:
        """
        Perform inference with your DINO model.
        """
        raise NotImplementedError("Implement DINO inference and return list of detections as dicts.")
