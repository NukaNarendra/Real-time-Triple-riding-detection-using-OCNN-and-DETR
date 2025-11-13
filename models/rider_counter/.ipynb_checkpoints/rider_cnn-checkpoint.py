"""
A compact PyTorch model for predicting rider counts from a cropped motorcycle image.
The model outputs logits for 3 classes:
    0 -> 1 rider
    1 -> 2 riders
    2 -> 3+ riders

This module includes:
- RiderCNN: a small CNN architecture (input-size agnostic via Global Avg Pool)
- RiderCounter: a wrapper with load/predict utilities
"""

from typing import Tuple
import numpy as np

# Try import torch; if not available, we provide an informative fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False


if TORCH_OK:
    class RiderCNN(nn.Module):
        """Tiny CNN for rider-count classification (1,2,3+)."""

        def __init__(self, num_classes: int = 3, dropout: float = 0.25, dropout2d: float = 0.10):
            """
            Args:
                num_classes: number of logits to output (default 3)
                dropout: dropout probability used in the MLP head
                dropout2d: spatial dropout after conv blocks
            """
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(2, 2)
            self.do2d1 = nn.Dropout2d(dropout2d)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.do2d2 = nn.Dropout2d(dropout2d)

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.do2d3 = nn.Dropout2d(dropout2d)

            # Resolution invariance: adaptive global average pool -> (B, C, 1, 1)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            # After GAP, feature dim is just the channel count (64 here)
            self.fc1 = nn.Linear(64, 128)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            # works for any HxW (e.g., 64, 96, 128…)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.do2d1(x)
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.do2d2(x)
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.do2d3(x)

            x = self.gap(x)            # (B, 64, 1, 1)
            x = torch.flatten(x, 1)    # (B, 64)
            x = F.relu(self.fc1(x))    # (B, 128)
            x = self.dropout(x)
            x = self.fc2(x)            # (B, num_classes)
            return x


    class RiderCounter:
        """
        Wrapper for RiderCNN with simple predict API.

        Usage:
            rc = RiderCounter(device='cpu', checkpoint='checkpoints/best.pth')
            count = rc.predict_count(image_crop)
        """

        def __init__(self, checkpoint: str = None, device: str = "cpu",
                     dropout: float = 0.25, dropout2d: float = 0.10):
            self.device = device
            self.num_classes = 3
            self.model = RiderCNN(num_classes=self.num_classes,
                                  dropout=dropout,
                                  dropout2d=dropout2d).to(self.device)
            if checkpoint:
                self.load_checkpoint(checkpoint)
            self.model.eval()

        def load_checkpoint(self, path: str):
            import torch
            state = torch.load(path, map_location=self.device)
            # Accept both pure state_dict and full checkpoint dicts
            if isinstance(state, dict) and "model" in state:
                self.model.load_state_dict(state["model"])
            else:
                self.model.load_state_dict(state)
            return True

        def preprocess(self, image_crop: np.ndarray, target_size: Tuple[int, int] = (128, 128)):
            """
            Convert numpy image (H,W,3, BGR or RGB) into torch tensor (1,3,H,W) normalized.
            """
            import torch
            import cv2

            img = image_crop
            # If BGR (common from cv2), flip channels -> RGB
            if img.shape[2] == 3:
                img = img[:, :, ::-1].copy()

            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(img_resized.astype("float32") / 255.0)  # H,W,3 in [0,1]
            tensor = tensor.permute(2, 0, 1)  # 3,H,W

            # Same normalization as training (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std

            return tensor.unsqueeze(0).to(self.device)

        def predict_count(self, image_crop: np.ndarray) -> int:
            """
            Predict rider count from crop.

            Returns:
                int in {1,2,3} where 3 means "3 or more"
            """
            import torch
            x = self.preprocess(image_crop)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                cls = int(torch.argmax(probs, dim=1).item())
            return {0: 1, 1: 2, 2: 3}[cls]

else:
    # Fallback non-torch deterministic stub
    class RiderCounter:
        def __init__(self, checkpoint: str = None, device: str = "cpu"):
            self.device = device

        def predict_count(self, image_crop):
            import numpy as _np
            h = _np.asarray(image_crop).shape[0]
            if h < 60:
                return 1
            elif h < 120:
                return 2
            else:
                return 3
