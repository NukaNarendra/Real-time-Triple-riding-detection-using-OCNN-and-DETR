#!/usr/bin/env python3
"""
scripts/export_model.py

Export detector and rider counter to TorchScript / ONNX where possible.

Usage examples:
    python scripts/export_model.py --detector-checkpoint checkpoints/detector_best.pth --rider-checkpoint checkpoints/rider_counter.pth --out-dir exported
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from utils.logger import get_logger
from models.detector.detr_wrapper import DetectorWrapper
from models.rider_counter.rider_cnn import RiderCNN

logger = get_logger("scripts.export_model")

def export_detector_to_torchscript(device: str, out_path: str):
    logger.info("Attempting to export detector (torchvision Faster R-CNN) to TorchScript...")
    try:
        wrapper = DetectorWrapper(device=device, model_name="fasterrcnn")
        model = wrapper.model
        model.eval()
        example = torch.zeros((3, 800, 800), dtype=torch.float32).unsqueeze(0).to(device)
        try:
            scripted = torch.jit.trace(model, [example])
            scripted.save(out_path)
            logger.info("Detector exported to TorchScript at %s", out_path)
            return True
        except Exception as e:
            logger.exception("TorchScript trace failed: %s", e)
            return False
    except Exception as e:
        logger.exception("Detector export failed: %s", e)
        return False

def export_rider_counter_to_onnx(checkpoint: str, out_path: str, device: str = "cpu"):
    logger.info("Exporting RiderCounter to ONNX...")
    try:
        model = RiderCNN(num_classes=3)
        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state)
        model.eval()
        dummy = torch.randn(1, 3, 64, 64, dtype=torch.float32)
        try:
            torch.onnx.export(model, dummy, out_path, opset_version=11, input_names=["input"], output_names=["output"], dynamic_axes=None)
            logger.info("Rider counter exported to ONNX at %s", out_path)
            return True
        except Exception as e:
            logger.exception("ONNX export failed: %s", e)
            return False
    except Exception as e:
        logger.exception("RiderCounter export failed: %s", e)
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-checkpoint", default=None)
    parser.add_argument("--rider-checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="exported")
    args = parser.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    det_out = outdir / "detector_ts.pt"
    rid_out = outdir / "rider_counter.onnx"
    ok1 = export_detector_to_torchscript(args.device, str(det_out))
    ok2 = export_rider_counter_to_onnx(args.rider_checkpoint, str(rid_out), device=args.device)
    if ok1 and ok2:
        logger.info("Exported models successfully to %s", outdir)
    else:
        logger.warning("Some exports failed. See logs for details.")

if __name__ == "__main__":
    main()
