#!/usr/bin/env bash
# scripts/start_training.sh
# Simple wrapper to launch detector and rider counter training sequentially.
# Usage: ./scripts/start_training.sh /path/to/data/motorcycle_rider_dataset /path/to/rider_counter_data

set -euo pipefail

DATASET_DIR="${1:-data/motorcycle_rider_dataset}"
RIDER_TRAIN_DIR="${2:-data/rider_counter/train}"
RIDER_VAL_DIR="${3:-data/rider_counter/val}"
DETECTOR_CHECKPOINT_DIR="${4:-checkpoints}"
RIDER_CHECKPOINT_DIR="${5:-checkpoints}"

mkdir -p "${DETECTOR_CHECKPOINT_DIR}"
mkdir -p "${RIDER_CHECKPOINT_DIR}"

echo "Starting detector training..."
python training/train_detector.py --dataset "${DATASET_DIR}" --epochs 10 --batch 4 --out "${DETECTOR_CHECKPOINT_DIR}/detector_best.pth"

echo "Detector training complete."

echo "Starting rider counter training..."
python training/train_rider_counter.py --train_dir "${RIDER_TRAIN_DIR}" --val_dir "${RIDER_VAL_DIR}" --epochs 20 --batch 32 --out "${RIDER_CHECKPOINT_DIR}/rider_counter.pth"

echo "Rider counter training complete."

echo "All training finished."
