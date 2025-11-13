"""
utils/logger.py
Centralized logging helper used across the project.
Provides a simple get_logger(name) that configures console logging only.
"""

import logging
import sys
from logging import Logger
from typing import Optional


def get_logger(name: str, level: int = logging.INFO, to_file: Optional[str] = None) -> Logger:
    """
    Returns a configured logger.

    Args:
        name: logger name (e.g., "triple_riding_detector.pipeline")
        level: logging level
        to_file: optional file path to write logs. If None, logs only to console.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid adding multiple handlers if called multiple times
    if not logger.handlers:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        formatter = logging.Formatter(fmt)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional file handler
        if to_file:
            fh = logging.FileHandler(to_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # Avoid propagate to root logger (prevent duplicate logs)
        logger.propagate = False

    return logger
