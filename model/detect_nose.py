"""Nose detection and cropping.

A tight crop of the nasal planum is the single biggest accuracy lever: a stray
background or a loose crop poisons the embedding. This wraps an Ultralytics YOLO
detector trained to find the nose.

Scaffold behaviour: if no trained detector weights are present, this falls back
to a centre crop and prints a warning, so the rest of the pipeline still runs
end to end. Replace NOSE_WEIGHTS with a trained YOLO nose model to make it real.
"""

from __future__ import annotations

import os
import sys

import numpy as np

NOSE_WEIGHTS = os.environ.get("ELLA_NOSE_WEIGHTS", "weights/nose_yolo.pt")

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(NOSE_WEIGHTS):
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[detect_nose] ultralytics not installed; using centre crop.", file=sys.stderr)
        return None
    _model = YOLO(NOSE_WEIGHTS)
    return _model


def _centre_crop(bgr: np.ndarray, frac: float = 0.6) -> np.ndarray:
    h, w = bgr.shape[:2]
    side = int(min(h, w) * frac)
    cy, cx = h // 2, w // 2
    half = side // 2
    return bgr[cy - half : cy + half, cx - half : cx + half]


def crop_nose(bgr: np.ndarray, pad: float = 0.1) -> np.ndarray:
    """Return a cropped BGR image around the detected nose.

    Falls back to a centre crop when no detector weights are available.
    """
    model = _load_model()
    if model is None:
        print(
            f"[detect_nose] No detector at {NOSE_WEIGHTS}; falling back to centre crop. "
            f"Train a YOLO nose model for real accuracy.",
            file=sys.stderr,
        )
        return _centre_crop(bgr)

    results = model.predict(bgr, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("[detect_nose] No nose detected; using centre crop.", file=sys.stderr)
        return _centre_crop(bgr)

    # Highest-confidence box.
    best = int(np.argmax(boxes.conf.cpu().numpy()))
    x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[best].astype(int)

    h, w = bgr.shape[:2]
    px = int((x2 - x1) * pad)
    py = int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return bgr[y1:y2, x1:x2]
