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


def detect_box(bgr: np.ndarray, conf: float = 0.25) -> tuple[int, int, int, int] | None:
    """Return the highest-confidence nose box (x1,y1,x2,y2), or None.

    None means either no detector weights are present or nothing was detected;
    the caller decides how to fall back.
    """
    model = _load_model()
    if model is None:
        return None
    results = model.predict(bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = int(np.argmax(boxes.conf.cpu().numpy()))
    x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[best].astype(int)
    return int(x1), int(y1), int(x2), int(y2)


def crop_nose(bgr: np.ndarray, pad: float = 0.1, conf: float = 0.25) -> np.ndarray:
    """Return a cropped BGR image around the detected nose.

    Falls back to a centre crop when no detector weights are available or no
    nose is found, so the rest of the pipeline always gets an image.
    """
    box = detect_box(bgr, conf=conf)
    if box is None:
        if _load_model() is None:
            print(
                f"[detect_nose] No detector at {NOSE_WEIGHTS}; falling back to centre crop. "
                f"Train one with train_detector.py for real accuracy.",
                file=sys.stderr,
            )
        else:
            print("[detect_nose] No nose detected; using centre crop.", file=sys.stderr)
        return _centre_crop(bgr)

    x1, y1, x2, y2 = box
    h, w = bgr.shape[:2]
    px = int((x2 - x1) * pad)
    py = int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return bgr[y1:y2, x1:x2]


def _main() -> None:
    import argparse

    from preprocess import load_bgr

    ap = argparse.ArgumentParser(description="Test the nose detector on an image.")
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out-crop", default="nose_crop.jpg")
    ap.add_argument("--out-annotated", default="nose_annotated.jpg")
    args = ap.parse_args()

    bgr = load_bgr(args.image)
    box = detect_box(bgr, conf=args.conf)
    if box is None:
        print("No detection (no weights, or nothing found). "
              "Centre-crop fallback would be used by the pipeline.")
        return

    import cv2

    x1, y1, x2, y2 = box
    cv2.imwrite(args.out_crop, bgr[y1:y2, x1:x2])
    annotated = bgr.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(args.out_annotated, annotated)
    print(f"Nose box: ({x1},{y1})-({x2},{y2})")
    print(f"  crop      -> {args.out_crop}")
    print(f"  annotated -> {args.out_annotated}")


if __name__ == "__main__":
    _main()
