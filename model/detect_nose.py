"""Nose detection and cropping with RT-DETR (via Hugging Face transformers).

This replaces the previous AGPL-licensed YOLO detector. RT-DETR is Apache-2.0 and
lives in `transformers`, so the whole ML stack stays in one library
(transformers + PyTorch) with a clean commercial licence.

Stage A of the two-stage funnel: find the bounding box of the bare nose pad
(rhinarium) and crop out everything else. A tight crop is what forces the
embedding model to look at ridge texture instead of breed, fur colour or
background. Out of the box RT-DETR only knows COCO classes, not noses, so this
loads a FINE-TUNED checkpoint (produced by train_detector.py) when present and
otherwise falls back to a centre crop so the pipeline still runs end to end.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# A fine-tuned RT-DETR checkpoint saved with save_pretrained() (a directory).
NOSE_WEIGHTS = os.environ.get("ELLA_NOSE_WEIGHTS", "weights/nose_rtdetr")

_model = None
_proc = None


def _load_model():
    global _model, _proc
    if _model is not None:
        return _model, _proc
    if not os.path.isdir(NOSE_WEIGHTS):
        return None, None
    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except ImportError:
        print("[detect_nose] transformers not installed; using centre crop.", file=sys.stderr)
        return None, None
    _proc = AutoImageProcessor.from_pretrained(NOSE_WEIGHTS)
    _model = AutoModelForObjectDetection.from_pretrained(NOSE_WEIGHTS).eval()
    return _model, _proc


def _centre_crop(bgr: np.ndarray, frac: float = 0.6) -> np.ndarray:
    h, w = bgr.shape[:2]
    side = int(min(h, w) * frac)
    cy, cx = h // 2, w // 2
    half = side // 2
    return bgr[cy - half : cy + half, cx - half : cx + half]


def detect_box(bgr: np.ndarray, conf: float = 0.5) -> tuple[int, int, int, int] | None:
    """Return the highest-confidence nose box (x1,y1,x2,y2), or None.

    None means no detector weights are present or nothing was found; the caller
    decides how to fall back.
    """
    model, proc = _load_model()
    if model is None:
        return None
    import cv2
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    inputs = proc(images=rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    res = proc.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=conf
    )[0]
    if len(res["scores"]) == 0:
        return None
    i = int(res["scores"].argmax())
    x1, y1, x2, y2 = (int(v) for v in res["boxes"][i].tolist())
    return x1, y1, x2, y2


def crop_nose(bgr: np.ndarray, pad: float = 0.1, conf: float = 0.5) -> np.ndarray:
    """Return a cropped BGR image around the detected nose.

    Falls back to a centre crop when no detector weights are available or no
    nose is found, so the rest of the pipeline always gets an image.
    """
    box = detect_box(bgr, conf=conf)
    if box is None:
        if _load_model()[0] is None:
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

    import cv2

    from preprocess import load_bgr

    ap = argparse.ArgumentParser(description="Test the RT-DETR nose detector on an image.")
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out-crop", default="nose_crop.jpg")
    ap.add_argument("--out-annotated", default="nose_annotated.jpg")
    args = ap.parse_args()

    bgr = load_bgr(args.image)
    box = detect_box(bgr, conf=args.conf)
    if box is None:
        print("No detection (no weights, or nothing found). "
              "The pipeline would use the centre-crop fallback.")
        return
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
