"""Nose-crop preprocessing: LAB + CLAHE to normalise lighting and bring out the
ridge texture, then a square resize for the backbone.

CLAHE (Contrast Limited Adaptive Histogram Equalisation) applied to the L
channel of LAB space is a standard, cheap trick for biometric ridge/texture
tasks. It makes the nose pattern far more consistent across phones and lighting,
which is one of the biggest accuracy levers in practice.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import INPUT_SIZE


def enhance(bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB. Input and output are BGR uint8."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def to_model_input(bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Enhance and resize to a square RGB uint8 array ready for the embedder."""
    enhanced = enhance(bgr)
    resized = cv2.resize(enhanced, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
