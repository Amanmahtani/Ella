"""Folder-per-identity dataset for fine-tuning.

Expected layout:

    data/
      pet_0001/  img1.jpg img2.jpg ...
      pet_0002/  img1.jpg ...

Each subfolder is one pet (one identity / class label). Several photos per pet
is what lets metric learning work: the loss pulls same-pet embeddings together
and pushes different pets apart.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from detect_nose import crop_nose
from preprocess import load_bgr, to_model_input

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class NoseDataset(Dataset):
    def __init__(self, data_dir: str, processor, detect: bool = True):
        self.processor = processor
        self.detect = detect
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []

        for cls in sorted(os.listdir(data_dir)):
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            imgs = [
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in IMG_EXT
            ]
            if not imgs:
                continue
            label = len(self.classes)
            self.classes.append(cls)
            for p in imgs:
                self.samples.append((p, label))

        if not self.samples:
            raise ValueError(f"No images found under {data_dir}")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        bgr = load_bgr(path)
        if self.detect:
            bgr = crop_nose(bgr)
        rgb = to_model_input(bgr)
        pixel_values = self.processor(images=rgb, return_tensors="pt")["pixel_values"][0]
        return pixel_values, torch.tensor(label, dtype=torch.long)
