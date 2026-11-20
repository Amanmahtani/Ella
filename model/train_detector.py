"""Fine-tune RT-DETR to detect the nose pad (Apache-2.0, via transformers).

Stage A of the funnel. The detector's only job is to draw a tight box around the
bare, textured nose (rhinarium) so the embedding model never sees breed, fur
colour or background. Base model is RT-DETR from Hugging Face (no new deps beyond
transformers, which we already use for DINOv2).

Dataset (standard YOLO export from CVAT / Label Studio / Roboflow):

    data_detector/
      images/train/*.jpg   labels/train/*.txt
      images/val/*.jpg     labels/val/*.txt

Each label line is one box:  `0 cx cy w h`  (class 0 = nose, coords normalised).
Labelling ~300 images is enough to auto-crop the rest of your photos.

Usage:
    python train_detector.py --data-dir data_detector --epochs 50
    # -> saves a checkpoint to weights/nose_rtdetr, which detect_nose.py loads.
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)

CHECKPOINT_DEFAULT = "PekingU/rtdetr_r18vd"  # small/fast; r50vd for more accuracy
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class NoseDetDataset(Dataset):
    """Reads YOLO-format labels and yields processor inputs for RT-DETR."""

    def __init__(self, root: str, split: str, processor):
        self.processor = processor
        self.items: list[tuple[str, str]] = []
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        for f in sorted(os.listdir(img_dir)):
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
            self.items.append((os.path.join(img_dir, f), lbl))
        if not self.items:
            raise ValueError(f"No images under {img_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        anns = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, w, h = (float(p) for p in parts)
                    # YOLO normalised cx,cy,w,h -> COCO absolute x,y,w,h
                    bw, bh = w * W, h * H
                    x, y = cx * W - bw / 2, cy * H - bh / 2
                    anns.append({"image_id": idx, "category_id": 0,
                                 "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})

        enc = self.processor(
            images=img,
            annotations={"image_id": idx, "annotations": anns},
            return_tensors="pt",
        )
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune RT-DETR to detect pet noses.")
    ap.add_argument("--data-dir", default="data_detector")
    ap.add_argument("--base", default=CHECKPOINT_DEFAULT)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="weights/nose_rtdetr")
    args = ap.parse_args()

    processor = AutoImageProcessor.from_pretrained(args.base)
    model = AutoModelForObjectDetection.from_pretrained(
        args.base,
        num_labels=1,
        id2label={0: "nose"},
        label2id={"nose": 0},
        ignore_mismatched_sizes=True,  # swap the COCO head for a 1-class head
    )

    train_ds = NoseDetDataset(args.data_dir, "train", processor)
    has_val = os.path.isdir(os.path.join(args.data_dir, "images", "val"))
    val_ds = NoseDetDataset(args.data_dir, "val", processor) if has_val else None
    print(f"[train_detector] train={len(train_ds)}" + (f" val={len(val_ds)}" if val_ds else ""))

    targs = TrainingArguments(
        output_dir="runs/nose_rtdetr",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=1e-4,
        warmup_steps=100,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate_fn, tokenizer=processor,
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"[train_detector] saved RT-DETR checkpoint to {args.out}")
    print("[train_detector] detect_nose.py will use it automatically.")


if __name__ == "__main__":
    main()
