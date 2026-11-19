"""Train the YOLO nose detector.

detect_nose.py already does inference; this produces the weights it loads. The
detector finds and tightly crops the nasal planum before embedding, which is the
single biggest accuracy lever: a loose or off-centre crop poisons the embedding.

Dataset format (standard Ultralytics YOLO layout), described in
detector_data.yaml:

    data_detector/
      images/train/*.jpg   labels/train/*.txt
      images/val/*.jpg     labels/val/*.txt

Each label file has one line per box: `0 cx cy w h` (class 0 = nose, coords
normalised 0..1). Label a few hundred pet photos (Roboflow, CVAT or Label
Studio all export this format). Even 200-300 boxed noses train a usable
detector, since it only has to find one obvious region.

Usage:
    python train_detector.py --data detector_data.yaml --epochs 100
    # -> writes weights/nose_yolo.pt, which detect_nose.py picks up automatically
"""

from __future__ import annotations

import argparse
import os
import shutil

from ultralytics import YOLO

from detect_nose import NOSE_WEIGHTS


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a YOLO nose detector.")
    ap.add_argument("--data", default="detector_data.yaml", help="dataset YAML")
    ap.add_argument("--base", default="yolov8n.pt",
                    help="base model to fine-tune (n=nano is plenty for one class)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=NOSE_WEIGHTS, help="where to copy the best weights")
    ap.add_argument("--project", default="runs/nose_detector")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"Dataset YAML not found: {args.data}. See detector_data.yaml and the "
            f"README for the expected images/labels layout."
        )

    model = YOLO(args.base)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name="train",
        exist_ok=True,
        single_cls=True,  # only one class: nose
    )

    best = os.path.join(results.save_dir, "weights", "best.pt")
    if not os.path.exists(best):
        raise SystemExit(f"Training finished but best weights not found at {best}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    shutil.copy(best, args.out)
    print(f"[train_detector] copied best weights to {args.out}")
    print("[train_detector] detect_nose.py will now use the trained detector.")


if __name__ == "__main__":
    main()
