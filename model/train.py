"""Fine-tune a DINOv2 backbone for pet-nose re-identification with ArcFace.

This is where accuracy comes from. The stock DINOv2 weights are a weak baseline
for fine-grained nose texture; fine-tuning with an angular-margin loss on real
pet data is what closes the gap to (non-commercial) MegaDescriptor while staying
Apache-2.0.

The saved file is just the backbone state_dict, which embed.py / identify.py load
via --weights. The ArcFace head is training-only and is discarded.

Usage:
    python train.py --data-dir data --model facebook/dinov2-small --epochs 30
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from arcface import ArcFace
from config import DINOV2_DIMS
from dataset import NoseDataset


def embed_batch(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    out = model(pixel_values=pixel_values)
    cls = out.last_hidden_state[:, 0]
    return F.normalize(cls, p=2, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune DINOv2 + ArcFace on pet noses.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="backbone LR (small, it is pretrained)")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="ArcFace head LR")
    ap.add_argument("--out", default="weights/dinov2_nose.pt")
    ap.add_argument("--no-detect", action="store_true", help="skip nose detector during training")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    ds = NoseDataset(args.data_dir, processor, detect=not args.no_detect)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    print(f"[train] {len(ds)} images across {ds.num_classes} pets")

    model = AutoModel.from_pretrained(args.model).to(device).train()
    dim = DINOV2_DIMS.get(args.model)
    if dim is None:
        dim = model.config.hidden_size
    head = ArcFace(dim, ds.num_classes).to(device).train()

    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        total, correct, running = 0, 0, 0.0
        for pixel_values, labels in tqdm(dl, desc=f"epoch {epoch}/{args.epochs}"):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            emb = embed_batch(model, pixel_values)
            logits = head(emb, labels)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"[train] epoch {epoch}: loss={running/total:.4f} train_acc={correct/total:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[train] saved backbone to {args.out}")
    print("[train] use it with: python identify.py <img> --weights " + args.out)


if __name__ == "__main__":
    main()
