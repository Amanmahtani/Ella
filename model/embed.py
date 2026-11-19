"""DINOv2 embedder: image -> L2-normalised embedding vector.

The embedding is L2-normalised so that cosine similarity (what the database
matches on) behaves correctly. The output dimension equals the backbone's native
dimension (see DINOV2_DIMS), which is why it lines up with the typed vector
columns in the schema (small -> 384, giant -> 1536).

Optionally loads fine-tuned backbone weights produced by train.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from config import DEFAULT_MODALITY, DEFAULT_MODEL_ID, DINOV2_DIMS, space_for
from detect_nose import crop_nose
from preprocess import load_bgr, to_model_input


class NoseEmbedder:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        finetuned_weights: str | None = None,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if finetuned_weights and os.path.exists(finetuned_weights):
            state = torch.load(finetuned_weights, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            print(f"[embed] loaded fine-tuned weights from {finetuned_weights}", file=sys.stderr)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed_rgb(self, rgb: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        # CLS token is the standard DINOv2 image embedding.
        cls = out.last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        return cls.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_image(self, path: str) -> np.ndarray:
        """Full pipeline: read -> detect nose -> preprocess -> embed."""
        bgr = load_bgr(path)
        nose = crop_nose(bgr)
        rgb = to_model_input(nose)
        return self.embed_rgb(rgb)


def to_pgvector(vec: np.ndarray) -> str:
    """Format as a pgvector literal: '[0.1,-0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed a pet-nose image.")
    ap.add_argument("image")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--modality", default=DEFAULT_MODALITY)
    ap.add_argument("--weights", default=None, help="fine-tuned backbone weights")
    args = ap.parse_args()

    space = space_for(args.modality, args.model)
    emb = NoseEmbedder(args.model, args.weights)
    vec = emb.embed_image(args.image)

    expected = DINOV2_DIMS.get(args.model, space.dim)
    if vec.shape[0] != expected:
        print(
            f"[embed] WARNING: dim {vec.shape[0]} != expected {expected} for {args.model}",
            file=sys.stderr,
        )
    print(to_pgvector(vec))


if __name__ == "__main__":
    main()
