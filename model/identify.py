"""CLI: identify a found pet from a new nose photo.

photo -> detect -> preprocess -> embed -> public_identify()

Runs as the anon role, exactly as a public web lookup would, so the result is
the PII-free public_pet_summary the database is designed to return. Owner phone,
email, address and the raw microchip number are not just hidden, they are
unrepresentable in the return type.
"""

from __future__ import annotations

import argparse

import db
from config import DEFAULT_MODALITY, DEFAULT_MODEL_ID
from embed import NoseEmbedder, to_pgvector


def main() -> None:
    ap = argparse.ArgumentParser(description="Identify a found pet by its nose.")
    ap.add_argument("image")
    ap.add_argument("--modality", default=DEFAULT_MODALITY)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--weights", default=None, help="fine-tuned backbone weights")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the registry match threshold")
    ap.add_argument("-k", type=int, default=3, help="top-k candidates")
    args = ap.parse_args()

    embedder = NoseEmbedder(args.model, args.weights)
    vec = embedder.embed_image(args.image)

    conn = db.connect(role="anon")
    rows = db.public_identify(
        conn,
        embedding=to_pgvector(vec),
        modality=args.modality,
        model_id=args.model,
        threshold=args.threshold,
        k=args.k,
    )

    if not rows:
        print("No match above threshold. This pet is not enrolled (or needs a clearer photo).")
        return

    print(f"Top {len(rows)} candidate(s):")
    for i, r in enumerate(rows, 1):
        sim = r.get("similarity")
        sim_s = f"{float(sim):.4f}" if sim is not None else "n/a"
        print(f"  {i}. {r.get('name')}  [{r.get('ella_uid')}]  "
              f"{r.get('species')}  similarity={sim_s}  "
              f"has_microchip={r.get('has_microchip')}")


if __name__ == "__main__":
    main()
