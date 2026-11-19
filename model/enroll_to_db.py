"""CLI: register a pet from one or more nose photos.

photo(s) -> detect -> preprocess -> embed -> enroll_pet()

Multiple photos of the same nose are averaged into one enrolment embedding
(then re-normalised), which is more robust than a single frame. The number of
frames is passed through to the database's frame_count column.
"""

from __future__ import annotations

import argparse

import numpy as np

import db
from config import DEFAULT_MODALITY, DEFAULT_MODEL_ID
from embed import NoseEmbedder, to_pgvector


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrol a pet from nose photo(s).")
    ap.add_argument("images", nargs="+", help="one or more photos of the same nose")
    ap.add_argument("--name", required=True)
    ap.add_argument("--species", default="dog")
    ap.add_argument("--breed", default=None)
    ap.add_argument("--modality", default=DEFAULT_MODALITY)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--weights", default=None, help="fine-tuned backbone weights")
    ap.add_argument("--microchip", default=None)
    ap.add_argument("--owner-name", default=None)
    ap.add_argument("--owner-phone", default=None)
    ap.add_argument("--owner-email", default=None)
    ap.add_argument("--region", default="BLR")
    args = ap.parse_args()

    embedder = NoseEmbedder(args.model, args.weights)
    vecs = [embedder.embed_image(p) for p in args.images]
    mean = np.mean(np.stack(vecs), axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)

    conn = db.connect(role="service_role")
    row = db.enroll_pet(
        conn,
        name=args.name,
        species=args.species,
        modality=args.modality,
        embedding=to_pgvector(mean.astype(np.float32)),
        model_id=args.model,
        breed=args.breed,
        microchip_number=args.microchip,
        owner_name=args.owner_name,
        owner_phone=args.owner_phone,
        owner_email=args.owner_email,
        region=args.region,
        frame_count=len(args.images),
    )
    print(f"Enrolled {args.name}")
    print(f"  ella_uid     : {row['ella_uid']}")
    print(f"  pet_id       : {row['pet_id']}")
    print(f"  has_microchip: {row['has_microchip']}")
    print(f"  frames used  : {len(args.images)}")


if __name__ == "__main__":
    main()
