# Ella — model/

The machine-learning side of Ella: turn a photo of a pet's nose into the
embedding vector that the database (`001_ella_init.sql` .. `003_ella_security.sql`)
stores and matches.

This is a **scaffold**. Every file runs, but accuracy only arrives after you
fine-tune on real pet-nose photos (see `train.py`). Out of the box it uses the
stock DINOv2 weights, which are commercially clean (Apache-2.0) but not yet
tuned for noses.

## How it maps to the database

The SQL side defines an `embedding_spaces` registry. This package mirrors it in
`config.py`, so the Python and the database always agree on which model produces
which dimension for which modality.

| modality       | model_id (default active) | dim  | db column        |
|----------------|---------------------------|------|------------------|
| `nasal_planum` | `facebook/dinov2-small`   | 384  | `embedding_384`  |
| `nasal_planum` | DINOv2-giant (launch)     | 1536 | `embedding_1536` |

A fine-tuned DINOv2-giant is 1536-d, the same width as the demo MegaDescriptor
model, so it drops into the existing `embedding_1536` column with no schema
change. That is the intended commercial launch model.

## The pipeline

```
photo ──▶ detect_nose.py ──▶ preprocess.py ──▶ embed.py ──▶ [0.12,-0.44,...]::vector
         (YOLO crop)         (LAB + CLAHE)      (DINOv2)              │
                                                                     ▼
                                          enroll_to_db.py  →  enroll_pet()   (register)
                                          identify.py      →  public_identify()  (find)
```

## Quick start

```bash
cd model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Point at your database (defaults to the standard PG* env vars)
export DATABASE_URL="postgresql://service_role@/ella?host=/var/lib/postgresql/ella-sock&port=5433"

# Register a pet from a photo (calls enroll_pet)
python enroll_to_db.py path/to/ella_nose.jpg \
    --name Ella --species dog --breed Indie --region BLR

# Identify a found pet from a new photo (calls public_identify)
python identify.py path/to/found_dog.jpg
```

## Training (where accuracy comes from)

```bash
# data/ is a folder per pet identity, each holding several nose photos:
#   data/pet_0001/*.jpg
#   data/pet_0002/*.jpg
python train.py --data-dir data --model facebook/dinov2-small --epochs 30
```

Datasets to start from are listed in the project discussion: the CVPR 2022 Pet
Biometric Challenge data is the main public benchmark. Your own collected
multi-frame enrolments matter more than any model swap.

## What is real vs stub in this scaffold

- `embed.py`, `preprocess.py`, `db.py`, `enroll_to_db.py`, `identify.py`,
  `dataset.py`, `arcface.py`, `train.py`: real, runnable.
- `detect_nose.py`: real YOLO wrapper, but with no trained nose-detector weights
  it falls back to a centre crop and warns. Train or drop in a YOLO nose model
  to make it real.
