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
        (RT-DETR crop)       (LAB + CLAHE)      (DINOv2)              │
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

## Nose detector (RT-DETR, Apache-2.0)

Stage A of the funnel: find and tightly crop the bare nose pad before embedding,
so the model can't cheat on breed/fur/background. The detector is RT-DETR via
`transformers` (Apache-2.0, no extra dependency, no AGPL). Until trained weights
exist, `detect_nose.py` falls back to a centre crop and warns.

Train it on boxed noses (one class), then it is picked up automatically:

```bash
# 1. Lay out labelled data (standard YOLO export from CVAT / Label Studio / Roboflow):
#      data_detector/images/{train,val}/*.jpg
#      data_detector/labels/{train,val}/*.txt   (lines: `0 cx cy w h`, normalised)

# 2. Fine-tune RT-DETR (writes a checkpoint dir weights/nose_rtdetr)
python train_detector.py --data-dir data_detector --epochs 50

# 3. Test detection on an image (saves a crop + an annotated preview)
python detect_nose.py path/to/pet.jpg
```

`detect_nose.py` looks for the checkpoint at `weights/nose_rtdetr`, overridable
via the `ELLA_NOSE_WEIGHTS` environment variable. Once present, `enroll_to_db.py`
and `identify.py` crop to the nose automatically with no code change.

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
  `dataset.py`, `arcface.py`, `train.py`, `train_detector.py`: real, runnable.
- `detect_nose.py`: real RT-DETR wrapper with a detection + crop CLI. With no
  trained weights yet it falls back to a centre crop and warns. Run
  `train_detector.py` on boxed noses to produce `weights/nose_rtdetr`.
