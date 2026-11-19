# Ella — Training Runbook

A step-by-step checklist to take the nose scanner from "stock model, demo-grade"
to "trained and accurate." Written so you, or an ML contractor, can run it start
to finish. Two models get trained: the **detector** (finds the nose) and the
**embedder** (recognises the nose).

> The whole point: a tight nose crop (detector) + fine-tuning on same-breed
> clusters (embedder) forces the model to learn the unique ridge pattern instead
> of breed, fur colour or background.

---

## 0. What you need

- **A GPU.** A single consumer/cloud GPU is plenty. Cheap options: RunPod,
  Lambda, Vast.ai, or Google Colab. Detector fine-tune ~20 min; embedder
  fine-tune under ~1 hour for a few hundred pets.
- Python deps: `pip install -r model/requirements.txt`.
- No paid APIs or licences — every model used is Apache-2.0 / open source.

---

## 1. Collect the data

Aim for **150–200 pets, 5+ photos each (~1,000 images)**.

Capture tips (these matter more than model choice):
- **Close-up of the nose**, filling the frame, in focus. The nose pad (the bare,
  bumpy skin) should be clearly visible.
- **Vary it**: different angles, distances, and lighting per pet. This teaches
  the model the pet is the same under different conditions.
- **Include look-alike clusters on purpose**: e.g. 15 black Labradors, 15 tuxedo
  cats, 15 Indies. This is the single most important data choice — it forces the
  model to tell near-identical noses apart instead of grouping by breed.
- Phone photos are fine. The app's burst capture already produces good frames.

Keep a simple spreadsheet mapping each photo to a pet id (pet_001, pet_002, …).

---

## 2. Label ~300 images for the detector

You only need to label a **subset** (~300 images); the trained detector then
auto-crops the rest.

1. Use a free tool: **CVAT**, **Label Studio**, or **Roboflow**.
2. Draw **one tight box** around the **bare nose pad only** (exclude eyes, fur,
   hands). One class, name it `nose`.
3. Export in **YOLO format** and lay it out like this:

```
data_detector/
  images/train/*.jpg   labels/train/*.txt
  images/val/*.jpg     labels/val/*.txt
```

Each `labels/*.txt` has one line per box: `0 cx cy w h` (normalised 0–1).
Put ~80% in `train`, ~20% in `val`.

---

## 3. Train the detector (RT-DETR)

```bash
cd model
python train_detector.py --data-dir data_detector --epochs 50
# -> saves a checkpoint to weights/nose_rtdetr
```

Smoke-test it on a photo before the full run:

```bash
python detect_nose.py path/to/a_pet.jpg
# writes nose_crop.jpg + nose_annotated.jpg — check the box hugs the nose
```

If boxes look loose or wrong, label more images (quality + count both help) and
retrain.

---

## 4. Auto-crop all your images

Once the detector is good, run every collected photo through it to produce tight
nose crops, organised **one folder per pet** (this is what the embedder trains
on):

```
data/
  pet_001/ img1.jpg img2.jpg ...
  pet_002/ img1.jpg ...
```

A tiny script using `detect_nose.crop_nose()` on each image does this; crop and
save into `data/<pet_id>/`. (Keeping the same-breed clusters from step 1 intact.)

---

## 5. Train the embedder (DINOv2 + ArcFace)

```bash
cd model
python train.py --data-dir data --model facebook/dinov2-small --epochs 30
# -> saves a fine-tuned backbone to weights/dinov2_nose.pt
```

- Augmentations (colour/brightness/blur/rotation) are **on by default** so the
  model ignores lighting and fur tint and focuses on ridge geometry.
- ArcFace separates every identity, so because your data has same-breed clusters,
  it is forced to distinguish look-alike noses (the "hard negative" effect — it
  comes from the data composition, not a special sampler).
- Want more accuracy later? Swap `--model facebook/dinov2-small` (384-d) for
  `facebook/dinov2-giant` (1536-d) — it drops into the schema's existing 1536
  column with no DB change.

---

## 6. Deploy the trained weights

Point the app at the new weights (both are picked up automatically):

- **Detector**: `weights/nose_rtdetr` (detect_nose.py looks here, or set
  `ELLA_NOSE_WEIGHTS`).
- **Embedder**: pass `--weights weights/dinov2_nose.pt` to `embed.py` /
  `identify.py` / `enroll_to_db.py`, or load it in the demo app's `NoseEmbedder`.

Important: a fine-tuned embedder produces **different** vectors from the stock
model, so **re-enrol every pet** with the new model (old vectors won't match new
ones). In practice: deploy the new weights, then re-register the pets.

---

## 7. Check it actually improved

Hold out a few pets the model never trained on, then:
- **Same pet, new photo** should match (high similarity, above the 0.86 threshold).
- **Different pets of the same breed** should NOT match (this is the real test —
  stock DINOv2 fails here; a well-trained model passes).

Tune the match threshold (currently `0.86`, stored in the `embedding_spaces`
table) using these held-out pairs: raise it to cut false matches, lower it to
cut false misses.

---

## Quick reference

| Step | Command |
|---|---|
| Train detector | `python train_detector.py --data-dir data_detector --epochs 50` |
| Test detector | `python detect_nose.py pet.jpg` |
| Train embedder | `python train.py --data-dir data --epochs 30` |
| Embed one image | `python embed.py pet.jpg --weights weights/dinov2_nose.pt` |

**Biggest levers, in order:** (1) same-breed clusters in the data, (2) tight nose
crops from a good detector, (3) enough photos per pet, (4) a larger backbone.
