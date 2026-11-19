# Ella — System Overview

A plain-language map of what Ella is, what we have built, the AI model behind the
nose scanner, and an honest list of where it can improve. Written for reading
through and planning, not as API docs.

---

## What Ella is

Ella identifies pets by their **nose print** instead of a microchip. Every
animal's nose has a unique ridge pattern, like a human fingerprint. You take a
photo of the nose, the system turns that pattern into a numeric "fingerprint,"
stores it, and can later recognise the same pet from a new photo. The pitch:
**Face ID, but for a pet's nose**, so a lost pet can be identified and its owner
reached without any implant or chip reader.

---

## How it is built (three layers)

1. **Database** (`001_*.sql` .. `003_*.sql`) — PostgreSQL with the `pgvector`
   extension. Stores pets, their nose-print vectors, owners, clinics, and a log
   of every scan. Built with a strict security model (see Privacy below).
2. **ML pipeline** (`model/`) — the Python that turns a nose photo into a vector
   and matches it. This is the "camera brain."
3. **Web app** (`demo/`) — the mobile-first app: landing page, scanner, pet
   profiles, and the vet dashboard. A thin layer over the database and the model.

Everything runs locally today with one command (`docker compose up`), which
bundles the database, the model, and the web app into one package.

---

## Features we have built

### The scanner (mobile-first, camera-based)
- Opens the phone/laptop camera with a live "scanning" overlay (red reticle,
  sweep line, feature points) styled like a biometric scan.
- **Identify** mode: scan a nose, get the matching pet.
- **Register** mode: enrol a new pet from the camera.
- **Multi-frame capture**: takes a short burst (5 frames to enrol, 3 to
  identify) and averages them, which makes recognition much more reliable.
- **Graceful fallback**: on a phone over plain `http` (where browsers block the
  live camera), the button opens the camera to snap a photo instead.
- Works over **https** on a phone so the full live camera runs on mobile.

### Pet profiles (`/pet/<id>`)
- Each pet has its own page: photo, species/breed, **Safe / LOST** and
  **Vet verified** badges, microchip, registering clinic, and scan history.
- **Owner contact**: name, phone (tap to call), email (tap to email), address.
- **Report lost / Mark safe** toggle, which drives the LOST badge shown on matches.

### Reunion flow (the point of the product)
- On a match, a **"Notify owner"** button alerts the owner and shows a **masked**
  contact (e.g. `+91****11`). The finder reaches the owner **without ever seeing
  the real number**. Every alert is logged.

### Vet clinic dashboard (`/dashboard`)
- A **clinic selector** scopes everything to one clinic.
- Stat tiles: registered pets, pets found, enrollments, scans today.
- Overall match-rate meter (system-wide).
- **Recent scans** table (every scan with result, similarity, time) and a
  **searchable registry** with pet photos. Mobile-first (tables become cards).

### Landing page (`/`)
- Hero with Ella's animated logo, live stats, a 3-step "how it works," and a
  Get Started button into the scanner.

### Design & branding
- Tuxedo-cat theme (black/white with a gold bow tie), drifting paw animations,
  and Ella's photo as the animated logo.

### Privacy model (important and already built)
- The public scan path returns **no owner personal data** by design. Owner phone,
  email and address are *unrepresentable* in the public result, not just hidden.
- A stranger who scans a found pet sees the pet plus the "Notify owner" relay.
- The full profile (with owner address) is the **owner/clinic view**. In
  production it must be gated behind owner/clinic login (not yet built).

### Infrastructure
- One-command Docker setup (Postgres + pgvector + app).
- Dual http (Mac) + https (phone) serving with a self-signed certificate.
- Pet photos stored on a Docker volume so they survive restarts.

---

## The ML model behind the nose scanner

### The approach: recognition, not classification
This is a **re-identification** problem. We do not train a model that "knows
Ella." We use a model that converts any nose photo into a list of ~384 numbers
(an **embedding**) such that two photos of the **same** nose produce **close**
numbers and different noses produce **far-apart** numbers. Matching is then a
distance comparison (cosine similarity) against everything in the database.

### The model we use: DINOv2 (stock, not yet trained)
- **Model:** `facebook/dinov2-small` — a general-purpose image-understanding
  model from Meta. Licence **Apache-2.0** (commercially clean).
- **Important, and the key place to improve:** we are using it **off the shelf,
  pre-trained, NOT fine-tuned on pet noses.** We have **not trained "the camera"
  on nose data yet.** It works for a controlled demo (re-scanning the same pet),
  but it has never been taught what makes two noses the same or different.

### How one scan works, end to end
1. **Detect the nose** in the photo and crop to it (`detect_nose.py`).
2. **Clean it up** — LAB + CLAHE contrast enhancement to bring out the ridges
   (`preprocess.py`).
3. **Embed** — DINOv2 turns the crop into a 384-number vector, normalised
   (`embed.py`).
4. **Average** several frames into one robust vector (multi-frame capture).
5. **Match** — compare against stored vectors in pgvector by cosine similarity;
   a match must clear a threshold (currently **0.86**).

### The nose detector (also not trained yet)
- There is a YOLO-based nose detector wrapper and a trainer (`train_detector.py`),
  but **no detector has been trained**, so today it falls back to a **centre
  crop** of the photo. A trained detector would tightly crop the actual nose and
  is one of the biggest accuracy levers.

### The training pipeline (built, never run)
- `train.py` + `arcface.py` implement **fine-tuning with ArcFace** (the standard
  technique for biometric embeddings). It is ready to run but needs a dataset of
  labelled pet-nose photos, which we do not have yet.

### What is NOT used
- The schema also references **MegaDescriptor** (a stronger animal-ID model), but
  it is **not used in the app** — it is non-commercial licensed and only kept as
  a benchmark reference. Nothing ships with it.

---

## Honest status: real vs demo-grade

| Part | Status |
|---|---|
| Database, security model, scan logging | Real, production-shaped |
| Web app, profiles, dashboard, reunion flow | Real and working |
| The recognition **pipeline** (detect → embed → match) | Real |
| Recognition **accuracy** | **Demo-grade** — stock model, no nose training |
| Nose detector | **Not trained** — centre-crop fallback |
| Owner/clinic login & PII gating | **Not built** (demo shows full profile) |
| Deployment | **Local only** (runs on your Mac via Docker) |

The one-line takeaway: **everything around the model is solid; the model itself
has not been trained on noses yet.** That is the honest frontier.

---

## Where we can improve (roughly in priority order)

1. **Collect real nose data and fine-tune the model.** The single biggest lever.
   Gather many nose photos (multiple per pet, varied lighting/angles) and run the
   existing `train.py`. This is what turns "demo-grade" into "actually accurate."
   Public starting point: the CVPR 2022 Pet Biometric Challenge dataset.
2. **Train the YOLO nose detector** so crops are tight and consistent (big,
   cheap accuracy gain; bad crops are the number-one accuracy killer).
3. **Calibrate the match threshold and add a quality gate** (reject blurry or
   badly-lit captures before matching) to cut false matches and false misses.
4. **Deploy to a real URL** — hosted Postgres (Supabase fits the schema) plus a
   container host (Railway / Render / Fly.io), so it is not tied to your Mac.
5. **Add owner/clinic login** and gate the full profile (owner PII) behind it, so
   only the owner and their clinic see contact details.
6. **Robust photo storage** — move pet photos to object storage (e.g. S3/Supabase
   Storage) instead of a local volume.
7. **Real lost-and-found alerts** — wire the "Notify owner" relay to actual
   SMS/email so owners are really contacted.
8. **Swap the AGPL-licensed YOLO** detector for a permissive alternative (or buy a
   commercial licence) before selling.
9. **Scale the model** — once data exists, move from DINOv2-small (384-d) to a
   larger DINOv2 (the schema already supports a 1536-d column) for more accuracy.

---

## Open-source components and licences

| Component | Role | Licence | Commercial? |
|---|---|---|---|
| PostgreSQL + pgvector | Database + vector search | Open source | Yes |
| DINOv2 (`facebook/dinov2-small`) | Nose embedding (the "brain") | Apache-2.0 | Yes |
| PyTorch, Hugging Face Transformers | Run the model | Open source | Yes |
| OpenCV | Image preprocessing | Open source | Yes |
| YOLO / Ultralytics | Nose detection | **AGPL-3.0** | **Needs care** — copyleft or paid licence |
| FastAPI, Uvicorn | Web server | Open source | Yes |
| Docker | Packaging / running | Open source | Yes |

The only licensing flag is **YOLO/Ultralytics (AGPL)** — fine for building and
demoing, but swap or licence it before a commercial launch.
