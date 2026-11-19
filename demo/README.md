# Ella — demo/

A small web app to preview the system: register a pet from a nose photo, then
identify one. It is a thin UI over the real pipeline, not a mock. It calls the
same `model/` code and the same SECURITY DEFINER functions as everything else:

- **Register** uploads a photo, embeds it (DINOv2), and calls `enroll_pet()` as
  `service_role`.
- **Find** uploads a photo, embeds it, and calls `public_identify()` as `anon`,
  so it only ever sees the PII-free summary.

This is for demos and manual testing, not production.

## Run with Docker (one command)

From the repo root, no local Postgres or Python needed:

```bash
docker compose up --build
# then open http://localhost:8000
```

This starts Postgres + pgvector, applies the migrations (001..003) automatically,
builds the app, and serves it. First build downloads torch + the DINOv2 weights,
so it takes a few minutes; later starts are fast. Stop with Ctrl+C.

## Run manually (without Docker)

```bash
# 1. A Postgres with the Ella schema applied (001..003) and the roles created.
export DATABASE_URL="postgresql://postgres@/ella?host=/path/to/socket&port=5432"

# 2. Install deps (needs model/requirements.txt too, for torch/transformers)
pip install -r ../model/requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart

# 3. Serve
cd demo
uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## Accuracy note

Runs on stock DINOv2 with a centre-crop fallback for the nose detector. Identity
separation works, but accuracy is not production-grade until the model is
fine-tuned on real pet-nose data (see `model/train.py`) and a nose detector is
trained (see `model/train_detector.py`).
