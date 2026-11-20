# Ella demo app image: the FastAPI preview over the real model/ pipeline.
FROM python:3.12-slim

# opencv-python-headless needs libglib; openssl makes the local self-signed cert;
# postgresql-client (psql) applies migrations on a cloud deploy.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 openssl postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch is pinned to 2.6.0: from this version back, its NVIDIA CUDA dependencies
# are gated to x86_64, so on arm64 (Apple Silicon containers) they are skipped
# entirely. torch 2.15 dropped that gate and drags ~5 GB of unused GPU libraries
# onto arm64. Pinning keeps the image small and CPU-only, and the arm64 wheel is
# ~96 MB. DINOv2 via transformers runs fine on this version.
# CPU-only torch keeps the image small on any host. The cpu index is CPU-only on
# both x86_64 (cloud) and arm64 (Apple Silicon); fall back to PyPI if the cpu
# index lacks a wheel for the build arch (PyPI torch 2.6.0 also has no CUDA deps
# on arm64, so the fallback stays slim on a Mac).
RUN pip install --no-cache-dir "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu \
      || pip install --no-cache-dir "torch==2.6.0"
RUN pip install --no-cache-dir \
      transformers numpy pillow opencv-python-headless qrcode \
      "psycopg[binary]" fastapi "uvicorn[standard]" python-multipart

# Bake the DINOv2 weights into the image so the first request is fast.
RUN python -c "from transformers import AutoModel, AutoImageProcessor; \
AutoModel.from_pretrained('facebook/dinov2-small'); \
AutoImageProcessor.from_pretrained('facebook/dinov2-small')"

# App code (demo/ imports ../model at runtime).
COPY model/ ./model/
COPY demo/ ./demo/
# Migration SQL, used by start.sh on a cloud deploy (RUN_MIGRATIONS=1).
COPY 001_ella_init.sql 002_ella_functions.sql 003_ella_security.sql /app/

WORKDIR /app/demo
EXPOSE 8000 8443
CMD ["bash", "start.sh"]
