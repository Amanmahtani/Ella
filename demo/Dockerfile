# Ella demo app image: the FastAPI preview over the real model/ pipeline.
FROM python:3.12-slim

# opencv-python-headless needs libglib; openssl generates the self-signed cert.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch is pinned to 2.6.0: from this version back, its NVIDIA CUDA dependencies
# are gated to x86_64, so on arm64 (Apple Silicon containers) they are skipped
# entirely. torch 2.15 dropped that gate and drags ~5 GB of unused GPU libraries
# onto arm64. Pinning keeps the image small and CPU-only, and the arm64 wheel is
# ~96 MB. DINOv2 via transformers runs fine on this version.
RUN pip install --no-cache-dir \
      "torch==2.6.0" transformers numpy pillow opencv-python-headless \
      "psycopg[binary]" fastapi "uvicorn[standard]" python-multipart

# Bake the DINOv2 weights into the image so the first request is fast.
RUN python -c "from transformers import AutoModel, AutoImageProcessor; \
AutoModel.from_pretrained('facebook/dinov2-small'); \
AutoImageProcessor.from_pretrained('facebook/dinov2-small')"

# App code (demo/ imports ../model at runtime).
COPY model/ ./model/
COPY demo/ ./demo/

WORKDIR /app/demo
EXPOSE 8000 8443
CMD ["bash", "start.sh"]
