# Ella demo app image: the FastAPI preview over the real model/ pipeline.
FROM python:3.12-slim

# opencv-python-headless needs libglib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch from PyPI works on both Intel and Apple-Silicon machines (the pytorch CPU
# index has no arm64 Linux wheels, which breaks the build on M1/M2/M3 Macs).
RUN pip install --no-cache-dir \
      torch transformers numpy pillow opencv-python-headless \
      "psycopg[binary]" fastapi "uvicorn[standard]" python-multipart

# Bake the DINOv2 weights into the image so the first request is fast.
RUN python -c "from transformers import AutoModel, AutoImageProcessor; \
AutoModel.from_pretrained('facebook/dinov2-small'); \
AutoImageProcessor.from_pretrained('facebook/dinov2-small')"

# App code (demo/ imports ../model at runtime).
COPY model/ ./model/
COPY demo/ ./demo/

WORKDIR /app/demo
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
