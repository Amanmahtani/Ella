"""Ella demo web app.

A thin preview UI over the real pipeline: upload a nose photo to register a pet
(enroll_pet, as service_role) or to identify one (public_identify, as anon). It
uses the same model/ code and the same SECURITY DEFINER functions as everything
else, so what you see here is the real system, not a mock.

Run:
    export DATABASE_URL="postgresql://postgres@/ella?host=/var/lib/postgresql/ella-sock&port=5433"
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Reuse the real pipeline in ../model
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))

import db  # noqa: E402
from config import DEFAULT_MODALITY, DEFAULT_MODEL_ID  # noqa: E402
from embed import NoseEmbedder, to_pgvector  # noqa: E402

app = FastAPI(title="Ella demo")

# Load the backbone once, not per request.
_embedder: NoseEmbedder | None = None


def embedder() -> NoseEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = NoseEmbedder(DEFAULT_MODEL_ID)
    return _embedder


def _embed_upload(upload: UploadFile) -> np.ndarray:
    suffix = os.path.splitext(upload.filename or "img.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.file.read())
        path = tmp.name
    try:
        return embedder().embed_image(path)
    finally:
        os.unlink(path)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/pets")
def list_pets() -> JSONResponse:
    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        cur.execute("select count(*) from pets")
        count = cur.fetchone()[0]
        cur.execute(
            "select name, ella_uid, species, coalesce(breed,'') "
            "from pets order by created_at desc limit 50"
        )
        pets = [
            {"name": r[0], "ella_uid": r[1], "species": r[2], "breed": r[3]}
            for r in cur.fetchall()
        ]
    conn.close()
    return JSONResponse({"count": count, "pets": pets})


@app.post("/api/enroll")
async def enroll(
    file: UploadFile = File(...),
    name: str = Form(...),
    species: str = Form("dog"),
    breed: str = Form(""),
    region: str = Form("BLR"),
    microchip: str = Form(""),
) -> JSONResponse:
    try:
        vec = _embed_upload(file)
        conn = db.connect(role="service_role")
        row = db.enroll_pet(
            conn,
            name=name,
            species=species,
            modality=DEFAULT_MODALITY,
            embedding=to_pgvector(vec),
            model_id=DEFAULT_MODEL_ID,
            breed=breed or None,
            microchip_number=microchip or None,
            region=region,
            frame_count=1,
        )
        conn.close()
        return JSONResponse({
            "ok": True,
            "ella_uid": row["ella_uid"],
            "pet_id": str(row["pet_id"]),
            "has_microchip": row["has_microchip"],
        })
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/identify")
async def identify(file: UploadFile = File(...), k: int = Form(3)) -> JSONResponse:
    try:
        vec = _embed_upload(file)
        conn = db.connect(role="anon")
        rows = db.public_identify(
            conn,
            embedding=to_pgvector(vec),
            modality=DEFAULT_MODALITY,
            model_id=DEFAULT_MODEL_ID,
            k=k,
        )
        conn.close()
        matches = [
            {
                "name": r.get("name"),
                "ella_uid": r.get("ella_uid"),
                "species": r.get("species"),
                "similarity": float(r["similarity"]) if r.get("similarity") is not None else None,
                "has_microchip": r.get("has_microchip"),
            }
            for r in rows
        ]
        return JSONResponse({"ok": True, "matches": matches})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
