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
import secrets
import string
import sys
import tempfile

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
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


PETS_DIR = os.path.join(HERE, "static", "pets")
os.makedirs(PETS_DIR, exist_ok=True)


def _embed_bytes(data: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return embedder().embed_image(path)
    finally:
        os.unlink(path)


def _embed_files(files: list[UploadFile]) -> tuple[np.ndarray, bytes]:
    """Embed every frame and average into one robust nose print (L2-normalised).

    Multi-frame capture is the cheapest reliability win: averaging several shots
    of the same nose cancels per-frame noise, so a re-scan matches far more
    reliably. Returns the mean vector and the first frame's bytes (for the photo).
    """
    vecs: list[np.ndarray] = []
    first: bytes | None = None
    for f in files:
        data = f.file.read()
        if not data:
            continue
        if first is None:
            first = data
        vecs.append(_embed_bytes(data))
    if not vecs:
        raise ValueError("no usable frames")
    mean = np.mean(np.stack(vecs), axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)
    return mean.astype(np.float32), first


def _save_pet_photo(pet_id: str, data: bytes) -> str:
    """Save a representative frame as the pet's photo and return its URL."""
    path = os.path.join(PETS_DIR, f"{pet_id}.jpg")
    with open(path, "wb") as fh:
        fh.write(data)
    return f"/static/pets/{pet_id}.jpg"


@app.on_event("startup")
def seed_clinics() -> None:
    """Seed demo vet clinics on first boot and attach any unassigned pets, so the
    clinic dashboard has data to show. Safe to run every start."""
    try:
        conn = db.connect(role="service_role")
        with conn.cursor() as cur:
            cur.execute("select count(*) from organizations")
            if cur.fetchone()[0] == 0:
                cur.execute("""
                    insert into organizations (name, org_type, city, state, is_verified) values
                      ('Adv3nture Vet Clinic', 'vet_clinic', 'Bangalore', 'KA', true),
                      ('Indiranagar Pet Hospital', 'vet_clinic', 'Bangalore', 'KA', true)
                """)
            cur.execute("select id from organizations order by created_at, name limit 1")
            first = cur.fetchone()[0]
            cur.execute("update pets set organization_id=%s where organization_id is null", (first,))
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        print("[seed_clinics] skipped:", e)


def _page(name: str) -> FileResponse:
    # no-cache so a rebuilt page is always picked up (avoids stale HTML calling a
    # changed API, e.g. the old "undefined% match rate").
    return FileResponse(os.path.join(HERE, "static", name),
                        headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/")
def landing() -> FileResponse:
    return _page("landing.html")


@app.get("/scan")
def scan() -> FileResponse:
    return _page("index.html")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return _page("dashboard.html")


@app.get("/pet/{uid}")
def pet_page(uid: str) -> FileResponse:
    return _page("pet.html")


@app.get("/api/pet")
def pet_detail(uid: str) -> JSONResponse:
    """Full pet profile including owner contact. This is the OWNER / clinic view.
    In production it must be gated to the owner or an authorised clinic; the
    public (anon) path stays PII-free and uses the relay instead."""
    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        cur.execute("""
            select p.name, p.species, p.breed, p.color_markings, p.photo_url,
                   p.microchip_number, p.is_lost, p.created_at,
                   p.owner_name, p.owner_phone, p.owner_email, p.owner_address,
                   o.name, o.is_verified
            from pets p left join organizations o on o.id = p.organization_id
            where p.ella_uid = %s
        """, (uid,))
        r = cur.fetchone()
        if not r:
            conn.close()
            return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
        cur.execute("""
            select action, outcome, similarity, created_at
            from verification_logs where pet_id = (select id from pets where ella_uid=%s)
            order by created_at desc limit 12
        """, (uid,))
        history = [
            {"action": h[0], "outcome": h[1],
             "similarity": float(h[2]) if h[2] is not None else None, "at": h[3].isoformat()}
            for h in cur.fetchall()
        ]
    conn.close()
    return JSONResponse({
        "ok": True, "ella_uid": uid,
        "name": r[0], "species": r[1], "breed": r[2], "color_markings": r[3], "photo_url": r[4],
        "microchip_number": r[5], "is_lost": r[6], "created_at": r[7].isoformat(),
        "owner": {"name": r[8], "phone": r[9], "email": r[10], "address": r[11]},
        "clinic": {"name": r[12], "verified": r[13]},
        "history": history,
    })


@app.get("/api/stats")
def stats(clinic: str | None = None) -> JSONResponse:
    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        if clinic:
            cur.execute("""
                select
                  (select count(*) from pets where organization_id=%(c)s::uuid),
                  (select count(*) from verification_logs v join pets p on p.id=v.pet_id
                     where v.action='identify' and v.outcome='match' and p.organization_id=%(c)s::uuid),
                  (select count(*) from verification_logs v join pets p on p.id=v.pet_id
                     where v.action='enroll' and p.organization_id=%(c)s::uuid),
                  (select count(*) from verification_logs v join pets p on p.id=v.pet_id
                     where p.organization_id=%(c)s::uuid and v.created_at::date=now()::date),
                  (select count(*) from verification_logs v join pets p on p.id=v.pet_id
                     where v.action='alert_relay' and v.outcome='success' and p.organization_id=%(c)s::uuid),
                  (select count(*) from verification_logs v join pets p on p.id=v.pet_id
                     where v.action='identify' and p.organization_id=%(c)s::uuid)
            """, {"c": clinic})
        else:
            cur.execute("""
                select
                  (select count(*) from pets),
                  (select count(*) from verification_logs where action='identify' and outcome='match'),
                  (select count(*) from verification_logs where action='enroll'),
                  (select count(*) from verification_logs where created_at::date=now()::date),
                  (select count(*) from verification_logs where action='alert_relay' and outcome='success'),
                  (select count(*) from verification_logs where action='identify')
            """)
        pets, found, enrollments, today, reunions, scans = cur.fetchone()
    conn.close()
    return JSONResponse({
        "pets": pets, "matches": found, "enrollments": enrollments,
        "today": today, "reunions": reunions, "scans": scans,
    })


@app.get("/api/scans")
def scans(clinic: str | None = None) -> JSONResponse:
    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        cur.execute("""
            select v.action, v.outcome, v.similarity, v.created_at, p.name, p.ella_uid, p.photo_url
            from verification_logs v
            left join pets p on p.id = v.pet_id
            where v.action in ('identify','lookup','enroll','alert_relay')
              and (%(c)s::uuid is null or p.organization_id = %(c)s::uuid)
            order by v.created_at desc
            limit 25
        """, {"c": clinic})
        rows = [
            {
                "action": r[0], "outcome": r[1],
                "similarity": float(r[2]) if r[2] is not None else None,
                "at": r[3].isoformat(), "name": r[4], "ella_uid": r[5], "photo_url": r[6],
            }
            for r in cur.fetchall()
        ]
    conn.close()
    return JSONResponse({"scans": rows})


@app.get("/api/clinics")
def clinics() -> JSONResponse:
    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        cur.execute("""
            select o.id, o.name, coalesce(o.city,''), o.is_verified,
                   (select count(*) from pets p where p.organization_id = o.id)
            from organizations o order by o.name
        """)
        rows = [
            {"id": str(r[0]), "name": r[1], "city": r[2], "verified": r[3], "pets": r[4]}
            for r in cur.fetchall()
        ]
    conn.close()
    return JSONResponse({"clinics": rows})


@app.get("/api/pets")
def list_pets(clinic: str | None = None) -> JSONResponse:
    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        cur.execute(
            "select name, ella_uid, species, coalesce(breed,''), photo_url, is_lost from pets "
            "where (%(c)s::uuid is null or organization_id = %(c)s::uuid) "
            "order by created_at desc limit 100",
            {"c": clinic},
        )
        pets = [
            {"name": r[0], "ella_uid": r[1], "species": r[2], "breed": r[3],
             "photo_url": r[4], "is_lost": r[5]}
            for r in cur.fetchall()
        ]
    conn.close()
    return JSONResponse({"count": len(pets), "pets": pets})


@app.post("/api/enroll")
async def enroll(
    files: list[UploadFile] = File(...),
    name: str = Form(...),
    species: str = Form("dog"),
    breed: str = Form(""),
    region: str = Form("BLR"),
    microchip: str = Form(""),
    clinic: str = Form(""),
    owner_name: str = Form(""),
    owner_phone: str = Form(""),
    owner_email: str = Form(""),
    owner_address: str = Form(""),
) -> JSONResponse:
    try:
        vec, first = _embed_files(files)
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
            owner_name=owner_name or None,
            owner_phone=owner_phone or None,
            owner_email=owner_email or None,
            owner_address=owner_address or None,
            region=region,
            frame_count=len(files),
            organization_id=clinic or None,
        )
        pet_id = str(row["pet_id"])
        photo_url = _save_pet_photo(pet_id, first) if first else None
        if photo_url:
            with conn.cursor() as cur:
                cur.execute("update pets set photo_url=%s where id=%s", (photo_url, pet_id))
            conn.commit()
        conn.close()
        return JSONResponse({
            "ok": True, "ella_uid": row["ella_uid"], "pet_id": pet_id,
            "has_microchip": row["has_microchip"], "photo_url": photo_url,
            "frames": len(files),
        })
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/identify")
async def identify(files: list[UploadFile] = File(...), k: int = Form(3)) -> JSONResponse:
    try:
        vec, _ = _embed_files(files)
        conn = db.connect(role="anon")
        rows = db.public_identify(
            conn, embedding=to_pgvector(vec),
            modality=DEFAULT_MODALITY, model_id=DEFAULT_MODEL_ID, k=k,
        )
        conn.close()
        matches = [
            {
                "name": r.get("name"), "ella_uid": r.get("ella_uid"),
                "species": r.get("species"), "breed": r.get("breed"),
                "similarity": float(r["similarity"]) if r.get("similarity") is not None else None,
                "has_microchip": r.get("has_microchip"), "photo_url": r.get("photo_url"),
                "is_lost": r.get("is_lost"), "is_vet_verified": r.get("is_vet_verified"),
                "relay_token": r.get("alert_relay_token"),
            }
            for r in rows
        ]
        return JSONResponse({"ok": True, "matches": matches})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/pet/lost")
async def set_lost(ella_uid: str = Form(...), lost: str = Form(...)) -> JSONResponse:
    is_lost = lost.lower() in ("1", "true", "yes", "on")
    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        cur.execute("update pets set is_lost=%s, lost_since=case when %s then now() else null end "
                    "where ella_uid=%s", (is_lost, is_lost, ella_uid))
        n = cur.rowcount
    conn.commit()
    conn.close()
    if not n:
        return JSONResponse({"ok": False, "error": "unknown pet"}, status_code=404)
    return JSONResponse({"ok": True, "is_lost": is_lost})


def _new_tag_code() -> str:
    """Unique tag code matching the schema's ^[A-Za-z0-9]{6,32}$ constraint."""
    alpha = string.ascii_uppercase + string.digits
    return "ELLA" + "".join(secrets.choice(alpha) for _ in range(8))


def _ensure_tag(cur, ella_uid: str) -> str | None:
    """Return the pet's active tag code, creating one if it has none."""
    cur.execute("select id, organization_id from pets where ella_uid=%s", (ella_uid,))
    r = cur.fetchone()
    if not r:
        return None
    pet_id, org_id = r
    cur.execute("select tag_code from qr_tags where pet_id=%s and status='active' limit 1", (pet_id,))
    t = cur.fetchone()
    if t:
        return t[0]
    code = _new_tag_code()
    cur.execute(
        "insert into qr_tags (tag_code, pet_id, organization_id, status, activated_at) "
        "values (%s, %s, %s, 'active', now())",
        (code, pet_id, org_id),
    )
    return code


@app.post("/api/pet/tag")
async def ensure_tag(ella_uid: str = Form(...)) -> JSONResponse:
    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        code = _ensure_tag(cur, ella_uid)
    if code is None:
        conn.close()
        return JSONResponse({"ok": False, "error": "unknown pet"}, status_code=404)
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "tag_code": code})


@app.get("/api/pet/qr")
def pet_qr(uid: str, origin: str = "") -> Response:
    """A PNG QR code encoding this pet's finder URL (origin/t/<tag>). Generated
    server-side, so no external CDN and it works offline. `origin` is the
    browser's own origin, so the QR points at whatever host the owner is on."""
    import io

    import qrcode

    conn = db.connect(role="service_role")
    with conn.cursor() as cur:
        code = _ensure_tag(cur, uid)
    if code is None:
        conn.close()
        return JSONResponse({"ok": False, "error": "unknown pet"}, status_code=404)
    conn.commit()
    conn.close()
    img = qrcode.make(f"{origin}/t/{code}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/t/{code}")
def tag_page(code: str) -> FileResponse:
    return _page("finder.html")


@app.get("/api/tag")
def tag_lookup(code: str) -> JSONResponse:
    """Finder-safe lookup for a scanned QR tag. Runs as anon via public_lookup_tag,
    so it returns NO owner PII — just the pet and the relay token."""
    conn = db.connect(role="anon")
    with conn.cursor() as cur:
        cur.execute(
            "select name, ella_uid, species, breed, photo_url, is_lost, alert_relay_token "
            "from public_lookup_tag(%s::text)",
            (code,),
        )
        r = cur.fetchone()
    conn.commit()
    conn.close()
    if not r:
        return JSONResponse({"ok": True, "found": False})
    return JSONResponse({
        "ok": True, "found": True,
        "name": r[0], "ella_uid": r[1], "species": r[2], "breed": r[3],
        "photo_url": r[4], "is_lost": r[5], "relay_token": r[6],
    })


@app.post("/api/relay")
async def relay(ella_uid: str = Form(...)) -> JSONResponse:
    """Reunion payoff: a finder alerts the owner via the relay, without ever
    seeing the owner's real contact details. Logs an alert_relay event and
    returns a masked confirmation (real messaging would hook in here)."""
    try:
        conn = db.connect(role="service_role")
        with conn.cursor() as cur:
            cur.execute("select id, name, owner_phone, owner_email from pets where ella_uid=%s", (ella_uid,))
            row = cur.fetchone()
            if not row:
                conn.close()
                return JSONResponse({"ok": False, "error": "unknown pet"}, status_code=404)
            pet_id, name, phone, email = row
            cur.execute(
                "insert into verification_logs (pet_id, action, outcome, modality) "
                "values (%s, 'alert_relay', 'success', %s)",
                (pet_id, DEFAULT_MODALITY),
            )
        conn.commit()
        conn.close()
        contact = phone or email or ""
        masked = (contact[:3] + "••••" + contact[-2:]) if len(contact) >= 5 else "on file"
        return JSONResponse({"ok": True, "name": name, "masked": masked})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
