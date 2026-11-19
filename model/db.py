"""Thin database helper around the SECURITY DEFINER functions in 002/003.

No ORM. We call the same functions the rest of the system calls, so the Python
path goes through exactly the privilege and RLS boundary the SQL tests verify.

Connection comes from DATABASE_URL, or the standard PG* environment variables.
enroll_pet requires the service_role; the public_* functions are callable by
anon/authenticated. Set the role on the connection accordingly.
"""

from __future__ import annotations

import os

import psycopg


def connect(role: str | None = None) -> psycopg.Connection:
    dsn = os.environ.get("DATABASE_URL")
    conn = psycopg.connect(dsn) if dsn else psycopg.connect()
    if role:
        with conn.cursor() as cur:
            cur.execute("set role %s" % psycopg.sql.Identifier(role).as_string(conn))
    return conn


def enroll_pet(
    conn: psycopg.Connection,
    *,
    name: str,
    species: str,
    modality: str,
    embedding: str,  # pgvector literal, e.g. '[0.1,-0.2,...]'
    model_id: str | None = None,
    microchip_number: str | None = None,
    breed: str | None = None,
    owner_name: str | None = None,
    owner_phone: str | None = None,
    owner_email: str | None = None,
    region: str = "BLR",
    frame_count: int | None = None,
    quality_score: float | None = None,
) -> dict:
    sql = """
        select pet_id, ella_uid, embedding_id, has_microchip
        from enroll_pet(
            p_name => %(name)s,
            p_species => %(species)s,
            p_modality => %(modality)s,
            p_embedding => %(embedding)s::vector,
            p_model_id => %(model_id)s,
            p_microchip_number => %(microchip_number)s,
            p_breed => %(breed)s,
            p_owner_name => %(owner_name)s,
            p_owner_phone => %(owner_phone)s,
            p_owner_email => %(owner_email)s,
            p_region => %(region)s,
            p_frame_count => %(frame_count)s,
            p_quality_score => %(quality_score)s
        )
    """
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, locals_to_params(locals()))
        row = cur.fetchone()
    conn.commit()
    return row


def public_identify(
    conn: psycopg.Connection,
    *,
    embedding: str,
    modality: str,
    model_id: str | None = None,
    threshold: float | None = None,
    k: int = 3,
) -> list[dict]:
    sql = """
        select *
        from public_identify(
            p_embedding => %(embedding)s::vector,
            p_modality => %(modality)s::text,
            p_model_id => %(model_id)s::text,
            p_threshold => %(threshold)s::numeric,
            p_k => %(k)s::integer
        )
    """
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, {
            "embedding": embedding, "modality": modality,
            "model_id": model_id, "threshold": threshold, "k": k,
        })
        rows = cur.fetchall()
    conn.commit()
    return rows


def locals_to_params(d: dict) -> dict:
    """Pull just the named-arg values out of enroll_pet's locals()."""
    keys = {
        "name", "species", "modality", "embedding", "model_id",
        "microchip_number", "breed", "owner_name", "owner_phone",
        "owner_email", "region", "frame_count", "quality_score",
    }
    return {k: d[k] for k in keys}
