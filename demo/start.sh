#!/usr/bin/env bash
set -e

# --- Cloud host (Render / Railway / Fly set $PORT and terminate TLS for you) ---
# One http server on $PORT; the platform provides the public https URL.
if [ -n "$PORT" ]; then
  if [ "$RUN_MIGRATIONS" = "1" ]; then
    echo "Applying migrations to the database..."
    for f in 001_ella_init.sql 002_ella_functions.sql 003_ella_security.sql; do
      psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f "/app/$f"
    done
  fi
  exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
fi

# --- Local (docker compose) ---
# http on :8000 for the Mac (localhost is a secure context), and self-signed
# https on :8443 so the live camera works on phones over the LAN.
if [ ! -f cert.pem ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
    -days 825 -subj "/CN=ella-demo" >/dev/null 2>&1
fi

uvicorn app:app --host 0.0.0.0 --port 8000 &
exec uvicorn app:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
