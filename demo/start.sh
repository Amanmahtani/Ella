#!/usr/bin/env bash
# Serve the demo two ways:
#   - http  on :8000  for the Mac at http://localhost:8000 (localhost is a secure
#     context, so the live camera works there with no warning)
#   - https on :8443  for phones on the LAN, where the live camera needs https.
#     The cert is self-signed, so the phone shows a one-time "proceed" warning.
set -e

if [ ! -f cert.pem ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
    -days 825 -subj "/CN=ella-demo" >/dev/null 2>&1
fi

uvicorn app:app --host 0.0.0.0 --port 8000 &
exec uvicorn app:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
