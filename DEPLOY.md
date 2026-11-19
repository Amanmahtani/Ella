# Deploying the Ella demo

Two ways to let other people open it, not just your phone on your wifi.

- **Path A — instant public link** (your Mac stays on). ~2 minutes. Best for
  "send someone the link right now."
- **Path B — real cloud deploy** (your Mac can be off). ~20 minutes, needs
  accounts and a small monthly cost. Best for a standing demo you share widely.

---

## Path A — Instant public link (Cloudflare Tunnel)

Your app keeps running locally; a tunnel gives it a public `https://` URL that
works on any phone or computer, with a real certificate (no "not secure"
warning).

1. Make sure the app is running locally (`docker compose up --build`).
2. Install cloudflared (one time), in a **new** terminal window:

   ```bash
   brew install cloudflared
   ```

3. Start the tunnel pointing at your local app:

   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```

4. It prints a URL like `https://random-words.trycloudflare.com`. **That's your
   shareable link** — send it to anyone. The live camera works there because it
   is real https.

Notes:
- Keep both terminals open (the app + the tunnel) and the Mac awake.
- The free "quick tunnel" URL changes each time you restart it. Fine for a live
  demo; for a stable URL use Path B or a named Cloudflare tunnel.

---

## Path B — Real cloud deploy (Supabase + Render)

The app runs in the cloud 24/7. Two pieces: a **database** (Supabase) and the
**app** (Render).

### 1. Database — Supabase (free)

Ella's schema was written in Supabase's style (it uses the same `anon` /
`service_role` roles), so this is a clean fit.

1. Create a free project at supabase.com.
2. In the project, open the **SQL Editor**.
3. Paste and run each of these three files from the repo, in order:
   `001_ella_init.sql`, then `002_ella_functions.sql`, then `003_ella_security.sql`.
   (pgvector and the roles already exist on Supabase, so this just builds the
   tables and functions.)
4. Get the connection string: **Project Settings -> Database -> Connection
   string -> URI**. Copy it (it looks like
   `postgresql://postgres:...@...supabase.com:5432/postgres`).

### 2. App — Render

1. Create an account at render.com and connect your GitHub.
2. **New -> Blueprint**, pick the `Amanmahtani/Ella` repo. Render reads
   `render.yaml` and sets up the web service.
3. When asked, set the environment variable **`DATABASE_URL`** to the Supabase
   connection string from step 1.
   - Since the database is already set up, you can leave `RUN_MIGRATIONS=1` (it
     re-applies the idempotent migrations on boot) or set it to `0` to skip.
4. Deploy. The first build takes a few minutes (it installs torch + the model).
   When it's live, Render gives you a public `https://ella-xxxx.onrender.com`
   URL — that's your demo.

### Sizing and cost (be realistic)
- The app loads PyTorch + DINOv2, so it needs about **1–2 GB RAM**. That's the
  **Standard** plan on Render (~$25/mo); the free/starter tiers (512 MB) are too
  small and will crash. Supabase's database free tier is fine.
- Cheaper hosts that also work: **Railway** or **Fly.io** (both run the same
  Dockerfile; set `DATABASE_URL` the same way).

### Known demo limitations in the cloud
- **Pet photos are not persisted** across redeploys (no attached disk here). For
  a demo that's fine; for production, add a Render disk or move photos to object
  storage (e.g. Supabase Storage).
- This deploy has **no login yet**, so anyone with the link can use it and the
  pet profile shows owner details. Add owner/clinic auth before a public launch.

---

## Which should I use?

- Showing someone this week → **Path A** (instant link, Mac on).
- A demo that's always up and you send around → **Path B** (Supabase + Render).
