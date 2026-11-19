-- =============================================================================
-- Ella — 003_ella_security.sql
-- Row-Level Security, privilege REVOKE/GRANT, and the public API surface.
--
-- This migration is where the security posture promised by 001 and 002 is
-- actually enforced. Until it is applied, the SECURITY DEFINER functions in 002
-- are callable by anyone and the base tables are readable by PUBLIC — so 003 is
-- not optional hardening, it is load-bearing.
--
-- Threat model & design
--   * LOCKDOWN. Every base table has FORCED row-level security with NO policy
--     for anon/authenticated. "No policy" under RLS means "deny all rows", so a
--     leaked anon key cannot `select * from pets`. There is no direct table read
--     path for the public at all — every public interaction goes through a
--     SECURITY DEFINER public_* function whose return type (public_pet_summary)
--     is PII-free by construction (002).
--   * FORCE (not just ENABLE) RLS so the policies also bind the table OWNER.
--     Without FORCE, the owner role — which migrations and some tooling run as —
--     silently bypasses RLS, which is exactly the footgun that leaks PII in dev
--     and then surprises everyone in prod.
--   * service_role is Supabase's trusted server identity and BYPASSES RLS by
--     design (it has rolbypassrls). The API gateway runs as service_role, so it
--     is the sole enroller. We therefore do NOT write permissive policies for
--     it; we simply grant it EXECUTE on the internal functions. Belt-and-braces:
--     we never GRANT table DML to anon/authenticated either, so even if a policy
--     were added by mistake, the privilege check still fails first.
--   * Default-deny privileges. We REVOKE ALL from PUBLIC on every object, then
--     grant back only the minimum: EXECUTE on the public_* read path to the
--     anonymous web, and EXECUTE on the internal enroll/match path to
--     service_role. No base-table SELECT/INSERT/UPDATE/DELETE is granted to
--     anon or authenticated, ever.
--   * Idempotent: every policy is dropped before create; grants/revokes are
--     naturally repeatable. Safe to re-apply.
--
-- Role assumptions (Supabase conventions)
--   anon           — unauthenticated web visitor (the lost-pet finder).
--   authenticated  — a signed-in Supabase Auth user. No owner model exists in
--                    the schema yet, so for now it has the SAME minimal surface
--                    as anon: the public_* read path only. Owner-scoped direct
--                    table policies are deferred until an auth.uid() linkage
--                    column is added.
--   service_role   — the trusted backend / API gateway. Bypasses RLS; is the
--                    only caller of enroll_pet / add_pet_embedding /
--                    match_biometric_embedding / log_verification.
--
-- These roles may not exist outside Supabase (e.g. a bare PGlite/Postgres used
-- for CI). The guarded block below creates them NOLOGIN if missing so this file
-- applies anywhere without turning CI red.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 0. Ensure the Supabase roles exist (no-op on a real Supabase project).
-- -----------------------------------------------------------------------------
do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then
    create role anon nologin noinherit;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin noinherit;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then
    -- bypassrls mirrors Supabase: the trusted backend sees through RLS.
    create role service_role nologin noinherit bypassrls;
  end if;
end;
$$;

-- -----------------------------------------------------------------------------
-- 1. Enable + FORCE row-level security on every base table.
-- -----------------------------------------------------------------------------
-- FORCE makes the (deny-all) policies apply to the table owner too, closing the
-- "owner silently bypasses RLS" gap. Only rolbypassrls roles (service_role) get
-- through, which is precisely what we want for the gateway.
alter table public.embedding_spaces     enable row level security;
alter table public.embedding_spaces     force  row level security;

alter table public.organizations        enable row level security;
alter table public.organizations        force  row level security;

alter table public.pets                  enable row level security;
alter table public.pets                  force  row level security;

alter table public.biometric_embeddings enable row level security;
alter table public.biometric_embeddings force  row level security;

alter table public.qr_tags               enable row level security;
alter table public.qr_tags               force  row level security;

alter table public.health_records        enable row level security;
alter table public.health_records        force  row level security;

alter table public.verification_logs     enable row level security;
alter table public.verification_logs     force  row level security;

-- -----------------------------------------------------------------------------
-- 2. Policies.
-- -----------------------------------------------------------------------------
-- Deliberately NO policy grants rows to anon/authenticated on any PII-bearing
-- table. Under enabled RLS, the absence of a permissive policy is a hard deny.
-- We make that intent explicit (rather than implicit-by-omission) with a single
-- restrictive-looking FALSE policy per table, so an auditor reading the policy
-- catalogue sees "deny" written down instead of inferring it from silence.
--
-- NOTE: `using (false)` is a *permissive* policy that matches no rows; combined
-- with "no other permissive policy exists", the effective result is deny-all
-- for every command for anon/authenticated. service_role is unaffected (it
-- bypasses RLS entirely).

-- embedding_spaces --------------------------------------------------------
drop policy if exists embedding_spaces_no_public_access on public.embedding_spaces;
create policy embedding_spaces_no_public_access
  on public.embedding_spaces
  for all
  to anon, authenticated
  using (false)
  with check (false);

-- organizations -----------------------------------------------------------
drop policy if exists organizations_no_public_access on public.organizations;
create policy organizations_no_public_access
  on public.organizations
  for all
  to anon, authenticated
  using (false)
  with check (false);

-- pets (owner PII) --------------------------------------------------------
drop policy if exists pets_no_public_access on public.pets;
create policy pets_no_public_access
  on public.pets
  for all
  to anon, authenticated
  using (false)
  with check (false);

-- biometric_embeddings ----------------------------------------------------
drop policy if exists biometric_embeddings_no_public_access on public.biometric_embeddings;
create policy biometric_embeddings_no_public_access
  on public.biometric_embeddings
  for all
  to anon, authenticated
  using (false)
  with check (false);

-- qr_tags -----------------------------------------------------------------
drop policy if exists qr_tags_no_public_access on public.qr_tags;
create policy qr_tags_no_public_access
  on public.qr_tags
  for all
  to anon, authenticated
  using (false)
  with check (false);

-- health_records (PII-adjacent) ------------------------------------------
drop policy if exists health_records_no_public_access on public.health_records;
create policy health_records_no_public_access
  on public.health_records
  for all
  to anon, authenticated
  using (false)
  with check (false);

-- verification_logs (append-only audit) ----------------------------------
drop policy if exists verification_logs_no_public_access on public.verification_logs;
create policy verification_logs_no_public_access
  on public.verification_logs
  for all
  to anon, authenticated
  using (false)
  with check (false);

-- -----------------------------------------------------------------------------
-- 3. Table-level privileges: default-deny.
-- -----------------------------------------------------------------------------
-- RLS gates *rows*; GRANTs gate *commands*. We want both lines of defence, so
-- strip the PUBLIC defaults and never hand anon/authenticated any table DML.
-- service_role is granted table DML because the gateway's SECURITY DEFINER
-- functions run as their definer (the migration owner), but direct service_role
-- maintenance queries (backfills, admin scripts) still need a privilege path.
revoke all on all tables    in schema public from public;
revoke all on all sequences in schema public from public;
revoke all on all functions in schema public from public;

-- service_role: full table DML (RLS-bypassing trusted backend).
grant usage on schema public to service_role;
grant all privileges on all tables    in schema public to service_role;
grant all privileges on all sequences in schema public to service_role;

-- anon / authenticated: schema usage only. No table privileges. Their ONLY
-- capability is EXECUTE on the public_* functions granted in section 4.
grant usage on schema public to anon, authenticated;

-- -----------------------------------------------------------------------------
-- 4. Function privileges: the real API surface.
-- -----------------------------------------------------------------------------
-- Because we revoked EXECUTE from PUBLIC above, every function is now deny-all
-- until explicitly granted. We grant the minimum.

-- 4a. Public read path — safe for the anonymous web. Each returns the PII-free
--     public_pet_summary (or a boolean verdict) and self-logs via the audit
--     function it owns. Granting these to anon is the entire public product.
grant execute on function
  public.public_identify(vector, text, text, numeric, int, text, text)
  to anon, authenticated;

grant execute on function
  public.public_lookup_pet(text, text, text, text)
  to anon, authenticated;

grant execute on function
  public.public_lookup_tag(text, text, text)
  to anon, authenticated;

grant execute on function
  public.verify_pet(text, vector, text, text, numeric, text, text)
  to anon, authenticated;

-- 4b. Internal write/matching path — service_role ONLY. These touch PII, mutate
--     state, or expose raw match internals (pet_id, embedding_id), so they must
--     never be reachable by anon/authenticated. REVOKE from the public roles is
--     redundant after the blanket revoke in section 3 but is stated explicitly
--     so the restriction survives any future broad re-grant.
revoke execute on function
  public.enroll_pet(text, text, text, vector, text, text, uuid, text, text, date, text, text, text, text, text, text, numeric, text, int, text)
  from anon, authenticated;
grant execute on function
  public.enroll_pet(text, text, text, vector, text, text, uuid, text, text, date, text, text, text, text, text, text, numeric, text, int, text)
  to service_role;

revoke execute on function
  public.add_pet_embedding(text, text, vector, text, numeric, text, int)
  from anon, authenticated;
grant execute on function
  public.add_pet_embedding(text, text, vector, text, numeric, text, int)
  to service_role;

revoke execute on function
  public.match_biometric_embedding(vector, text, text, numeric, int)
  from anon, authenticated;
grant execute on function
  public.match_biometric_embedding(vector, text, text, numeric, int)
  to service_role;

-- 4c. Supporting internal helpers — service_role only. resolve_embedding_space
--     and log_verification are called internally by the definer functions (they
--     run with the definer's rights regardless of the invoker's grants), so the
--     public roles have no reason to call them directly.
revoke execute on function public.resolve_embedding_space(text, text)
  from anon, authenticated;
grant execute on function public.resolve_embedding_space(text, text)
  to service_role;

revoke execute on function public.log_verification(text, text, uuid, uuid, uuid, numeric, text, text, text, text)
  from anon, authenticated;
grant execute on function public.log_verification(text, text, uuid, uuid, uuid, numeric, text, text, text, text)
  to service_role;

-- 4d. UID / token generators are SECURITY DEFINER defaults on pets. They are not
--     part of any caller's API; keep them service_role-only. (The pets defaults
--     still fire because defaults execute with the inserting function's rights.)
revoke execute on function public.generate_ella_uid(text)
  from anon, authenticated;
grant  execute on function public.generate_ella_uid(text)
  to service_role;

revoke execute on function public.generate_alert_relay_token()
  from anon, authenticated;
grant  execute on function public.generate_alert_relay_token()
  to service_role;

-- set_updated_at is a trigger function; it is never called directly. Trigger
-- execution does not consult EXECUTE privilege, so no grant is required and we
-- leave it revoked from PUBLIC.

-- -----------------------------------------------------------------------------
-- 5. Default privileges for objects created later in this schema.
-- -----------------------------------------------------------------------------
-- So that a future `create table` / `create function` in public does not
-- silently re-open access to PUBLIC, pin the default-deny posture going forward.
-- Scoped to the role running this migration (the object owner).
alter default privileges in schema public
  revoke all on tables    from public;
alter default privileges in schema public
  revoke all on sequences from public;
alter default privileges in schema public
  revoke execute on functions from public;

-- Future functions should NOT be auto-executable by the web roles; each new
-- public_* entry point must be granted deliberately (as in section 4a).
-- We intentionally do not add a blanket default EXECUTE grant to anon here.

-- =============================================================================
-- End 003_ella_security.sql
-- =============================================================================
