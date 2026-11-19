-- =============================================================================
-- Ella — 001_ella_init.sql
-- Core schema: organizations, pets, biometric embeddings, QR tags,
-- health records, verification logs.
--
-- Design notes
--   * Model-agnostic embeddings. pgvector needs a FIXED dimension per column to
--     build an index, so a single "any dimension" column is physically
--     impossible. Instead: an `embedding_spaces` registry (thresholds become
--     DATA, not code) plus one nullable typed vector column per space, each with
--     its own partial HNSW index scoped by model. Both 384-d (DINOv2-small) and
--     1536-d (MegaDescriptor-L-384 / Swin-L) are pre-provisioned, so the planned
--     demo model swap needs ZERO schema change. A genuinely new dimension needs
--     one additive ADD COLUMN + CREATE INDEX — no backfill, no data loss.
--   * PII (owner_*) lives on `pets` and is never returned by the public_*
--     accessors in 002. Biometric vectors live in a separate table so they can
--     be granted/revoked independently of owner contact details.
--   * gen_random_uuid() is core PostgreSQL 13+; pgcrypto is deliberately NOT a
--     dependency.
--   * Idempotent: safe to re-apply.
-- =============================================================================

create extension if not exists vector;

-- -----------------------------------------------------------------------------
-- Helper functions (created before tables: used as column defaults / triggers)
-- -----------------------------------------------------------------------------

-- Public-facing pet identifier, e.g. 'ELLA-BLR-849201'.
--
-- SECURITY DEFINER is required, not cosmetic: `pets` has FORCED row-level
-- security (003), so an invoker-rights collision pre-check would see zero rows
-- and silently stop deduplicating. The function returns only a random string
-- and leaks no row data. The UNIQUE index on pets.ella_uid remains the
-- authoritative guarantee; this loop just avoids insert retries.
create or replace function public.generate_ella_uid(p_region text default 'BLR')
returns text
language plpgsql
volatile
security definer
set search_path = public, pg_temp
as $$
declare
  v_region    text := upper(regexp_replace(coalesce(p_region, 'BLR'), '[^A-Za-z]', '', 'g'));
  v_candidate text;
  v_attempt   int := 0;
begin
  if length(v_region) <> 3 then
    v_region := 'BLR';
  end if;

  loop
    v_attempt := v_attempt + 1;

    -- 100000..999999 — always 6 digits, no leading-zero ambiguity.
    v_candidate := format(
      'ELLA-%s-%s',
      v_region,
      (floor(random() * 900000) + 100000)::bigint::text
    );

    if not exists (select 1 from public.pets where ella_uid = v_candidate) then
      return v_candidate;
    end if;

    if v_attempt >= 50 then
      raise exception
        'generate_ella_uid: could not allocate a unique UID for region % after % attempts',
        v_region, v_attempt;
    end if;
  end loop;
end;
$$;

comment on function public.generate_ella_uid(text) is
  'Allocates a collision-checked public pet UID in the form ELLA-<REGION>-<6 digits>.';

-- Opaque token used by the public lost-pet flow to relay a message to the owner
-- WITHOUT ever exposing their phone number or address. 2x UUIDv4 => 244 bits.
create or replace function public.generate_alert_relay_token()
returns text
language sql
volatile
set search_path = public, pg_temp
as $$
  select replace(gen_random_uuid()::text, '-', '')
      || replace(gen_random_uuid()::text, '-', '');
$$;

comment on function public.generate_alert_relay_token() is
  'Generates a 64-char opaque relay token so finders can contact an owner without seeing their PII.';

create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

comment on function public.set_updated_at() is
  'BEFORE UPDATE trigger: maintains updated_at server-side so clients cannot spoof it.';

-- -----------------------------------------------------------------------------
-- embedding_spaces — the model registry
-- -----------------------------------------------------------------------------
-- One row per (modality, model, version). Match thresholds live here as data so
-- tuning never requires a code deploy. `column_name` tells the matcher which
-- physical vector column on biometric_embeddings holds this space's vectors.
create table if not exists public.embedding_spaces (
  id              uuid primary key default gen_random_uuid(),
  modality        text not null,
  model_id        text not null,
  model_version   text not null default '1',
  dim             int  not null,
  column_name     text not null,
  is_active       boolean not null default false,
  match_threshold numeric(5, 4) not null,
  license         text,
  notes           text,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),

  constraint embedding_spaces_modality_check
    check (modality in ('nasal_planum', 'cat_face', 'coat_pattern', 'iris')),
  constraint embedding_spaces_dim_check
    check (dim > 0 and dim <= 2000),
  constraint embedding_spaces_column_name_check
    check (column_name in ('embedding_384', 'embedding_1536')),
  constraint embedding_spaces_column_matches_dim_check
    check (
      (dim = 384  and column_name = 'embedding_384')
      or (dim = 1536 and column_name = 'embedding_1536')
    ),
  constraint embedding_spaces_threshold_check
    check (match_threshold > 0 and match_threshold <= 1),
  constraint embedding_spaces_identity_key
    unique (modality, model_id, model_version)
);

comment on table public.embedding_spaces is
  'Registry of biometric embedding spaces. Match thresholds are data, not code.';
comment on column public.embedding_spaces.column_name is
  'Physical vector column on biometric_embeddings backing this space.';
comment on column public.embedding_spaces.license is
  'Model licence. Non-commercial licences are demo-only and must not ship commercially.';

-- -----------------------------------------------------------------------------
-- organizations — vet clinics, shelters, insurers
-- -----------------------------------------------------------------------------
create table if not exists public.organizations (
  id                  uuid primary key default gen_random_uuid(),
  name                text not null,
  org_type            text not null,
  registration_number text,
  contact_email       text,
  contact_phone       text,
  address             text,
  city                text,
  state               text,
  country             text not null default 'IN',
  is_verified         boolean not null default false,
  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now(),

  constraint organizations_name_check
    check (length(btrim(name)) > 0),
  constraint organizations_org_type_check
    check (org_type in ('vet_clinic', 'shelter', 'insurer', 'ngo', 'government', 'other'))
);

comment on table public.organizations is
  'Tenants of the identity rail: vet clinics, shelters, insurers, NGOs.';

-- -----------------------------------------------------------------------------
-- pets — the identity record (holds owner PII)
-- -----------------------------------------------------------------------------
create table if not exists public.pets (
  id              uuid primary key default gen_random_uuid(),
  ella_uid        text not null default public.generate_ella_uid(),
  organization_id uuid references public.organizations (id) on delete set null,

  name            text not null,
  species         text not null,
  breed           text,
  sex             text not null default 'unknown',
  date_of_birth   date,
  color_markings  text,
  photo_url       text,

  -- Hardware-to-biometric bridge. Strict ISO 11784/11785 15-digit transponder
  -- code. NULL means "biometric-only" — the common case, and the whole point of
  -- the product. Uniqueness is enforced by a PARTIAL index below so unlimited
  -- rows may stay NULL.
  microchip_number        text,
  microchip_registered_at timestamptz,

  owner_name      text,
  owner_phone     text,
  owner_email     text,
  owner_address   text,

  is_lost         boolean not null default false,
  lost_since      timestamptz,

  is_vet_verified boolean not null default false,
  vet_verified_at timestamptz,
  vet_verified_by uuid references public.organizations (id) on delete set null,

  -- Lets a finder reach the owner without the owner's PII ever being disclosed.
  alert_relay_token text not null default public.generate_alert_relay_token(),

  notes           text,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),

  constraint pets_ella_uid_unique
    unique (ella_uid),
  constraint pets_alert_relay_token_unique
    unique (alert_relay_token),
  constraint pets_name_check
    check (length(btrim(name)) > 0),
  constraint pets_species_check
    check (species in ('dog', 'cat')),
  constraint pets_sex_check
    check (sex in ('male', 'female', 'unknown')),
  constraint pets_ella_uid_format_check
    check (ella_uid ~ '^ELLA-[A-Z]{3}-[0-9]{6}$'),
  -- Strict 15-digit ISO microchip. Rejects 14/16 digits, letters, spaces,
  -- hyphens and empty strings. Normalisation happens in the API layer before
  -- the value ever reaches the database.
  constraint pets_microchip_number_check
    check (microchip_number is null or microchip_number ~ '^[0-9]{15}$'),
  constraint pets_lost_since_check
    check (is_lost = false or lost_since is not null)
);

comment on table public.pets is
  'Canonical animal identity. Owner PII lives here and is never exposed by the public_* accessors.';
comment on column public.pets.microchip_number is
  'Optional ISO 11784/11785 15-digit transponder code. Never returned to unauthenticated callers.';
comment on column public.pets.alert_relay_token is
  'Opaque token for PII-free owner contact during a lost-pet scan.';

-- Partial UNIQUE: one pet per physical microchip, unlimited NULLs.
create unique index if not exists idx_pets_microchip_number_unique
  on public.pets (microchip_number)
  where microchip_number is not null;

create index if not exists idx_pets_organization_id
  on public.pets (organization_id);

create index if not exists idx_pets_is_lost
  on public.pets (is_lost)
  where is_lost = true;

create index if not exists idx_pets_species
  on public.pets (species);

-- -----------------------------------------------------------------------------
-- biometric_embeddings — vectors, decoupled from PII
-- -----------------------------------------------------------------------------
-- (modality, model_id, model_version) is a composite FK into embedding_spaces,
-- so a vector can never claim a model/modality combination that is not
-- registered. All matching is scoped by modality + model_id, which makes
-- cross-model comparison structurally impossible rather than merely discouraged.
create table if not exists public.biometric_embeddings (
  id            uuid primary key default gen_random_uuid(),
  pet_id        uuid not null references public.pets (id) on delete cascade,

  modality      text not null,
  model_id      text not null,
  model_version text not null default '1',
  embedding_dim int  not null,

  -- Exactly one of these is populated, per the check constraint below.
  embedding_384  vector(384),
  embedding_1536 vector(1536),

  quality_score numeric(5, 4),
  source_kind   text,
  frame_count   int,
  is_active     boolean not null default true,
  captured_at   timestamptz not null default now(),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),

  constraint biometric_embeddings_space_fk
    foreign key (modality, model_id, model_version)
    references public.embedding_spaces (modality, model_id, model_version)
    on update cascade,
  -- Guarantees exactly one vector column is populated AND that it agrees with
  -- the declared dimension. One constraint, both invariants.
  constraint biometric_embeddings_single_vector_check
    check (
      (embedding_dim = 384  and embedding_384  is not null and embedding_1536 is null)
      or (embedding_dim = 1536 and embedding_1536 is not null and embedding_384  is null)
    ),
  constraint biometric_embeddings_quality_score_check
    check (quality_score is null or (quality_score >= 0 and quality_score <= 1)),
  constraint biometric_embeddings_source_kind_check
    check (source_kind is null or source_kind in ('image', 'video')),
  constraint biometric_embeddings_frame_count_check
    check (frame_count is null or frame_count > 0)
);

comment on table public.biometric_embeddings is
  'Biometric vectors. Many per pet and per modality; always matched scoped by modality + model_id.';
comment on column public.biometric_embeddings.embedding_dim is
  'Declared dimension; a check constraint ties it to the populated vector column.';

create index if not exists idx_biometric_embeddings_pet_id
  on public.biometric_embeddings (pet_id);

create index if not exists idx_biometric_embeddings_lookup
  on public.biometric_embeddings (modality, model_id, is_active);

-- Partial HNSW (cosine) per embedding space. Matching queries must filter on
-- the same predicate for the planner to use these.
create index if not exists idx_biometric_embeddings_384_hnsw
  on public.biometric_embeddings
  using hnsw (embedding_384 vector_cosine_ops)
  where embedding_384 is not null and is_active = true;

create index if not exists idx_biometric_embeddings_1536_hnsw
  on public.biometric_embeddings
  using hnsw (embedding_1536 vector_cosine_ops)
  where embedding_1536 is not null and is_active = true;

-- -----------------------------------------------------------------------------
-- qr_tags — physical tags; pet_id stays NULL for unclaimed clinic stock
-- -----------------------------------------------------------------------------
create table if not exists public.qr_tags (
  id              uuid primary key default gen_random_uuid(),
  tag_code        text not null,
  pet_id          uuid references public.pets (id) on delete set null,
  organization_id uuid references public.organizations (id) on delete set null,

  status          text not null default 'unassigned',
  batch_id        text,
  activated_at    timestamptz,
  scan_count      int not null default 0,
  last_scanned_at timestamptz,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),

  constraint qr_tags_tag_code_unique
    unique (tag_code),
  constraint qr_tags_tag_code_format_check
    check (tag_code ~ '^[A-Za-z0-9]{6,32}$'),
  constraint qr_tags_status_check
    check (status in ('unassigned', 'active', 'deactivated', 'lost_mode')),
  constraint qr_tags_scan_count_check
    check (scan_count >= 0),
  -- An assigned status requires a pet; unassigned stock must not have one.
  constraint qr_tags_assignment_check
    check (
      (status = 'unassigned' and pet_id is null)
      or (status <> 'unassigned' and pet_id is not null)
    )
);

comment on table public.qr_tags is
  'Physical QR tags. pet_id is NULL for unclaimed clinic stock; binding is Phase 4.';

create index if not exists idx_qr_tags_pet_id
  on public.qr_tags (pet_id);

create index if not exists idx_qr_tags_organization_id
  on public.qr_tags (organization_id);

create index if not exists idx_qr_tags_batch_id
  on public.qr_tags (batch_id);

-- -----------------------------------------------------------------------------
-- health_records
-- -----------------------------------------------------------------------------
create table if not exists public.health_records (
  id               uuid primary key default gen_random_uuid(),
  pet_id           uuid not null references public.pets (id) on delete cascade,
  organization_id  uuid references public.organizations (id) on delete set null,

  record_type      text not null,
  title            text not null,
  description      text,
  administered_at  timestamptz,
  next_due_at      timestamptz,
  veterinarian_name text,
  document_url     text,
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now(),

  constraint health_records_title_check
    check (length(btrim(title)) > 0),
  constraint health_records_record_type_check
    check (record_type in (
      'vaccination', 'deworming', 'surgery', 'checkup',
      'diagnosis', 'prescription', 'allergy', 'weight', 'other'
    ))
);

comment on table public.health_records is
  'Clinical history. Treated as PII-adjacent: never exposed to anonymous callers.';

create index if not exists idx_health_records_pet_id
  on public.health_records (pet_id);

create index if not exists idx_health_records_organization_id
  on public.health_records (organization_id);

create index if not exists idx_health_records_next_due_at
  on public.health_records (next_due_at)
  where next_due_at is not null;

-- -----------------------------------------------------------------------------
-- verification_logs — append-only audit trail
-- -----------------------------------------------------------------------------
-- Deliberately stores ip_hash, never a raw IP address.
create table if not exists public.verification_logs (
  id              uuid primary key default gen_random_uuid(),
  pet_id          uuid references public.pets (id) on delete set null,
  qr_tag_id       uuid references public.qr_tags (id) on delete set null,
  organization_id uuid references public.organizations (id) on delete set null,

  action          text not null,
  outcome         text not null,
  similarity      numeric(6, 5),
  modality        text,
  model_id        text,
  ip_hash         text,
  user_agent      text,
  created_at      timestamptz not null default now(),

  constraint verification_logs_action_check
    check (action in ('enroll', 'identify', 'verify', 'lookup', 'tag_scan', 'alert_relay')),
  constraint verification_logs_outcome_check
    check (outcome in ('match', 'no_match', 'success', 'denied', 'error')),
  constraint verification_logs_similarity_check
    check (similarity is null or (similarity >= -1 and similarity <= 1))
);

comment on table public.verification_logs is
  'Append-only audit trail of identity operations. Stores ip_hash, never a raw IP.';

create index if not exists idx_verification_logs_pet_id
  on public.verification_logs (pet_id);

create index if not exists idx_verification_logs_created_at
  on public.verification_logs (created_at desc);

create index if not exists idx_verification_logs_action
  on public.verification_logs (action, outcome);

-- -----------------------------------------------------------------------------
-- updated_at triggers
-- -----------------------------------------------------------------------------
drop trigger if exists trg_embedding_spaces_updated_at on public.embedding_spaces;
create trigger trg_embedding_spaces_updated_at
  before update on public.embedding_spaces
  for each row execute function public.set_updated_at();

drop trigger if exists trg_organizations_updated_at on public.organizations;
create trigger trg_organizations_updated_at
  before update on public.organizations
  for each row execute function public.set_updated_at();

drop trigger if exists trg_pets_updated_at on public.pets;
create trigger trg_pets_updated_at
  before update on public.pets
  for each row execute function public.set_updated_at();

drop trigger if exists trg_biometric_embeddings_updated_at on public.biometric_embeddings;
create trigger trg_biometric_embeddings_updated_at
  before update on public.biometric_embeddings
  for each row execute function public.set_updated_at();

drop trigger if exists trg_qr_tags_updated_at on public.qr_tags;
create trigger trg_qr_tags_updated_at
  before update on public.qr_tags
  for each row execute function public.set_updated_at();

drop trigger if exists trg_health_records_updated_at on public.health_records;
create trigger trg_health_records_updated_at
  before update on public.health_records
  for each row execute function public.set_updated_at();

-- -----------------------------------------------------------------------------
-- Seed the embedding space registry
-- -----------------------------------------------------------------------------
-- Phase 1's shipping embedder is DINOv2-small (384-d, Apache-2.0).
-- MegaDescriptor-L-384 (1536-d) is pre-registered but INACTIVE: its
-- CC-BY-NC-4.0 licence makes it demo-only and it must not ship commercially.
insert into public.embedding_spaces
  (modality, model_id, model_version, dim, column_name, is_active, match_threshold, license, notes)
values
  ('nasal_planum', 'facebook/dinov2-small', '1', 384, 'embedding_384', true, 0.8600,
   'Apache-2.0', 'Phase 1 baseline. Commercially clean.'),
  ('cat_face', 'facebook/dinov2-small', '1', 384, 'embedding_384', true, 0.8600,
   'Apache-2.0', 'Phase 1 baseline. Commercially clean.'),
  ('nasal_planum', 'BVRA/MegaDescriptor-L-384', '1', 1536, 'embedding_1536', false, 0.7500,
   'CC-BY-NC-4.0', 'DEMO ONLY — non-commercial licence. Must be swapped before launch.'),
  ('cat_face', 'BVRA/MegaDescriptor-L-384', '1', 1536, 'embedding_1536', false, 0.7500,
   'CC-BY-NC-4.0', 'DEMO ONLY — non-commercial licence. Must be swapped before launch.')
on conflict on constraint embedding_spaces_identity_key do nothing;
