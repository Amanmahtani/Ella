-- =============================================================================
-- Ella — 002_ella_functions.sql
-- Matching, atomic enrolment, and the PII-free public accessors.
--
-- Security model
--   * Every public_* function is SECURITY DEFINER with a pinned search_path and
--     returns `public.public_pet_summary` — a composite type that contains NO
--     PII column at all. Owner phone, email, address and the raw microchip
--     number are not merely unselected, they are UNREPRESENTABLE in the return
--     value. That is safety by type rather than by developer discipline.
--   * `has_microchip` is a boolean. The 15-digit number itself is never
--     returned to an unauthenticated caller.
--   * `enroll_pet` and `match_biometric_embedding` are internal. 003 grants
--     EXECUTE on them to service_role only; the API gateway is the sole
--     enroller and performs its own authorization.
--   * Idempotent: the type is dropped CASCADE and rebuilt, so re-applying this
--     file is safe even after a signature change.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- The public contract. Dropping CASCADE also drops the dependent functions,
-- which are recreated below — this is what makes the file re-appliable.
-- -----------------------------------------------------------------------------
drop type if exists public.public_pet_summary cascade;

create type public.public_pet_summary as (
  ella_uid          text,
  name              text,
  species           text,
  breed             text,
  color_markings    text,
  photo_url         text,
  has_microchip     boolean,     -- never the number itself
  is_lost           boolean,
  lost_since        timestamptz,
  is_vet_verified   boolean,
  alert_relay_token text,        -- PII-free channel to reach the owner
  similarity        numeric      -- NULL unless this row came from a biometric match
);

comment on type public.public_pet_summary is
  'Public-safe pet projection. Contains no owner PII and no raw microchip number by construction.';

-- -----------------------------------------------------------------------------
-- Internal helpers
-- -----------------------------------------------------------------------------

-- Resolves which embedding space to use. With p_model_id NULL the active space
-- for the modality is chosen; passing an explicit model_id is a deliberate
-- opt-in and may select an inactive (e.g. demo-licensed) space.
create or replace function public.resolve_embedding_space(
  p_modality text,
  p_model_id text default null
)
returns public.embedding_spaces
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select es.*
  from public.embedding_spaces es
  where es.modality = p_modality
    and (p_model_id is null or es.model_id = p_model_id)
    and (p_model_id is not null or es.is_active = true)
  order by es.is_active desc, es.model_version desc
  limit 1;
$$;

comment on function public.resolve_embedding_space(text, text) is
  'Returns the embedding_spaces row to use for a modality, defaulting to the active model.';

create or replace function public.log_verification(
  p_action          text,
  p_outcome         text,
  p_pet_id          uuid default null,
  p_qr_tag_id       uuid default null,
  p_organization_id uuid default null,
  p_similarity      numeric default null,
  p_modality        text default null,
  p_model_id        text default null,
  p_ip_hash         text default null,
  p_user_agent      text default null
)
returns void
language sql
volatile
security definer
set search_path = public, pg_temp
as $$
  insert into public.verification_logs
    (action, outcome, pet_id, qr_tag_id, organization_id,
     similarity, modality, model_id, ip_hash, user_agent)
  values
    (p_action, p_outcome, p_pet_id, p_qr_tag_id, p_organization_id,
     p_similarity, p_modality, p_model_id, p_ip_hash, p_user_agent);
$$;

comment on function public.log_verification is
  'Appends an audit row. ip_hash is expected to already be hashed by the caller; never pass a raw IP.';

-- -----------------------------------------------------------------------------
-- match_biometric_embedding — internal k-NN, strictly scoped
-- -----------------------------------------------------------------------------
-- Scoping by modality + model_id makes cross-model comparison structurally
-- impossible: a 384-d DINOv2 vector can never be ranked against a 1536-d
-- MegaDescriptor vector, because they live in different columns AND the
-- modality/model filter excludes them anyway.
create or replace function public.match_biometric_embedding(
  p_query     vector,
  p_modality  text,
  p_model_id  text default null,
  p_threshold numeric default null,
  p_k         int default 5
)
returns table (
  pet_id       uuid,
  embedding_id uuid,
  similarity   numeric
)
language plpgsql
stable
security definer
set search_path = public, pg_temp
as $$
declare
  v_space     public.embedding_spaces;
  v_dim       int := vector_dims(p_query);
  v_threshold numeric;
  v_k         int := greatest(1, least(coalesce(p_k, 5), 100));
  v_q384      vector(384);
  v_q1536     vector(1536);
begin
  v_space := public.resolve_embedding_space(p_modality, p_model_id);

  if v_space.id is null then
    raise exception
      'match_biometric_embedding: no embedding space registered for modality=%, model_id=%',
      p_modality, coalesce(p_model_id, '<active>')
      using errcode = 'no_data_found';
  end if;

  if v_space.dim <> v_dim then
    raise exception
      'match_biometric_embedding: query vector has % dimensions but embedding space % expects %',
      v_dim, v_space.model_id, v_space.dim
      using errcode = 'data_exception';
  end if;

  v_threshold := coalesce(p_threshold, v_space.match_threshold);

  if v_dim = 384 then
    v_q384 := p_query::vector(384);
    return query
      select be.pet_id,
             be.id,
             round((1 - (be.embedding_384 <=> v_q384))::numeric, 5)
      from public.biometric_embeddings be
      where be.embedding_384 is not null
        and be.is_active = true
        and be.modality = v_space.modality
        and be.model_id = v_space.model_id
        and (1 - (be.embedding_384 <=> v_q384)) >= v_threshold
      order by be.embedding_384 <=> v_q384
      limit v_k;

  elsif v_dim = 1536 then
    v_q1536 := p_query::vector(1536);
    return query
      select be.pet_id,
             be.id,
             round((1 - (be.embedding_1536 <=> v_q1536))::numeric, 5)
      from public.biometric_embeddings be
      where be.embedding_1536 is not null
        and be.is_active = true
        and be.modality = v_space.modality
        and be.model_id = v_space.model_id
        and (1 - (be.embedding_1536 <=> v_q1536)) >= v_threshold
      order by be.embedding_1536 <=> v_q1536
      limit v_k;

  else
    raise exception
      'match_biometric_embedding: unsupported query dimension % (expected 384 or 1536)', v_dim
      using errcode = 'data_exception';
  end if;
end;
$$;

comment on function public.match_biometric_embedding is
  'Internal cosine k-NN scoped by modality + model_id. Service-role only; returns pet ids, not PII.';

-- -----------------------------------------------------------------------------
-- enroll_pet — atomic pet + embedding (+ optional microchip) in one transaction
-- -----------------------------------------------------------------------------
create or replace function public.enroll_pet(
  p_name             text,
  p_species          text,
  p_modality         text,
  p_embedding        vector,
  p_model_id         text default null,
  p_microchip_number text default null,
  p_organization_id  uuid default null,
  p_breed            text default null,
  p_sex              text default 'unknown',
  p_date_of_birth    date default null,
  p_color_markings   text default null,
  p_photo_url        text default null,
  p_owner_name       text default null,
  p_owner_phone      text default null,
  p_owner_email      text default null,
  p_owner_address    text default null,
  p_quality_score    numeric default null,
  p_source_kind      text default null,
  p_frame_count      int default null,
  p_region           text default 'BLR'
)
returns table (
  pet_id        uuid,
  ella_uid      text,
  embedding_id  uuid,
  has_microchip boolean
)
language plpgsql
volatile
security definer
set search_path = public, pg_temp
as $$
declare
  v_space        public.embedding_spaces;
  v_dim          int := vector_dims(p_embedding);
  v_pet_id       uuid;
  v_ella_uid     text;
  v_embedding_id uuid;
begin
  v_space := public.resolve_embedding_space(p_modality, p_model_id);

  if v_space.id is null then
    raise exception
      'enroll_pet: no embedding space registered for modality=%, model_id=%',
      p_modality, coalesce(p_model_id, '<active>')
      using errcode = 'no_data_found';
  end if;

  if v_space.dim <> v_dim then
    raise exception
      'enroll_pet: embedding has % dimensions but embedding space % expects %',
      v_dim, v_space.model_id, v_space.dim
      using errcode = 'data_exception';
  end if;

  -- Validate the microchip here so the caller gets an actionable message rather
  -- than a raw CHECK-constraint violation.
  if p_microchip_number is not null and p_microchip_number !~ '^[0-9]{15}$' then
    raise exception
      'enroll_pet: microchip_number must be exactly 15 numeric digits (ISO 11784/11785)'
      using errcode = 'invalid_parameter_value',
            hint = 'Strip spaces and hyphens before submitting.';
  end if;

  -- Clear, non-leaky duplicate error: states the fact without disclosing which
  -- pet or owner already holds the chip.
  if p_microchip_number is not null
     and exists (select 1 from public.pets p where p.microchip_number = p_microchip_number) then
    raise exception
      'enroll_pet: microchip_number is already registered in the Ella registry'
      using errcode = 'unique_violation',
            hint = 'Look the animal up by microchip instead of enrolling a duplicate.';
  end if;

  insert into public.pets (
    ella_uid, organization_id, name, species, breed, sex, date_of_birth,
    color_markings, photo_url, microchip_number, microchip_registered_at,
    owner_name, owner_phone, owner_email, owner_address
  )
  values (
    public.generate_ella_uid(p_region), p_organization_id, p_name, p_species,
    p_breed, coalesce(p_sex, 'unknown'), p_date_of_birth,
    p_color_markings, p_photo_url, p_microchip_number,
    case when p_microchip_number is not null then now() end,
    p_owner_name, p_owner_phone, p_owner_email, p_owner_address
  )
  returning public.pets.id, public.pets.ella_uid
  into v_pet_id, v_ella_uid;

  insert into public.biometric_embeddings (
    pet_id, modality, model_id, model_version, embedding_dim,
    embedding_384, embedding_1536,
    quality_score, source_kind, frame_count
  )
  values (
    v_pet_id, v_space.modality, v_space.model_id, v_space.model_version, v_dim,
    case when v_dim = 384  then p_embedding::vector(384)  end,
    case when v_dim = 1536 then p_embedding::vector(1536) end,
    p_quality_score, p_source_kind, p_frame_count
  )
  returning public.biometric_embeddings.id into v_embedding_id;

  perform public.log_verification(
    p_action          => 'enroll',
    p_outcome         => 'success',
    p_pet_id          => v_pet_id,
    p_organization_id => p_organization_id,
    p_modality        => v_space.modality,
    p_model_id        => v_space.model_id
  );

  return query select v_pet_id, v_ella_uid, v_embedding_id, (p_microchip_number is not null);
end;
$$;

comment on function public.enroll_pet is
  'Atomically creates a pet, its first biometric embedding and the optional microchip link.';

-- -----------------------------------------------------------------------------
-- add_pet_embedding — append another capture to an existing pet
-- -----------------------------------------------------------------------------
create or replace function public.add_pet_embedding(
  p_ella_uid      text,
  p_modality      text,
  p_embedding     vector,
  p_model_id      text default null,
  p_quality_score numeric default null,
  p_source_kind   text default null,
  p_frame_count   int default null
)
returns table (
  pet_id       uuid,
  embedding_id uuid
)
language plpgsql
volatile
security definer
set search_path = public, pg_temp
as $$
declare
  v_space        public.embedding_spaces;
  v_dim          int := vector_dims(p_embedding);
  v_pet_id       uuid;
  v_embedding_id uuid;
begin
  select p.id into v_pet_id from public.pets p where p.ella_uid = p_ella_uid;

  if v_pet_id is null then
    raise exception 'add_pet_embedding: no pet found for ella_uid %', p_ella_uid
      using errcode = 'no_data_found';
  end if;

  v_space := public.resolve_embedding_space(p_modality, p_model_id);

  if v_space.id is null then
    raise exception
      'add_pet_embedding: no embedding space registered for modality=%, model_id=%',
      p_modality, coalesce(p_model_id, '<active>')
      using errcode = 'no_data_found';
  end if;

  if v_space.dim <> v_dim then
    raise exception
      'add_pet_embedding: embedding has % dimensions but embedding space % expects %',
      v_dim, v_space.model_id, v_space.dim
      using errcode = 'data_exception';
  end if;

  insert into public.biometric_embeddings (
    pet_id, modality, model_id, model_version, embedding_dim,
    embedding_384, embedding_1536, quality_score, source_kind, frame_count
  )
  values (
    v_pet_id, v_space.modality, v_space.model_id, v_space.model_version, v_dim,
    case when v_dim = 384  then p_embedding::vector(384)  end,
    case when v_dim = 1536 then p_embedding::vector(1536) end,
    p_quality_score, p_source_kind, p_frame_count
  )
  returning public.biometric_embeddings.id into v_embedding_id;

  return query select v_pet_id, v_embedding_id;
end;
$$;

comment on function public.add_pet_embedding is
  'Appends an additional biometric capture to an existing pet, keeping enrolment multi-sample.';

-- -----------------------------------------------------------------------------
-- public_identify — 1:N lost-pet scan. PII-free by construction.
-- -----------------------------------------------------------------------------
create or replace function public.public_identify(
  p_embedding  vector,
  p_modality   text,
  p_model_id   text default null,
  p_threshold  numeric default null,
  p_k          int default 3,
  p_ip_hash    text default null,
  p_user_agent text default null
)
returns setof public.public_pet_summary
language plpgsql
volatile
security definer
set search_path = public, pg_temp
as $$
declare
  v_space    public.embedding_spaces;
  v_best_pet uuid;
  v_best_sim numeric;
  v_rows     int := 0;
begin
  v_space := public.resolve_embedding_space(p_modality, p_model_id);

  for v_best_pet, v_best_sim in
    select m.pet_id, max(m.similarity)
    from public.match_biometric_embedding(p_embedding, p_modality, p_model_id, p_threshold, p_k) m
    group by m.pet_id
    order by max(m.similarity) desc
  loop
    v_rows := v_rows + 1;

    return query
      select p.ella_uid,
             p.name,
             p.species,
             p.breed,
             p.color_markings,
             p.photo_url,
             (p.microchip_number is not null),
             p.is_lost,
             p.lost_since,
             p.is_vet_verified,
             p.alert_relay_token,
             v_best_sim
      from public.pets p
      where p.id = v_best_pet;
  end loop;

  perform public.log_verification(
    p_action     => 'identify',
    p_outcome    => case when v_rows > 0 then 'match' else 'no_match' end,
    p_pet_id     => v_best_pet,
    p_similarity => v_best_sim,
    p_modality   => coalesce(v_space.modality, p_modality),
    p_model_id   => v_space.model_id,
    p_ip_hash    => p_ip_hash,
    p_user_agent => p_user_agent
  );

  return;
end;
$$;

comment on function public.public_identify is
  'Anonymous 1:N biometric identification. Returns only public-safe fields plus an alert relay token.';

-- -----------------------------------------------------------------------------
-- public_lookup_pet — by ELLA UID or by microchip number
-- -----------------------------------------------------------------------------
-- Accepts a microchip number as INPUT (the finder read it off a scanner) but
-- never returns one. Exactly one identifier must be supplied.
create or replace function public.public_lookup_pet(
  p_ella_uid         text default null,
  p_microchip_number text default null,
  p_ip_hash          text default null,
  p_user_agent       text default null
)
returns setof public.public_pet_summary
language plpgsql
volatile
security definer
set search_path = public, pg_temp
as $$
declare
  v_pet_id uuid;
begin
  if (p_ella_uid is null) = (p_microchip_number is null) then
    raise exception
      'public_lookup_pet: supply exactly one of ella_uid or microchip_number'
      using errcode = 'invalid_parameter_value';
  end if;

  if p_microchip_number is not null and p_microchip_number !~ '^[0-9]{15}$' then
    raise exception
      'public_lookup_pet: microchip_number must be exactly 15 numeric digits (ISO 11784/11785)'
      using errcode = 'invalid_parameter_value';
  end if;

  select p.id into v_pet_id
  from public.pets p
  where (p_ella_uid is not null and p.ella_uid = p_ella_uid)
     or (p_microchip_number is not null and p.microchip_number = p_microchip_number);

  perform public.log_verification(
    p_action     => 'lookup',
    p_outcome    => case when v_pet_id is null then 'no_match' else 'success' end,
    p_pet_id     => v_pet_id,
    p_ip_hash    => p_ip_hash,
    p_user_agent => p_user_agent
  );

  return query
    select p.ella_uid,
           p.name,
           p.species,
           p.breed,
           p.color_markings,
           p.photo_url,
           (p.microchip_number is not null),
           p.is_lost,
           p.lost_since,
           p.is_vet_verified,
           p.alert_relay_token,
           null::numeric
    from public.pets p
    where p.id = v_pet_id;
end;
$$;

comment on function public.public_lookup_pet is
  'Public lookup by ELLA UID or microchip number. The microchip number is accepted as input, never returned.';

-- -----------------------------------------------------------------------------
-- public_lookup_tag — the /t/{tag_code} scan path
-- -----------------------------------------------------------------------------
create or replace function public.public_lookup_tag(
  p_tag_code   text,
  p_ip_hash    text default null,
  p_user_agent text default null
)
returns setof public.public_pet_summary
language plpgsql
volatile
security definer
set search_path = public, pg_temp
as $$
declare
  v_tag_id uuid;
  v_pet_id uuid;
  v_status text;
begin
  select t.id, t.pet_id, t.status
  into v_tag_id, v_pet_id, v_status
  from public.qr_tags t
  where t.tag_code = p_tag_code;

  if v_tag_id is not null then
    update public.qr_tags t
    set scan_count = t.scan_count + 1,
        last_scanned_at = now()
    where t.id = v_tag_id;
  end if;

  perform public.log_verification(
    p_action     => 'tag_scan',
    p_outcome    => case
                      when v_tag_id is null then 'no_match'
                      when v_pet_id is null or v_status = 'deactivated' then 'denied'
                      else 'success'
                    end,
    p_pet_id     => v_pet_id,
    p_qr_tag_id  => v_tag_id,
    p_ip_hash    => p_ip_hash,
    p_user_agent => p_user_agent
  );

  -- Unassigned clinic stock and deactivated tags resolve to no pet.
  if v_pet_id is null or v_status = 'deactivated' then
    return;
  end if;

  return query
    select p.ella_uid,
           p.name,
           p.species,
           p.breed,
           p.color_markings,
           p.photo_url,
           (p.microchip_number is not null),
           p.is_lost,
           p.lost_since,
           p.is_vet_verified,
           p.alert_relay_token,
           null::numeric
    from public.pets p
    where p.id = v_pet_id;
end;
$$;

comment on function public.public_lookup_tag is
  'Resolves a scanned QR tag code to a public-safe pet record and increments the scan counter.';

-- -----------------------------------------------------------------------------
-- verify_pet — 1:1 confirmation that a capture belongs to a claimed identity
-- -----------------------------------------------------------------------------
create or replace function public.verify_pet(
  p_ella_uid   text,
  p_embedding  vector,
  p_modality   text,
  p_model_id   text default null,
  p_threshold  numeric default null,
  p_ip_hash    text default null,
  p_user_agent text default null
)
returns table (
  ella_uid    text,
  is_verified boolean,
  similarity  numeric,
  threshold   numeric
)
language plpgsql
volatile
security definer
set search_path = public, pg_temp
as $$
declare
  v_space      public.embedding_spaces;
  v_dim        int := vector_dims(p_embedding);
  v_pet_id     uuid;
  v_threshold  numeric;
  v_similarity numeric;
  v_q384       vector(384);
  v_q1536      vector(1536);
begin
  select p.id into v_pet_id from public.pets p where p.ella_uid = p_ella_uid;

  if v_pet_id is null then
    raise exception 'verify_pet: no pet found for ella_uid %', p_ella_uid
      using errcode = 'no_data_found';
  end if;

  v_space := public.resolve_embedding_space(p_modality, p_model_id);

  if v_space.id is null then
    raise exception
      'verify_pet: no embedding space registered for modality=%, model_id=%',
      p_modality, coalesce(p_model_id, '<active>')
      using errcode = 'no_data_found';
  end if;

  if v_space.dim <> v_dim then
    raise exception
      'verify_pet: embedding has % dimensions but embedding space % expects %',
      v_dim, v_space.model_id, v_space.dim
      using errcode = 'data_exception';
  end if;

  v_threshold := coalesce(p_threshold, v_space.match_threshold);

  if v_dim = 384 then
    v_q384 := p_embedding::vector(384);
    select max(round((1 - (be.embedding_384 <=> v_q384))::numeric, 5))
    into v_similarity
    from public.biometric_embeddings be
    where be.pet_id = v_pet_id
      and be.is_active = true
      and be.modality = v_space.modality
      and be.model_id = v_space.model_id
      and be.embedding_384 is not null;
  else
    v_q1536 := p_embedding::vector(1536);
    select max(round((1 - (be.embedding_1536 <=> v_q1536))::numeric, 5))
    into v_similarity
    from public.biometric_embeddings be
    where be.pet_id = v_pet_id
      and be.is_active = true
      and be.modality = v_space.modality
      and be.model_id = v_space.model_id
      and be.embedding_1536 is not null;
  end if;

  perform public.log_verification(
    p_action     => 'verify',
    p_outcome    => case
                      when v_similarity is null then 'no_match'
                      when v_similarity >= v_threshold then 'match'
                      else 'no_match'
                    end,
    p_pet_id     => v_pet_id,
    p_similarity => v_similarity,
    p_modality   => v_space.modality,
    p_model_id   => v_space.model_id,
    p_ip_hash    => p_ip_hash,
    p_user_agent => p_user_agent
  );

  return query
    select p_ella_uid,
           coalesce(v_similarity >= v_threshold, false),
           v_similarity,
           v_threshold;
end;
$$;

comment on function public.verify_pet is
  '1:1 verification that a fresh capture matches the pet claimed by ella_uid. Returns no PII.';
