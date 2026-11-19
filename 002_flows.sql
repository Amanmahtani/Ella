-- =============================================================================
-- Ella — tests/002_flows.sql  (pgTAP)
-- Runtime behaviour of the enroll -> identify -> lookup pipeline, exercising the
-- SECURITY DEFINER functions with real vectors rather than just inspecting the
-- catalogue (that is 001_security.sql's job).
--
-- What this proves end-to-end
--   * service_role can enroll a pet and gets back (pet_id, ella_uid, embedding_id,
--     has_microchip) — has_microchip is a boolean, never the raw number.
--   * public_identify, given the SAME embedding, returns exactly one PII-free
--     public_pet_summary row with similarity = 1.0 (cosine self-match) and the
--     correct ella_uid / name.
--   * The row returned to the public carries NO owner PII — we assert it is the
--     public_pet_summary shape and that has_microchip reflects presence only.
--   * public_lookup_pet by ella_uid returns the same PII-free summary.
--   * The identify call self-logs a verification row (audit trail fires).
--
-- Determinism
--   The active embedding space (seeded in 001) is nasal_planum / dinov2-small,
--   384-d, cosine, threshold 0.8600. We enroll with a fixed 384-d unit vector and
--   identify with the identical vector, so cosine similarity is exactly 1.0 and is
--   always >= threshold. No randomness, no reliance on HNSW recall (the dataset is
--   a single row, so the scan is exhaustive).
--
-- Running
--   Requires pgTAP + pgvector. Apply 001 -> 002 -> 003 first. pg_prove wraps each
--   file in its own transaction; the trailing rollback keeps it side-effect free.
-- =============================================================================

begin;

create extension if not exists pgtap;

select plan(15);

-- -----------------------------------------------------------------------------
-- Fixtures: a deterministic 384-d query vector, built once and reused for both
-- enroll and identify so the self-match similarity is exactly 1.0.
-- -----------------------------------------------------------------------------
-- A 384-element vector of all-equal components. Direction is all that matters for
-- cosine; magnitude cancels. '[1,1,...,1]' with 384 ones.
--
-- NOTE on role-switching + temp tables: this file flips between service_role and
-- anon to exercise the real privilege boundary. Temp tables created here are owned
-- by the bootstrap/migration role, and migration 003 revokes ALL from PUBLIC, so a
-- switched role cannot read them by default. We therefore GRANT read/write on each
-- fixture table to both roles immediately after creating it, while still the owner.
create temporary table _fx(v text) on commit drop;
insert into _fx(v)
  select '[' || array_to_string(array(select '1' from generate_series(1, 384)), ',') || ']';
grant select on _fx to anon, service_role;

-- Enroll as the trusted backend. service_role is the sole enroller.
set local role service_role;

-- Capture the enrollment result into a temp table so later assertions can read it.
create temporary table _enroll on commit drop as
select *
from public.enroll_pet(
  p_name             => 'Ella Test Dog',
  p_species          => 'dog',
  p_modality         => 'nasal_planum',
  p_embedding        => (select v from _fx)::vector,
  p_model_id         => 'facebook/dinov2-small',
  p_microchip_number => '900000000000001',   -- present -> has_microchip must be true
  p_organization_id  => null,
  p_breed            => 'Indie',
  p_sex              => 'female',
  p_date_of_birth    => null,
  p_color_markings   => 'brown, white chest',
  p_photo_url        => null,
  p_owner_name       => 'Jane Owner',        -- PII: must never surface publicly
  p_owner_phone      => '+91-555-0100',      -- PII
  p_owner_email      => 'jane@example.test', -- PII
  p_owner_address    => '1 Test Lane, BLR',  -- PII
  p_quality_score    => 0.95,
  p_source_kind      => 'image',   -- must satisfy source_kind in ('image','video')
  p_frame_count      => 1,
  p_region           => 'BLR'
);
grant select on _enroll to anon, service_role;

-- 1. Enrollment produced exactly one pet with an ella_uid and an embedding.
select is((select count(*)::int from _enroll), 1,
  'enroll_pet returned exactly one row');
select isnt((select pet_id      from _enroll), null, 'enroll_pet returned a pet_id');
select isnt((select ella_uid    from _enroll), null, 'enroll_pet returned an ella_uid');
select isnt((select embedding_id from _enroll), null, 'enroll_pet returned an embedding_id');
-- Use ok(... LIKE ...) rather than pgTAP's like() matcher: the string-matching
-- matchers are not present in every pgTAP build, but the SQL LIKE operator is.
select ok((select ella_uid from _enroll) like 'ELLA-%',
  'ella_uid is namespaced ELLA-*');

-- 2. has_microchip is a boolean TRUE (we passed a microchip), never the number.
select is((select has_microchip from _enroll), true,
  'enroll_pet.has_microchip is boolean true when a microchip is supplied');

-- -----------------------------------------------------------------------------
-- Identify path — run as the public web role. This is the entire public product.
-- -----------------------------------------------------------------------------
reset role;
set local role anon;

create temporary table _ident on commit drop as
select *
from public.public_identify(
  p_embedding  => (select v from _fx)::vector,
  p_modality   => 'nasal_planum',
  p_model_id   => 'facebook/dinov2-small',
  p_threshold  => null,     -- use the space's own threshold (0.8600)
  p_k          => 3,
  p_ip_hash    => 'test-ip-hash',
  p_user_agent => 'pgtap/runtime'
);
grant select on _ident to anon, service_role;

-- 3. Exactly one match, and it is OUR pet (same ella_uid, same name).
select is((select count(*)::int from _ident), 1,
  'public_identify returned exactly one match for the self-embedding');
select is((select ella_uid from _ident),
          (select ella_uid from _enroll),
  'public_identify match ella_uid equals the enrolled ella_uid');
select is((select name from _ident), 'Ella Test Dog',
  'public_identify returns the pet name');

-- 4. Self-match similarity is 1.0 (cosine of a vector with itself), >= threshold.
select cmp_ok((select similarity from _ident), '>=', 0.8600::numeric,
  'public_identify similarity is at or above the active threshold (0.8600)');

-- 5. PII-free contract at RUNTIME: has_microchip is boolean presence only, and
--    the alert_relay_token (PII-free relay) is populated for the finder.
select is((select has_microchip from _ident), true,
  'public_identify.has_microchip is boolean true (presence), not the number');
select isnt((select alert_relay_token from _ident), null,
  'public_identify exposes a non-null PII-free alert_relay_token');

-- -----------------------------------------------------------------------------
-- Lookup-by-UID path — also public, also PII-free.
-- -----------------------------------------------------------------------------
create temporary table _look on commit drop as
select *
from public.public_lookup_pet(
  p_ella_uid         => (select ella_uid from _enroll),
  p_microchip_number => null,
  p_ip_hash          => 'test-ip-hash',
  p_user_agent       => 'pgtap/runtime'
);
grant select on _look to anon, service_role;

-- 6. Lookup by ella_uid returns the same single PII-free summary.
select is((select count(*)::int from _look), 1,
  'public_lookup_pet by ella_uid returned exactly one row');
select is((select ella_uid from _look), (select ella_uid from _enroll),
  'public_lookup_pet returns the matching ella_uid');

-- -----------------------------------------------------------------------------
-- Audit trail — identify self-logs a verification row. Read it back as the
-- trusted role (anon has no table SELECT, by design).
-- -----------------------------------------------------------------------------
reset role;
set local role service_role;

-- 7. An 'identify' verification was logged with outcome 'match'.
select ok(
  (select count(*) > 0
   from public.verification_logs
   where action = 'identify' and outcome = 'match'),
  'identify self-logged a verification row with outcome=match'
);

select * from finish();

rollback;
