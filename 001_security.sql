-- =============================================================================
-- Ella — tests/001_security.sql  (pgTAP)
-- Verifies the lockdown security posture established by migration 003 and the
-- PII-free-by-type contract from 002.
--
-- Scope of this file
--   * RLS is ENABLED and FORCED on every base table.
--   * Exactly one deny-all policy exists per table for the public web roles.
--   * Table privileges: anon/authenticated hold NO DML on any base table;
--     service_role holds table DML.
--   * Function privileges (the real API surface): anon/authenticated may EXECUTE
--     only the four public_* entry points; the internal write/match helpers are
--     service_role-only.
--   * The public contract type carries no owner PII column by construction.
--
-- Running
--   Requires the pgTAP extension. Apply 001 -> 002 -> 003 first, then this file
--   inside a transaction (pg_prove wraps each file in its own transaction).
--
-- These are catalogue/privilege assertions — they inspect pg_class, pg_policies,
-- pg_proc and the information_schema privilege views. They do not depend on row
-- data, so they are deterministic and fast.
-- =============================================================================

begin;

create extension if not exists pgtap;

select plan(54);

-- -----------------------------------------------------------------------------
-- 0. The schema objects under test exist.
-- -----------------------------------------------------------------------------
select has_table('public', 'embedding_spaces',     'embedding_spaces exists');
select has_table('public', 'organizations',        'organizations exists');
select has_table('public', 'pets',                 'pets exists');
select has_table('public', 'biometric_embeddings', 'biometric_embeddings exists');
select has_table('public', 'qr_tags',              'qr_tags exists');
select has_table('public', 'health_records',       'health_records exists');
select has_table('public', 'verification_logs',    'verification_logs exists');

-- -----------------------------------------------------------------------------
-- 1. RLS is ENABLED and FORCED on every base table.
-- -----------------------------------------------------------------------------
-- relrowsecurity = RLS enabled; relforcerowsecurity = applies to the owner too.
-- FORCE is the load-bearing bit: without it the migration owner silently
-- bypasses the deny-all policies and leaks PII.
select is(
  (select bool_and(relrowsecurity and relforcerowsecurity)
   from pg_class
   where relnamespace = 'public'::regnamespace
     and relkind = 'r'
     and relname in ('embedding_spaces','organizations','pets',
                     'biometric_embeddings','qr_tags','health_records',
                     'verification_logs')),
  true,
  'RLS is ENABLED and FORCED on all 7 base tables'
);

-- Spell it out per PII-bearing table so a failure names the offender.
select is((select relforcerowsecurity from pg_class where oid = 'public.pets'::regclass),
          true, 'pets: RLS is FORCED (owner cannot bypass)');
select is((select relforcerowsecurity from pg_class where oid = 'public.biometric_embeddings'::regclass),
          true, 'biometric_embeddings: RLS is FORCED');
select is((select relforcerowsecurity from pg_class where oid = 'public.health_records'::regclass),
          true, 'health_records: RLS is FORCED');
select is((select relforcerowsecurity from pg_class where oid = 'public.verification_logs'::regclass),
          true, 'verification_logs: RLS is FORCED');

-- -----------------------------------------------------------------------------
-- 2. Deny-all policies: exactly one per table, scoped to the web roles.
-- -----------------------------------------------------------------------------
select is(
  (select count(*)::int from pg_policies where schemaname = 'public'),
  7,
  'exactly 7 policies exist in schema public (one per table)'
);

-- Each policy denies both read (USING false) and write (WITH CHECK false) and
-- targets anon + authenticated only. service_role is intentionally absent (it
-- bypasses RLS entirely).
select is(
  (select count(*)::int
   from pg_policies
   where schemaname = 'public'
     and qual = 'false'
     and with_check = 'false'
     and roles @> array['anon','authenticated']::name[]
     and not (roles @> array['service_role']::name[])),
  7,
  'all 7 policies are deny-all (false/false) for anon+authenticated, not service_role'
);

-- Name-level presence check for the policy auditors read first.
select policy_cmd_is('public', 'pets', 'pets_no_public_access', 'ALL',
  'pets deny-all policy applies to ALL commands');

-- -----------------------------------------------------------------------------
-- 3. Table privileges — anon/authenticated have NO DML on any base table.
-- -----------------------------------------------------------------------------
-- RLS gates rows; GRANTs gate commands. This is the second, independent line of
-- defence: even if a permissive policy were added by mistake, there is no
-- table-level privilege for the web roles to exercise.

-- pets: the crown-jewel PII table.
select ok( not has_table_privilege('anon',          'public.pets', 'SELECT'), 'anon has NO SELECT on pets');
select ok( not has_table_privilege('anon',          'public.pets', 'INSERT'), 'anon has NO INSERT on pets');
select ok( not has_table_privilege('anon',          'public.pets', 'UPDATE'), 'anon has NO UPDATE on pets');
select ok( not has_table_privilege('anon',          'public.pets', 'DELETE'), 'anon has NO DELETE on pets');
select ok( not has_table_privilege('authenticated', 'public.pets', 'SELECT'), 'authenticated has NO SELECT on pets');
select ok( not has_table_privilege('authenticated', 'public.pets', 'INSERT'), 'authenticated has NO INSERT on pets');

-- The other PII-bearing tables: no anon SELECT.
select ok( not has_table_privilege('anon', 'public.biometric_embeddings', 'SELECT'),
           'anon has NO SELECT on biometric_embeddings');
select ok( not has_table_privilege('anon', 'public.health_records',       'SELECT'),
           'anon has NO SELECT on health_records');
select ok( not has_table_privilege('anon', 'public.organizations',        'SELECT'),
           'anon has NO SELECT on organizations');
select ok( not has_table_privilege('anon', 'public.qr_tags',              'SELECT'),
           'anon has NO SELECT on qr_tags');
select ok( not has_table_privilege('anon', 'public.verification_logs',    'SELECT'),
           'anon has NO SELECT on verification_logs');
select ok( not has_table_privilege('anon', 'public.embedding_spaces',     'SELECT'),
           'anon has NO SELECT on embedding_spaces');

-- service_role holds full DML (trusted backend maintenance path).
select ok( has_table_privilege('service_role', 'public.pets', 'SELECT'), 'service_role CAN SELECT pets');
select ok( has_table_privilege('service_role', 'public.pets', 'INSERT'), 'service_role CAN INSERT pets');
select ok( has_table_privilege('service_role', 'public.pets', 'UPDATE'), 'service_role CAN UPDATE pets');
select ok( has_table_privilege('service_role', 'public.pets', 'DELETE'), 'service_role CAN DELETE pets');
select ok( has_table_privilege('service_role', 'public.biometric_embeddings', 'INSERT'),
           'service_role CAN INSERT biometric_embeddings');

-- -----------------------------------------------------------------------------
-- 4a. Function privileges — the public read path IS executable by the web roles.
-- -----------------------------------------------------------------------------
select function_privs_are(
  'public', 'public_identify',
  array['vector','text','text','numeric','int','text','text'],
  'anon', array['EXECUTE'],
  'anon CAN EXECUTE public_identify'
);
select function_privs_are(
  'public', 'public_lookup_pet',
  array['text','text','text','text'],
  'anon', array['EXECUTE'],
  'anon CAN EXECUTE public_lookup_pet'
);
select function_privs_are(
  'public', 'public_lookup_tag',
  array['text','text','text'],
  'anon', array['EXECUTE'],
  'anon CAN EXECUTE public_lookup_tag'
);
select function_privs_are(
  'public', 'verify_pet',
  array['text','vector','text','text','numeric','text','text'],
  'anon', array['EXECUTE'],
  'anon CAN EXECUTE verify_pet'
);
-- authenticated shares the same public read surface.
select function_privs_are(
  'public', 'public_identify',
  array['vector','text','text','numeric','int','text','text'],
  'authenticated', array['EXECUTE'],
  'authenticated CAN EXECUTE public_identify'
);

-- -----------------------------------------------------------------------------
-- 4b. Internal write/match path — NOT executable by the web roles.
-- -----------------------------------------------------------------------------
-- An empty privilege array asserts "no privileges held".
select function_privs_are(
  'public', 'enroll_pet',
  array['text','text','text','vector','text','text','uuid','text','text','date',
        'text','text','text','text','text','text','numeric','text','int','text'],
  'anon', array[]::text[],
  'anon CANNOT EXECUTE enroll_pet'
);
select function_privs_are(
  'public', 'enroll_pet',
  array['text','text','text','vector','text','text','uuid','text','text','date',
        'text','text','text','text','text','text','numeric','text','int','text'],
  'authenticated', array[]::text[],
  'authenticated CANNOT EXECUTE enroll_pet'
);
select function_privs_are(
  'public', 'match_biometric_embedding',
  array['vector','text','text','numeric','int'],
  'anon', array[]::text[],
  'anon CANNOT EXECUTE match_biometric_embedding'
);
select function_privs_are(
  'public', 'add_pet_embedding',
  array['text','text','vector','text','numeric','text','int'],
  'anon', array[]::text[],
  'anon CANNOT EXECUTE add_pet_embedding'
);
select function_privs_are(
  'public', 'resolve_embedding_space',
  array['text','text'],
  'anon', array[]::text[],
  'anon CANNOT EXECUTE resolve_embedding_space'
);
select function_privs_are(
  'public', 'log_verification',
  array['text','text','uuid','uuid','uuid','numeric','text','text','text','text'],
  'anon', array[]::text[],
  'anon CANNOT EXECUTE log_verification'
);

-- service_role CAN execute the internal path.
select function_privs_are(
  'public', 'enroll_pet',
  array['text','text','text','vector','text','text','uuid','text','text','date',
        'text','text','text','text','text','text','numeric','text','int','text'],
  'service_role', array['EXECUTE'],
  'service_role CAN EXECUTE enroll_pet'
);
select function_privs_are(
  'public', 'match_biometric_embedding',
  array['vector','text','text','numeric','int'],
  'service_role', array['EXECUTE'],
  'service_role CAN EXECUTE match_biometric_embedding'
);

-- -----------------------------------------------------------------------------
-- 5. PII-free-by-type: the public contract type carries no owner PII column.
-- -----------------------------------------------------------------------------
-- The whole safety argument rests on public_pet_summary being UNREPRESENTABLE
-- for PII. Assert the type exists, has the expected columns, and — critically —
-- has none of the PII column names.
select has_type('public', 'public_pet_summary', 'public_pet_summary type exists');

select hasnt_column('public', 'public_pet_summary', 'owner_name',
  'public_pet_summary has NO owner_name column');
select hasnt_column('public', 'public_pet_summary', 'owner_phone',
  'public_pet_summary has NO owner_phone column');
select hasnt_column('public', 'public_pet_summary', 'owner_email',
  'public_pet_summary has NO owner_email column');
select hasnt_column('public', 'public_pet_summary', 'owner_address',
  'public_pet_summary has NO owner_address column');
select hasnt_column('public', 'public_pet_summary', 'microchip_number',
  'public_pet_summary has NO raw microchip_number column');

-- What it DOES expose: a boolean has_microchip (presence, not the number) and a
-- PII-free relay token.
select has_column('public', 'public_pet_summary', 'has_microchip',
  'public_pet_summary exposes has_microchip (boolean presence only)');
select has_column('public', 'public_pet_summary', 'alert_relay_token',
  'public_pet_summary exposes the PII-free alert_relay_token');
select col_type_is('public', 'public_pet_summary', 'has_microchip', 'boolean',
  'has_microchip is a boolean, not the 15-digit number');

select * from finish();

rollback;
