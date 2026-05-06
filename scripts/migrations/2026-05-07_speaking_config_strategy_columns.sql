-- Migration: add strategy columns to pte_speaking_scoring_config
-- Date: 2026-05-07
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Adds the columns that let one generic _score_speaking_v2 function pick
-- the right content / pronunciation / cross-penalty strategy per task
-- type. RA already runs through the v2 path; this migration just makes
-- the strategy choice explicit-in-data instead of implicit-in-code, so
-- adding new task types becomes a row INSERT rather than a code change.
--
-- Defaults are chosen so an unseeded row matches RA's behaviour exactly.
--
-- Idempotent — safe to re-run.

BEGIN;

ALTER TABLE pte_speaking_scoring_config
    ADD COLUMN IF NOT EXISTS content_method       TEXT NOT NULL DEFAULT 'lcs_k2',
    ADD COLUMN IF NOT EXISTS uses_reference_text  BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS uses_cross_penalty   BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS pronunciation_source TEXT NOT NULL DEFAULT 'azure_assessment';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_content_method'
    ) THEN
        ALTER TABLE pte_speaking_scoring_config
            ADD CONSTRAINT chk_content_method CHECK (
                content_method IN ('lcs_k2','llm_keypoints','regex_match','binary')
            );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_pronunciation_source'
    ) THEN
        ALTER TABLE pte_speaking_scoring_config
            ADD CONSTRAINT chk_pronunciation_source CHECK (
                pronunciation_source IN ('azure_assessment','azure_freeform')
            );
    END IF;
END$$;

-- Make the existing read_aloud row's strategy explicit (defaults already match,
-- so this is a no-op semantically — kept for clarity / future audits).
UPDATE pte_speaking_scoring_config
SET content_method       = 'lcs_k2',
    uses_reference_text  = TRUE,
    uses_cross_penalty   = TRUE,
    pronunciation_source = 'azure_assessment'
WHERE task_type = 'read_aloud';

COMMIT;

-- Verify:
--   SELECT task_type, content_method, uses_reference_text,
--          uses_cross_penalty, pronunciation_source
--   FROM pte_speaking_scoring_config;
