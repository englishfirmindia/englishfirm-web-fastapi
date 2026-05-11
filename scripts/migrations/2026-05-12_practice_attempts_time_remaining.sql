-- Migration: add time_remaining_seconds column to practice_attempts
-- Date: 2026-05-12
-- Applies to: postgres database on database-1 (ap-southeast-2)
--
-- Persists the on-screen sectional exam countdown so a tab-close / refresh
-- / device switch resumes the timer at the exact value the user last saw.
-- Written by every speaking sectional submit; read by the resume endpoint.
--
-- NULL means "use the module's fresh-start default" — for speaking
-- sectional that's 40 min (2400 s). The fresh-start path leaves the row
-- with NULL on attempt creation; the first submit writes the value.
--
-- Idempotent — IF NOT EXISTS makes re-running safe.

BEGIN;

ALTER TABLE practice_attempts
    ADD COLUMN IF NOT EXISTS time_remaining_seconds INTEGER;

COMMIT;

-- Verify:
--   \d practice_attempts
--   SELECT count(*) FILTER (WHERE time_remaining_seconds IS NULL) AS unset,
--          count(*) FILTER (WHERE time_remaining_seconds IS NOT NULL) AS set
--   FROM practice_attempts;
