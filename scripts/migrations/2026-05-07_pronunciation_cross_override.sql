-- Migration: pronunciation cross-penalty override (gated by fluency)
-- Date: 2026-05-07
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Adds 4 nullable columns letting pronunciation use a wider 0–100
-- content-driven cross-penalty curve when fluency is healthy, while
-- falling back to today's fluency-driven 0–20 rule when fluency is low.
--
-- All NULLs = override disabled → today's symmetric rule. The seed UPDATE
-- at the bottom opts read_aloud in.
--
-- Idempotent — safe to re-run.

BEGIN;

ALTER TABLE pte_speaking_scoring_config
    ADD COLUMN IF NOT EXISTS pronunciation_fluency_gate      REAL,
    ADD COLUMN IF NOT EXISTS pronunciation_content_threshold REAL,
    ADD COLUMN IF NOT EXISTS pronunciation_content_floor     REAL,
    ADD COLUMN IF NOT EXISTS pronunciation_content_slope     REAL;

-- Seed RA: when fluency >= 20, pronunciation is damped by the 0–100
-- content curve (max 50% drop at content=0). When fluency < 20, fall
-- back to today's fluency-driven mP (unchanged).
UPDATE pte_speaking_scoring_config
SET pronunciation_fluency_gate       = 20,
    pronunciation_content_threshold  = 100,
    pronunciation_content_floor      = 0.5,
    pronunciation_content_slope      = 0.005
WHERE task_type = 'read_aloud';

COMMIT;

-- Verify:
--   SELECT task_type, pronunciation_fluency_gate, pronunciation_content_threshold,
--          pronunciation_content_floor, pronunciation_content_slope
--   FROM pte_speaking_scoring_config WHERE task_type = 'read_aloud';
