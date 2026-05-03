-- Migration: WPM-based scoring rules for Practice Read Aloud
-- Date: 2026-05-03
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Idempotent — safe to re-run. Creates the table if missing, and seeds 5 rows
-- only if the table is empty.

BEGIN;

CREATE TABLE IF NOT EXISTS wpm_scoring_rules (
    id        SERIAL PRIMARY KEY,
    min_wpm   NUMERIC NULL,
    max_wpm   NUMERIC NULL,
    mode      VARCHAR(16) NOT NULL CHECK (mode IN ('zero_out','subtract')),
    penalty   NUMERIC NOT NULL DEFAULT 0,
    label     TEXT
);

INSERT INTO wpm_scoring_rules (min_wpm, max_wpm, mode, penalty, label)
SELECT * FROM (VALUES
    (NULL::numeric, 80::numeric,  'zero_out', 0::numeric,  'very slow → score voided'),
    (80::numeric,   90::numeric,  'subtract', 25::numeric, 'slow'),
    (90::numeric,   110::numeric, 'subtract', 20::numeric, 'mild slow'),
    (110::numeric,  170::numeric, 'subtract', 0::numeric,  'ideal'),
    (170::numeric,  NULL::numeric,'subtract', 20::numeric, 'too fast')
) AS v(min_wpm, max_wpm, mode, penalty, label)
WHERE NOT EXISTS (SELECT 1 FROM wpm_scoring_rules);

COMMIT;

-- Lookup convention used by services/speaking_scorer.py:
--   half-open ranges [min_wpm, max_wpm)
--   NULL min = open lower bound; NULL max = open upper bound
--
-- Example rows:
--   min=NULL max=80   → matches WPM < 80          → zero_out (kill switch)
--   min=80   max=90   → matches 80 <= WPM < 90    → subtract 25
--   min=90   max=110  → matches 90 <= WPM < 110   → subtract 20
--   min=110  max=170  → matches 110 <= WPM < 170  → ideal (no penalty)
--   min=170  max=NULL → matches WPM >= 170        → subtract 20 (too fast)
