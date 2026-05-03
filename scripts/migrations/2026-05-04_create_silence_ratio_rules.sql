-- Migration: silence_ratio_rules table for speaking-penalty stack
-- Date: 2026-05-04
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Pairs with services/speaking_scorer.py changes that:
--   1. Compute within-speech silence ratio via pydub silence detection.
--   2. Look up this table to get a fluency penalty for the silence band.
--   3. Sum WPM penalty + silence penalty as the total fluency reduction.
--   4. Spill any excess (overflow) onto pronunciation when fluency floors at 0.
--   5. Apply zero_out kill switch if either WPM or silence rule says so.
--
-- Idempotent — safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS silence_ratio_rules (
    id        SERIAL PRIMARY KEY,
    min_pct   NUMERIC NULL,
    max_pct   NUMERIC NULL,
    mode      VARCHAR(16) NOT NULL CHECK (mode IN ('zero_out','subtract')),
    penalty   NUMERIC NOT NULL DEFAULT 0,
    label     TEXT
);

INSERT INTO silence_ratio_rules (min_pct, max_pct, mode, penalty, label)
SELECT * FROM (VALUES
    (NULL::numeric, 15::numeric,  'subtract', 0::numeric,  'fluent'),
    (15::numeric,   25::numeric,  'subtract', 15::numeric, 'mildly hesitant'),
    (25::numeric,   40::numeric,  'subtract', 75::numeric, 'hesitant'),
    (40::numeric,   60::numeric,  'subtract', 85::numeric, 'very fragmented'),
    (60::numeric,   NULL::numeric,'zero_out', 0::numeric,  'mostly silent → score voided')
) AS v(min_pct, max_pct, mode, penalty, label)
WHERE NOT EXISTS (SELECT 1 FROM silence_ratio_rules);

COMMIT;

-- Bands:
--   < 15%   subtract  penalty=0   → 'fluent'
--   [15,25) subtract  penalty=15  → 'mildly hesitant'
--   [25,40) subtract  penalty=75  → 'hesitant'
--   [40,60) subtract  penalty=85  → 'very fragmented'
--   >= 60%  zero_out               → 'mostly silent → score voided' (kill switch)
