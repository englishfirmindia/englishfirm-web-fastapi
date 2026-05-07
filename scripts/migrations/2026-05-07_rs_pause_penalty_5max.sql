-- Migration: tighten RS pause penalty curve (5 pauses → 0 fluency)
-- Date: 2026-05-07
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- RS is a 9-second single-sentence task; the inherited RA curve
-- (max=10 pauses) was too lenient. New curve, with sentence_count=1
-- which gives s_clamped=1:
--    pauses=0 or 1 → 100
--    pauses=2     → 75
--    pauses=3     → 50
--    pauses=4     → 25
--    pauses=5     → 0
--    pauses>5     → 0
--
-- Code change: NONE — formula is already config-driven via
-- pte_speaking_scoring_config; this is a pure data update.
--
-- Idempotent — re-running just rewrites the same values.

BEGIN;

UPDATE pte_speaking_scoring_config
SET pause_penalty_max_pauses        = 5,
    pause_penalty_formula_constant  = 5
WHERE task_type = 'repeat_sentence';

COMMIT;

-- Verify:
--   SELECT task_type, pause_penalty_max_pauses, pause_penalty_formula_constant,
--          pause_penalty_sentence_clamp_min, pause_penalty_sentence_clamp_max
--   FROM pte_speaking_scoring_config WHERE task_type = 'repeat_sentence';
