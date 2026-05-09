-- Migration: tighten RS pause penalty further (4 pauses → 0 fluency)
-- Date: 2026-05-09
-- Applies to: postgres database on database-1 (ap-southeast-2)
--
-- Drops the RS pause-penalty floor from 5 pauses → 4. With sentence_count=1
-- (RS is single-sentence) → s_clamped=1, the curve becomes:
--
--    pauses=0 or 1 → 100
--    pauses=2     → 67   (was 75)
--    pauses=3     → 33   (was 50)
--    pauses=4     → 0    (was 25)
--    pauses>=5    → 0    (was 0 at 5 already)
--
-- Net effect vs today: scores at p=2..4 drop by 8 / 17 / 25 points
-- respectively. Roughly equivalent to a 10-15% scale-down across the
-- penalty band. Applied per user request to make RS slightly stricter
-- on real-world hesitation density without going as far as a max=3 cliff.
--
-- Pure data change — _score_speaking_v2 reads both columns at runtime.
-- Idempotent: re-running just rewrites the same values.

BEGIN;

UPDATE pte_speaking_scoring_config
SET pause_penalty_max_pauses        = 4,
    pause_penalty_formula_constant  = 4
WHERE task_type = 'repeat_sentence';

COMMIT;

-- Verify:
--   SELECT task_type, pause_penalty_max_pauses, pause_penalty_formula_constant,
--          pause_penalty_sentence_clamp_min, pause_penalty_sentence_clamp_max
--   FROM pte_speaking_scoring_config WHERE task_type = 'repeat_sentence';
