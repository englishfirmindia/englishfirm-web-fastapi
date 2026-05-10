-- Migration: dynamic max_pauses scaled by sentence count
-- Date: 2026-05-10
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Adds `pause_penalty_max_pauses_mult` so DI / RTL / RTS / ptea_RTS / SGD
-- can derive their pause-cliff from the user's actual sentence count
-- (1.5× sentences) instead of the static max=10. Long-form free-speech
-- tasks naturally have more sentences and more natural pauses; the static
-- ceiling under-rewards normal long responses.
--
-- NULL on the column → use the existing static pause_penalty_max_pauses
-- and pause_penalty_formula_constant columns, so RA / RS / ASQ behaviour
-- is unchanged.
--
-- Idempotent — safe to re-run.

BEGIN;

ALTER TABLE pte_speaking_scoring_config
    ADD COLUMN IF NOT EXISTS pause_penalty_max_pauses_mult REAL;

UPDATE pte_speaking_scoring_config
SET pause_penalty_max_pauses_mult = 1.5
WHERE task_type IN (
    'describe_image',
    'retell_lecture',
    'respond_to_situation',
    'ptea_respond_situation',
    'summarize_group_discussion'
);

COMMIT;

-- Verify:
--   SELECT task_type, pause_penalty_max_pauses, pause_penalty_formula_constant,
--          pause_penalty_max_pauses_mult
--   FROM pte_speaking_scoring_config ORDER BY task_type;
