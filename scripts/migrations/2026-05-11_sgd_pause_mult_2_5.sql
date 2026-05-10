-- Migration: bump SGD pause-penalty multiplier 2.0 → 2.5
-- Date: 2026-05-11
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- SGD responses are summaries of a multi-speaker discussion — students
-- pause more often than in monologic tasks (DI / RTL / RTS) while they
-- recall what each speaker said. The shared 2.0 multiplier was still
-- clipping otherwise-fluent SGD responses, so SGD gets its own wider
-- tolerance: max_pauses = ceil(sentence_count × 2.5).
--
-- DI / RTL / RTS / ptea_RTS stay at 2.0.
-- RA / RS / ASQ are unaffected (mult is NULL → static columns used).
--
-- Idempotent — safe to re-run.

BEGIN;

UPDATE pte_speaking_scoring_config
SET pause_penalty_max_pauses_mult = 2.5
WHERE task_type = 'summarize_group_discussion';

COMMIT;

-- Verify:
--   SELECT task_type, pause_penalty_max_pauses_mult
--   FROM pte_speaking_scoring_config
--   WHERE pause_penalty_max_pauses_mult IS NOT NULL
--   ORDER BY task_type;
