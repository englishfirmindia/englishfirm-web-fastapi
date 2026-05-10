-- Migration: bump pause_penalty_max_pauses_mult 1.5 → 2.0
-- Date: 2026-05-10
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- DI / RTL / RTS / ptea_RTS / SGD now allow ~2× sentence_count pauses
-- before fluency hits 0. e.g. a 9-sentence SGD response: cliff moves
-- 14 → 18 pauses, slope softens (~5.6 pts per pause beyond free zone).
--
-- Pure data update; the v2 scorer reads this column at run time. Effect
-- lands within the 5 min cfg cache TTL; no code deploy needed.
--
-- Idempotent — safe to re-run.

BEGIN;

UPDATE pte_speaking_scoring_config
SET pause_penalty_max_pauses_mult = 2.0
WHERE task_type IN (
    'describe_image',
    'retell_lecture',
    'respond_to_situation',
    'ptea_respond_situation',
    'summarize_group_discussion'
);

COMMIT;

-- Verify:
--   SELECT task_type, pause_penalty_max_pauses_mult
--   FROM pte_speaking_scoring_config ORDER BY task_type;
