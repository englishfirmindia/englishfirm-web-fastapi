-- Migration: add content_curve_exponent + seed 0.5 (sqrt) for LLM-scored types
-- Date: 2026-05-09
-- Applies to: postgres database on database-1 (ap-southeast-2)
--
-- Adds an optional softening curve to the LLM content score:
--
--   final = round(100 · (raw / 100) ** content_curve_exponent, 1)
--
-- Empirical comparison across 66 historical DI submissions
-- (services/llm_content_scoring_service.score_content_with_llm called via
-- gpt-4o-mini with the existing baseline prompt) showed:
--
--   exponent = 1.0  (today)         avg 39.3 — strict rubric, almost no
--                                    submissions land above 60.
--   exponent = 0.5  (sqrt curve)    avg 59.3 — +20 average, 0 stays 0,
--                                    100 stays 100; mid-range responses
--                                    (raw 30-50) get +20-25.
--
-- The deterministic curve outperformed every prompt-tweaking variant we
-- tested (softer wording, explicit anchors, mechanical count-based) —
-- those either net to ≈0 or *lower* the average because the LLM gets
-- stricter when given anchors. Post-LLM math is the right knob.
--
-- NULL = no transform (today's behaviour). RA, RS, ASQ keep NULL because
-- they don't use the LLM keypoints path:
--   RA / RS  → lcs_k2 content method (positional/LCS, not LLM)
--   ASQ      → binary regex match
--
-- 0.5 selected (vs. 0.7 gentler) because the strict-rubric problem is
-- pronounced — DI baseline avg 39.3 with zero submissions above 80 means
-- the LLM is heavily under-scoring partial coverage. 0.5 lifts the band
-- where the bulk of submissions sit, leaving zeros and ceilings alone.
--
-- Idempotent — safe to re-run; ADD COLUMN IF NOT EXISTS + targeted UPDATE.

BEGIN;

ALTER TABLE pte_speaking_scoring_config
    ADD COLUMN IF NOT EXISTS content_curve_exponent REAL;

UPDATE pte_speaking_scoring_config
SET content_curve_exponent = 0.5
WHERE task_type IN (
    'describe_image',
    'retell_lecture',
    'respond_to_situation',
    'ptea_respond_situation',
    'summarize_group_discussion'
);

COMMIT;

-- Verify:
--   SELECT task_type, content_method, content_curve_exponent
--   FROM pte_speaking_scoring_config
--   ORDER BY task_type;
--
-- Expected:
--   answer_short_question        binary          NULL
--   describe_image               llm_keypoints   0.5
--   ptea_respond_situation       llm_keypoints   0.5
--   read_aloud                   lcs_k2          NULL
--   repeat_sentence              lcs_k2          NULL
--   respond_to_situation         llm_keypoints   0.5
--   retell_lecture               llm_keypoints   0.5
--   summarize_group_discussion   llm_keypoints   0.5
