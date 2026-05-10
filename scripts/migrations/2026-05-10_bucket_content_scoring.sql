-- Migration: bucket-based content scoring + length cap, disable sqrt curve
-- Date: 2026-05-10
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Replaces LLM holistic 0–100 content scoring with deterministic
-- per-keypoint bucket classification:
--   covered            → 1.00 × (100 / n_keypoints)
--   mentioned_explicit → 0.50 × (100 / n_keypoints)
--   mentioned_partial  → 0.25 × (100 / n_keypoints)
--   missed             → 0
--
-- Plus a length-floor cap: if word_count < length_floor_words, content
-- is capped at length_floor_cap regardless of LLM judgement.
--
-- The sqrt softening curve (content_curve_exponent) is set to NULL for
-- DI / RTL / RTS / ptea_RTS / SGD so it stops firing — bucket scoring
-- already grades each keypoint explicitly, no mid-band drift to soften.
-- Column kept on the table for rollback.
--
-- RA / RS / ASQ are unaffected.
--
-- Idempotent — safe to re-run.

BEGIN;

ALTER TABLE pte_speaking_scoring_config
    ADD COLUMN IF NOT EXISTS length_floor_words INT,
    ADD COLUMN IF NOT EXISTS length_floor_cap   INT;

UPDATE pte_speaking_scoring_config
SET length_floor_words = 50,
    length_floor_cap   = 30,
    content_curve_exponent = NULL
WHERE task_type IN (
    'describe_image',
    'retell_lecture',
    'respond_to_situation',
    'ptea_respond_situation',
    'summarize_group_discussion'
);

COMMIT;

-- Verify:
--   SELECT task_type, content_curve_exponent,
--          length_floor_words, length_floor_cap
--   FROM pte_speaking_scoring_config ORDER BY task_type;
