-- Migration: revert LLM content scoring to holistic 0–100 + sqrt curve,
--            keep the length-floor cap from the bucket-scoring iteration.
-- Date: 2026-05-10
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Why:
--   Bucket classification (covered / mentioned_explicit / mentioned_partial /
--   missed) was over-strict on responses that hit all keywords but lacked
--   full-sentence explanation — e.g. Nimisha's last RTS scored 62.5
--   (PTE 69) despite covering every key point.
--
-- What this migration does:
--   1. Re-enables the sqrt softening curve for DI / RTL / RTS / ptea_RTS /
--      SGD by setting content_curve_exponent = 0.5.
--   2. Keeps length_floor_words = 50, length_floor_cap = 30 unchanged.
--      The cap still prevents short keyword-drop responses from being
--      lifted into the high band by the sqrt curve.
--
-- Code-side change (commit shipping with this migration) reverts
-- score_content_with_llm() to its pre-bucket implementation: holistic
-- 0–100 with one-sentence reasoning, no per-keypoint classifications.
--
-- RA / RS / ASQ are unaffected (content_method != 'llm_keypoints').
--
-- Idempotent — safe to re-run.

BEGIN;

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
--   SELECT task_type, content_curve_exponent,
--          length_floor_words, length_floor_cap
--   FROM pte_speaking_scoring_config ORDER BY task_type;
