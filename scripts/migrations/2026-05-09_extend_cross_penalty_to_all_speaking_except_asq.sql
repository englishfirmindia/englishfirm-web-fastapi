-- Migration: extend RA/RS cross-penalty to DI / RL / RTS / ptea_RTS / SGD
-- Date: 2026-05-09
-- Applies to: postgres database on database-1 (ap-southeast-2)
--
-- Today only read_aloud and repeat_sentence have uses_cross_penalty=true.
-- The 5 free-speech types (DI, RL, RTS, ptea_RTS, SGD) keep the three
-- pillars independent: when fluency is zeroed by the WPM/pause rule,
-- pronunciation and content are unaffected. Reviewer feedback (Nimisha's
-- last RL/RTS attempts: fluency 0 but pron 84 / 90 untouched) showed
-- this lets dysfluent recordings keep full pronunciation credit, which
-- isn't the intent.
--
-- This migration brings the 5 types in line with RA/RS, applying the
-- same dual-curve mP override:
--
--   uses_cross_penalty                = TRUE
--   pronunciation_fluency_gate        = 20    -- branch threshold
--   pronunciation_content_threshold   = 100   -- 0–100 wide curve
--   pronunciation_content_floor       = 0.5   -- 50 % floor at C=0
--   pronunciation_content_slope       = 0.005 -- gradual ramp
--
-- mC and mF columns (cross_penalty_healthy_threshold/floor/slope) are
-- already seeded at 20 / 0.5 / 0.025 — same as RA/RS.
--
-- Net behaviour after migration:
--   F < 20  → mP = _cm(F)        (fluency-driven, ×0.5 at F=0)
--   F ≥ 20  → mP = cross_mult(C, healthy=100, floor=0.5, slope=0.005)
--   mC = _cm(min(F, P))
--   mF = _cm(min(C, P))
--
-- ASQ (answer_short_question) is intentionally excluded — its rubric
-- ignores fluency + pronunciation entirely (1.0/0/0 weights), so the
-- cross-penalty would have no observable effect on the final score.
--
-- Idempotent — safe to re-run; the WHERE clause filters by task_type.

BEGIN;

UPDATE pte_speaking_scoring_config
SET uses_cross_penalty                = TRUE,
    pronunciation_fluency_gate        = 20,
    pronunciation_content_threshold   = 100,
    pronunciation_content_floor       = 0.5,
    pronunciation_content_slope       = 0.005
WHERE task_type IN (
    'describe_image',
    'retell_lecture',
    'respond_to_situation',
    'ptea_respond_situation',
    'summarize_group_discussion'
);

COMMIT;

-- Verify all 7 fluency-formula types now share the same cross-penalty config:
--   SELECT task_type, uses_cross_penalty,
--          pronunciation_fluency_gate, pronunciation_content_threshold,
--          pronunciation_content_floor, pronunciation_content_slope
--   FROM pte_speaking_scoring_config
--   WHERE task_type <> 'answer_short_question'
--   ORDER BY task_type;
--
-- Expected: 7 rows, all uses_cross_penalty = t, fluency_gate = 20,
--           content_threshold = 100, content_floor = 0.5, content_slope = 0.005.
