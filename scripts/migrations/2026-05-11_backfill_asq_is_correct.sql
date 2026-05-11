-- Migration: backfill is_correct flag on historic ASQ attempt_answers rows.
-- Date: 2026-05-11
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Bug: update_speaking_score_in_db never persisted is_correct into
-- result_json, so every historic ASQ row has result_json->'is_correct'
-- = NULL even when content_score = 100 (correct answer). The sectional
-- and mock review screens key off is_correct to render the ✓/✗ badge,
-- so correctly-answered ASQ questions show as wrong.
--
-- Code-side fix (shipping with this migration): is_correct now flows
-- from _run_scoring → update_speaking_score_in_db → result_json on every
-- future ASQ submission.
--
-- This migration backfills the boolean for existing rows by deriving it
-- from content_score (binary 0/100 for ASQ via regex_match). Rows where
-- content_score IS NULL (5 of 109 — partial pipeline failures) are left
-- alone since there's no signal to derive from.
--
-- Idempotent — only updates rows where is_correct is still NULL.

BEGIN;

UPDATE attempt_answers
   SET result_json = jsonb_set(
       COALESCE(result_json, '{}'::jsonb),
       '{is_correct}',
       CASE WHEN content_score >= 100 THEN 'true'::jsonb ELSE 'false'::jsonb END
   )
 WHERE question_type    = 'answer_short_question'
   AND scoring_status   = 'complete'
   AND (result_json -> 'is_correct') IS NULL
   AND content_score IS NOT NULL;

COMMIT;

-- Verify:
--   SELECT (result_json->>'is_correct')::boolean AS is_correct,
--          COUNT(*)
--   FROM attempt_answers
--   WHERE question_type='answer_short_question' AND scoring_status='complete'
--   GROUP BY is_correct;
