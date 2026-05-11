-- Migration: add sectional_test_questions table
-- Date: 2026-05-12
-- Applies to: postgres database on database-1 (ap-southeast-2)
--
-- Locks the question set for each (module, test_number) sectional slot.
-- Source of truth replacing the per-call `random.sample` selection in the
-- four sectional services. Seeded once via
-- scripts/migrations/2026-05-12_seed_sectional_test_questions.py and never
-- re-randomised — letting users redo a sectional always yields the same
-- canonical questions.
--
-- Catalog size: 4 modules × 40 test_numbers = 160 rows.
--
-- Idempotent — IF NOT EXISTS makes re-running safe.

BEGIN;

CREATE TABLE IF NOT EXISTS sectional_test_questions (
    module        TEXT       NOT NULL,
    test_number   INTEGER    NOT NULL,
    question_ids  INTEGER[]  NOT NULL,
    seeded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (module, test_number)
);

COMMIT;

-- Verify:
--   \d sectional_test_questions
--   SELECT module, count(*) FROM sectional_test_questions GROUP BY module;
