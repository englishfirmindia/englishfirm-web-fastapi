-- Migration: pte_speaking_scoring_config — per-task tunables for speaking scorer
-- Date: 2026-05-07
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Replaces hardcoded WPM band / pause-penalty / cross-penalty / pause-detection
-- constants in services/speaking_scorer.py with a per-task config row. Backend
-- reads with a TTL cache and falls back to compiled defaults if the row is
-- missing or DB is unreachable, so a misconfig never breaks scoring.
--
-- Adding a new question type to the scorer is now a single INSERT here, no
-- code deploy required.
--
-- Idempotent — safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS pte_speaking_scoring_config (
    task_type                          TEXT PRIMARY KEY,
    wpm_floor                          REAL NOT NULL,
    wpm_ceiling                        REAL NOT NULL,
    wpm_plateau_low                    REAL NOT NULL,
    wpm_plateau_high                   REAL NOT NULL,
    wpm_slope_per_wpm                  REAL NOT NULL,
    wpm_peak_score                     REAL NOT NULL,
    pause_min_ms                       INT  NOT NULL,
    pause_leading_tol_ms               INT  NOT NULL,
    pause_trailing_tol_ms              INT  NOT NULL,
    silence_thresh_dbfs                REAL NOT NULL,
    content_insertion_penalty_k        REAL NOT NULL,
    pause_penalty_max_pauses           INT  NOT NULL,
    pause_penalty_sentence_clamp_min   INT  NOT NULL,
    pause_penalty_sentence_clamp_max   INT  NOT NULL,
    pause_penalty_formula_constant     INT  NOT NULL,
    cross_penalty_healthy_threshold    REAL NOT NULL,
    cross_penalty_floor_multiplier     REAL NOT NULL,
    cross_penalty_slope                REAL NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO pte_speaking_scoring_config (
    task_type,
    wpm_floor, wpm_ceiling,
    wpm_plateau_low, wpm_plateau_high,
    wpm_slope_per_wpm, wpm_peak_score,
    pause_min_ms, pause_leading_tol_ms, pause_trailing_tol_ms,
    silence_thresh_dbfs,
    content_insertion_penalty_k,
    pause_penalty_max_pauses,
    pause_penalty_sentence_clamp_min, pause_penalty_sentence_clamp_max,
    pause_penalty_formula_constant,
    cross_penalty_healthy_threshold,
    cross_penalty_floor_multiplier,
    cross_penalty_slope
) VALUES (
    'read_aloud',
    80, 270,
    130, 220,
    2.0, 100.0,
    500, 200, 200,
    -30,
    2.0,
    10,
    1, 10,
    11,
    20.0, 0.5, 0.025
) ON CONFLICT (task_type) DO NOTHING;

COMMIT;

-- Verify:
--   SELECT * FROM pte_speaking_scoring_config WHERE task_type = 'read_aloud';
