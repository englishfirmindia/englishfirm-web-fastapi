-- Migration: seed pte_speaking_scoring_config for repeat_sentence
-- Date: 2026-05-07
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- RS already routes through _score_speaking_v2 since rev 32 but had no
-- row, so it was inheriting RA's compiled fallback (no pronunciation
-- override). This INSERTs an RS row that mirrors RA exactly, including
-- the new pronunciation cross-override columns. RS scoring becomes
-- RDS-tunable independently from RA from this point on.
--
-- Idempotent: ON CONFLICT DO NOTHING so re-runs are safe. Tunables can
-- still be changed afterwards via UPDATE.

BEGIN;

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
    cross_penalty_slope,
    content_method, uses_reference_text, uses_cross_penalty, pronunciation_source,
    pronunciation_fluency_gate, pronunciation_content_threshold,
    pronunciation_content_floor, pronunciation_content_slope
)
SELECT
    'repeat_sentence',
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
    cross_penalty_slope,
    content_method, uses_reference_text, uses_cross_penalty, pronunciation_source,
    pronunciation_fluency_gate, pronunciation_content_threshold,
    pronunciation_content_floor, pronunciation_content_slope
FROM pte_speaking_scoring_config WHERE task_type = 'read_aloud'
ON CONFLICT (task_type) DO NOTHING;

COMMIT;

-- Verify:
--   SELECT task_type, content_method, uses_cross_penalty,
--          wpm_plateau_low, wpm_plateau_high,
--          pronunciation_fluency_gate, pronunciation_content_threshold
--   FROM pte_speaking_scoring_config ORDER BY task_type;
