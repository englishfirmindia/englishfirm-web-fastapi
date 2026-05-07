-- Migration: seed pte_speaking_scoring_config for the remaining 6 types
-- Date: 2026-05-07
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- After this migration every speaking task type has a config row, so
-- _score_speaking_v2 routes them all through one code path. Strategy
-- switches per type:
--
--   describe_image, retell_lecture, respond_to_situation,
--   ptea_respond_situation, summarize_group_discussion:
--     content_method        = llm_keypoints
--     pronunciation_source  = azure_freeform
--     uses_reference_text   = false
--     uses_cross_penalty    = false   (legacy formula didn't apply it)
--     pronunciation override columns = NULL (no override)
--
--   answer_short_question:
--     content_method        = binary  (regex match against expected_answers)
--     pronunciation_source  = azure_freeform
--     uses_reference_text   = false
--     uses_cross_penalty    = false
--     fluency / pronunciation are computed but the rubric ignores them
--     (RUBRIC['answer_short_question'] = 1.0/0/0 split).
--
-- WPM/pause defaults mirror RA except:
--   summarize_group_discussion: plateau 70-200 (longer audio, slower OK)
--
-- Fluency numbers will shift on the LLM-scored types vs today's legacy
-- formula (Azure raw fluency minus band penalties → uniform WPM band +
-- pause penalty). This is the calibration step that follows; tune any
-- column via UPDATE without code change.
--
-- Idempotent: ON CONFLICT DO NOTHING.

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
    content_method, uses_reference_text, uses_cross_penalty, pronunciation_source
) VALUES
    -- DI / RL / RTS / ptea_RTS: RA-like WPM curve, LLM content, free pronunciation, no cross-penalty
    ('describe_image',
        80, 270, 130, 220, 2.0, 100.0,
        500, 200, 200, -30.0, 2.0,
        10, 1, 10, 11,
        20.0, 0.5, 0.025,
        'llm_keypoints', FALSE, FALSE, 'azure_freeform'),
    ('retell_lecture',
        80, 270, 130, 220, 2.0, 100.0,
        500, 200, 200, -30.0, 2.0,
        10, 1, 10, 11,
        20.0, 0.5, 0.025,
        'llm_keypoints', FALSE, FALSE, 'azure_freeform'),
    ('respond_to_situation',
        80, 270, 130, 220, 2.0, 100.0,
        500, 200, 200, -30.0, 2.0,
        10, 1, 10, 11,
        20.0, 0.5, 0.025,
        'llm_keypoints', FALSE, FALSE, 'azure_freeform'),
    ('ptea_respond_situation',
        80, 270, 130, 220, 2.0, 100.0,
        500, 200, 200, -30.0, 2.0,
        10, 1, 10, 11,
        20.0, 0.5, 0.025,
        'llm_keypoints', FALSE, FALSE, 'azure_freeform'),
    -- SGD: longer audio, slower delivery acceptable — plateau 70-200
    ('summarize_group_discussion',
        50, 250, 70, 200, 2.0, 100.0,
        500, 200, 200, -30.0, 2.0,
        10, 1, 10, 11,
        20.0, 0.5, 0.025,
        'llm_keypoints', FALSE, FALSE, 'azure_freeform'),
    -- ASQ: binary correctness; fluency/pronunciation values computed but rubric ignores them
    ('answer_short_question',
        80, 270, 130, 220, 2.0, 100.0,
        500, 200, 200, -30.0, 2.0,
        10, 1, 10, 11,
        20.0, 0.5, 0.025,
        'binary', FALSE, FALSE, 'azure_freeform')
ON CONFLICT (task_type) DO NOTHING;

COMMIT;

-- Verify all 8 task types are present:
--   SELECT task_type, content_method, pronunciation_source,
--          uses_reference_text, uses_cross_penalty
--   FROM pte_speaking_scoring_config ORDER BY task_type;
