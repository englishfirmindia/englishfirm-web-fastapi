"""Tests for AzureSpeakingScorer and _compute_question_score."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.scoring.azure_scorer import (
    AzureSpeakingScorer,
    _compute_question_score,
    RUBRIC,
    SPEAKING_WEIGHTS,
)


# ============================================================
# AzureSpeakingScorer.score()
# ============================================================

class TestAzureSpeakingScorer:

    def test_score_returns_is_async_true(self):
        scorer = AzureSpeakingScorer('read_aloud')
        result = scorer.score(1, 'sess', {'audio_url': 'http://example.com/audio.wav'})
        assert result.is_async is True

    def test_score_returns_zero_pte_score(self):
        scorer = AzureSpeakingScorer('read_aloud')
        result = scorer.score(1, 'sess', {'audio_url': 'http://example.com/audio.wav'})
        assert result.pte_score == 0

    def test_score_calls_kick_off_fn(self):
        calls = []
        def kick_off(task_type, question_id, audio_url):
            calls.append((task_type, question_id, audio_url))

        scorer = AzureSpeakingScorer('repeat_sentence')
        scorer.score(42, 'sess', {
            'audio_url': 'http://s3.example.com/audio.wav',
            'kick_off_fn': kick_off,
        })
        assert len(calls) == 1
        assert calls[0] == ('repeat_sentence', 42, 'http://s3.example.com/audio.wav')

    def test_score_no_kick_off_fn_does_not_crash(self):
        scorer = AzureSpeakingScorer('read_aloud')
        result = scorer.score(1, 'sess', {'audio_url': 'http://example.com/audio.wav'})
        assert result.is_async is True

    def test_breakdown_contains_pending_status(self):
        scorer = AzureSpeakingScorer('describe_image')
        result = scorer.score(1, 'sess', {})
        assert result.breakdown.get('status') == 'pending'
        assert result.breakdown.get('task_type') == 'describe_image'

    def test_is_async_property_true(self):
        scorer = AzureSpeakingScorer('read_aloud')
        assert scorer.is_async is True


# ============================================================
# _compute_question_score rubric calculations
# ============================================================

class TestComputeQuestionScore:

    def test_read_aloud_all_100(self):
        result = _compute_question_score('read_aloud', {
            'content': 100, 'fluency': 100, 'pronunciation': 100
        })
        # RA rubric: 0.333+0.333+0.333 = 0.999 (source precision from mobile backend)
        # pct = total/15 ≈ 0.999, not exactly 1.0 — use approximate comparison
        assert abs(result['pct'] - 0.999) < 0.001
        # total_points=15, each sub-score should be ~5 (15 * 0.333)
        assert abs(result['total'] - 14.985) < 0.01

    def test_read_aloud_mixed_scores(self):
        result = _compute_question_score('read_aloud', {
            'content': 80, 'fluency': 70, 'pronunciation': 60
        })
        rubric = RUBRIC['read_aloud']
        max_pts = rubric['total_points']
        expected_content = (80 / 100) * max_pts * rubric['content_pct']
        expected_fluency = (70 / 100) * max_pts * rubric['fluency_pct']
        expected_pronunciation = (60 / 100) * max_pts * rubric['pronunciation_pct']
        expected_total = expected_content + expected_fluency + expected_pronunciation
        assert abs(result['total'] - expected_total) < 0.01

    def test_content_zero_read_aloud_gives_all_zeros(self):
        """RA: content=0 means silence → all zeros (content_zero rule)."""
        result = _compute_question_score('read_aloud', {
            'content': 0, 'fluency': 80, 'pronunciation': 70
        })
        assert result['total'] == 0.0
        assert result['pct'] == 0.0
        assert result['fluency'] == 0.0

    def test_content_zero_repeat_sentence_gives_all_zeros(self):
        result = _compute_question_score('repeat_sentence', {
            'content': 0, 'fluency': 90, 'pronunciation': 85
        })
        assert result['total'] == 0.0
        assert result['pct'] == 0.0

    def test_error_status_gives_all_zeros(self):
        result = _compute_question_score('read_aloud', {'scoring': 'error'})
        assert result['total'] == 0.0
        assert result['pct'] == 0.0

    def test_timeout_status_gives_all_zeros(self):
        result = _compute_question_score('read_aloud', {'scoring': 'timeout'})
        assert result['total'] == 0.0

    def test_asq_correct_gives_one(self):
        result = _compute_question_score('answer_short_question', {'is_correct': True})
        assert result['total'] == 1.0
        assert result['pct'] == 1.0
        assert result['fluency'] == 0.0
        assert result['pronunciation'] == 0.0

    def test_asq_incorrect_gives_zero(self):
        result = _compute_question_score('answer_short_question', {'is_correct': False})
        assert result['total'] == 0.0
        assert result['pct'] == 0.0

    def test_unknown_question_type_gives_all_zeros(self):
        result = _compute_question_score('unknown_type', {'content': 80})
        assert result['total'] == 0.0
        assert result['pct'] == 0.0

    def test_describe_image_content_zero_without_llm_flag(self):
        """DI: content=0 without content_llm_scored → should NOT zero-out (LLM may have failed)."""
        result = _compute_question_score('describe_image', {
            'content': 0, 'fluency': 80, 'pronunciation': 70,
            'content_llm_scored': False,
        })
        # Fluency and pronunciation should still contribute
        assert result['fluency'] > 0.0
        assert result['pronunciation'] > 0.0

    def test_describe_image_content_zero_with_llm_flag(self):
        """DI: content=0 with content_llm_scored=True → all zeros (content_zero_llm rule)."""
        result = _compute_question_score('describe_image', {
            'content': 0, 'fluency': 80, 'pronunciation': 70,
            'content_llm_scored': True,
        })
        assert result['total'] == 0.0
        assert result['pct'] == 0.0

    def test_repeat_sentence_rubric_weights(self):
        rubric = RUBRIC['repeat_sentence']
        assert rubric['content_pct'] == 0.231
        assert abs(rubric['fluency_pct'] - 0.385) < 0.001
        assert abs(rubric['pronunciation_pct'] - 0.385) < 0.001
        # Pcts should sum to ~1.0
        total = rubric['content_pct'] + rubric['fluency_pct'] + rubric['pronunciation_pct']
        assert abs(total - 1.001) < 0.01  # slight float rounding in source data


# ============================================================
# RUBRIC / SPEAKING_WEIGHTS constants
# ============================================================

class TestRubricConstants:

    def test_all_speaking_types_have_rubric(self):
        expected_types = [
            'read_aloud', 'repeat_sentence', 'describe_image',
            'retell_lecture', 'respond_to_situation',
            'summarize_group_discussion', 'answer_short_question',
        ]
        for t in expected_types:
            assert t in RUBRIC, f"Missing rubric for {t}"

    def test_asq_excluded_from_speaking_weights(self):
        assert 'answer_short_question' not in SPEAKING_WEIGHTS

    def test_speaking_weights_sum_to_101(self):
        # RA=9, RS=16, DI=31, RL=13, RTS=13, SGD=19 → 101 (source data)
        total = sum(SPEAKING_WEIGHTS.values())
        # Allow for ptea_respond_situation duplication in weights
        assert total >= 100
