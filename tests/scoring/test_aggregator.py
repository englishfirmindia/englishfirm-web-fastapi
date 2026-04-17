"""Tests for SectionalAggregator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.scoring.aggregator import SectionalAggregator, AggregatedResult
from services.scoring.base import ScoringResult, to_pte_score
from services.scoring.azure_scorer import RUBRIC, SPEAKING_WEIGHTS


def make_result(raw_score: float, is_async: bool = False, breakdown: dict = None) -> ScoringResult:
    if breakdown is None:
        breakdown = {}
    return ScoringResult(
        pte_score=to_pte_score(raw_score),
        raw_score=raw_score,
        is_async=is_async,
        breakdown=breakdown,
    )


# ============================================================
# Simple aggregation (reading / writing / listening)
# ============================================================

class TestSimpleAggregation:

    def test_reading_three_questions_average(self):
        """Average of 0.6 + 0.8 + 1.0 = 0.8 → pte_score = to_pte_score(0.8)"""
        agg = SectionalAggregator('reading')
        results = {
            'reading_mcs': [make_result(0.6), make_result(0.8), make_result(1.0)],
        }
        aggregated = agg.aggregate(results)
        expected_avg = (0.6 + 0.8 + 1.0) / 3
        assert aggregated.pte_score == to_pte_score(expected_avg)
        assert aggregated.module == 'reading'

    def test_mixed_types_simple_average(self):
        """Multiple question types — flattened average."""
        agg = SectionalAggregator('reading')
        results = {
            'reading_mcs': [make_result(1.0)],
            'reading_mcm': [make_result(0.0)],
            'reorder_paragraphs': [make_result(0.5)],
        }
        aggregated = agg.aggregate(results)
        expected_avg = (1.0 + 0.0 + 0.5) / 3
        assert aggregated.pte_score == to_pte_score(expected_avg)

    def test_all_zero_raw_scores_gives_floor(self):
        """All zero raw scores → raw_avg=0.0 → pte_score=10 (floor)."""
        agg = SectionalAggregator('listening')
        results = {
            'listening_wfd': [make_result(0.0), make_result(0.0)],
        }
        aggregated = agg.aggregate(results)
        assert aggregated.pte_score == 10

    def test_all_perfect_raw_scores_gives_ceiling(self):
        agg = SectionalAggregator('reading')
        results = {
            'reading_mcs': [make_result(1.0), make_result(1.0)],
        }
        aggregated = agg.aggregate(results)
        assert aggregated.pte_score == 90

    def test_empty_results_graceful(self):
        agg = SectionalAggregator('reading')
        aggregated = agg.aggregate({})
        assert aggregated.pte_score == 10  # floor
        assert aggregated.breakdown['question_count'] == 0

    def test_empty_list_per_type_graceful(self):
        agg = SectionalAggregator('reading')
        aggregated = agg.aggregate({'reading_mcs': []})
        assert aggregated.pte_score == 10
        assert aggregated.breakdown['question_count'] == 0

    def test_breakdown_contains_question_count(self):
        agg = SectionalAggregator('reading')
        results = {
            'reading_mcs': [make_result(0.5), make_result(0.5)],
        }
        aggregated = agg.aggregate(results)
        assert aggregated.breakdown['question_count'] == 2

    def test_breakdown_contains_average_raw(self):
        agg = SectionalAggregator('reading')
        results = {
            'reading_mcs': [make_result(0.4), make_result(0.6)],
        }
        aggregated = agg.aggregate(results)
        assert abs(aggregated.breakdown['average_raw'] - 0.5) < 0.001


# ============================================================
# Speaking aggregation (weighted rubric)
# ============================================================

class TestSpeakingAggregation:

    def _speaking_result(self, question_type: str, content: float, fluency: float,
                         pronunciation: float) -> ScoringResult:
        """Create a ScoringResult where breakdown mimics a resolved Azure score."""
        return ScoringResult(
            pte_score=0,
            raw_score=0.0,
            is_async=True,
            breakdown={
                'content': content,
                'fluency': fluency,
                'pronunciation': pronunciation,
            },
        )

    def test_speaking_perfect_read_aloud(self):
        """All scores 100 on read_aloud → pct=1.0 → pte_score=90."""
        agg = SectionalAggregator('speaking')
        results = {
            'read_aloud': [self._speaking_result('read_aloud', 100, 100, 100)],
        }
        aggregated = agg.aggregate(results)
        # Only one task type, weight is 9 out of total 9 used → weighted_pct = 1.0
        assert aggregated.pte_score == 90
        assert aggregated.module == 'speaking'

    def test_speaking_all_zero_gives_floor(self):
        agg = SectionalAggregator('speaking')
        results = {
            'read_aloud': [self._speaking_result('read_aloud', 0, 0, 0)],
        }
        aggregated = agg.aggregate(results)
        assert aggregated.pte_score == 10

    def test_speaking_uses_rubric_weights(self):
        """Test that task weights affect the final score."""
        agg = SectionalAggregator('speaking')
        # RA weight=9, RS weight=16 in SPEAKING_WEIGHTS
        # Perfect RA, zero RS → weighted_pct = (1.0 * 100 * 9 + 0 * 100 * 16) / (9+16) / 100
        #   = 900 / 25 / 100 = 0.36
        results = {
            'read_aloud':      [self._speaking_result('read_aloud', 100, 100, 100)],
            'repeat_sentence': [self._speaking_result('repeat_sentence', 0, 0, 0)],
        }
        aggregated = agg.aggregate(results)
        expected_weighted_pct = (1.0 * 9) / (9 + 16)   # = 9/25 = 0.36
        expected_pte = to_pte_score(expected_weighted_pct)
        assert aggregated.pte_score == expected_pte

    def test_speaking_breakdown_has_task_breakdown_key(self):
        agg = SectionalAggregator('speaking')
        results = {
            'read_aloud': [self._speaking_result('read_aloud', 80, 70, 60)],
        }
        aggregated = agg.aggregate(results)
        assert 'task_breakdown' in aggregated.breakdown
        assert 'read_aloud' in aggregated.breakdown['task_breakdown']

    def test_speaking_pending_result_treated_as_zero(self):
        """Unresolved async results (status=pending) are counted as zero."""
        agg = SectionalAggregator('speaking')
        results = {
            'read_aloud': [
                ScoringResult(
                    pte_score=0, raw_score=0.0, is_async=True,
                    breakdown={'status': 'pending', 'task_type': 'read_aloud'},
                )
            ],
        }
        aggregated = agg.aggregate(results)
        # Pending → scored as not_found → zero
        assert aggregated.pte_score == 10

    def test_aggregated_result_is_dataclass(self):
        agg = SectionalAggregator('reading')
        result = agg.aggregate({'reading_mcs': [make_result(0.5)]})
        assert isinstance(result, AggregatedResult)
        assert hasattr(result, 'pte_score')
        assert hasattr(result, 'module')
        assert hasattr(result, 'breakdown')
