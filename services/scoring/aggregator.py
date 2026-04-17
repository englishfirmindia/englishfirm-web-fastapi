"""
Sectional exam aggregator — collects per-question ScoringResults and
computes a final module-level PTE score.

For speaking: uses RUBRIC + SPEAKING_WEIGHTS from azure_scorer (exact port
of _aggregate_speaking_score from speaking_scoring_service.py).

For reading / writing / listening: simple average of raw_scores.
"""

from dataclasses import dataclass
from typing import Dict, List

from .azure_scorer import RUBRIC, SPEAKING_WEIGHTS, _compute_question_score
from .base import ScoringResult, to_pte_score


@dataclass
class AggregatedResult:
    pte_score: int
    module: str           # 'speaking' | 'reading' | 'writing' | 'listening'
    breakdown: dict       # per-type or summary breakdown


class SectionalAggregator:
    def __init__(self, module: str):
        self.module = module  # 'speaking' | 'reading' | 'writing' | 'listening'

    def aggregate(self, question_results: Dict[str, List[ScoringResult]]) -> AggregatedResult:
        """
        question_results: {question_type: [ScoringResult, ...]}

        For speaking: uses RUBRIC + SPEAKING_WEIGHTS for weighted PTE score.
        For others:   simple average of raw_scores across all questions.
        """
        if self.module == 'speaking':
            return self._aggregate_speaking(question_results)
        return self._aggregate_simple(question_results)

    # ------------------------------------------------------------------
    # Speaking aggregation — weighted by RUBRIC + SPEAKING_WEIGHTS
    # ------------------------------------------------------------------

    def _aggregate_speaking(self, results: Dict[str, List[ScoringResult]]) -> AggregatedResult:
        """
        Port of _aggregate_speaking_score from speaking_scoring_service.py.

        Each ScoringResult.breakdown for async results contains the raw Azure
        score dict (after scoring completes). For aggregation we expect callers
        to pass results whose breakdown includes 'content', 'fluency',
        'pronunciation' keys (the raw_score dict from Azure).

        If breakdown contains 'status': 'pending', the question is treated as
        scored 0 (timed-out / not yet resolved).
        """
        by_task: Dict[str, list] = {}

        for question_type, scoring_results in results.items():
            for sr in scoring_results:
                # Extract raw Azure score dict from breakdown.
                # After async scoring completes the caller should populate
                # breakdown with the Azure raw scores. If still pending treat
                # as {'scoring': 'not_found'}.
                raw_score = sr.breakdown if sr.breakdown and 'status' not in sr.breakdown else {}
                if sr.breakdown.get('status') == 'pending':
                    raw_score = {'scoring': 'not_found'}

                computed = _compute_question_score(question_type, raw_score)
                rubric = RUBRIC.get(question_type, {'total_points': 1})

                by_task.setdefault(question_type, []).append({
                    'content':       computed['content'],
                    'fluency':       computed['fluency'],
                    'pronunciation': computed['pronunciation'],
                    'total':         computed['total'],
                    'max_total':     rubric['total_points'],
                    'pct':           computed['pct'],
                })

        # Per-task averages
        task_breakdown = {}
        for task, qs in by_task.items():
            rubric = RUBRIC.get(task, {'total_points': 1})
            n = len(qs)
            avg_total         = sum(q['total'] for q in qs) / n
            avg_content       = sum(q['content'] for q in qs) / n
            avg_fluency       = sum(q['fluency'] for q in qs) / n
            avg_pronunciation = sum(q['pronunciation'] for q in qs) / n
            pct               = avg_total / rubric['total_points'] if rubric['total_points'] > 0 else 0.0

            task_breakdown[task] = {
                'count':             n,
                'max_points':        rubric['total_points'],
                'avg_total':         round(avg_total, 2),
                'avg_content':       round(avg_content, 2),
                'avg_fluency':       round(avg_fluency, 2),
                'avg_pronunciation': round(avg_pronunciation, 2),
                'pct':               round(pct, 4),
                'weight':            SPEAKING_WEIGHTS.get(task),
                'questions':         qs,
            }

        # Weighted speaking score (ASQ excluded — weight is None)
        total_weight = 0
        weighted_sum = 0.0
        for task, data in task_breakdown.items():
            w = SPEAKING_WEIGHTS.get(task)
            if w is None:
                continue
            weighted_sum += data['pct'] * 100.0 * w
            total_weight += w

        weighted_pct = (weighted_sum / total_weight / 100.0) if total_weight > 0 else 0.0
        pte_score = to_pte_score(weighted_pct)

        return AggregatedResult(
            pte_score=pte_score,
            module=self.module,
            breakdown={
                'weighted_pct': round(weighted_pct, 4),
                'task_breakdown': task_breakdown,
            },
        )

    # ------------------------------------------------------------------
    # Simple aggregation — average raw_score across all questions
    # ------------------------------------------------------------------

    def _aggregate_simple(self, results: Dict[str, List[ScoringResult]]) -> AggregatedResult:
        all_scores = [r.raw_score for rs in results.values() for r in rs]
        if not all_scores:
            return AggregatedResult(
                pte_score=to_pte_score(0.0),
                module=self.module,
                breakdown={'question_count': 0, 'average_raw': 0.0},
            )
        avg = sum(all_scores) / len(all_scores)
        return AggregatedResult(
            pte_score=to_pte_score(avg),
            module=self.module,
            breakdown={'question_count': len(all_scores), 'average_raw': avg},
        )
