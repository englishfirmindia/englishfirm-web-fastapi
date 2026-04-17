"""
AI-based scorer for writing/listening tasks that require open-ended evaluation.

Covers:
  summarize_written_text (swt)
  write_essay (we)
  listening_sst (sst)

_score_with_ai raises NotImplementedError — real implementation
will call Claude API when credentials are configured. Tests mock it.
"""

from .base import ScoringResult, ScoringStrategy, to_pte_score


class AIScorer(ScoringStrategy):
    """
    Sync AI scorer. Calls _score_with_ai(text, prompt) → 0-100 raw score,
    then converts to PTE scale via to_pte_score.

    answer dict:
      text: str   (user's written response)
      prompt: str (original question/passage for context)
    """

    is_async = False  # class-level

    def __init__(self, task_type: str):
        self.task_type = task_type  # 'swt' | 'we' | 'sst'

    @property
    def is_async(self) -> bool:
        return False

    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        text = answer.get('text', '')
        prompt = answer.get('prompt', '')
        try:
            raw = self._score_with_ai(text, prompt)
        except NotImplementedError:
            raise
        except Exception as e:
            return ScoringResult(
                pte_score=to_pte_score(0.0),
                raw_score=0.0,
                is_async=False,
                breakdown={'ai_raw': 0, 'task_type': self.task_type},
                error=str(e),
            )

        pct = raw / 100.0
        return ScoringResult(
            pte_score=to_pte_score(pct),
            raw_score=pct,
            is_async=False,
            breakdown={'ai_raw': raw, 'task_type': self.task_type},
        )

    def _score_with_ai(self, text: str, prompt: str) -> float:
        """
        Returns 0-100. Override in tests. Real impl calls Claude API.
        Raises NotImplementedError until configured.
        """
        raise NotImplementedError(
            f"AIScorer._score_with_ai not implemented for task_type='{self.task_type}'. "
            "Configure the Claude API key and implement this method."
        )
