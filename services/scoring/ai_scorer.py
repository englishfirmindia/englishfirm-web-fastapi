"""
AI-based scorer for writing/listening tasks that require open-ended evaluation.

Covers:
  summarize_written_text (swt)
  write_essay (we)
  listening_sst (sst)

Uses the Anthropic Claude API (claude-haiku-4-5-20251001) to evaluate responses
on a 0-100 scale, then converts to PTE scale via to_pte_score.
"""

import os
import anthropic

from .base import ScoringResult, ScoringStrategy, to_pte_score

_TASK_INSTRUCTIONS = {
    "swt": (
        "Summarize Written Text — one sentence summary of a passage. "
        "Score on: (1) content accuracy 0-40, (2) grammar/structure 0-30, (3) conciseness 0-30."
    ),
    "we": (
        "Write Essay — argumentative essay on a topic. "
        "Score on: (1) content/argument 0-35, (2) structure/cohesion 0-25, "
        "(3) grammar 0-20, (4) vocabulary 0-20."
    ),
    "sst": (
        "Summarize Spoken Text — written summary of audio content. "
        "Score on: (1) content accuracy 0-40, (2) grammar 0-30, (3) coherence 0-30."
    ),
}


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
        Returns 0-100. Calls Claude Haiku via the Anthropic API.
        Falls back to 0.0 if the API key is not configured.
        """
        if not text or not text.strip():
            return 0.0

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — cannot score with AI")

        instruction = _TASK_INSTRUCTIONS.get(self.task_type, _TASK_INSTRUCTIONS["we"])

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a PTE Academic examiner. Score this student response.\n\n"
                    f"Task type: {instruction}\n"
                    f"Original prompt/passage: {prompt[:500] if prompt else 'N/A'}\n"
                    f"Student response: {text[:1000]}\n\n"
                    "Reply with ONLY a number between 0 and 100 representing the score. "
                    "Nothing else."
                ),
            }],
        )
        score_text = message.content[0].text.strip()
        return float(score_text.split()[0])
