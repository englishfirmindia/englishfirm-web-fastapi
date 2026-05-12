"""
Claude-backed scoring service for SWT sub-scores (content, grammar, vocabulary).

Replaces the gpt-4o-mini content-only call in the SWT path with a single
Claude Haiku 4.5 call that returns content + grammar + vocabulary in one
JSON response. Each sub-score is `{score, reasoning}` so trainer review can
show *why* a score was given.

Prompt caching: the system-prompt rubric explanation and the passage prefix
are marked with `cache_control: {"type": "ephemeral"}`. The minimum cacheable
prefix on Haiku 4.5 is 4096 tokens; shorter SWT prefixes won't hit the cache
(silent no-op, no error). This is forward-compatible: longer passages and
future migrations to Sonnet (2048-token minimum) will cache without code
changes.

Retry envelope: 3 attempts with linear 2s backoff (mirrors `_call_llm` in
llm_content_scoring_service.py). AuthenticationError doesn't retry. On full
failure returns `scored=False` so the caller can fall back to the heuristic.
"""
import json
import os
import time
from typing import Optional

import anthropic
from anthropic import Anthropic

from core.logging_config import get_logger

log = get_logger(__name__)

_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
_MODEL = "claude-haiku-4-5"

_SWT_SYSTEM_PROMPT = """You are scoring a PTE Academic Summarize Written Text response.

The student is given a passage of ~200–300 words and must write a one-sentence summary in 5–75 words. You score three sub-scores — Form is scored deterministically by the system before reaching you and is not your job.

CONTENT (0–4):
  4 = Provides a good summary; all main aspects of the passage are covered concisely.
  3 = Provides a fair summary; one main aspect is missing, underdeveloped, or unclear.
  2 = Captures only some main aspects; misses key points or includes irrelevant detail.
  1 = Mentions only a tangential aspect; mostly off-topic.
  0 = Completely off-topic, irrelevant, or a near-verbatim copy of the source passage.

GRAMMAR & SPELLING (0–2):
  2 = Correct grammatical structure and spelling throughout.
  1 = One or two minor grammar/spelling errors (e.g. a typo, a capitalisation slip, a single agreement issue).
  0 = Three or more errors, or any error that obscures meaning.

VOCABULARY (0–2):
  2 = Appropriate, varied word choice that demonstrates paraphrasing of the passage.
  1 = Mostly appropriate but limited paraphrasing or some imprecise word choices.
  0 = Inappropriate word choice, or near-verbatim vocabulary copied from the passage.

For each sub-score, give the integer score AND a one-sentence reasoning that cites specific evidence (a word, an error, an aspect of the response). Keep reasoning under 25 words. Be honest about errors — students rely on the feedback.

Return JSON only, in this exact shape:
{
  "content":    {"score": <int 0-4>, "reasoning": "<one sentence>"},
  "grammar":    {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "vocabulary": {"score": <int 0-2>, "reasoning": "<one sentence>"}
}
"""


def score_swt_subscores_with_claude(passage: str, user_text: str) -> dict:
    """Score SWT sub-scores via Claude Haiku 4.5.

    Returns a dict carrying per-sub-score `{score, reasoning}` plus a
    `scored` flag and (when relevant) a `warning_code` for the partial-
    success plumbing the rest of the pipeline already understands:

        {
            "content":    {"score": float, "reasoning": str|None},
            "grammar":    {"score": float, "reasoning": str|None},
            "vocabulary": {"score": float, "reasoning": str|None},
            "scored":     bool,
            "warning_code": Optional[str],  # "content_llm_unavailable" when scored=False
        }

    On any failure (auth, parse, timeout) returns scored=False so the caller
    can decide whether to fall back to a heuristic or surface a warning.
    Never raises.
    """
    if not passage or not passage.strip():
        return _failure("content_llm_unavailable", reason="empty passage")
    if not user_text or not user_text.strip():
        # No student text to score — return zeros without an LLM call.
        return {
            "content": {"score": 0.0, "reasoning": None},
            "grammar": {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "scored": True,
            "warning_code": None,
        }

    last_exc: Exception = RuntimeError("score_swt_subscores_with_claude: no attempts made")
    for attempt in range(1, 4):
        try:
            response = _client.messages.create(
                model=_MODEL,
                max_tokens=600,
                system=[
                    {
                        "type": "text",
                        "text": _SWT_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"PASSAGE:\n{passage}",
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": f"STUDENT SUMMARY:\n{user_text}",
                            },
                        ],
                    }
                ],
                timeout=30.0,
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break
            if not text:
                last_exc = RuntimeError("Claude returned no text content")
                if attempt < 3:
                    time.sleep(2)
                continue

            # Strip markdown fences if present.
            text = text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

            parsed = json.loads(text)

            content_score = _clamp(parsed.get("content", {}).get("score"), 0, 4)
            content_reasoning = _reasoning(parsed.get("content", {}).get("reasoning"))
            grammar_score = _clamp(parsed.get("grammar", {}).get("score"), 0, 2)
            grammar_reasoning = _reasoning(parsed.get("grammar", {}).get("reasoning"))
            vocab_score = _clamp(parsed.get("vocabulary", {}).get("score"), 0, 2)
            vocab_reasoning = _reasoning(parsed.get("vocabulary", {}).get("reasoning"))

            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            log.info(
                "[CLAUDE-SWT] content=%.1f grammar=%.1f vocab=%.1f "
                "in=%d out=%d cache_read=%d cache_write=%d",
                content_score, grammar_score, vocab_score,
                response.usage.input_tokens, response.usage.output_tokens,
                cache_read, cache_write,
            )

            return {
                "content": {"score": content_score, "reasoning": content_reasoning},
                "grammar": {"score": grammar_score, "reasoning": grammar_reasoning},
                "vocabulary": {"score": vocab_score, "reasoning": vocab_reasoning},
                "scored": True,
                "warning_code": None,
            }
        except anthropic.AuthenticationError as exc:
            log.error("[CLAUDE-SWT] AuthenticationError — not retrying: %s", exc)
            return _failure("content_llm_unavailable", reason=f"auth: {exc}")
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[CLAUDE-SWT] attempt=%d/3 failed — %s: %s",
                attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)

    log.error(
        "[CLAUDE-SWT] failed after 3 attempts — exception=%s: %s",
        type(last_exc).__name__, last_exc,
    )
    return _failure("content_llm_unavailable", reason=str(last_exc))


def _clamp(value, lo: int, hi: int) -> float:
    try:
        return max(float(lo), min(float(hi), float(value)))
    except (TypeError, ValueError):
        return 0.0


def _reasoning(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _failure(warning_code: str, reason: str = "") -> dict:
    return {
        "content": {"score": 0.0, "reasoning": None},
        "grammar": {"score": 0.0, "reasoning": None},
        "vocabulary": {"score": 0.0, "reasoning": None},
        "scored": False,
        "warning_code": warning_code,
    }
