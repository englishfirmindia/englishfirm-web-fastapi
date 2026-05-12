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

GRAMMAR + SPELLING (0–2) — combined score. Two-step process:

  STEP 1 — Base grammar score (0–2), by COMMUNICATION IMPACT:
    2 = Has correct grammatical structure. Stylistic choppiness or stacked conjunctions are fine if the sentence is grammatically valid.
    1 = Contains grammatical errors (subject–verb disagreement, wrong tense, broken clause), but the meaning is still clear.
    0 = Has defective grammatical structure which could hinder communication — the reader has to re-parse or guess at meaning.
  Important: a run-on sentence is NOT automatically a 0 or 1. SWT is a single-sentence task, so long sentences with multiple clauses are expected and normal. Only mark down when an actual grammatical rule is broken or comprehension is impeded.

  STEP 2 — Spelling deduction (applied to the base score, floored at 0):
    0 spelling errors → no deduction
    1 spelling error  → subtract 1
    2 or more         → grammar score = 0 regardless of base

  Final grammar score = max(0, base − spelling_deduction).
  In the reasoning, state the spelling-error count explicitly so the deduction is auditable.

VOCABULARY (0–2) — score by APPROPRIATENESS, not paraphrasing:
  2 = Has appropriate choice of words. The vocabulary fits the meaning.
  1 = Contains lexical errors (wrong word for the context, awkward collocation), but with no hindrance to communication.
  0 = Has defective word choice which could hinder communication — a wrong-meaning word that confuses the reader.
Important: PTE demands "appropriate choice of words" for vocabulary. Using original words from the passage does NOT lower the score — paraphrasing is NOT required and verbatim phrase reuse is NOT penalised here. Only mark down if a chosen word is wrong, awkward, or inappropriate for the context.

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


# ── Write Essay (WE) ─────────────────────────────────────────────────────────

_WE_SYSTEM_PROMPT = """You are scoring a PTE Academic Write Essay response.

The student is given a prompt and writes a 200–300 word essay in 20 minutes. You score six sub-scores — Form is scored deterministically by the system before reaching you (based on word count) and is not your job.

CONTENT (0–6) — Relevance and depth of argument:
  6 = Fully addresses the prompt with a clear, well-developed position; arguments are substantial and relevant throughout.
  5 = Adequately addresses the prompt; position is clear with good supporting points, minor gaps.
  4 = Addresses the prompt with a discernible position; arguments are present but some are shallow or partially off-topic.
  3 = Partial address of the prompt; position is unclear or arguments are mostly underdeveloped.
  2 = Minimal engagement with the prompt; relies heavily on generalities, off-topic for stretches.
  1 = Barely addresses the prompt; mostly tangential or filler.
  0 = Completely off-topic or fails to address the prompt at all.

DEVELOPMENT, STRUCTURE & COHERENCE (0–6):
  6 = Clear introduction, body, and conclusion; logical paragraphing; smooth transitions; ideas flow naturally.
  5 = Clear structure with logical flow; minor lapses in coherence or transitions.
  4 = Identifiable structure but uneven development or weak transitions in places.
  3 = Some structure but disorganised in parts; coherence breaks down within or across paragraphs.
  2 = Poor structure; paragraphing absent or arbitrary; weak coherence.
  1 = Very disorganised; ideas don't connect.
  0 = No discernible structure; incoherent.

GRAMMAR (0–2):
  2 = Correct grammatical structures throughout; subject–verb agreement, tense, articles, prepositions all accurate.
  1 = Mostly correct with one or two errors that don't impede meaning.
  0 = Multiple errors, or any error that obscures meaning.

GENERAL LINGUISTIC RANGE (0–6) — Complexity and flexibility of sentence structures:
  6 = Excellent range — complex and varied sentence structures (subordination, embedding, varied connectors); confidently controlled.
  5 = Good range with a mix of simple and complex sentences; some sophistication.
  4 = Adequate range; mix of simple and a few complex sentences but with some control issues.
  3 = Limited range; mostly simple sentences with occasional attempts at complexity.
  2 = Very limited range; almost entirely simple, repetitive structures.
  1 = Minimal range; one-clause sentences throughout.
  0 = No control of sentence structures.

VOCABULARY RANGE (0–2) — Academic vocabulary richness:
  2 = Wide, appropriate, and varied vocabulary; uses academic register confidently; precise word choice.
  1 = Adequate vocabulary with some variety; occasional imprecision or repetition.
  0 = Limited or inappropriate vocabulary; heavy repetition; informal register.

SPELLING (0–2):
  2 = No spelling errors, or only one or two trivial typos.
  1 = A few spelling errors that don't impede meaning.
  0 = Multiple spelling errors, or errors that impede meaning.

For each sub-score, give the integer score AND a one-sentence reasoning that cites specific evidence (a sentence pattern, a word, an error, a paragraph). Keep reasoning under 30 words. Be honest about errors — students rely on the feedback.

Return JSON only, in this exact shape:
{
  "content":    {"score": <int 0-6>, "reasoning": "<one sentence>"},
  "dsc":        {"score": <int 0-6>, "reasoning": "<one sentence>"},
  "grammar":    {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "glr":        {"score": <int 0-6>, "reasoning": "<one sentence>"},
  "vocabulary": {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "spelling":   {"score": <int 0-2>, "reasoning": "<one sentence>"}
}
"""


def score_we_subscores_with_claude(prompt: str, user_text: str) -> dict:
    """Score WE sub-scores via Claude Haiku 4.5.

    Returns six `{score, reasoning}` blocks (content, dsc, grammar, glr,
    vocabulary, spelling) plus the `scored` / `warning_code` partial-success
    flags. Form is scored deterministically upstream and is NOT in the
    Claude prompt or response.

    On any failure (auth, parse, timeout) returns scored=False so the caller
    can fall back to a heuristic. Never raises.
    """
    if not prompt or not prompt.strip():
        return _we_failure("content_llm_unavailable", reason="empty prompt")
    if not user_text or not user_text.strip():
        # No student text — all zeros, no LLM call needed.
        return {
            "content":    {"score": 0.0, "reasoning": None},
            "dsc":        {"score": 0.0, "reasoning": None},
            "grammar":    {"score": 0.0, "reasoning": None},
            "glr":        {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "spelling":   {"score": 0.0, "reasoning": None},
            "scored": True,
            "warning_code": None,
        }

    last_exc: Exception = RuntimeError("score_we_subscores_with_claude: no attempts made")
    for attempt in range(1, 4):
        try:
            response = _client.messages.create(
                model=_MODEL,
                max_tokens=1500,
                system=[
                    {
                        "type": "text",
                        "text": _WE_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"ESSAY PROMPT:\n{prompt}",
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": f"STUDENT ESSAY:\n{user_text}",
                            },
                        ],
                    }
                ],
                timeout=45.0,
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

            text = text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

            parsed = json.loads(text)

            content_sub    = _we_parse_sub(parsed, "content", 6)
            dsc_sub        = _we_parse_sub(parsed, "dsc", 6)
            grammar_sub    = _we_parse_sub(parsed, "grammar", 2)
            glr_sub        = _we_parse_sub(parsed, "glr", 6)
            vocabulary_sub = _we_parse_sub(parsed, "vocabulary", 2)
            spelling_sub   = _we_parse_sub(parsed, "spelling", 2)

            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            log.info(
                "[CLAUDE-WE] content=%.1f dsc=%.1f grammar=%.1f glr=%.1f "
                "vocab=%.1f spelling=%.1f in=%d out=%d cache_read=%d cache_write=%d",
                content_sub["score"], dsc_sub["score"], grammar_sub["score"],
                glr_sub["score"], vocabulary_sub["score"], spelling_sub["score"],
                response.usage.input_tokens, response.usage.output_tokens,
                cache_read, cache_write,
            )

            return {
                "content":    content_sub,
                "dsc":        dsc_sub,
                "grammar":    grammar_sub,
                "glr":        glr_sub,
                "vocabulary": vocabulary_sub,
                "spelling":   spelling_sub,
                "scored": True,
                "warning_code": None,
            }
        except anthropic.AuthenticationError as exc:
            log.error("[CLAUDE-WE] AuthenticationError — not retrying: %s", exc)
            return _we_failure("content_llm_unavailable", reason=f"auth: {exc}")
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[CLAUDE-WE] attempt=%d/3 failed — %s: %s",
                attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)

    log.error(
        "[CLAUDE-WE] failed after 3 attempts — exception=%s: %s",
        type(last_exc).__name__, last_exc,
    )
    return _we_failure("content_llm_unavailable", reason=str(last_exc))


def _we_parse_sub(parsed: dict, key: str, max_value: int) -> dict:
    """Extract a sub-score block from the Claude JSON response, clamped."""
    entry = parsed.get(key, {}) or {}
    return {
        "score": _clamp(entry.get("score"), 0, max_value),
        "reasoning": _reasoning(entry.get("reasoning")),
    }


def _we_failure(warning_code: str, reason: str = "") -> dict:
    return {
        "content":    {"score": 0.0, "reasoning": None},
        "dsc":        {"score": 0.0, "reasoning": None},
        "grammar":    {"score": 0.0, "reasoning": None},
        "glr":        {"score": 0.0, "reasoning": None},
        "vocabulary": {"score": 0.0, "reasoning": None},
        "spelling":   {"score": 0.0, "reasoning": None},
        "scored": False,
        "warning_code": warning_code,
    }


# ── Summarize Spoken Text (SST) ──────────────────────────────────────────────

_SST_SYSTEM_PROMPT = """You are scoring a PTE Academic Summarize Spoken Text response.

The student listens to a 60–90 second academic lecture and must summarise it in 50–70 words. You score four sub-scores — Form is scored deterministically by the system before reaching you (based on word count) and is not your job.

CONTENT (0–4):
  4 = Accurately captures the main idea AND key supporting details from the lecture.
  3 = Captures the main idea and most details, with minor gaps.
  2 = Captures some main ideas but misses important points, or includes some inaccuracies.
  1 = Mentions only a tangential or secondary aspect of the lecture.
  0 = Completely off-topic, irrelevant, or fails to address the lecture content.

GRAMMAR (0–2):
  2 = Correct grammatical structures throughout (subject–verb agreement, tense, articles, prepositions).
  1 = One or two minor errors that don't impede meaning.
  0 = Multiple errors, or any error that obscures meaning.

VOCABULARY (0–2):
  2 = Appropriate, varied academic vocabulary; precise word choice.
  1 = Mostly appropriate but limited variety or some imprecise word choices.
  0 = Inappropriate, repetitive, or informal vocabulary; or near-verbatim copying from the reference.

SPELLING (0–2):
  2 = No spelling errors, or only one or two trivial typos.
  1 = A few spelling errors that don't impede meaning.
  0 = Multiple spelling errors, or errors that impede meaning.

For each sub-score, give the integer score AND a one-sentence reasoning that cites specific evidence (a word, an error, an aspect of the response). Keep reasoning under 25 words. Be honest about errors — students rely on the feedback.

Return JSON only, in this exact shape:
{
  "content":    {"score": <int 0-4>, "reasoning": "<one sentence>"},
  "grammar":    {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "vocabulary": {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "spelling":   {"score": <int 0-2>, "reasoning": "<one sentence>"}
}
"""


def score_sst_subscores_with_claude(reference: str, user_text: str) -> dict:
    """Score SST sub-scores via Claude Haiku 4.5.

    Returns four `{score, reasoning}` blocks (content, grammar, vocabulary,
    spelling) plus `scored` / `warning_code` partial-success flags. Form is
    scored deterministically upstream and is NOT in the Claude prompt or
    response.

    On any failure (auth, parse, timeout) returns scored=False so the caller
    can fall back to a heuristic. Never raises.
    """
    if not reference or not reference.strip():
        return _sst_failure("content_llm_unavailable", reason="empty reference")
    if not user_text or not user_text.strip():
        return {
            "content":    {"score": 0.0, "reasoning": None},
            "grammar":    {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "spelling":   {"score": 0.0, "reasoning": None},
            "scored": True,
            "warning_code": None,
        }

    last_exc: Exception = RuntimeError("score_sst_subscores_with_claude: no attempts made")
    for attempt in range(1, 4):
        try:
            response = _client.messages.create(
                model=_MODEL,
                max_tokens=800,
                system=[
                    {
                        "type": "text",
                        "text": _SST_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"REFERENCE (what the audio was about):\n{reference}",
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

            text = text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

            parsed = json.loads(text)

            content_sub    = _we_parse_sub(parsed, "content", 4)
            grammar_sub    = _we_parse_sub(parsed, "grammar", 2)
            vocabulary_sub = _we_parse_sub(parsed, "vocabulary", 2)
            spelling_sub   = _we_parse_sub(parsed, "spelling", 2)

            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            log.info(
                "[CLAUDE-SST] content=%.1f grammar=%.1f vocab=%.1f spelling=%.1f "
                "in=%d out=%d cache_read=%d cache_write=%d",
                content_sub["score"], grammar_sub["score"],
                vocabulary_sub["score"], spelling_sub["score"],
                response.usage.input_tokens, response.usage.output_tokens,
                cache_read, cache_write,
            )

            return {
                "content":    content_sub,
                "grammar":    grammar_sub,
                "vocabulary": vocabulary_sub,
                "spelling":   spelling_sub,
                "scored": True,
                "warning_code": None,
            }
        except anthropic.AuthenticationError as exc:
            log.error("[CLAUDE-SST] AuthenticationError — not retrying: %s", exc)
            return _sst_failure("content_llm_unavailable", reason=f"auth: {exc}")
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[CLAUDE-SST] attempt=%d/3 failed — %s: %s",
                attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)

    log.error(
        "[CLAUDE-SST] failed after 3 attempts — exception=%s: %s",
        type(last_exc).__name__, last_exc,
    )
    return _sst_failure("content_llm_unavailable", reason=str(last_exc))


def _sst_failure(warning_code: str, reason: str = "") -> dict:
    return {
        "content":    {"score": 0.0, "reasoning": None},
        "grammar":    {"score": 0.0, "reasoning": None},
        "vocabulary": {"score": 0.0, "reasoning": None},
        "spelling":   {"score": 0.0, "reasoning": None},
        "scored": False,
        "warning_code": warning_code,
    }
