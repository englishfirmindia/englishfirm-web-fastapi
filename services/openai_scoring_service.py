"""OpenAI gpt-5-nano scoring service for SWT, WE, and SST sub-scores.

Replaces the Claude Haiku 4.5 calls as the primary scorer. Claude functions in
anthropic_scoring_service.py remain as the fallback when gpt-5-nano fails
after 3 retries; the heuristic in ai_scorer.py is the final fallback.

Why gpt-5-nano: stress test against 5 sample SWT responses showed nano
correctly applies the deduction-based grammar rubric (it doesn't fabricate
mistakes outside the four named categories the way Claude Haiku 4.5 does on
long single-sentence summaries), and it doesn't fabricate vocabulary
penalties for paraphrasing the way Claude and full GPT-5 do. Also ~5×
cheaper than gpt-5-mini and ~50× cheaper than full gpt-5.

Latency: ~17s per call. Acceptable for SWT/WE/SST since scoring runs
post-submit, not during typing.

Retry envelope: 3 attempts with 2s linear backoff. AuthenticationError
doesn't retry. On full failure returns scored=False so ai_scorer can fall
back to Claude.

Shared rubric (used in all three task prompts):
  GRAMMAR (0–2): deduction-based, 4 named categories only (capitalisation,
                 punctuation, spacing, verb use). Default = 0 mistakes.
                 Stylistic awkwardness / run-on length / "wrong conjunction"
                 are explicitly NOT mistakes.
  SPELLING (0–2): per-typo deduction, floor 0. Each misspelled English word
                  counts once. Proper-noun typos count.
  VOCABULARY (0–2): appropriateness, not paraphrasing. Reusing passage
                    words verbatim does NOT lower the score.
"""
import json
import os
import time
from typing import Optional

from openai import OpenAI, AuthenticationError

from core.logging_config import get_logger

log = get_logger(__name__)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = "gpt-5-nano"
# gpt-5 family burns "reasoning tokens" before emitting visible JSON. Default
# reasoning_effort=medium routinely consumed 4500–6000 reasoning tokens per
# SWT call, leaving little or no room for the JSON output — many calls hit
# finish_reason=length with empty content. The deduction-based rubric is fully
# spelled out in the prompt, so "minimal" reasoning is plenty: live testing
# on Nimisha's last 5 SWTs showed identical sub-scores at ~12× the speed and
# zero finish=length failures. 8000-token cap is the safety net.
_MAX_COMPLETION_TOKENS = 8000
_REASONING_EFFORT = "minimal"


# ── Shared rubric language used in all three task prompts ────────────────────
_GRAMMAR_DEDUCTION_RULES = """GRAMMAR (0–2) — DEDUCTION-BASED, four-step procedure. Follow it EXACTLY.

  GROUND RULES — read these before counting:
    • The default is ZERO mistakes. Only count a mistake if you can QUOTE the exact offending characters AND name which of the four categories (a/b/c/d) it falls under. If you cannot do both, count zero.
    • Long sentences with many clauses are EXPECTED. Length, stacked `and`s, choppy flow, and "lacks subordination" are NOT grammar mistakes — do not count them.
    • Comma usage in English: commas are OPTIONAL before "and", "so", "but", "so that", "because", "although". Their absence is NEVER a grammar mistake.
    • Word-choice issues belong to VOCABULARY, not GRAMMAR. If your only complaint about a word is its appropriateness, do not count it under grammar.

  STEP A — Walk through the four grammar categories ONE BY ONE. Quote any violations:
    (a) CAPITALISATION — lowercase at sentence start; lowercase proper noun; mid-sentence common noun capitalised with no reason.
    (b) PUNCTUATION — missing terminal mark (. ! ?); wrong punctuation mark used; OBVIOUSLY-required comma missing (e.g. between two independent clauses joined without a conjunction). Optional commas don't count.
    (c) SPACING — double space; missing space after a punctuation mark.
    (d) VERB USE — wrong tense / wrong aspect / broken auxiliary chain.
        EXAMPLES (these are the only kind of verb mistake that counts):
          "I had breakfast." — NO mistake.
          "I had breakfast today." — NO mistake.
          "I did had breakfast." — ONE mistake (auxiliary `did` cannot take past-form `had`).
          "She have gone home." — ONE mistake (subject-verb agreement).
          "They eats lunch." — ONE mistake (subject-verb agreement).

  STEP B — Compute: grammar_score = max(0, 2 - grammar_count).

  STEP C — Category check. For EACH candidate mistake, ask: "Which of (a)(b)(c)(d) does this violate?" If the answer is "none / it feels awkward / poorly constructed / illogical / wrong conjunction", DELETE it — not a grammar mistake under this rubric.

  STEP D — Output the integer score. In `reasoning`, list each counted mistake with its quoted text and rule tag, then state the arithmetic:
    "grammar=N [each: quoted + (rule x)] or 'none' -> grammar_remaining=max(0,2-N)=X"
  The `score` field MUST equal X.

  ALSO populate `mistake_quotes`: a JSON array of the exact verbatim substrings from the student response that you flagged. Each entry MUST appear verbatim in the student response (case-sensitive, exact characters) so the UI can locate it. Empty list if no mistakes."""

_SPELLING_DEDUCTION_RULES = """SPELLING (0–2) — DEDUCTION-BASED, per-typo:

  Start at 2.0. Subtract 1.0 for each misspelled English word (floor at 0).
  US and British spelling are both acceptable — do not penalise either variant.
  Proper-noun typos count.

  spelling_score = max(0, 2 - spelling_count)

  In `reasoning`, list each misspelled word quoted, then state the arithmetic:
    "spelling=M [\\"misspelled1\\", \\"misspelled2\\", ...] or 'none' -> spelling_remaining=max(0,2-M)=Y"
  The `score` field MUST equal Y.

  ALSO populate `mistake_quotes`: a JSON array of each misspelled word as it appears verbatim in the student response (case-sensitive, exact characters). Empty list if no misspellings."""

_VOCABULARY_APPROPRIATENESS_RULES = """VOCABULARY (0–2) — score by APPROPRIATENESS, not paraphrasing:
  2 = Has appropriate choice of words. The vocabulary fits the meaning.
  1 = Contains lexical errors (wrong word for context, awkward collocation), but with no hindrance to communication.
  0 = Has defective word choice which could hinder communication — a wrong-meaning word that confuses the reader.

  IMPORTANT: PTE demands "appropriate choice of words" for vocabulary. Using original words from the source passage does NOT lower the score — paraphrasing is NOT required, and verbatim phrase reuse is NOT penalised. Only mark down if a chosen word is wrong, awkward, or inappropriate for the context."""


# ── SWT prompt (3 sub-scores: content, grammar+spelling combined, vocabulary) ─
# For SWT the rubric reports a single combined grammar+spelling sub-score
# (max 2). Both counters start at 2; final = min(grammar_remaining, spelling_remaining).
_SWT_SYSTEM_PROMPT = f"""You are scoring a PTE Academic Summarize Written Text response.

The student is given a passage of ~200–300 words and must write a one-sentence summary in 5–75 words. You score three sub-scores — Form is scored deterministically by the system before reaching you and is not your job.

CONTENT (0–4):
  4 = Provides a good summary; all main aspects of the passage are covered concisely.
  3 = Provides a fair summary; one main aspect is missing, underdeveloped, or unclear.
  2 = Captures only some main aspects; misses key points or includes irrelevant detail.
  1 = Mentions only a tangential aspect; mostly off-topic.
  0 = Completely off-topic, irrelevant, or a near-verbatim copy of the source passage.

GRAMMAR + SPELLING (0–2) — COMBINED sub-score, computed as min(grammar_remaining, spelling_remaining).

  Both GRAMMAR and SPELLING start at 2. Each independent mistake deducts 1 from its respective counter, floored at 0. The final reported `grammar` score is min(grammar_remaining, spelling_remaining).

{_GRAMMAR_DEDUCTION_RULES}

  Then ALSO count spelling mistakes (per-typo, floor 0; US/UK both fine; proper-noun typos count).

  Final output:
    grammar_remaining   = max(0, 2 - grammar_count)
    spelling_remaining  = max(0, 2 - spelling_count)
    grammar_field_score = min(grammar_remaining, spelling_remaining)

  In `reasoning`, write EXACTLY this format on one line:
    "grammar=N [each: quoted + (rule x)] or 'none', spelling=M [quoted misspellings] or 'none' -> grammar_remaining=X, spelling_remaining=Y -> min=Z"
  The `score` field MUST equal Z.

{_VOCABULARY_APPROPRIATENESS_RULES}

Return JSON only, in this exact shape:
{{
  "content":    {{"score": <number 0-4>, "reasoning": "<one sentence>"}},
  "grammar":    {{
    "score": <number 0-2>,
    "reasoning": "grammar=N [...], spelling=M [...] -> grammar_remaining=X, spelling_remaining=Y -> min=Z",
    "grammar_mistake_quotes":  [<exact verbatim substrings flagged as grammar>],
    "spelling_mistake_quotes": [<exact verbatim misspelled words>]
  }},
  "vocabulary": {{"score": <number 0-2>, "reasoning": "<one sentence>"}}
}}
"""


# ── WE prompt (6 sub-scores: content, dsc, grammar, glr, vocabulary, spelling) ─
# WE keeps grammar and spelling as INDEPENDENT sub-scores (not combined).
# Each is deduction-based at its own 0–2 scale.
_WE_SYSTEM_PROMPT = f"""You are scoring a PTE Academic Write Essay response.

The student is given a prompt and writes a 200–300 word essay in 20 minutes. You score six sub-scores — Form is scored deterministically by the system before reaching you (based on word count) and is not your job.

CONTENT (0–6) — Relevance and depth of argument:
  6 = Fully addresses the prompt with a clear, well-developed position; arguments are substantial and relevant throughout.
  5 = Adequately addresses the prompt; position is clear with good supporting points, minor gaps.
  4 = Addresses the prompt with a discernible position; arguments are present but some are shallow or partially off-topic.
  3 = Partial address of the prompt; position is unclear or arguments are mostly underdeveloped.
  2 = Minimal engagement with the prompt; relies heavily on generalities, off-topic for stretches.
  1 = Barely addresses the prompt; mostly tangential or filler.
  0 = Completely off-topic or fails to address the prompt at all.

DEVELOPMENT, STRUCTURE & COHERENCE (DSC, 0–6):
  6 = Clear introduction, body, and conclusion; logical paragraphing; smooth transitions; ideas flow naturally.
  5 = Clear structure with logical flow; minor lapses in coherence or transitions.
  4 = Identifiable structure but uneven development or weak transitions in places.
  3 = Some structure but disorganised in parts; coherence breaks down within or across paragraphs.
  2 = Poor structure; paragraphing absent or arbitrary; weak coherence.
  1 = Very disorganised; ideas don't connect.
  0 = No discernible structure; incoherent.

{_GRAMMAR_DEDUCTION_RULES}

GENERAL LINGUISTIC RANGE (GLR, 0–6) — Complexity and flexibility of sentence structures:
  6 = Excellent range — complex and varied sentence structures (subordination, embedding, varied connectors); confidently controlled.
  5 = Good range with a mix of simple and complex sentences; some sophistication.
  4 = Adequate range; mix of simple and a few complex sentences but with some control issues.
  3 = Limited range; mostly simple sentences with occasional attempts at complexity.
  2 = Very limited range; almost entirely simple, repetitive structures.
  1 = Minimal range; one-clause sentences throughout.
  0 = No control of sentence structures.

{_VOCABULARY_APPROPRIATENESS_RULES}

{_SPELLING_DEDUCTION_RULES}

Return JSON only, in this exact shape:
{{
  "content":    {{"score": <number 0-6>, "reasoning": "<one sentence>"}},
  "dsc":        {{"score": <number 0-6>, "reasoning": "<one sentence>"}},
  "grammar":    {{
    "score": <number 0-2>,
    "reasoning": "grammar=N [...] -> grammar_remaining=X",
    "mistake_quotes": [<exact verbatim substrings flagged as grammar>]
  }},
  "glr":        {{"score": <number 0-6>, "reasoning": "<one sentence>"}},
  "vocabulary": {{"score": <number 0-2>, "reasoning": "<one sentence>"}},
  "spelling":   {{
    "score": <number 0-2>,
    "reasoning": "spelling=M [...] -> spelling_remaining=Y",
    "mistake_quotes": [<exact verbatim misspelled words>]
  }}
}}
"""


# ── SST prompt (4 sub-scores: content, grammar, vocabulary, spelling) ────────
_SST_SYSTEM_PROMPT = f"""You are scoring a PTE Academic Summarize Spoken Text response.

The student listens to a 60–90 second academic lecture and must summarise it in 50–70 words. You score four sub-scores — Form is scored deterministically by the system before reaching you (based on word count) and is not your job.

CONTENT (0–4):
  4 = Accurately captures the main idea AND key supporting details from the lecture.
  3 = Captures the main idea and most details, with minor gaps.
  2 = Captures some main ideas but misses important points, or includes some inaccuracies.
  1 = Mentions only a tangential or secondary aspect of the lecture.
  0 = Completely off-topic, irrelevant, or fails to address the lecture content.

{_GRAMMAR_DEDUCTION_RULES}

{_VOCABULARY_APPROPRIATENESS_RULES}

{_SPELLING_DEDUCTION_RULES}

Return JSON only, in this exact shape:
{{
  "content":    {{"score": <number 0-4>, "reasoning": "<one sentence>"}},
  "grammar":    {{
    "score": <number 0-2>,
    "reasoning": "grammar=N [...] -> grammar_remaining=X",
    "mistake_quotes": [<exact verbatim substrings flagged as grammar>]
  }},
  "vocabulary": {{"score": <number 0-2>, "reasoning": "<one sentence>"}},
  "spelling":   {{
    "score": <number 0-2>,
    "reasoning": "spelling=M [...] -> spelling_remaining=Y",
    "mistake_quotes": [<exact verbatim misspelled words>]
  }}
}}
"""


# ── Public scoring functions ─────────────────────────────────────────────────

def score_swt_subscores_with_openai(passage: str, user_text: str) -> dict:
    """Score SWT sub-scores via gpt-5-nano. Mirrors the return shape of
    score_swt_subscores_with_claude so ai_scorer.py can swap them with no
    other changes.

        {
            "content":    {"score": float, "reasoning": str|None},
            "grammar":    {"score": float, "reasoning": str|None},
            "vocabulary": {"score": float, "reasoning": str|None},
            "scored":     bool,
            "warning_code": Optional[str],
        }

    On any failure returns scored=False so the caller can fall back to Claude.
    Never raises.
    """
    if not passage or not passage.strip():
        return _swt_failure("content_llm_unavailable", reason="empty passage")
    if not user_text or not user_text.strip():
        return {
            "content":    {"score": 0.0, "reasoning": None},
            "grammar":    {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "scored": True,
            "warning_code": None,
        }
    user_block = f"PASSAGE:\n{passage}\n\nSTUDENT SUMMARY:\n{user_text}"
    parsed = _call_openai(_SWT_SYSTEM_PROMPT, user_block, label="SWT")
    if parsed is None:
        return _swt_failure("content_llm_unavailable", reason="gpt-5-nano failed after retries")
    return {
        "content":    _parse_sub(parsed, "content", 4),
        "grammar":    _parse_sub(parsed, "grammar", 2),
        "vocabulary": _parse_sub(parsed, "vocabulary", 2),
        "scored": True,
        "warning_code": None,
    }


def score_we_subscores_with_openai(prompt: str, user_text: str) -> dict:
    """Score WE sub-scores via gpt-5-nano. Six sub-scores: content, dsc,
    grammar, glr, vocabulary, spelling. Form is deterministic upstream and
    NOT in the prompt or response.
    """
    if not prompt or not prompt.strip():
        return _we_failure("content_llm_unavailable", reason="empty prompt")
    if not user_text or not user_text.strip():
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
    user_block = f"ESSAY PROMPT:\n{prompt}\n\nSTUDENT ESSAY:\n{user_text}"
    parsed = _call_openai(_WE_SYSTEM_PROMPT, user_block, label="WE")
    if parsed is None:
        return _we_failure("content_llm_unavailable", reason="gpt-5-nano failed after retries")
    return {
        "content":    _parse_sub(parsed, "content", 6),
        "dsc":        _parse_sub(parsed, "dsc", 6),
        "grammar":    _parse_sub(parsed, "grammar", 2),
        "glr":        _parse_sub(parsed, "glr", 6),
        "vocabulary": _parse_sub(parsed, "vocabulary", 2),
        "spelling":   _parse_sub(parsed, "spelling", 2),
        "scored": True,
        "warning_code": None,
    }


def score_sst_subscores_with_openai(reference: str, user_text: str) -> dict:
    """Score SST sub-scores via gpt-5-nano. Four sub-scores: content, grammar,
    vocabulary, spelling. Form is deterministic upstream.
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
    user_block = f"REFERENCE (what the audio was about):\n{reference}\n\nSTUDENT SUMMARY:\n{user_text}"
    parsed = _call_openai(_SST_SYSTEM_PROMPT, user_block, label="SST")
    if parsed is None:
        return _sst_failure("content_llm_unavailable", reason="gpt-5-nano failed after retries")
    return {
        "content":    _parse_sub(parsed, "content", 4),
        "grammar":    _parse_sub(parsed, "grammar", 2),
        "vocabulary": _parse_sub(parsed, "vocabulary", 2),
        "spelling":   _parse_sub(parsed, "spelling", 2),
        "scored": True,
        "warning_code": None,
    }


# ── Internals ────────────────────────────────────────────────────────────────

def _call_openai(system_prompt: str, user_block: str, label: str) -> Optional[dict]:
    """Call gpt-5-nano with 3-retry envelope. Returns parsed JSON dict on
    success, None on full failure. Never raises."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(1, 4):
        try:
            resp = _client.chat.completions.create(
                model=_MODEL,
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                reasoning_effort=_REASONING_EFFORT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_block},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            if not raw:
                finish = resp.choices[0].finish_reason
                last_exc = RuntimeError(f"empty content; finish_reason={finish}")
                # finish_reason=length is deterministic for a given (prompt, model,
                # max_tokens) pair — retrying with identical inputs produces an
                # identical empty response and just wastes ~30s per attempt before
                # the Claude fallback finally runs. Break out early so the caller
                # falls to Claude in seconds, not minutes.
                if finish == "length":
                    log.warning(
                        "[OPENAI-%s] finish=length on attempt=%d/3 — skipping remaining retries (deterministic failure)",
                        label, attempt,
                    )
                    break
                if attempt < 3:
                    time.sleep(2)
                continue
            raw = raw.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            parsed = json.loads(raw)

            usage = resp.usage
            log.info(
                "[OPENAI-%s] model=%s in=%d out=%d reasoning=%d",
                label, _MODEL,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                getattr(getattr(usage, "completion_tokens_details", None),
                        "reasoning_tokens", 0) or 0,
            )
            return parsed
        except AuthenticationError as exc:
            log.error("[OPENAI-%s] AuthenticationError — not retrying: %s", label, exc)
            return None
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[OPENAI-%s] attempt=%d/3 failed — %s: %s",
                label, attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)
    log.error(
        "[OPENAI-%s] failed after 3 attempts — %s: %s",
        label, type(last_exc).__name__, last_exc,
    )
    return None


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


def _parse_sub(parsed: dict, key: str, max_value: int) -> dict:
    entry = parsed.get(key, {}) or {}
    out = {
        "score": _clamp(entry.get("score"), 0, max_value),
        "reasoning": _reasoning(entry.get("reasoning")),
    }
    # Pass through any mistake-quote arrays the LLM emits for highlight
    # builder. SWT grammar field returns two arrays (grammar_mistake_quotes,
    # spelling_mistake_quotes) because it combines both sub-scores; WE/SST
    # grammar and spelling each return a single mistake_quotes array.
    for k in ("mistake_quotes", "grammar_mistake_quotes", "spelling_mistake_quotes"):
        v = entry.get(k)
        if isinstance(v, list):
            out[k] = [str(x) for x in v if x]
    return out


def _swt_failure(warning_code: str, reason: str = "") -> dict:
    return {
        "content":    {"score": 0.0, "reasoning": None},
        "grammar":    {"score": 0.0, "reasoning": None},
        "vocabulary": {"score": 0.0, "reasoning": None},
        "scored": False,
        "warning_code": warning_code,
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


def _sst_failure(warning_code: str, reason: str = "") -> dict:
    return {
        "content":    {"score": 0.0, "reasoning": None},
        "grammar":    {"score": 0.0, "reasoning": None},
        "vocabulary": {"score": 0.0, "reasoning": None},
        "spelling":   {"score": 0.0, "reasoning": None},
        "scored": False,
        "warning_code": warning_code,
    }
