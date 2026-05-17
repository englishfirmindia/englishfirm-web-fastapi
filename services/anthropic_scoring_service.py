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
from concurrent.futures import ThreadPoolExecutor
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

  STEP 1 — Base grammar score (0–2). Score by COMMUNICATION IMPACT only.

    WORKED EXAMPLES — calibrate against these:

    Score 2/2 (long compound, valid):
    "While major events worldwide are joining a climate network, the sporting events are the latest participants, and are important for inspiring global action, said Steiner, and organizers say they will spend one billion on infrastructure."
    → Compound, multiple 'and', mid-sentence attribution. Grammatically valid. Score: 2.

    Score 1/2 (real grammatical error):
    "The organizers says they will spent one billion on infrastructure that are important."
    → Subject-verb disagreement (organizers says, that are), wrong tense (spent for spend). Score: 1.

    Score 0/2 (defective):
    "While events striving carbon, the participants joining important inspire global, organizer say spend billion."
    → Missing verbs/articles, broken clauses, reader cannot parse. Score: 0.

  The "long sentence with many 'and's" pattern is NORMAL for SWT and is NOT a deduction trigger.

  STEP 2 — Spelling deduction (applied to the base score, floored at 0):
    0 spelling errors → no deduction
    1 spelling error  → subtract 1
    2 or more         → grammar score = 0 regardless of base

  Final grammar score = max(0, base − spelling_deduction).
  In the reasoning, name the specific violation type (or state "no violations") and the spelling-error count.

VOCABULARY (0–2) — score by APPROPRIATENESS, not paraphrasing:
  2 = Has appropriate choice of words. The vocabulary fits the meaning.
  1 = Contains lexical errors (wrong word for the context, awkward collocation), but with no hindrance to communication.
  0 = Has defective word choice which could hinder communication — a wrong-meaning word that confuses the reader.
Important: PTE demands "appropriate choice of words" for vocabulary. Using original words from the passage does NOT lower the score — paraphrasing is NOT required and verbatim phrase reuse is NOT penalised here. Only mark down if a chosen word is wrong, awkward, or inappropriate for the context.

For each sub-score, give the integer score AND a one-sentence reasoning that cites specific evidence (a word, an error, an aspect of the response). Keep reasoning under 25 words. Be honest about errors — students rely on the feedback.

PARAPHRASING AUDIT (informational, not a sub-score):
  Also count the student's paraphrasing effort vs the passage:
    synonyms_count — substantive content words (nouns/verbs/adjectives/adverbs) from the passage that the student replaced with synonyms. Skip function words. Each distinct swap counts at most once.
    paraphrased_phrases_count — multi-word phrases (2+ content words) from the passage that the student restructured. Single-word swaps don't count here.
    examples — up to 5 short "src → user" audit strings.

Return JSON only, in this exact shape:
{
  "content":    {"score": <int 0-4>, "reasoning": "<one sentence>"},
  "grammar":    {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "vocabulary": {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "paraphrasing": {
    "synonyms_count": <int>,
    "paraphrased_phrases_count": <int>,
    "examples": [<short strings>]
  }
}
"""


# ── Split SWT prompts (content v3 / grammar / vocab v3.1) ────────────────────
# Each prompt is focused on ONE sub-score so the rubric can be iterated without
# regression risk on the others. All three are fired in parallel via
# ThreadPoolExecutor — see score_swt_subscores_with_claude below.

_SWT_CONTENT_V3_PROMPT = """You are scoring CONTENT only for a PTE Academic Summarize Written Text response.

The student is given a passage and writes a one-sentence summary in 5–75 words. Form, Grammar, and Vocabulary are scored separately — ignore them. Content is scored on coverage of key points AND paraphrasing effort.

CONTENT (0–4, half-point granularity allowed: 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4):

SCORE 4 — All key points covered AND paraphrasing threshold met (4+ content-word synonym swaps OR 2+ phrases visibly restructured — clause reorder, connector reshape, parenthetical → relative clause). Function-word changes don't count.

SCORE 3.5 — Full key-point coverage AND partial paraphrasing (1–3 word swaps OR 1 phrase restructure). OR full verbatim coverage capturing BOTH the central insight AND the resolution.

SCORE 3 — All key points covered, sentences copied exactly from the passage (verbatim phrasing throughout). No paraphrasing effort.

SCORE 2.5 — Key points partially covered (more than "very few" but less than full), verbatim or near-verbatim. OR fact-listing without synthesis. OR ≥3 of 4 arc pieces verbatim, missing one.

SCORE 2 — Very few key points conveyed (only 1–2 main propositions), OR full attempt <50 words. Generic mentions ("numerous advantages" without specifics) count as named-not-conveyed → here.

SCORE 1.5 — Very few key points AND presentation issues (5+ spelling errors, broken connector chain) blocking comprehension.

SCORE 1 — Very few key points AND (<30 words OR leads with illustrative content — metaphor/example/quote — while omitting the central thesis). OR has only half of a two-part central insight.

SCORE 0 — Off-topic or blank.

CRITICAL-MOVE RULES:
  • Factual misrepresentation of source's central claim → cap at 0.5.
  • Named-without-specifics: arc pieces that are NAMED ("numerous advantages") without ANY concrete content count as half-present, not present.
  • List-style penalty: shallow named-only mentions chained with list connectors (and… furthermore… moreover…) cap at 2.5 regardless of how many arc pieces are nominally touched. Breadth without depth ≠ coverage.

In the reasoning, name which key points were covered vs missing, paraphrasing evidence (cite a swap or note "verbatim"), and any rule applied.

Return JSON only, exactly this shape:
{
  "content": {"score": <number 0-4 in 0.5 steps>, "reasoning": "<one sentence>"},
  "paraphrasing": {
    "synonyms_count": <int>,
    "paraphrased_phrases_count": <int>,
    "examples": [<up to 5 short "src → user" strings>]
  }
}
"""

_SWT_GRAMMAR_PROMPT = """You are scoring GRAMMAR + SPELLING only for a PTE Academic Summarize Written Text response.

The student writes a one-sentence summary in 5–75 words. You return a single 0–2 score reflecting grammatical structure with a spelling deduction layered on top.

GRAMMAR + SPELLING (0–2) — combined score. Two-step process:

  STEP 1 — Base grammar score (0–2). Score by COMMUNICATION IMPACT only.

    WORKED EXAMPLES — calibrate against these:

    Score 2/2 (long compound, valid):
    "While major events worldwide are joining a climate network, the sporting events are the latest participants, and are important for inspiring global action, said Steiner, and organizers say they will spend one billion on infrastructure."
    → Compound, multiple 'and', mid-sentence attribution. Grammatically valid. Score: 2.

    Score 1/2 (real grammatical error):
    "The organizers says they will spent one billion on infrastructure that are important."
    → Subject-verb disagreement (organizers says, that are), wrong tense (spent for spend). Score: 1.

    Score 0/2 (defective):
    "While events striving carbon, the participants joining important inspire global, organizer say spend billion."
    → Missing verbs/articles, broken clauses, reader cannot parse. Score: 0.

  The "long sentence with many 'and's" pattern is NORMAL for SWT and is NOT a deduction trigger.

  STEP 2 — Spelling deduction (applied to the base score, floored at 0):
    0 spelling errors → no deduction
    1 spelling error  → subtract 1
    2 or more         → grammar score = 0 regardless of base

  Final grammar score = max(0, base − spelling_deduction).
  In the reasoning, name the specific violation type (or state "no violations") and the spelling-error count.

ALSO populate `mistakes`: a JSON array of objects, one per grammar error.
Each object MUST have these keys:
  quote      — the offending verbatim substring from the student summary
               (case-sensitive, exact characters; the UI substring-matches
               this back into the body to underline it).
  correction — the corrected text the student should have written. Keep
               the correction minimal — replace only the offending span,
               do not rewrite the whole sentence.
  reason     — a short label naming the error type (e.g. "Singular
               required", "Subject-verb agreement", "Wrong tense").
Empty list if no errors. Order should match the natural reading order.

For backward compatibility ALSO emit `mistake_quotes` as a flat array of
the same verbatim substrings (the same `quote` values, in the same order).

Return JSON only, exactly this shape:
{
  "grammar": {
    "score": <number 0-2 in 0.5 steps>,
    "reasoning": "<one sentence>",
    "mistake_quotes": [<exact verbatim substrings>],
    "mistakes": [
      {"quote": "<exact substring>",
       "correction": "<corrected text>",
       "reason": "<short label>"}
    ]
  }
}
"""

_SWT_VOCAB_V31_PROMPT = """You are scoring VOCABULARY only for a PTE Academic Summarize Written Text response.

The student writes a one-sentence summary. You return a single 0–2 vocabulary score reflecting APPROPRIATENESS and PRECISION of word choice. Paraphrasing is NOT required and verbatim phrase reuse is NOT penalised here — that's a content-side concern.

VOCABULARY (0–2, half-points allowed: 0, 0.5, 1, 1.5, 2):

  SCORE 2.0 — 0 vocabulary errors. Word choices are appropriate and precise.

  SCORE 1.5 — Exactly 1 minor issue: a single non-word typo (e.g. "noctural" for "nocturnal") OR one mildly awkward collocation. Meaning fully clear.

  SCORE 1.0 — Either 1 wrong-meaning content word (e.g. "developed the public's imagination" when "captivated" is required) OR 2–3 awkward word choices. Meaning still recoverable.

  SCORE 0.5 — 3 wrong-meaning / awkward choices, at least one comprehension-blocking. Reader has to re-parse.

  SCORE 0 — 4+ wrong-meaning / awkward choices, OR pervasive non-word typos producing unparseable text.

REPETITION PENALTY:
  If the summary is short (≤60 words) AND mostly verbatim AND repeats the SAME content word ≥2× (e.g. "scientists" used twice, "advantages" used twice), subtract 1.0 from the base. This catches over-reliance on a single lexical anchor.

TYPOS AS VOCAB ERRORS:
  Typos that produce non-English non-words (e.g. "demirts", "carfull", "sustanblity", "prevelent") COUNT as vocabulary errors here. A typo that produces a different real word (e.g. "their" for "there") counts as grammar, not vocab.

HARD EXCLUSIONS — do NOT deduct vocabulary for:
  • Spelling that produces a real word ("their"/"there") — that's grammar.
  • Verbatim reuse of source vocabulary — verbatim is not a vocab issue.
  • Stylistic length / sentence structure — that's grammar.

In the reasoning, count: "N vocab errors found: [list]; repetition penalty: yes/no."

Return JSON only, exactly this shape:
{
  "vocabulary": {"score": <number 0-2 in 0.5 steps>, "reasoning": "<one sentence>"}
}
"""


def _swt_call_one(system_prompt: str, passage: str, user_text: str,
                  max_tokens: int, log_tag: str) -> Optional[dict]:
    """Single Claude call for one SWT sub-score. Returns parsed dict on
    success, None on failure after 3 retries. Never raises.
    """
    last_exc: Exception = RuntimeError(f"{log_tag}: no attempts made")
    for attempt in range(1, 4):
        try:
            response = _client.messages.create(
                model=_MODEL,
                temperature=0,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
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
                last_exc = RuntimeError(f"{log_tag}: empty response")
                if attempt < 3:
                    time.sleep(2)
                continue
            text = text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
            parsed = json.loads(text)
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            log.info(
                "[CLAUDE-SWT-%s] in=%d out=%d cache_read=%d",
                log_tag, response.usage.input_tokens, response.usage.output_tokens, cache_read,
            )
            return parsed
        except anthropic.AuthenticationError as exc:
            log.error("[CLAUDE-SWT-%s] AuthenticationError — not retrying: %s", log_tag, exc)
            return None
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[CLAUDE-SWT-%s] attempt=%d/3 failed — %s: %s",
                log_tag, attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)
    log.error("[CLAUDE-SWT-%s] failed after 3 attempts: %s", log_tag, last_exc)
    return None


def score_swt_subscores_with_claude(passage: str, user_text: str) -> dict:
    """Score SWT sub-scores via Claude Haiku 4.5.

    Fires THREE parallel calls — content (v3 rubric), grammar (worked-examples
    rubric), vocabulary (v3.1 rubric) — via ThreadPoolExecutor. Each rubric
    lives in its own focused prompt so we can iterate on one without
    regression risk on the others.

    Returns a dict carrying per-sub-score `{score, reasoning}` plus a
    `scored` flag and (when relevant) a `warning_code` for the partial-
    success plumbing the rest of the pipeline already understands:

        {
            "content":    {"score": float, "reasoning": str|None},
            "grammar":    {"score": float, "reasoning": str|None},
            "vocabulary": {"score": float, "reasoning": str|None},
            "scored":     bool,
            "warning_code": Optional[str],  # "content_llm_unavailable" when content sub-call failed
        }

    `scored` follows the content sub-call only — that's the load-bearing one
    that callers gate fallback paths on. Grammar/vocab failures are tolerated
    by returning score=0 with reasoning=None.

    Never raises.
    """
    if not passage or not passage.strip():
        return _failure("content_llm_unavailable", reason="empty passage")
    if not user_text or not user_text.strip():
        # No student text to score — return zeros without any LLM call.
        return {
            "content": {"score": 0.0, "reasoning": None},
            "grammar": {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "scored": True,
            "warning_code": None,
        }

    # Fire all three sub-calls in parallel. Each call has its own retry loop
    # and returns None on failure. Total wall-time = max(per-call latency).
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        # Token budgets are headroomed: content reasoning + paraphrasing
        # audit can easily exceed 400 tokens; first ship had 300 which
        # truncated JSON mid-string → JSONDecodeError → false fallback.
        f_content = pool.submit(_swt_call_one, _SWT_CONTENT_V3_PROMPT,
                                passage, user_text, 800, "CONTENT")
        f_grammar = pool.submit(_swt_call_one, _SWT_GRAMMAR_PROMPT,
                                passage, user_text, 400, "GRAMMAR")
        f_vocab   = pool.submit(_swt_call_one, _SWT_VOCAB_V31_PROMPT,
                                passage, user_text, 400, "VOCAB")
        content_parsed = f_content.result()
        grammar_parsed = f_grammar.result()
        vocab_parsed   = f_vocab.result()
    dt_ms = int((time.monotonic() - t0) * 1000)

    # Content is the load-bearing sub-call — its failure flips scored=False.
    if content_parsed is None:
        log.error("[CLAUDE-SWT] content sub-call failed; returning fallback")
        return _failure("content_llm_unavailable", reason="content sub-call failed")

    content_score = _clamp(content_parsed.get("content", {}).get("score"), 0, 4)
    content_reasoning = _reasoning(content_parsed.get("content", {}).get("reasoning"))
    paraphrasing = _parse_paraphrasing_block(content_parsed)

    if grammar_parsed is not None:
        gr = grammar_parsed.get("grammar", {})
        grammar_score = _clamp(gr.get("score"), 0, 2)
        grammar_reasoning = _reasoning(gr.get("reasoning"))
        grammar_mistakes = _parse_mistakes(gr)
        grammar_quotes = [m["quote"] for m in grammar_mistakes]
    else:
        grammar_score, grammar_reasoning = 0.0, None
        grammar_mistakes, grammar_quotes = [], []

    if vocab_parsed is not None:
        vocab_score = _clamp(vocab_parsed.get("vocabulary", {}).get("score"), 0, 2)
        vocab_reasoning = _reasoning(vocab_parsed.get("vocabulary", {}).get("reasoning"))
    else:
        vocab_score, vocab_reasoning = 0.0, None

    log.info(
        "[CLAUDE-SWT] split-parallel content=%.1f grammar=%.1f vocab=%.1f total_ms=%d",
        content_score, grammar_score, vocab_score, dt_ms,
    )

    return {
        "content": {
            "score": content_score,
            "reasoning": content_reasoning,
            "paraphrasing": paraphrasing,
        },
        "grammar": {
            "score": grammar_score,
            "reasoning": grammar_reasoning,
            "mistake_quotes": grammar_quotes,
            "mistakes": grammar_mistakes,
        },
        "vocabulary": {"score": vocab_score, "reasoning": vocab_reasoning},
        "scored": True,
        "warning_code": None,
    }


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


def _parse_mistakes(sub: dict) -> list:
    """Extract a normalised list of mistakes from a grammar/spelling sub-block.

    Accepts EITHER:
      - new shape  → `mistakes: [{quote, correction, reason}, …]`
      - legacy     → `mistake_quotes: ["word", "phrase", …]`

    Returns a list of `{"quote": str, "correction": str|None, "reason": str|None}`.
    Empty quotes filtered out.
    """
    out: list = []
    raw = sub.get("mistakes")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                quote = str(item.get("quote") or "").strip()
                if not quote:
                    continue
                correction = item.get("correction")
                correction = str(correction).strip() if correction else None
                reason = item.get("reason")
                reason = str(reason).strip() if reason else None
                out.append({"quote": quote, "correction": correction or None, "reason": reason or None})
            elif isinstance(item, str) and item.strip():
                out.append({"quote": item.strip(), "correction": None, "reason": None})
    if out:
        return out
    # Legacy fallback: flat list of strings.
    for q in (sub.get("mistake_quotes") or []):
        s = str(q or "").strip()
        if s:
            out.append({"quote": s, "correction": None, "reason": None})
    return out


# ── Grammar-only validators (cross-LLM fallback) ────────────────────────────
# Fire when the primary GPT-4o split returned grammar.score<2 with no itemised
# mistakes — a known compliance issue where the model agrees there are errors
# but skips the enumeration. These functions re-run JUST the grammar sub-call
# through Claude (which has been reliable at producing the mistakes array),
# leaving the other sub-scores untouched.

def score_swt_grammar_only(passage: str, user_text: str) -> dict:
    """Single Claude call for SWT grammar only. Returns
        {"scored": bool, "score": float, "reasoning": str|None, "mistakes": [...]}
    Never raises. Used as validator fallback after GPT-4o primary."""
    if not user_text or not user_text.strip():
        return {"scored": True, "score": 0.0, "reasoning": None, "mistakes": []}
    parsed = _swt_call_one(
        _SWT_GRAMMAR_PROMPT, passage or "", user_text, 500, "GRAMMAR-VALIDATOR"
    )
    if parsed is None:
        return {"scored": False, "score": 0.0, "reasoning": None, "mistakes": []}
    gr = parsed.get("grammar") or {}
    return {
        "scored": True,
        "score": _clamp(gr.get("score"), 0, 2),
        "reasoning": _reasoning(gr.get("reasoning")),
        "mistakes": _parse_mistakes(gr),
    }


def score_we_grammar_only(prompt: str, user_text: str) -> dict:
    """Single Claude call for WE grammar only. Mirror of score_swt_grammar_only."""
    if not user_text or not user_text.strip():
        return {"scored": True, "score": 0.0, "reasoning": None, "mistakes": []}
    parsed = _we_call_one(
        _WE_GRAMMAR_PROMPT, prompt or "", user_text, 500, "GRAMMAR-VALIDATOR"
    )
    if parsed is None:
        return {"scored": False, "score": 0.0, "reasoning": None, "mistakes": []}
    gr = parsed.get("grammar") or {}
    return {
        "scored": True,
        "score": _clamp(gr.get("score"), 0, 2),
        "reasoning": _reasoning(gr.get("reasoning")),
        "mistakes": _parse_mistakes(gr),
    }


def _parse_paraphrasing_block(parsed: dict) -> dict:
    """Pull the paraphrasing-audit block off the Claude response. Mirrors
    services.openai_scoring_service._parse_paraphrasing."""
    p = parsed.get("paraphrasing") or {}
    try:
        synonyms = int(p.get("synonyms_count") or 0)
    except (TypeError, ValueError):
        synonyms = 0
    try:
        phrases = int(p.get("paraphrased_phrases_count") or 0)
    except (TypeError, ValueError):
        phrases = 0
    examples = [str(x) for x in (p.get("examples") or []) if x][:5]
    return {
        "synonyms_count": max(0, synonyms),
        "paraphrased_phrases_count": max(0, phrases),
        "examples": examples,
    }


# ── Grammar-only judge (SWT / WE / SST) ───────────────────────────────────────
# Routed through Claude because benchmarking showed gpt-5-nano (with
# reasoning_effort=minimal) gives full marks on 17/20 submissions even when
# real grammar errors are present (subject-verb agreement, wrong tense,
# awkward word choice). Claude catches those reliably in ~0.8s.

_GRAMMAR_VOCAB_SYSTEM_PROMPT = """You are a grammar AND vocabulary judge for a PTE Academic writing response.

You return TWO sub-scores, GRAMMAR (0-2) and VOCABULARY (0-2). Each follows its own strict rubric.

══════════════════════════════════════════════════════════════════════════════
GRAMMAR (0-2) — STRUCTURAL ERRORS ONLY
══════════════════════════════════════════════════════════════════════════════

A word being misspelled does NOT count under grammar. Only structural errors count.

Categories to count (one deduction per distinct error):
  (a) CAPITALISATION — sentence start lowercase; proper noun lowercase; mid-sentence common noun capitalised without reason.
  (b) PUNCTUATION — missing terminal mark; wrong punctuation; obviously required comma missing (between independent clauses without a conjunction).
  (c) SPACING — double space; missing space after punctuation.
  (d) VERB USE — wrong tense; subject-verb disagreement; broken auxiliary chain ("did had", "have went").
  (e) AGREEMENT / DETERMINERS — wrong article; pronoun mismatch ("there" vs "their"); count/non-count mismatch.
  (f) WORD-FORM — wrong part of speech ("becoming environmental performance better"), awkward verb-noun pairing.
  (g) PLURALS — wrong plural ("transportations", "electricitys", "infrastructures" when "infrastructure" is the standard mass noun).

DO NOT count any of these as grammar errors:
  - Stylistic length / "lacks subordination" / "choppy flow" — long single-sentence answers are expected in SWT.
  - Optional commas before "and", "so", "but", "because".
  - Word choice that is inappropriate or imprecise — that's vocabulary, not grammar.
  - Spelling typos — handled separately.

grammar_score = max(0, 2 - grammar_count)

══════════════════════════════════════════════════════════════════════════════
VOCABULARY (0-2) — APPROPRIATENESS OF WORD CHOICE ONLY
══════════════════════════════════════════════════════════════════════════════

  2 = Appropriate choice of words. Vocabulary fits the meaning.
  1 = Contains lexical errors (wrong word for context, awkward collocation), but with no hindrance to communication.
  0 = Defective word choice which could hinder communication — a wrong-meaning word that confuses the reader.

HARD EXCLUSIONS — do NOT deduct vocabulary for any of these:
  • Spelling mistakes (typos) — that is the Spelling sub-score, NOT vocabulary. A correctly-chosen but misspelled word like "Whilew" is correct word choice and should not affect vocabulary.
  • Numerical or factual omissions ("one billion" vs "$1.75 billion") — that is the Content sub-score, NOT vocabulary. The student picked the right word ("billion"); the wrong number is a content/accuracy issue.
  • Using verbatim words from the passage — paraphrasing is NOT required. Verbatim phrase reuse must NOT lower vocabulary.
  • Awkward or choppy sentence flow caused by structure rather than word choice — that is Grammar / GLR, not vocabulary.

Only mark down vocabulary when an actual word is wrong, awkward, or inappropriate for its context.

══════════════════════════════════════════════════════════════════════════════
OUTPUT
══════════════════════════════════════════════════════════════════════════════

Return JSON ONLY in this exact shape:
{
  "grammar": {
    "score": <int 0-2>,
    "reasoning": "<one short sentence citing the specific error(s); say 'no errors' if none>",
    "mistakes": [
      {"quote": "<exact verbatim substring>", "correction": "<fixed wording>", "reason": "<short rule label, ≤6 words>"}
    ]
  },
  "vocabulary": {
    "score": <int 0-2>,
    "reasoning": "<one short sentence justifying the score>"
  }
}

Each grammar.mistakes[].quote MUST appear verbatim in the student's text (case-sensitive). `correction` is the minimal fixed fragment (e.g. "Researchers believe" for the quote "Researchers believes"). `reason` is a short rule label like "subject-verb agreement" or "wrong tense". Empty list if no errors."""


def score_grammar_and_vocab_with_claude(passage: str, user_text: str) -> dict:
    """Grammar + vocabulary judge via Claude Haiku 4.5 in one call.

    Returns:
        {
          "grammar":    {"score": float, "reasoning": str|None, "mistake_quotes": [str, ...]},
          "vocabulary": {"score": float, "reasoning": str|None},
          "scored":     bool,
          "warning_code": Optional[str],
        }
    Never raises.
    """
    if not user_text or not user_text.strip():
        return {
            "grammar": {"score": 0.0, "reasoning": None, "mistake_quotes": []},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "scored": True,
            "warning_code": None,
        }

    last_exc: Exception = RuntimeError("score_grammar_and_vocab_with_claude: no attempts made")
    for attempt in range(1, 4):
        try:
            response = _client.messages.create(
                model=_MODEL,
                temperature=0,
                max_tokens=600,
                system=[
                    {
                        "type": "text",
                        "text": _GRAMMAR_VOCAB_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"PASSAGE (context only, do NOT score):\n{passage or '(none)'}\n\n"
                                    f"STUDENT TEXT:\n{user_text}\n\n"
                                    "Reply with ONLY the JSON object."
                                ),
                            }
                        ],
                    }
                ],
                timeout=20.0,
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break
            if not text:
                last_exc = RuntimeError("Claude grammar+vocab judge returned no text")
                if attempt < 3:
                    time.sleep(2)
                continue

            text = text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

            parsed = json.loads(text)
            g = parsed.get("grammar") or {}
            v = parsed.get("vocabulary") or {}
            g_score = _clamp(g.get("score"), 0, 2)
            g_reasoning = _reasoning(g.get("reasoning"))
            mistakes = _parse_mistakes(g)
            quotes = [m["quote"] for m in mistakes]
            v_score = _clamp(v.get("score"), 0, 2)
            v_reasoning = _reasoning(v.get("reasoning"))

            log.info(
                "[CLAUDE-GRAMMAR-VOCAB] g=%s v=%s mistakes=%d in_tok=%s out_tok=%s",
                int(g_score), int(v_score), len(quotes),
                getattr(response.usage, "input_tokens", 0),
                getattr(response.usage, "output_tokens", 0),
            )
            return {
                "grammar": {
                    "score": g_score,
                    "reasoning": g_reasoning,
                    "mistake_quotes": quotes,
                    "mistakes": mistakes,
                },
                "vocabulary": {
                    "score": v_score,
                    "reasoning": v_reasoning,
                },
                "scored": True,
                "warning_code": None,
            }
        except anthropic.AuthenticationError as exc:
            log.error("[CLAUDE-GRAMMAR-VOCAB] AuthenticationError — not retrying: %s", exc)
            return {
                "grammar": {"score": 0.0, "reasoning": None, "mistake_quotes": []},
                "vocabulary": {"score": 0.0, "reasoning": None},
                "scored": False,
                "warning_code": "grammar_vocab_claude_unavailable",
            }
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[CLAUDE-GRAMMAR-VOCAB] attempt=%d/3 failed — %s: %s",
                attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)
    log.error(
        "[CLAUDE-GRAMMAR-VOCAB] failed after 3 attempts — %s: %s",
        type(last_exc).__name__, last_exc,
    )
    return {
        "grammar": {"score": 0.0, "reasoning": None, "mistake_quotes": []},
        "vocabulary": {"score": 0.0, "reasoning": None},
        "scored": False,
        "warning_code": "grammar_vocab_claude_unavailable",
    }


# ── Write Essay (WE) — 5-prompt split ────────────────────────────────────────
# Mirrors the SWT split architecture. Each sub-score has its own focused
# prompt so the rubric can be iterated without regression risk on the others.
# Spelling is NOT here — that runs as a deterministic check in ai_scorer.py.
# Form is computed deterministically upstream from word count.

_WE_CONTENT_PROMPT = """You are scoring CONTENT only for a PTE Academic Write Essay response.

The student is given a prompt and writes a 200–300 word essay in 20 minutes. DSC, Grammar, GLR, Vocabulary, and Spelling are scored separately — ignore them. Content is about RELEVANCE TO THE PROMPT and DEPTH OF ARGUMENT.

CONTENT (0–6, half-point granularity allowed):

  SCORE 6 — Fully addresses the prompt with a clear, well-developed position. Arguments are substantial and relevant throughout. Multiple supporting points with concrete examples.

  SCORE 5 — Adequately addresses the prompt; position is clear with good supporting points; minor gaps in argument depth or relevance.

  SCORE 4 — Addresses the prompt with a discernible position; arguments are present but some are shallow or partially off-topic.

  SCORE 3 — Partial address of the prompt; position is unclear OR arguments are mostly underdeveloped OR drifts off-topic in places.

  SCORE 2 — Minimal engagement with the prompt; relies heavily on generalities; off-topic for stretches.

  SCORE 1 — Barely addresses the prompt; mostly tangential or filler.

  SCORE 0 — Completely off-topic or fails to address the prompt at all.

CRITICAL-MOVE RULES:
  • Factual misrepresentation of the prompt's central claim → cap at 1.0.
  • Off-prompt: if the essay answers a DIFFERENT question than the one asked → cap at 1.0.
  • Generic filler ("technology has changed our lives", "many people think") without engaging the specific prompt → cap at 3.0.

In the reasoning, name what the prompt asked, whether the student took a position, what their main arguments are, and whether examples support the position.

Return JSON only, exactly this shape:
{
  "content": {"score": <number 0-6 in 0.5 steps>, "reasoning": "<one sentence>"}
}
"""

_WE_DSC_PROMPT = """You are scoring DEVELOPMENT, STRUCTURE & COHERENCE (DSC) only for a PTE Academic Write Essay response.

DSC measures PARAGRAPHING and how ideas connect within and across paragraphs. Content, Grammar, GLR, Vocabulary, and Spelling are scored separately — ignore them.

DSC (0–6, half-point granularity allowed):

  SCORE 6 — Clear introduction, body paragraphs, and conclusion. Multiple paragraphs separated by blank lines. Smooth transitions between paragraphs (connectives, topic sentences). Ideas flow naturally.

  SCORE 5 — Clear paragraphed structure with logical flow. Minor lapses in transitions or coherence.

  SCORE 4 — Identifiable paragraphed structure but uneven development or weak transitions in places.

  SCORE 3 — HARD CAP for single-paragraph essays. Some structure within the block but no paragraph separation. Also fires when paragraphs exist but are disorganised in parts; coherence breaks down within or across paragraphs.

  SCORE 2 — Poor structure; paragraphing absent or arbitrary; weak coherence throughout.

  SCORE 1 — Very disorganised; ideas don't connect.

  SCORE 0 — No discernible structure; incoherent.

CRITICAL-MOVE RULE — SINGLE PARAGRAPH:
  If the entire essay is written as ONE continuous block (no blank lines separating paragraphs), the DSC score CANNOT exceed 3.0 regardless of how well the ideas inside are organised. Paragraphing is a required signal in PTE essays.

In the reasoning, name how many paragraphs you see, whether the intro/body/conclusion are distinguishable, and quality of transitions.

Return JSON only, exactly this shape:
{
  "dsc": {"score": <number 0-6 in 0.5 steps>, "reasoning": "<one sentence>"}
}
"""

_WE_GRAMMAR_PROMPT = """You are scoring GRAMMAR only for a PTE Academic Write Essay response.

You return a single 0–2 score reflecting grammatical structure. Spelling is scored separately and is NOT your concern here.

GRAMMAR (0–2, half-points allowed):

  WORKED EXAMPLES — calibrate against these:

  Score 2/2 (valid grammar throughout):
  "Technology has transformed modern education in profound ways. Although traditional methods remain valuable, many institutions are now adopting blended approaches that combine both. This shift, which has accelerated since 2020, demonstrates that pedagogical evolution requires balancing innovation with proven practice."
  → Subordination, varied structures, correct subject-verb agreement, correct tenses. Score: 2.

  Score 1/2 (some real grammatical errors, meaning recoverable):
  "Technology has transform education in many way. Many institution is adopting new approach. This shift demonstrate that evolution require balance."
  → Subject-verb disagreement (institution is, shift demonstrate, evolution require), wrong tense (has transform), missing articles. Score: 1.

  Score 0/2 (multiple errors blocking comprehension):
  "Technology transforming education, institution adopting approach, shift evolution requiring balance traditional innovation."
  → Missing verbs, broken clauses, reader cannot parse. Score: 0.

Long sentences with subordination and multiple coordinating conjunctions are NORMAL essay style and NOT a deduction trigger.

ALSO populate `mistakes`: a JSON array of objects, one per flagged grammar error. Each object MUST have:
  - "quote":      exact verbatim substring from the essay (case-sensitive).
  - "correction": the minimal corrected wording (e.g. "institutions are" for "institution is").
  - "reason":     a short rule label (≤6 words), e.g. "subject-verb agreement", "wrong tense".

MANDATORY ITEMISATION RULE:
  • If `score == 2`  → `mistakes` MUST be `[]` (no errors).
  • If `score < 2`   → `mistakes` MUST contain at least one entry naming a
    concrete error in the essay. You may NOT score below 2 without listing
    the specific quotes that drove the deduction. A holistic "feels off"
    verdict is NOT acceptable — point to the words.
  • Aim for 1–6 entries; cap at 10. Cover the most damaging errors first.

In the reasoning, name the specific violation types (or state "no violations") in one sentence.

Return JSON only, exactly this shape:
{
  "grammar": {
    "score": <number 0-2 in 0.5 steps>,
    "reasoning": "<one sentence>",
    "mistakes": [
      {"quote": "<verbatim substring>", "correction": "<fixed wording>", "reason": "<short rule label>"}
    ]
  }
}
"""

_WE_GLR_PROMPT = """You are scoring GENERAL LINGUISTIC RANGE (GLR) only for a PTE Academic Write Essay response.

GLR measures COMPLEXITY AND VARIETY of sentence structures across the essay. Content, DSC, Grammar, Vocabulary, and Spelling are scored separately — ignore them.

GLR (0–6, half-point granularity allowed):

  SCORE 6 — Excellent range: complex and varied sentence structures (subordination, embedding, relative clauses, varied connectors); confidently controlled. Mix of simple, compound, complex, and compound-complex sentences.

  SCORE 5 — Good range with a mix of simple and complex sentences; some sophistication.

  SCORE 4 — Adequate range; mix of simple and a few complex sentences but with some control issues.

  SCORE 3 — Limited range; mostly simple sentences with occasional attempts at complexity.

  SCORE 2 — Very limited range; almost entirely simple, repetitive structures.

  SCORE 1 — Minimal range; one-clause sentences throughout.

  SCORE 0 — No control of sentence structures.

In the reasoning, name examples of sentence variety (or lack thereof) — cite specific structures used.

Return JSON only, exactly this shape:
{
  "glr": {"score": <number 0-6 in 0.5 steps>, "reasoning": "<one sentence>"}
}
"""

_WE_VOCAB_PROMPT = """You are scoring VOCABULARY only for a PTE Academic Write Essay response.

Vocabulary measures APPROPRIATENESS and PRECISION of word choice. Paraphrasing the prompt is NOT required — verbatim reuse of prompt vocabulary is NOT penalised here.

VOCABULARY (0–2, half-points allowed: 0, 0.5, 1, 1.5, 2):

  SCORE 2.0 — 0 vocabulary errors. Word choices are appropriate, varied, and precise. Academic register.

  SCORE 1.5 — Exactly 1 minor issue: a single non-word typo (e.g. "noctural" for "nocturnal") OR one mildly awkward collocation. Meaning fully clear.

  SCORE 1.0 — Either 1 wrong-meaning content word (e.g. "developed the public's imagination" when "captivated" is required) OR 2–3 awkward word choices. Meaning still recoverable.

  SCORE 0.5 — 3 wrong-meaning / awkward choices, at least one comprehension-blocking.

  SCORE 0 — 4+ wrong-meaning / awkward choices, OR pervasive non-word typos producing unparseable text.

REPETITION PENALTY:
  If the essay heavily repeats the same content word ≥5× (e.g. "important" used 6 times where synonyms would help), subtract 0.5. Essay-length texts have more room for repetition than summaries, so the threshold is higher than for SWT.

TYPOS AS VOCAB ERRORS:
  Typos that produce non-English non-words COUNT as vocabulary errors. A typo that produces a different real word counts as grammar, not vocab.

HARD EXCLUSIONS — do NOT deduct vocabulary for:
  • Spelling that produces a real word — that's grammar/spelling.
  • Verbatim reuse of prompt vocabulary — verbatim is not a vocab issue.
  • Stylistic length / sentence structure — that's GLR.

In the reasoning, count: "N vocab errors found: [list]; repetition: yes/no."

Return JSON only, exactly this shape:
{
  "vocabulary": {"score": <number 0-2 in 0.5 steps>, "reasoning": "<one sentence>"}
}
"""


def _we_call_one(system_prompt: str, prompt: str, user_text: str,
                 max_tokens: int, log_tag: str) -> Optional[dict]:
    """Single Claude call for one WE sub-score. Returns parsed dict on
    success, None on failure after 3 retries. Never raises.
    """
    last_exc: Exception = RuntimeError(f"{log_tag}: no attempts made")
    for attempt in range(1, 4):
        try:
            response = _client.messages.create(
                model=_MODEL,
                temperature=0,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
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
                timeout=60.0,
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break
            if not text:
                last_exc = RuntimeError(f"{log_tag}: empty response")
                if attempt < 3:
                    time.sleep(2)
                continue
            text = text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
            parsed = json.loads(text)
            return parsed
        except anthropic.AuthenticationError as exc:
            log.error("[CLAUDE-WE-%s] AuthenticationError — not retrying: %s", log_tag, exc)
            return None
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[CLAUDE-WE-%s] attempt=%d/3 failed — %s: %s",
                log_tag, attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)
    log.error("[CLAUDE-WE-%s] failed after 3 attempts: %s", log_tag, last_exc)
    return None


def score_we_subscores_with_claude_split(prompt: str, user_text: str) -> dict:
    """Score WE sub-scores via Claude Haiku 4.5, 5 parallel calls.

    Returns content/dsc/grammar/glr/vocabulary sub-scores. Spelling is NOT
    returned here — it's computed deterministically upstream in ai_scorer.py
    and merged into the breakdown there.

    Content is load-bearing — if it fails, scored=False; other sub-call
    failures return 0/None for that sub-score.
    """
    if not prompt or not prompt.strip():
        return _we_split_failure("empty prompt")
    if not user_text or not user_text.strip():
        return {
            "content":    {"score": 0.0, "reasoning": None},
            "dsc":        {"score": 0.0, "reasoning": None},
            "grammar":    {"score": 0.0, "reasoning": None, "mistake_quotes": []},
            "glr":        {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "scored": True,
            "warning_code": None,
        }

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as pool:
        f_c = pool.submit(_we_call_one, _WE_CONTENT_PROMPT, prompt, user_text, 600, "CONTENT")
        f_d = pool.submit(_we_call_one, _WE_DSC_PROMPT,     prompt, user_text, 400, "DSC")
        f_g = pool.submit(_we_call_one, _WE_GRAMMAR_PROMPT, prompt, user_text, 500, "GRAMMAR")
        f_l = pool.submit(_we_call_one, _WE_GLR_PROMPT,     prompt, user_text, 400, "GLR")
        f_v = pool.submit(_we_call_one, _WE_VOCAB_PROMPT,   prompt, user_text, 400, "VOCAB")
        content_parsed = f_c.result()
        dsc_parsed     = f_d.result()
        grammar_parsed = f_g.result()
        glr_parsed     = f_l.result()
        vocab_parsed   = f_v.result()
    dt_ms = int((time.monotonic() - t0) * 1000)

    if content_parsed is None:
        log.error("[CLAUDE-WE-SPLIT] content sub-call failed; returning fallback")
        return _we_split_failure("content sub-call failed")

    def _r(v):
        if v is None: return None
        s = str(v).strip()
        return s if s else None

    content_score = _clamp(content_parsed.get("content", {}).get("score"), 0, 6)
    content_reasoning = _r(content_parsed.get("content", {}).get("reasoning"))

    if dsc_parsed is not None:
        dsc_score = _clamp(dsc_parsed.get("dsc", {}).get("score"), 0, 6)
        dsc_reasoning = _r(dsc_parsed.get("dsc", {}).get("reasoning"))
    else:
        dsc_score, dsc_reasoning = 0.0, None

    if grammar_parsed is not None:
        gr = grammar_parsed.get("grammar", {})
        grammar_score = _clamp(gr.get("score"), 0, 2)
        grammar_reasoning = _r(gr.get("reasoning"))
        grammar_mistakes = _parse_mistakes(gr)
        grammar_quotes = [m["quote"] for m in grammar_mistakes]
    else:
        grammar_score, grammar_reasoning = 0.0, None
        grammar_mistakes, grammar_quotes = [], []

    if glr_parsed is not None:
        glr_score = _clamp(glr_parsed.get("glr", {}).get("score"), 0, 6)
        glr_reasoning = _r(glr_parsed.get("glr", {}).get("reasoning"))
    else:
        glr_score, glr_reasoning = 0.0, None

    if vocab_parsed is not None:
        vocab_score = _clamp(vocab_parsed.get("vocabulary", {}).get("score"), 0, 2)
        vocab_reasoning = _r(vocab_parsed.get("vocabulary", {}).get("reasoning"))
    else:
        vocab_score, vocab_reasoning = 0.0, None

    log.info(
        "[CLAUDE-WE-SPLIT] content=%.1f dsc=%.1f grammar=%.1f glr=%.1f vocab=%.1f total_ms=%d",
        content_score, dsc_score, grammar_score, glr_score, vocab_score, dt_ms,
    )

    return {
        "content":    {"score": content_score,    "reasoning": content_reasoning},
        "dsc":        {"score": dsc_score,        "reasoning": dsc_reasoning},
        "grammar":    {"score": grammar_score,    "reasoning": grammar_reasoning,
                        "mistake_quotes": grammar_quotes,
                        "mistakes": grammar_mistakes},
        "glr":        {"score": glr_score,        "reasoning": glr_reasoning},
        "vocabulary": {"score": vocab_score,      "reasoning": vocab_reasoning},
        "scored": True,
        "warning_code": None,
    }


def _we_split_failure(reason: str) -> dict:
    return {
        "content":    {"score": 0.0, "reasoning": None},
        "dsc":        {"score": 0.0, "reasoning": None},
        "grammar":    {"score": 0.0, "reasoning": None, "mistake_quotes": []},
        "glr":        {"score": 0.0, "reasoning": None},
        "vocabulary": {"score": 0.0, "reasoning": None},
        "scored": False,
        "warning_code": "content_llm_unavailable",
    }


# ── Legacy combined WE prompt (kept for the existing fallback path) ──────────

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

GRAMMAR (0–2). Score by COMMUNICATION IMPACT only.

  WORKED EXAMPLES — calibrate against these:

  Score 2/2 (varied, valid):
  "While technology has transformed modern education, some argue that traditional methods remain essential, and many institutions are now adopting blended approaches that combine both. This shift, which has accelerated since 2020, demonstrates that pedagogical evolution requires balancing innovation with proven practice."
  → Subordination, embedded clauses, varied structure, multiple 'and' connectors. All grammatically valid. Score: 2.

  Score 1/2 (real grammatical error):
  "Technology has transform modern education, and many institution is adopting new approach. This shift demonstrate that evolution require balance."
  → Subject-verb disagreement (institution is, shift demonstrate, evolution require), wrong tense (has transform), missing articles. Score: 1.

  Score 0/2 (defective):
  "Technology transforming education, institution adopting approach, shift evolution requiring balance traditional innovation."
  → Missing verbs, broken clauses, reader cannot parse. Score: 0.

  Long sentences with subordination and multiple 'and' connectors are NORMAL essay style and NOT a deduction trigger.

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

PARAPHRASING AUDIT (informational, not a sub-score):
  Also count the student's paraphrasing vs the essay PROMPT (and common-knowledge phrasing of the topic):
    synonyms_count — substantive content words from the prompt/topic that the student replaced with synonyms.
    paraphrased_phrases_count — multi-word phrases the student restructured rather than echoing verbatim.
    examples — up to 5 short "src → user" audit strings.

Return JSON only, in this exact shape:
{
  "content":    {"score": <int 0-6>, "reasoning": "<one sentence>"},
  "dsc":        {"score": <int 0-6>, "reasoning": "<one sentence>"},
  "grammar":    {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "glr":        {"score": <int 0-6>, "reasoning": "<one sentence>"},
  "vocabulary": {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "spelling":   {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "paraphrasing": {
    "synonyms_count": <int>,
    "paraphrased_phrases_count": <int>,
    "examples": [<short strings>]
  }
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
                temperature=0,
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

            content_sub["paraphrasing"] = _parse_paraphrasing_block(parsed)
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

GRAMMAR (0–2). Score by COMMUNICATION IMPACT only.

  WORKED EXAMPLES — calibrate against these:

  Score 2/2 (long compound, valid):
  "The lecturer explains that climate change affects polar regions through melting ice, rising sea levels, and disrupted ecosystems, and notes that adaptation strategies must address both mitigation and resilience."
  → Compound, multiple 'and', coordinated clauses. Grammatically valid. Score: 2.

  Score 1/2 (real grammatical error):
  "The lecturer explain that climate change affect polar region and adaptation strategies must addresses mitigation."
  → Subject-verb disagreement (lecturer explain, change affect, strategies addresses), missing articles. Score: 1.

  Score 0/2 (defective):
  "Lecturer explaining climate affect polar, adaptation strategies addressing mitigation resilience."
  → Missing verbs/articles, broken clauses, reader cannot parse. Score: 0.

  The "long sentence with many 'and's" pattern is NORMAL for SST and is NOT a deduction trigger.

VOCABULARY (0–2):
  2 = Appropriate, varied academic vocabulary; precise word choice.
  1 = Mostly appropriate but limited variety or some imprecise word choices.
  0 = Inappropriate, repetitive, or informal vocabulary; or near-verbatim copying from the reference.

SPELLING (0–2):
  2 = No spelling errors, or only one or two trivial typos.
  1 = A few spelling errors that don't impede meaning.
  0 = Multiple spelling errors, or errors that impede meaning.

For each sub-score, give the integer score AND a one-sentence reasoning that cites specific evidence (a word, an error, an aspect of the response). Keep reasoning under 25 words. Be honest about errors — students rely on the feedback.

PARAPHRASING AUDIT (informational, not a sub-score):
  Also count the student's paraphrasing effort vs the spoken-text REFERENCE:
    synonyms_count — substantive content words from the reference that the student replaced with synonyms.
    paraphrased_phrases_count — multi-word phrases the student restructured.
    examples — up to 5 short "src → user" audit strings.

Return JSON only, in this exact shape:
{
  "content":    {"score": <int 0-4>, "reasoning": "<one sentence>"},
  "grammar":    {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "vocabulary": {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "spelling":   {"score": <int 0-2>, "reasoning": "<one sentence>"},
  "paraphrasing": {
    "synonyms_count": <int>,
    "paraphrased_phrases_count": <int>,
    "examples": [<short strings>]
  }
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
                temperature=0,
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

            content_sub["paraphrasing"] = _parse_paraphrasing_block(parsed)
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
