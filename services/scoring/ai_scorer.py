"""
Hybrid AI scorer for writing/listening tasks.
Covers: summarize_written_text (swt), write_essay (we), listening_sst (sst).

SWT (current): deterministic form gate + Claude Haiku 4.5 for content,
grammar, and vocabulary in a single call. Sub-scores carry per-component
reasoning strings (W7-style). Max points come from RDS via rubric_cache.

WE / SST (still legacy): heuristics for form/grammar/vocab + gpt-4o-mini
for the content subscore. Fallback to word-count heuristic if LLM fails.
"""
import re

from .base import ScoringResult, ScoringStrategy, to_pte_score


# ── SWT heuristic (max 10 pts) ────────────────────────────────────────────────

def _score_swt_heuristic(user_text: str) -> tuple:
    """Returns (earned, max=10). Matches mobile _score_swt_heuristic."""
    if not user_text or not user_text.strip():
        return (0, 10)
    text = user_text.strip()
    words = text.split()
    wc = len(words)

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentences_score = 1 if len(sentences) == 1 else 0
    words_score     = 1 if 5 <= wc <= 75 else 0

    if sentences_score == 0 and words_score == 0:
        return (0, 10)

    content_score = 0
    if sentences_score == 1 and words_score == 1:
        content_score = min(4, max(1, wc // 10))

    grammar_score = 0
    if text[0].isupper():
        grammar_score += 1
    if text[-1] in '.!?':
        grammar_score += 1

    unique_words = len(set(w.lower().strip('.,!?;:()') for w in words if len(w) > 3))
    vocab_score = min(2, unique_words // 5)

    earned = sentences_score + words_score + content_score + grammar_score + vocab_score
    return (earned, 10)


# ── WE heuristic (max 15 pts) ─────────────────────────────────────────────────

def _score_we_heuristic(user_text: str) -> tuple:
    """Returns (earned, max=15). Matches mobile _score_we_heuristic."""
    if not user_text or not user_text.strip():
        return (0, 15)
    text = user_text.strip()
    words = text.split()
    wc = len(words)

    if 200 <= wc <= 300:
        form_score = 2
    elif (120 <= wc <= 199) or (301 <= wc <= 380):
        form_score = 1
    else:
        form_score = 0

    if form_score == 0:
        return (0, 15)

    content_score = 3 if wc >= 200 else 2 if wc >= 120 else 1 if wc >= 60 else 0

    grammar_score = 0
    if text[0].isupper():
        grammar_score += 1
    if text[-1] in '.!?':
        grammar_score += 1
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) >= 4:
        grammar_score += 1

    unique_words = len(set(w.lower().strip('.,!?;:()"\'- ') for w in words if len(w) > 3))
    if unique_words >= 60:
        vocab_score = 3
    elif unique_words >= 35:
        vocab_score = 2
    elif unique_words >= 15:
        vocab_score = 1
    else:
        vocab_score = 0

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paragraphs) >= 4:
        structure_score = 4
    elif len(paragraphs) == 3:
        structure_score = 3
    elif len(paragraphs) == 2:
        structure_score = 2
    elif len(sentences) >= 3:
        structure_score = 1
    else:
        structure_score = 0

    earned = form_score + content_score + grammar_score + vocab_score + structure_score
    return (earned, 15)


# ── SST heuristic (max 10 pts) ────────────────────────────────────────────────

def _score_sst_heuristic(user_text: str) -> dict:
    """Returns component scores dict for SST (max 10 pts)."""
    if not user_text or not user_text.strip():
        return {'earned': 0, 'max_pts': 10, 'form': 0, 'content': 0, 'grammar': 0, 'vocabulary': 0, 'word_count': 0}
    text = user_text.strip()
    words = text.split()
    wc = len(words)

    if 50 <= wc <= 70:
        form_score = 2
    elif 30 <= wc <= 49 or 71 <= wc <= 90:
        form_score = 1
    else:
        form_score = 0

    if form_score == 0:
        return {'earned': 0, 'max_pts': 10, 'form': 0, 'content': 0, 'grammar': 0, 'vocabulary': 0, 'word_count': wc}

    content_score = min(4, max(1, wc // 12))

    grammar_score = 0
    if text[0].isupper():
        grammar_score += 1
    if text[-1] in '.!?':
        grammar_score += 1

    unique_words = len(set(w.lower().strip('.,!?;:()') for w in words if len(w) > 3))
    vocab_score = min(2, unique_words // 5)

    earned = form_score + content_score + grammar_score + vocab_score
    return {
        'earned': earned, 'max_pts': 10,
        'form': form_score, 'content': content_score,
        'grammar': grammar_score, 'vocabulary': vocab_score,
        'word_count': wc,
    }


# ── SWT — hybrid form gate + Claude Haiku 4.5 ─────────────────────────────────

def _swt_heuristic_fallback(user_text: str, content_max: int,
                            grammar_max: int, vocab_max: int) -> dict:
    """Heuristic sub-scores used only when Claude is unreachable. Same logic
    as the pre-Claude scorer (kept verbatim so a fallback doesn't shift the
    score distribution): word-count-banded content, first-cap + final-punct
    grammar, unique-word-count vocab."""
    text = (user_text or '').strip()
    if not text:
        return {
            "content": {"score": 0.0, "reasoning": None},
            "grammar": {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
        }
    words = text.split()
    wc = len(words)

    content_score = min(content_max, max(1, wc // 10))

    g = 0
    if text[0].isupper():
        g += 1
    if text[-1] in '.!?':
        g += 1
    grammar_score = min(grammar_max, g)

    unique_words = len(set(w.lower().strip('.,!?;:()') for w in words if len(w) > 3))
    vocab_score = min(vocab_max, unique_words // 5)

    return {
        "content": {"score": float(content_score), "reasoning": None},
        "grammar": {"score": float(grammar_score), "reasoning": None},
        "vocabulary": {"score": float(vocab_score), "reasoning": None},
    }


def _score_swt_with_claude(text: str, prompt: str) -> ScoringResult:
    """SWT scoring path: deterministic form gate + Claude for the rest.

    Pipeline:
      1. Load max points from RDS via rubric_cache (form / content / grammar /
         vocabulary). Total max is the sum.
      2. Form gate — single sentence AND 5–75 words. Either fails → earned=0,
         PTE floors to 10.
      3. Claude Haiku 4.5 scores content + grammar + vocabulary in one call,
         returning per-sub-score `{score, reasoning}`.
      4. If LLM content == 0 → off-topic → earned=0, PTE floor.
      5. Otherwise earned = form_max + content + grammar + vocab; PTE = formula.

    If Claude is unreachable after 3 retries, falls back to the legacy
    heuristic for grammar/vocab/content (same numbers the pre-Claude scorer
    produced) and surfaces a `content_llm_unavailable` warning in the result.
    """
    from services.rubric_cache import get_rubric_max

    form_max    = get_rubric_max('summarize_written_text', 'form_max')
    content_max = get_rubric_max('summarize_written_text', 'content_max')
    grammar_max = get_rubric_max('summarize_written_text', 'grammar_max')
    vocab_max   = get_rubric_max('summarize_written_text', 'vocabulary_max')
    max_pts = form_max + content_max + grammar_max + vocab_max  # default = 9

    # ── Form gate (deterministic) ──────────────────────────────────────────
    body = (text or '').strip()
    wc = len(body.split())
    sentences = [s for s in re.split(r'[.!?]+', body) if s.strip()]
    form_ok = body and len(sentences) == 1 and 5 <= wc <= 75

    if not form_ok:
        if not body:
            reason = "Empty response."
        elif len(sentences) != 1:
            reason = f"Form gate failed — response has {len(sentences)} sentences, requires exactly 1."
        else:
            reason = f"Form gate failed — {wc} words is outside the 5–75 range."
        breakdown = {
            "form": 0,
            "content":    {"score": 0.0, "reasoning": "Not scored — form gate failed."},
            "grammar":    {"score": 0.0, "reasoning": "Not scored — form gate failed."},
            "vocabulary": {"score": 0.0, "reasoning": "Not scored — form gate failed."},
            "earned": 0,
            "max_pts": max_pts,
            "task_type": "swt",
            "scorer": "form-gate-floor",
            "scoring_warnings": [reason],
        }
        return ScoringResult(
            pte_score=to_pte_score(0.0),
            raw_score=0.0,
            is_async=False,
            breakdown=breakdown,
        )

    # ── Claude scores content + grammar + vocabulary in one call ───────────
    scoring_warnings: list = []
    scorer_label = "claude-haiku-4-5"
    try:
        from services.anthropic_scoring_service import score_swt_subscores_with_claude
        claude_result = score_swt_subscores_with_claude(prompt or '', body)
    except Exception as e:
        # The service itself shouldn't raise (it catches everything), but be
        # defensive in case an import or boot-time failure surfaces here.
        from core.logging_config import get_logger
        get_logger(__name__).error(f"[SWT] Claude call raised unexpectedly: {e}")
        claude_result = {"scored": False, "warning_code": "content_llm_unavailable"}

    if not claude_result.get("scored"):
        fallback = _swt_heuristic_fallback(body, content_max, grammar_max, vocab_max)
        content_sub    = fallback["content"]
        grammar_sub    = fallback["grammar"]
        vocabulary_sub = fallback["vocabulary"]
        scoring_warnings.append(
            claude_result.get("warning_code") or "content_llm_unavailable"
        )
        scorer_label = "heuristic-fallback"
    else:
        content_sub    = claude_result["content"]
        grammar_sub    = claude_result["grammar"]
        vocabulary_sub = claude_result["vocabulary"]

    # ── Off-topic floor: LLM content == 0 zeroes everything ───────────────
    if claude_result.get("scored") and content_sub["score"] == 0:
        breakdown = {
            "form": form_max,
            "content":    content_sub,
            "grammar":    grammar_sub,
            "vocabulary": vocabulary_sub,
            "earned": 0,
            "max_pts": max_pts,
            "task_type": "swt",
            "scorer": scorer_label,
            "scoring_warnings": ["content_off_topic"],
        }
        return ScoringResult(
            pte_score=to_pte_score(0.0),
            raw_score=0.0,
            is_async=False,
            breakdown=breakdown,
        )

    # ── Aggregate + PTE ────────────────────────────────────────────────────
    earned = (
        form_max
        + content_sub["score"]
        + grammar_sub["score"]
        + vocabulary_sub["score"]
    )
    earned = max(0.0, min(float(max_pts), earned))
    pct = earned / max_pts if max_pts > 0 else 0.0

    breakdown = {
        "form": form_max,
        "content":    content_sub,
        "grammar":    grammar_sub,
        "vocabulary": vocabulary_sub,
        "earned": earned,
        "max_pts": max_pts,
        "task_type": "swt",
        "scorer": scorer_label,
    }
    if scoring_warnings:
        breakdown["scoring_warnings"] = scoring_warnings

    return ScoringResult(
        pte_score=to_pte_score(pct),
        raw_score=pct,
        is_async=False,
        breakdown=breakdown,
    )


# ── Scorer class ──────────────────────────────────────────────────────────────

class AIScorer(ScoringStrategy):
    """
    Hybrid scorer: heuristics for form/grammar/vocab,
    GPT-4o-mini replaces only the content subscore.

    answer dict:
      text:   str  (user's written response)
      prompt: str  (original passage/question for content scoring)
    """

    is_async = False

    def __init__(self, task_type: str):
        self.task_type = task_type  # 'swt' | 'we' | 'sst'

    @property
    def is_async(self) -> bool:
        return False

    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        text   = answer.get('text', '')
        prompt = answer.get('prompt', '')

        if self.task_type == 'swt':
            # SWT now uses a hybrid path: deterministic form gate +
            # Claude Haiku 4.5 for content / grammar / vocabulary (each
            # sub-score returned with a per-component reasoning string).
            # See services/anthropic_scoring_service.py.
            return _score_swt_with_claude(text, prompt)

        elif self.task_type == 'we':
            earned, max_pts = _score_we_heuristic(text)
            wc = len(text.split())
            form_ok = 120 <= wc <= 380
            if not form_ok:
                # Form gate failed (word count outside 120–380) — floor to PTE 10
                earned = 0
            elif prompt and text.strip():
                from services.llm_content_scoring_service import score_we_content
                heuristic_content = 3 if wc >= 200 else 2 if wc >= 120 else 1 if wc >= 60 else 0
                llm_content = score_we_content(prompt, text)
                if llm_content == 0:
                    # LLM judged content completely off-topic — floor to PTE 10
                    earned = 0
                else:
                    earned = max(0, min(max_pts, earned - heuristic_content + llm_content))

        else:  # sst
            sst = _score_sst_heuristic(text)
            earned = sst['earned']
            max_pts = sst['max_pts']
            wc = sst['word_count']
            form_ok = 30 <= wc <= 90
            if not form_ok:
                # Form gate failed (word count outside 30–90) — floor to PTE 10
                earned = 0
                sst['content'] = 0
                sst['earned'] = 0
            elif prompt and text.strip():
                from services.llm_content_scoring_service import score_sst_content
                heuristic_content = sst['content']
                llm_content = score_sst_content(prompt, text)
                if llm_content == 0:
                    # LLM judged content completely off-topic — floor to PTE 10
                    earned = 0
                    sst['content'] = 0
                    sst['earned'] = 0
                else:
                    earned = max(0, min(max_pts, earned - heuristic_content + llm_content))
                    sst['content'] = llm_content
                    sst['earned'] = earned

            pct = earned / max_pts if max_pts > 0 else 0.0
            return ScoringResult(
                pte_score=to_pte_score(pct),
                raw_score=pct,
                is_async=False,
                breakdown={
                    'earned': earned,
                    'max_pts': max_pts,
                    'form': sst['form'],
                    'content': sst['content'],
                    'grammar': sst['grammar'],
                    'vocabulary': sst['vocabulary'],
                    'word_count': wc,
                    'task_type': self.task_type,
                },
            )

        pct = earned / max_pts if max_pts > 0 else 0.0
        return ScoringResult(
            pte_score=to_pte_score(pct),
            raw_score=pct,
            is_async=False,
            breakdown={
                'earned': earned,
                'max_pts': max_pts,
                'task_type': self.task_type,
            },
        )
