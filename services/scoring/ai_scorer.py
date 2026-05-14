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


_BUMP_SYNONYMS_THRESHOLD = 4   # ≥ 4 synonym swaps qualifies for +1
_BUMP_PHRASES_THRESHOLD  = 2   # OR ≥ 2 phrase rewrites qualifies for +1


def _apply_content_bump(content_sub: dict, content_max: int) -> dict:
    """Paraphrase-gated content bump.

    The historical unconditional +1 was unfair to copy-paste submissions and
    too stingy on light-rewording ones. Now the bump fires only when the LLM
    reports meaningful paraphrasing effort against the source — either
    ≥ _BUMP_SYNONYMS_THRESHOLD substantive synonym swaps OR
    ≥ _BUMP_PHRASES_THRESHOLD multi-word phrase rewrites.

    Still gated by CONTENT_BUMP_ENABLED and still skipped when content == 0
    so the off-topic / verbatim-copy floor keeps firing. Preserves the
    pre-bump LLM score in `llm_score` for trainer audit either way.
    Returns a new dict; never mutates the input.
    """
    from core.config import CONTENT_BUMP_ENABLED
    llm_score = float(content_sub.get("score", 0) or 0)
    if not CONTENT_BUMP_ENABLED or llm_score < 1:
        return content_sub

    para = content_sub.get("paraphrasing") or {}
    synonyms = int(para.get("synonyms_count") or 0)
    phrases  = int(para.get("paraphrased_phrases_count") or 0)
    qualifies = (synonyms >= _BUMP_SYNONYMS_THRESHOLD
                 or phrases >= _BUMP_PHRASES_THRESHOLD)

    if not qualifies:
        out = dict(content_sub)
        out["llm_score"] = llm_score
        out["bump_applied"] = False
        out["bump_reason"] = (
            f"No bump — paraphrasing below threshold "
            f"(synonyms={synonyms}, phrases={phrases})."
        )
        return out

    final = min(float(content_max), llm_score + 1.0)
    if final == llm_score:
        return content_sub  # already at cap; nothing to bump
    out = dict(content_sub)
    out["score"] = final
    out["llm_score"] = llm_score
    out["bump_applied"] = True
    out["bump_reason"] = (
        f"+1 for paraphrasing — {synonyms} synonym(s), {phrases} phrase rewrite(s)."
    )
    return out


# NOTE: _apply_verbatim_copy_floor() lived here briefly. It was a deterministic
# guard that floored Content to 0 when ≥70% of the student's words sat inside
# verbatim 6-grams from the passage. Removed by product call — the LLM rubric
# already covers near-verbatim copies, and the 70% threshold was creating
# false-positive PTE 10s on submissions that were heavily quoted but
# legitimate paraphrases. The detector module (services/copy_detector.py)
# is kept around in case we want to bring this back behind a soft warning.


def _split_sentences(text: str) -> list:
    """Split into sentences while ignoring decimals (e.g. '$1.75') so a number
    like '1.75 billion' isn't counted as two sentences. Common bug surfaced
    by SWT submissions that quote dollar figures from the source passage.
    Abbreviations like 'Mr.' / 'e.g.' are rare in PTE summaries and not
    handled here — flag separately if they show up in practice."""
    # Mask digit-dot-digit so the regex below doesn't fragment on decimals.
    safe = re.sub(r'(\d)\.(\d)', r'\1 \2', text or '')
    return [s for s in re.split(r'[.!?]+', safe) if s.strip()]


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
    sentences = _split_sentences(body)
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

    # ── Three calls in parallel: nano (full SWT, used for content only),
    # Claude grammar+vocab judge (overrides nano on both sub-scores),
    # hybrid spelling check. Wall-clock = max of the three (~2s). ─────────
    from concurrent.futures import ThreadPoolExecutor
    from services.openai_scoring_service import score_swt_subscores_with_openai
    from services.anthropic_scoring_service import score_grammar_and_vocab_with_claude
    from services.spelling_checker import check_spelling, format_spelling_reasoning

    scoring_warnings: list = []
    scorer_label = "nano(content)+claude(grammar+vocab)+hybrid(spell)"

    def _run_nano():
        try:
            return score_swt_subscores_with_openai(prompt or '', body)
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[SWT] OpenAI call raised unexpectedly: {e}")
            return {"scored": False, "warning_code": "content_llm_unavailable"}

    def _run_claude_grammar_vocab():
        try:
            return score_grammar_and_vocab_with_claude(prompt or '', body)
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[SWT] Claude grammar+vocab raised unexpectedly: {e}")
            return {"scored": False, "warning_code": "grammar_vocab_claude_unavailable"}

    def _run_spell():
        return check_spelling(body, prompt or "")

    with ThreadPoolExecutor(max_workers=3) as _ex:
        f_llm = _ex.submit(_run_nano)
        f_gv = _ex.submit(_run_claude_grammar_vocab)
        f_spell = _ex.submit(_run_spell)
        llm_result = f_llm.result()
        gv_result = f_gv.result()
        spell_result = f_spell.result()

    # Nano fallback chain: if nano failed, fall back to Claude full SWT.
    if not llm_result.get("scored"):
        try:
            from services.anthropic_scoring_service import score_swt_subscores_with_claude
            llm_result = score_swt_subscores_with_claude(prompt or '', body)
            if llm_result.get("scored"):
                scorer_label = "claude-full-fallback+hybrid"
                scoring_warnings.append("openai_unavailable_used_claude")
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[SWT] Claude SWT fallback raised: {e}")
            llm_result = {"scored": False, "warning_code": "content_llm_unavailable"}

    if not llm_result.get("scored"):
        fallback = _swt_heuristic_fallback(body, content_max, grammar_max, vocab_max)
        content_sub    = fallback["content"]
        grammar_sub    = fallback["grammar"]
        vocabulary_sub = fallback["vocabulary"]
        scoring_warnings.append(
            llm_result.get("warning_code") or "content_llm_unavailable"
        )
        scorer_label = "heuristic-fallback"
    else:
        content_sub    = llm_result["content"]
        grammar_sub    = llm_result["grammar"]
        vocabulary_sub = llm_result["vocabulary"]

    # Override grammar + vocabulary with Claude's grammar+vocab judgment
    # when available. Preserves nano scores in audit fields.
    nano_grammar_score = float(grammar_sub.get("score", 0) or 0)
    nano_vocab_score = float(vocabulary_sub.get("score", 0) or 0)
    if gv_result.get("scored"):
        cg = gv_result["grammar"]
        cv = gv_result["vocabulary"]
        grammar_sub = dict(grammar_sub)
        grammar_sub["nano_grammar_score"] = nano_grammar_score
        grammar_sub["score"] = float(cg.get("score", 0) or 0)
        grammar_sub["reasoning"] = cg.get("reasoning") or grammar_sub.get("reasoning")
        cg_quotes = list(cg.get("mistake_quotes") or [])
        grammar_sub["mistake_quotes"] = cg_quotes
        grammar_sub["grammar_mistake_quotes"] = cg_quotes
        grammar_sub["grammar_scorer"] = "claude-haiku-4-5"

        vocabulary_sub = dict(vocabulary_sub)
        vocabulary_sub["nano_vocab_score"] = nano_vocab_score
        vocabulary_sub["score"] = float(cv.get("score", 0) or 0)
        vocabulary_sub["reasoning"] = cv.get("reasoning") or vocabulary_sub.get("reasoning")
        vocabulary_sub["vocab_scorer"] = "claude-haiku-4-5"
    else:
        if gv_result.get("warning_code"):
            scoring_warnings.append(gv_result["warning_code"])
        grammar_sub = dict(grammar_sub)
        grammar_sub["grammar_scorer"] = "nano-fallback"
        vocabulary_sub = dict(vocabulary_sub)
        vocabulary_sub["vocab_scorer"] = "nano-fallback"

    # ── 3-way floor on Grammar & Spelling: min(heuristic, grammar_score,
    # spelling_remaining). The heuristic catches deterministic surface bugs
    # (extra spaces, missing initial cap, missing terminal, improper ALL-CAPS);
    # the spelling check is the parallel hybrid pipeline result computed
    # above. None of these can raise the score — only lower it. ─────────
    from services.grammar_heuristic import grammar_heuristic, format_findings
    heur_score, heur_findings = grammar_heuristic(body)
    grammar_judge_score = float(grammar_sub.get("score", 0) or 0)
    grammar_only_score = min(float(heur_score), grammar_judge_score)
    grammar_sub["heuristic_findings"] = heur_findings
    grammar_sub["heuristic_score"] = heur_score
    grammar_sub["llm_score"] = grammar_judge_score

    spell_count = len(spell_result.get("mistakes") or [])
    spell_remaining = float(max(0, grammar_max - spell_count))
    final_grammar = min(grammar_only_score, spell_remaining)
    grammar_sub["spelling_check"] = spell_result
    grammar_sub["spelling_remaining"] = spell_remaining
    if spell_result.get("warning_code"):
        scoring_warnings.append(spell_result["warning_code"])
    grammar_sub["score"] = final_grammar
    spell_summary = format_spelling_reasoning(spell_result.get("mistakes") or [])
    heur_summary = format_findings(heur_findings)
    grammar_reasoning = grammar_sub.get("reasoning") or "no grammar reasoning"
    grammar_sub["reasoning"] = (
        f"Grammar & Spelling: {int(final_grammar)}/{grammar_max}. "
        f"Heuristic: {heur_summary}. "
        f"Spelling: {spell_summary} "
        f"Grammar: {grammar_reasoning}"
    )

    # ── Off-topic floor: LLM content == 0 zeroes everything ───────────────
    # (verbatim-copy floor removed — rely on LLM rubric to flag copies.)
    if llm_result.get("scored") and content_sub["score"] == 0:
        # Still build highlights for the student answer even on off-topic.
        from services.highlight_builder import build_highlights
        ot_spelling = [m["word"] for m in (spell_result.get("mistakes") or [])]
        ot_grammar = grammar_sub.get("grammar_mistake_quotes") or []
        ot_highlights = build_highlights(body, heur_findings, ot_spelling, ot_grammar)
        breakdown = {
            "form": form_max,
            "content":    content_sub,
            "grammar":    grammar_sub,
            "vocabulary": vocabulary_sub,
            "earned": 0,
            "max_pts": max_pts,
            "task_type": "swt",
            "scorer": scorer_label,
            "scoring_warnings": ["content_off_topic"] + scoring_warnings,
            "highlights": ot_highlights,
            "highlight_text": body,
        }
        return ScoringResult(
            pte_score=to_pte_score(0.0),
            raw_score=0.0,
            is_async=False,
            breakdown=breakdown,
        )

    # ── Content +1 generosity bump ──────────────────────────────────────
    # Applied after the off-topic check (so content==0 still floors PTE 10)
    # and only when llm_content >= 1. Original LLM score is preserved for
    # trainer audit.
    content_sub = _apply_content_bump(content_sub, content_max)

    # ── Aggregate + PTE ────────────────────────────────────────────────────
    earned = (
        form_max
        + content_sub["score"]
        + grammar_sub["score"]
        + vocabulary_sub["score"]
    )
    earned = max(0.0, min(float(max_pts), earned))
    pct = earned / max_pts if max_pts > 0 else 0.0

    # ── Build highlights (hybrid spelling words + LLM grammar quotes +
    # heuristic positions) ──────────────────────────────────────────────
    from services.highlight_builder import build_highlights
    spelling_quotes = [
        m["word"] for m in (grammar_sub.get("spelling_check", {}).get("mistakes") or [])
    ]
    grammar_quotes = grammar_sub.get("grammar_mistake_quotes") or []
    heur_findings = grammar_sub.get("heuristic_findings") or {}
    highlights = build_highlights(body, heur_findings, spelling_quotes, grammar_quotes)

    breakdown = {
        "form": form_max,
        "content":    content_sub,
        "grammar":    grammar_sub,
        "vocabulary": vocabulary_sub,
        "earned": earned,
        "max_pts": max_pts,
        "task_type": "swt",
        "scorer": scorer_label,
        "highlights": highlights,
        "highlight_text": body,
    }
    if scoring_warnings:
        breakdown["scoring_warnings"] = scoring_warnings

    return ScoringResult(
        pte_score=to_pte_score(pct),
        raw_score=pct,
        is_async=False,
        breakdown=breakdown,
    )


# ── WE — hybrid form gate + Claude Haiku 4.5 ─────────────────────────────────

def _we_heuristic_fallback(user_text: str) -> dict:
    """Heuristic sub-scores used only when Claude is unreachable. Six
    sub-scores match the real PTE WE rubric. Three of them (DSC, GLR,
    spelling) have no clean rule-based proxy — they default to a midpoint
    so the score is plausible rather than zero, with the warning surfaced
    in result_json so the trainer knows the attempt was degraded."""
    text = (user_text or '').strip()
    if not text:
        return {
            "content":    {"score": 0.0, "reasoning": None},
            "dsc":        {"score": 0.0, "reasoning": None},
            "grammar":    {"score": 0.0, "reasoning": None},
            "glr":        {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "spelling":   {"score": 0.0, "reasoning": None},
        }
    words = text.split()
    wc = len(words)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]

    # Content (0–6): word-count banded.
    content_score = 6 if wc >= 250 else 5 if wc >= 200 else 4 if wc >= 150 else 2 if wc >= 100 else 1

    # DSC (0–6): paragraph-count proxy.
    if len(paragraphs) >= 4:
        dsc_score = 6
    elif len(paragraphs) == 3:
        dsc_score = 4
    elif len(paragraphs) == 2:
        dsc_score = 2
    elif len(sentences) >= 4:
        dsc_score = 1
    else:
        dsc_score = 0

    # Grammar (0–2): first-cap + ends-punct (same proxy as before).
    g = 0
    if text[0].isupper():
        g += 1
    if text[-1] in '.!?':
        g += 1
    grammar_score = g

    # GLR (0–6): no rule-based proxy — midpoint default. Flagged via
    # scoring_warnings upstream so the trainer knows this is a fallback.
    glr_score = 3

    # Vocabulary (0–2): unique-word density.
    unique_words = len(set(w.lower().strip('.,!?;:()"\'- ') for w in words if len(w) > 3))
    if unique_words >= 60:
        vocab_score = 2
    elif unique_words >= 35:
        vocab_score = 1
    else:
        vocab_score = 0

    # Spelling (0–2): heuristic can't detect spelling — assume correct.
    spelling_score = 2

    return {
        "content":    {"score": float(content_score), "reasoning": None},
        "dsc":        {"score": float(dsc_score), "reasoning": None},
        "grammar":    {"score": float(grammar_score), "reasoning": None},
        "glr":        {"score": float(glr_score), "reasoning": None},
        "vocabulary": {"score": float(vocab_score), "reasoning": None},
        "spelling":   {"score": float(spelling_score), "reasoning": None},
    }


def _score_we_with_claude(text: str, prompt: str) -> ScoringResult:
    """WE scoring path — mirrors the SWT shape:
      1. Load form-band rubric from RDS (200–300 → 2, 120–199 / 301–380 → 1,
         outside → 0). Form-zero kills the attempt (PTE 10 floor).
      2. Claude Haiku 4.5 scores content + dsc + grammar + glr + vocabulary
         + spelling in a single call with per-sub-score reasoning.
      3. Off-topic floor: LLM content == 0 → PTE 10.
      4. Otherwise earned = form + sum(sub-scores); PTE via standard formula.

    Claude unreachable → six-sub-score heuristic fallback + warning code in
    result_json. Never raises.
    """
    from services.rubric_cache import (
        get_rubric_max, get_we_form_max, get_we_form_score,
    )

    content_max = get_rubric_max('write_essay', 'content_max')
    grammar_max = get_rubric_max('write_essay', 'grammar_max')
    vocab_max   = get_rubric_max('write_essay', 'vocabulary_max')
    dsc_max     = get_rubric_max('write_essay', 'coherence_max')
    glr_max     = get_rubric_max('write_essay', 'glr_max')
    spell_max   = get_rubric_max('write_essay', 'spelling_max')
    form_max    = get_we_form_max()
    max_pts = (
        form_max + content_max + dsc_max + grammar_max + glr_max
        + vocab_max + spell_max
    )  # default = 26

    body = (text or '').strip()
    wc = len(body.split())
    form_score = get_we_form_score(wc) if body else 0

    # ── Form-zero floor ────────────────────────────────────────────────────
    if form_score == 0:
        if not body:
            reason = "Empty response."
        elif wc < 120:
            reason = f"Form-zero — {wc} words is below the 120-word minimum."
        else:
            reason = f"Form-zero — {wc} words exceeds the 380-word maximum."
        breakdown = {
            "form": 0,
            "content":    {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "dsc":        {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "grammar":    {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "glr":        {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "vocabulary": {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "spelling":   {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "earned": 0,
            "max_pts": max_pts,
            "task_type": "we",
            "scorer": "form-gate-floor",
            "scoring_warnings": [reason],
        }
        return ScoringResult(
            pte_score=to_pte_score(0.0),
            raw_score=0.0,
            is_async=False,
            breakdown=breakdown,
        )

    # ── Parallel: nano (full WE — used for content/dsc/glr/spelling),
    # Claude grammar+vocab judge (overrides nano on grammar AND vocab),
    # hybrid spelling. ────────────────────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor
    from services.openai_scoring_service import score_we_subscores_with_openai
    from services.anthropic_scoring_service import score_grammar_and_vocab_with_claude
    from services.spelling_checker import check_spelling, format_spelling_reasoning

    scoring_warnings: list = []
    scorer_label = "nano(content+dsc+glr+spell)+claude(grammar+vocab)+hybrid(spell)"

    def _run_nano_we():
        try:
            return score_we_subscores_with_openai(prompt or '', body)
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[WE] OpenAI raised: {e}")
            return {"scored": False, "warning_code": "content_llm_unavailable"}

    def _run_claude_grammar_vocab_we():
        try:
            return score_grammar_and_vocab_with_claude(prompt or '', body)
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[WE] Claude grammar+vocab raised: {e}")
            return {"scored": False, "warning_code": "grammar_vocab_claude_unavailable"}

    def _run_spell_we():
        return check_spelling(body, prompt or "")

    with ThreadPoolExecutor(max_workers=3) as _ex:
        f_llm = _ex.submit(_run_nano_we)
        f_gv = _ex.submit(_run_claude_grammar_vocab_we)
        f_spell = _ex.submit(_run_spell_we)
        llm_result = f_llm.result()
        gv_result = f_gv.result()
        spell_result = f_spell.result()

    if not llm_result.get("scored"):
        try:
            from services.anthropic_scoring_service import score_we_subscores_with_claude
            llm_result = score_we_subscores_with_claude(prompt or '', body)
            if llm_result.get("scored"):
                scorer_label = "claude-full-fallback+hybrid"
                scoring_warnings.append("openai_unavailable_used_claude")
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[WE] Claude full fallback raised: {e}")
            llm_result = {"scored": False, "warning_code": "content_llm_unavailable"}

    if not llm_result.get("scored"):
        fallback = _we_heuristic_fallback(body)
        content_sub    = fallback["content"]
        dsc_sub        = fallback["dsc"]
        grammar_sub    = fallback["grammar"]
        glr_sub        = fallback["glr"]
        vocabulary_sub = fallback["vocabulary"]
        spelling_sub   = fallback["spelling"]
        scoring_warnings.append(
            llm_result.get("warning_code") or "content_llm_unavailable"
        )
        scorer_label = "heuristic-fallback"
    else:
        content_sub    = llm_result["content"]
        dsc_sub        = llm_result["dsc"]
        grammar_sub    = llm_result["grammar"]
        glr_sub        = llm_result["glr"]
        vocabulary_sub = llm_result["vocabulary"]
        spelling_sub   = llm_result["spelling"]

    # ── Grammar + vocab: override nano with Claude judge + heuristic floor.
    from services.grammar_heuristic import grammar_heuristic, format_findings
    heur_score, heur_findings = grammar_heuristic(body)
    nano_grammar_score = float(grammar_sub.get("score", 0) or 0)
    nano_vocab_score = float(vocabulary_sub.get("score", 0) or 0)
    grammar_sub = dict(grammar_sub)
    grammar_sub["heuristic_findings"] = heur_findings
    grammar_sub["heuristic_score"] = heur_score
    grammar_sub["nano_grammar_score"] = nano_grammar_score

    if gv_result.get("scored"):
        cg = gv_result["grammar"]
        cv = gv_result["vocabulary"]
        claude_g_score = float(cg.get("score", 0) or 0)
        cg_quotes = list(cg.get("mistake_quotes") or [])
        grammar_sub["llm_score"] = claude_g_score
        grammar_sub["mistake_quotes"] = cg_quotes
        grammar_sub["grammar_scorer"] = "claude-haiku-4-5"
        chosen_g_score = claude_g_score
        chosen_g_reasoning = cg.get("reasoning") or grammar_sub.get("reasoning")

        vocabulary_sub = dict(vocabulary_sub)
        vocabulary_sub["nano_vocab_score"] = nano_vocab_score
        vocabulary_sub["score"] = float(cv.get("score", 0) or 0)
        vocabulary_sub["reasoning"] = cv.get("reasoning") or vocabulary_sub.get("reasoning")
        vocabulary_sub["vocab_scorer"] = "claude-haiku-4-5"
    else:
        if gv_result.get("warning_code"):
            scoring_warnings.append(gv_result["warning_code"])
        grammar_sub["llm_score"] = nano_grammar_score
        grammar_sub["grammar_scorer"] = "nano-fallback"
        chosen_g_score = nano_grammar_score
        chosen_g_reasoning = grammar_sub.get("reasoning") or "no grammar reasoning"
        vocabulary_sub = dict(vocabulary_sub)
        vocabulary_sub["vocab_scorer"] = "nano-fallback"

    final_grammar = min(float(heur_score), chosen_g_score)
    grammar_sub["score"] = final_grammar
    grammar_sub["reasoning"] = (
        f"Grammar: {int(final_grammar)}/2. "
        f"Heuristic: {format_findings(heur_findings)}. "
        f"Judge: {chosen_g_reasoning}"
    )

    # ── Hybrid spelling override on WE's dedicated spelling sub-score. ──
    spell_count = len(spell_result.get("mistakes") or [])
    spell_max_pts = 2  # WE spelling sub-score max
    spell_remaining = float(max(0, spell_max_pts - spell_count))
    llm_spell_score = float(spelling_sub.get("score", 0) or 0)
    final_spelling = min(llm_spell_score, spell_remaining)
    spelling_sub = dict(spelling_sub)
    spelling_sub["llm_score"] = llm_spell_score
    spelling_sub["hybrid_remaining"] = spell_remaining
    spelling_sub["spelling_check"] = spell_result
    if spell_result.get("warning_code"):
        scoring_warnings.append(spell_result["warning_code"])
    spelling_sub["score"] = final_spelling
    spelling_sub["reasoning"] = (
        f"Spelling: {int(final_spelling)}/{spell_max_pts}. "
        f"{format_spelling_reasoning(spell_result.get('mistakes') or [])}"
    )

    # ── Verbatim-copy floor REMOVED — rely on LLM rubric to flag copies.
    # (Off-topic floor below still fires when LLM content == 0.)

    # ── Off-topic floor: LLM content == 0 zeroes everything ───────────────
    if llm_result.get("scored") and content_sub["score"] == 0:
        from services.highlight_builder import build_highlights
        ot_spelling = [m["word"] for m in (spell_result.get("mistakes") or [])]
        ot_grammar = (grammar_sub.get("mistake_quotes") or [])
        ot_highlights = build_highlights(body, heur_findings, ot_spelling, ot_grammar)
        breakdown = {
            "form": form_score,
            "content":    content_sub,
            "dsc":        dsc_sub,
            "grammar":    grammar_sub,
            "glr":        glr_sub,
            "vocabulary": vocabulary_sub,
            "spelling":   spelling_sub,
            "earned": 0,
            "max_pts": max_pts,
            "task_type": "we",
            "scorer": scorer_label,
            "scoring_warnings": ["content_off_topic"] + scoring_warnings,
            "highlights": ot_highlights,
            "highlight_text": body,
        }
        return ScoringResult(
            pte_score=to_pte_score(0.0),
            raw_score=0.0,
            is_async=False,
            breakdown=breakdown,
        )

    # ── Content +1 generosity bump ──────────────────────────────────────
    content_sub = _apply_content_bump(content_sub, content_max)

    # ── Aggregate + PTE ────────────────────────────────────────────────────
    earned = (
        form_score
        + content_sub["score"]
        + dsc_sub["score"]
        + grammar_sub["score"]
        + glr_sub["score"]
        + vocabulary_sub["score"]
        + spelling_sub["score"]
    )
    earned = max(0.0, min(float(max_pts), earned))
    pct = earned / max_pts if max_pts > 0 else 0.0

    # ── Build highlights ────────────────────────────────────────────────
    from services.highlight_builder import build_highlights
    spelling_quotes = [
        m["word"] for m in (spelling_sub.get("spelling_check", {}).get("mistakes") or [])
    ]
    grammar_quotes = grammar_sub.get("mistake_quotes") or []
    highlights = build_highlights(body, heur_findings, spelling_quotes, grammar_quotes)

    breakdown = {
        "form": form_score,
        "content":    content_sub,
        "dsc":        dsc_sub,
        "grammar":    grammar_sub,
        "glr":        glr_sub,
        "vocabulary": vocabulary_sub,
        "spelling":   spelling_sub,
        "earned": earned,
        "max_pts": max_pts,
        "task_type": "we",
        "scorer": scorer_label,
        "highlights": highlights,
        "highlight_text": body,
    }
    if scoring_warnings:
        breakdown["scoring_warnings"] = scoring_warnings

    return ScoringResult(
        pte_score=to_pte_score(pct),
        raw_score=pct,
        is_async=False,
        breakdown=breakdown,
    )


# ── SST — hybrid form gate + Claude Haiku 4.5 ────────────────────────────────

def _sst_heuristic_fallback(user_text: str) -> dict:
    """Heuristic sub-scores used only when Claude is unreachable. Four
    sub-scores match the real PTE SST rubric. Spelling defaults to 2 (the
    heuristic can't detect spelling errors — same compromise as WE)."""
    text = (user_text or '').strip()
    if not text:
        return {
            "content":    {"score": 0.0, "reasoning": None},
            "grammar":    {"score": 0.0, "reasoning": None},
            "vocabulary": {"score": 0.0, "reasoning": None},
            "spelling":   {"score": 0.0, "reasoning": None},
        }
    words = text.split()
    wc = len(words)

    # Content (0–4): word-count banded (legacy heuristic).
    content_score = min(4, max(1, wc // 12))

    # Grammar (0–2): first-cap + ends-punct.
    g = 0
    if text[0].isupper():
        g += 1
    if text[-1] in '.!?':
        g += 1
    grammar_score = g

    # Vocabulary (0–2): unique-word density.
    unique_words = len(set(w.lower().strip('.,!?;:()') for w in words if len(w) > 3))
    vocab_score = min(2, unique_words // 5)

    # Spelling (0–2): heuristic blind — assume correct.
    spelling_score = 2

    return {
        "content":    {"score": float(content_score), "reasoning": None},
        "grammar":    {"score": float(grammar_score), "reasoning": None},
        "vocabulary": {"score": float(vocab_score), "reasoning": None},
        "spelling":   {"score": float(spelling_score), "reasoning": None},
    }


def _score_sst_with_claude(text: str, prompt: str) -> ScoringResult:
    """SST scoring path — mirrors the SWT/WE shape:
      1. Load form-band rubric from RDS (50–70 → 2, 40–49 / 71–100 → 1,
         outside → 0). Form-zero kills the attempt (PTE 10 floor).
      2. Claude Haiku 4.5 scores content + grammar + vocabulary + spelling
         in a single call with per-sub-score reasoning.
      3. Off-topic floor: LLM content == 0 → PTE 10.
      4. Otherwise earned = form + sum(sub-scores); PTE via standard formula.

    Claude unreachable → four-sub-score heuristic fallback + warning code in
    result_json. Never raises.
    """
    from services.rubric_cache import (
        get_rubric_max, get_sst_form_max, get_sst_form_score,
    )

    content_max = get_rubric_max('summarize_spoken_text', 'content_max')
    grammar_max = get_rubric_max('summarize_spoken_text', 'grammar_max')
    vocab_max   = get_rubric_max('summarize_spoken_text', 'vocabulary_max')
    spell_max   = get_rubric_max('summarize_spoken_text', 'spelling_max')
    form_max    = get_sst_form_max()
    max_pts = (
        form_max + content_max + grammar_max + vocab_max + spell_max
    )  # default = 12

    body = (text or '').strip()
    wc = len(body.split())
    form_score = get_sst_form_score(wc) if body else 0

    # ── Form-zero floor ───────────────────────────────────────────────────
    if form_score == 0:
        if not body:
            reason = "Empty response."
        elif wc < 40:
            reason = f"Form-zero — {wc} words is below the 40-word minimum."
        else:
            reason = f"Form-zero — {wc} words exceeds the 100-word maximum."
        breakdown = {
            "form": 0,
            "content":    {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "grammar":    {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "vocabulary": {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "spelling":   {"score": 0.0, "reasoning": "Not scored — form-zero."},
            "earned": 0,
            "max_pts": max_pts,
            "task_type": "sst",
            "scorer": "form-gate-floor",
            "scoring_warnings": [reason],
            "word_count": wc,
        }
        return ScoringResult(
            pte_score=to_pte_score(0.0),
            raw_score=0.0,
            is_async=False,
            breakdown=breakdown,
        )

    # ── Parallel: nano (full SST — used for content/spelling), Claude
    # grammar+vocab judge (overrides nano on both), hybrid spelling. ──────
    from concurrent.futures import ThreadPoolExecutor
    from services.openai_scoring_service import score_sst_subscores_with_openai
    from services.anthropic_scoring_service import score_grammar_and_vocab_with_claude
    from services.spelling_checker import check_spelling, format_spelling_reasoning

    scoring_warnings: list = []
    scorer_label = "nano(content+spell)+claude(grammar+vocab)+hybrid(spell)"

    def _run_nano_sst():
        try:
            return score_sst_subscores_with_openai(prompt or '', body)
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[SST] OpenAI raised: {e}")
            return {"scored": False, "warning_code": "content_llm_unavailable"}

    def _run_claude_grammar_vocab_sst():
        try:
            return score_grammar_and_vocab_with_claude(prompt or '', body)
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[SST] Claude grammar+vocab raised: {e}")
            return {"scored": False, "warning_code": "grammar_vocab_claude_unavailable"}

    def _run_spell_sst():
        return check_spelling(body, prompt or "")

    with ThreadPoolExecutor(max_workers=3) as _ex:
        f_llm = _ex.submit(_run_nano_sst)
        f_gv = _ex.submit(_run_claude_grammar_vocab_sst)
        f_spell = _ex.submit(_run_spell_sst)
        llm_result = f_llm.result()
        gv_result = f_gv.result()
        spell_result = f_spell.result()

    if not llm_result.get("scored"):
        try:
            from services.anthropic_scoring_service import score_sst_subscores_with_claude
            llm_result = score_sst_subscores_with_claude(prompt or '', body)
            if llm_result.get("scored"):
                scorer_label = "claude-full-fallback+hybrid"
                scoring_warnings.append("openai_unavailable_used_claude")
        except Exception as e:
            from core.logging_config import get_logger
            get_logger(__name__).error(f"[SST] Claude full fallback raised: {e}")
            llm_result = {"scored": False, "warning_code": "content_llm_unavailable"}

    if not llm_result.get("scored"):
        fallback = _sst_heuristic_fallback(body)
        content_sub    = fallback["content"]
        grammar_sub    = fallback["grammar"]
        vocabulary_sub = fallback["vocabulary"]
        spelling_sub   = fallback["spelling"]
        scoring_warnings.append(
            llm_result.get("warning_code") or "content_llm_unavailable"
        )
        scorer_label = "heuristic-fallback"
    else:
        content_sub    = llm_result["content"]
        grammar_sub    = llm_result["grammar"]
        vocabulary_sub = llm_result["vocabulary"]
        spelling_sub   = llm_result["spelling"]

    # ── Grammar + vocab: override nano with Claude judge + heuristic floor.
    from services.grammar_heuristic import grammar_heuristic, format_findings
    heur_score, heur_findings = grammar_heuristic(body)
    nano_grammar_score = float(grammar_sub.get("score", 0) or 0)
    nano_vocab_score = float(vocabulary_sub.get("score", 0) or 0)
    grammar_sub = dict(grammar_sub)
    grammar_sub["heuristic_findings"] = heur_findings
    grammar_sub["heuristic_score"] = heur_score
    grammar_sub["nano_grammar_score"] = nano_grammar_score

    if gv_result.get("scored"):
        cg = gv_result["grammar"]
        cv = gv_result["vocabulary"]
        claude_g_score = float(cg.get("score", 0) or 0)
        cg_quotes = list(cg.get("mistake_quotes") or [])
        grammar_sub["llm_score"] = claude_g_score
        grammar_sub["mistake_quotes"] = cg_quotes
        grammar_sub["grammar_scorer"] = "claude-haiku-4-5"
        chosen_g_score = claude_g_score
        chosen_g_reasoning = cg.get("reasoning") or grammar_sub.get("reasoning")

        vocabulary_sub = dict(vocabulary_sub)
        vocabulary_sub["nano_vocab_score"] = nano_vocab_score
        vocabulary_sub["score"] = float(cv.get("score", 0) or 0)
        vocabulary_sub["reasoning"] = cv.get("reasoning") or vocabulary_sub.get("reasoning")
        vocabulary_sub["vocab_scorer"] = "claude-haiku-4-5"
    else:
        if gv_result.get("warning_code"):
            scoring_warnings.append(gv_result["warning_code"])
        grammar_sub["llm_score"] = nano_grammar_score
        grammar_sub["grammar_scorer"] = "nano-fallback"
        chosen_g_score = nano_grammar_score
        chosen_g_reasoning = grammar_sub.get("reasoning") or "no grammar reasoning"
        vocabulary_sub = dict(vocabulary_sub)
        vocabulary_sub["vocab_scorer"] = "nano-fallback"

    final_grammar = min(float(heur_score), chosen_g_score)
    grammar_sub["score"] = final_grammar
    grammar_sub["reasoning"] = (
        f"Grammar: {int(final_grammar)}/2. "
        f"Heuristic: {format_findings(heur_findings)}. "
        f"Judge: {chosen_g_reasoning}"
    )

    # ── Hybrid spelling override on SST's dedicated spelling sub-score. ──
    spell_count = len(spell_result.get("mistakes") or [])
    spell_remaining = float(max(0, spell_max - spell_count))
    llm_spell_score = float(spelling_sub.get("score", 0) or 0)
    final_spelling = min(llm_spell_score, spell_remaining)
    spelling_sub = dict(spelling_sub)
    spelling_sub["llm_score"] = llm_spell_score
    spelling_sub["hybrid_remaining"] = spell_remaining
    spelling_sub["spelling_check"] = spell_result
    if spell_result.get("warning_code"):
        scoring_warnings.append(spell_result["warning_code"])
    spelling_sub["score"] = final_spelling
    spelling_sub["reasoning"] = (
        f"Spelling: {int(final_spelling)}/{spell_max}. "
        f"{format_spelling_reasoning(spell_result.get('mistakes') or [])}"
    )

    # ── Verbatim-copy floor REMOVED — rely on LLM rubric to flag copies.

    # ── Off-topic floor: LLM content == 0 zeroes everything ──────────────
    if llm_result.get("scored") and content_sub["score"] == 0:
        from services.highlight_builder import build_highlights
        ot_spelling = [m["word"] for m in (spell_result.get("mistakes") or [])]
        ot_grammar = (grammar_sub.get("mistake_quotes") or [])
        ot_highlights = build_highlights(body, heur_findings, ot_spelling, ot_grammar)
        breakdown = {
            "form": form_score,
            "content":    content_sub,
            "grammar":    grammar_sub,
            "vocabulary": vocabulary_sub,
            "spelling":   spelling_sub,
            "earned": 0,
            "max_pts": max_pts,
            "task_type": "sst",
            "scorer": scorer_label,
            "scoring_warnings": ["content_off_topic"] + scoring_warnings,
            "word_count": wc,
            "highlights": ot_highlights,
            "highlight_text": body,
        }
        return ScoringResult(
            pte_score=to_pte_score(0.0),
            raw_score=0.0,
            is_async=False,
            breakdown=breakdown,
        )

    # ── Content +1 generosity bump ──────────────────────────────────────
    content_sub = _apply_content_bump(content_sub, content_max)

    # ── Aggregate + PTE ──────────────────────────────────────────────────
    earned = (
        form_score
        + content_sub["score"]
        + grammar_sub["score"]
        + vocabulary_sub["score"]
        + spelling_sub["score"]
    )
    earned = max(0.0, min(float(max_pts), earned))
    pct = earned / max_pts if max_pts > 0 else 0.0

    # Build highlights
    from services.highlight_builder import build_highlights
    spelling_quotes = [
        m["word"] for m in (spelling_sub.get("spelling_check", {}).get("mistakes") or [])
    ]
    grammar_quotes = grammar_sub.get("mistake_quotes") or []
    highlights = build_highlights(body, heur_findings, spelling_quotes, grammar_quotes)

    breakdown = {
        "form": form_score,
        "content":    content_sub,
        "grammar":    grammar_sub,
        "vocabulary": vocabulary_sub,
        "spelling":   spelling_sub,
        "earned": earned,
        "max_pts": max_pts,
        "task_type": "sst",
        "scorer": scorer_label,
        "word_count": wc,
        "highlights": highlights,
        "highlight_text": body,
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
            # WE now uses a hybrid path identical in shape to SWT: RDS-driven
            # form gate (word-count bands) + Claude Haiku 4.5 for the six
            # substantive sub-scores (content / DSC / grammar / GLR /
            # vocabulary / spelling), each returned with a reasoning string.
            return _score_we_with_claude(text, prompt)

        else:  # sst
            # SST mirrors SWT/WE: RDS-driven form band gate + Claude Haiku 4.5
            # for the four substantive sub-scores (content, grammar,
            # vocabulary, spelling). Spelling is now a real sub-score; form
            # bands tighten to RDS spec (40–49 / 71–100 → 1; 50–70 → 2).
            return _score_sst_with_claude(text, prompt)

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
