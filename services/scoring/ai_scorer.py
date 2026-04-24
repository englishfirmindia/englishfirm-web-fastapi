"""
Hybrid AI scorer for writing/listening tasks.
Covers: summarize_written_text (swt), write_essay (we), listening_sst (sst).

Uses heuristics for form/grammar/vocabulary (same as mobile),
GPT-4o-mini replaces only the content subscore.
Falls back to word-count heuristic if LLM call fails.
Ported from englishfirm-app-fastapi/services/question_service.py.
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
            earned, max_pts = _score_swt_heuristic(text)
            wc = len(text.split())
            form_ok = (
                len([s for s in re.split(r'[.!?]+', text.strip()) if s.strip()]) == 1
                and 5 <= wc <= 75
            )
            if form_ok and prompt and text.strip():
                from services.llm_content_scoring_service import score_swt_content
                heuristic_content = min(4, max(1, wc // 10))
                llm_content = score_swt_content(prompt, text)
                earned = max(0, min(max_pts, earned - heuristic_content + llm_content))

        elif self.task_type == 'we':
            earned, max_pts = _score_we_heuristic(text)
            wc = len(text.split())
            form_ok = 120 <= wc <= 380
            if form_ok and prompt and text.strip():
                from services.llm_content_scoring_service import score_we_content
                heuristic_content = 3 if wc >= 200 else 2 if wc >= 120 else 1 if wc >= 60 else 0
                llm_content = score_we_content(prompt, text)
                earned = max(0, min(max_pts, earned - heuristic_content + llm_content))

        else:  # sst
            sst = _score_sst_heuristic(text)
            earned = sst['earned']
            max_pts = sst['max_pts']
            wc = sst['word_count']
            form_ok = 30 <= wc <= 90
            if form_ok and prompt and text.strip():
                from services.llm_content_scoring_service import score_sst_content
                heuristic_content = sst['content']
                llm_content = score_sst_content(prompt, text)
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
