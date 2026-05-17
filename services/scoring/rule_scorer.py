"""
Rule-based scorers — pure Python, no DB access, no FastAPI imports.

Each scorer receives:
  question_id  : int
  session_id   : str
  answer       : dict with type-specific fields including 'evaluation_json'

The 'evaluation_json' key is the question's evaluation data from the DB,
passed in by the caller — scorers never touch the DB themselves.
"""

import re
from collections import Counter
from typing import List

from .base import ScoringResult, ScoringStrategy, to_pte_score


# ---------------------------------------------------------------------------
# FIBScorer — reading_fib, reading_fib_drop_down, listening_fib
# ---------------------------------------------------------------------------

class FIBScorer(ScoringStrategy):
    """
    Fill-in-the-Blanks scorer.

    Reading FIB evaluation_json shape:
      correctAnswers: {blank_id: correct_value, ...}
      scoringRules:
        marksPerBlank: int (default 1)
        isCaseSensitive: bool (default False)
        trimWhitespace: bool (default True)

    Listening FIB evaluation_json shape (different — blanks list):
      correctAnswers:
        blanks: [{blankId: int|str, answer: str}, ...]
      scoringRules:
        marksPerCorrect: int (default 1)

    answer dict:
      evaluation_json: dict
      user_answers: {blank_id_str: value_str, ...}
    """

    @property
    def is_async(self) -> bool:
        return False

    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        evaluation_json: dict = answer.get('evaluation_json', {})
        user_answers: dict = answer.get('user_answers', {})

        correct_answers_raw = evaluation_json.get('correctAnswers', {})
        scoring_rules = evaluation_json.get('scoringRules', {})

        # Listening FIB uses a 'blanks' list; Reading FIB uses a flat dict
        blanks_list = correct_answers_raw.get('blanks') if isinstance(correct_answers_raw, dict) else None
        if blanks_list is not None:
            # Listening FIB format
            marks_per_blank = scoring_rules.get('marksPerCorrect', 1)
            correct_map = {str(b['blankId']): b['answer'] for b in blanks_list}
            case_sensitive = False
            trim_whitespace = True
        else:
            # Reading FIB format
            marks_per_blank = scoring_rules.get('marksPerBlank', 1)
            correct_map = {str(k): str(v) for k, v in correct_answers_raw.items()}
            case_sensitive = scoring_rules.get('isCaseSensitive', False)
            trim_whitespace = scoring_rules.get('trimWhitespace', True)

        total_blanks = len(correct_map)
        if total_blanks == 0:
            return ScoringResult(
                pte_score=to_pte_score(0.0),
                raw_score=0.0,
                is_async=False,
                breakdown={'hits': 0, 'total': 0},
            )

        hits = 0
        blank_results = {}

        for blank_id, correct_value in correct_map.items():
            user_value = user_answers.get(blank_id)

            if user_value is None:
                blank_results[blank_id] = False
                continue

            user_value = str(user_value)
            correct_value = str(correct_value)

            if trim_whitespace:
                user_value = user_value.strip()
                correct_value = correct_value.strip()

            if not case_sensitive:
                user_value = user_value.lower()
                correct_value = correct_value.lower()

            is_correct = user_value == correct_value
            blank_results[blank_id] = is_correct
            if is_correct:
                hits += marks_per_blank

        max_score = total_blanks * marks_per_blank
        raw_score = min(1.0, max(0.0, hits / max_score)) if max_score > 0 else 0.0

        return ScoringResult(
            pte_score=to_pte_score(raw_score),
            raw_score=raw_score,
            is_async=False,
            breakdown={
                'hits': hits,
                'total': max_score,
                'blank_results': blank_results,
            },
        )


# ---------------------------------------------------------------------------
# MCQScorer — reading_mcs, reading_mcm, listening_mcs, listening_mcm,
#             listening_hcs, listening_smw
# ---------------------------------------------------------------------------

class MCQScorer(ScoringStrategy):
    """
    Multiple-choice scorer supporting single and multi-answer questions.

    Single (single=True):
      evaluation_json.correctAnswers.correctOption: str
      scoringRules.marksPerCorrect: int (default 1)
      answer.selected_option: str

    Multi (single=False):
      evaluation_json.correctAnswers.correctOptions: [str, ...]
      scoringRules.marksPerCorrect: int (default 1)
      scoringRules.deductPerWrong: int (default 0)
      answer.selected_options: [str, ...]
    """

    def __init__(self, single: bool = True):
        self.single = single

    @property
    def is_async(self) -> bool:
        return False

    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        evaluation_json: dict = answer.get('evaluation_json', {})
        scoring_rules = evaluation_json.get('scoringRules', {})
        correct_answers = evaluation_json.get('correctAnswers', {})

        if self.single:
            return self._score_single(correct_answers, scoring_rules, answer)
        else:
            return self._score_multi(correct_answers, scoring_rules, answer)

    def _score_single(self, correct_answers: dict, scoring_rules: dict, answer: dict) -> ScoringResult:
        marks_correct = scoring_rules.get('marksPerCorrect', 1)

        # Support both 'correctOption' and 'options' (list with single item, as in HCS/SMW)
        correct_option = correct_answers.get('correctOption')
        if correct_option is None:
            options = correct_answers.get('options', [])
            correct_option = str(options[0]) if options else None

        selected_option = answer.get('selected_option', '')
        if correct_option is None:
            return ScoringResult(
                pte_score=to_pte_score(0.0),
                raw_score=0.0,
                is_async=False,
                breakdown={'error': 'no correct option in evaluation_json'},
            )

        is_correct = str(selected_option).strip() == str(correct_option).strip()
        score = marks_correct if is_correct else 0
        raw_score = score / marks_correct if marks_correct > 0 else 0.0

        return ScoringResult(
            pte_score=to_pte_score(raw_score),
            raw_score=raw_score,
            is_async=False,
            breakdown={
                'is_correct': is_correct,
                'correct_option': correct_option,
                'selected_option': selected_option,
            },
        )

    def _score_multi(self, correct_answers: dict, scoring_rules: dict, answer: dict) -> ScoringResult:
        marks_per_correct = scoring_rules.get('marksPerCorrect', 1)
        deduct_per_wrong = scoring_rules.get('deductPerWrong', 0)

        correct_options = set(correct_answers.get('correctOptions', []))
        selected_options = set(answer.get('selected_options', []))

        correct_selected = selected_options & correct_options
        wrong_selected = selected_options - correct_options

        score = len(correct_selected) * marks_per_correct - len(wrong_selected) * deduct_per_wrong
        score = max(score, 0)

        max_possible = len(correct_options) * marks_per_correct if correct_options else 1
        raw_score = min(1.0, max(0.0, score / max_possible)) if max_possible > 0 else 0.0

        return ScoringResult(
            pte_score=to_pte_score(raw_score),
            raw_score=raw_score,
            is_async=False,
            breakdown={
                'correct_selected': list(correct_selected),
                'wrong_selected': list(wrong_selected),
                'score': score,
                'max_possible': max_possible,
            },
        )


# ---------------------------------------------------------------------------
# ReorderScorer — reorder_paragraphs
# ---------------------------------------------------------------------------

class ReorderScorer(ScoringStrategy):
    """
    Reorder Paragraphs scorer.

    evaluation_json:
      correctAnswers.correctSequence: [str, ...]
      scoringRules.marksPerAdjacentPair: int (default 1)

    answer dict:
      evaluation_json: dict
      user_sequence: [str, ...]
    """

    @property
    def is_async(self) -> bool:
        return False

    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        evaluation_json: dict = answer.get('evaluation_json', {})
        user_sequence: List[str] = answer.get('user_sequence', [])

        correct_sequence = evaluation_json.get('correctAnswers', {}).get('correctSequence', [])
        scoring_rules = evaluation_json.get('scoringRules', {})
        marks_per_pair = scoring_rules.get('marksPerAdjacentPair', 1)

        total_pairs = len(correct_sequence) - 1
        if total_pairs <= 0:
            return ScoringResult(
                pte_score=to_pte_score(0.0),
                raw_score=0.0,
                is_async=False,
                breakdown={'hits': 0, 'total_pairs': 0},
            )

        # PTE rule: 1 mark per adjacent pair in the user's answer that is
        # also a correct adjacency anywhere in the reference sequence — NOT
        # a slot-by-slot positional match. Two paragraphs correctly placed
        # next to each other should score, even if the rest of the sequence
        # is shifted. Flagged in production by user attempts where (X, Y)
        # was an adjacency in both sequences but landed at different
        # positions (e.g. correct=[2,4,3,1,5], user=[4,2,5,3,1] shares the
        # (3,1) adjacency — old positional algo scored 0/4 instead of 1/4).
        correct_pairs = set(zip(correct_sequence, correct_sequence[1:]))
        hits = 0
        pair_results = []
        for i in range(len(user_sequence) - 1):
            user_pair = (user_sequence[i], user_sequence[i + 1])
            is_correct = user_pair in correct_pairs
            pair_results.append(is_correct)
            if is_correct:
                hits += marks_per_pair

        max_score = total_pairs * marks_per_pair
        raw_score = min(1.0, max(0.0, hits / max_score)) if max_score > 0 else 0.0

        return ScoringResult(
            pte_score=to_pte_score(raw_score),
            raw_score=raw_score,
            is_async=False,
            breakdown={
                'hits': hits,
                'total_pairs': total_pairs,
                'max_score': max_score,
                'pair_results': pair_results,
            },
        )


# ---------------------------------------------------------------------------
# WFDScorer — listening_wfd (Write from Dictation)
# ---------------------------------------------------------------------------

def _normalise_word(w: str) -> str:
    """Lowercase, strip punctuation — exact port from mobile backend."""
    return re.sub(r"[^a-z0-9']", "", w.lower().strip())


class WFDScorer(ScoringStrategy):
    """
    Write From Dictation scorer.

    evaluation_json:
      correctAnswers.transcript: str

    answer dict:
      evaluation_json: dict
      user_text: str

    Scoring: each correctly-spelled reference word the student typed earns
    one point, regardless of position. Multiset-aware (a reference word
    appearing twice must be typed twice to earn both points). Extra words
    the student typed that aren't in the reference do not subtract — this
    matches the official PTE WFD rubric.
    """

    @property
    def is_async(self) -> bool:
        return False

    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        evaluation_json: dict = answer.get('evaluation_json', {})
        user_text: str = answer.get('user_text', '')

        transcript = evaluation_json.get('correctAnswers', {}).get('transcript', '')

        correct_words = [w for w in (_normalise_word(t) for t in transcript.split()) if w]
        user_words = [w for w in (_normalise_word(t) for t in user_text.split()) if w]

        if not correct_words:
            return ScoringResult(
                pte_score=to_pte_score(0.0),
                raw_score=0.0,
                is_async=False,
                breakdown={'hits': 0, 'total': 0, 'word_results': {}, 'extras': []},
            )

        remaining_user = Counter(user_words)
        hits = 0
        word_results: dict = {}
        for i, cw in enumerate(correct_words):
            if remaining_user[cw] > 0:
                remaining_user[cw] -= 1
                hits += 1
                word_results[str(i)] = {'correct': cw, 'user': cw, 'match': True}
            else:
                word_results[str(i)] = {'correct': cw, 'user': '', 'match': False}

        leftover = Counter(remaining_user)
        extras: List[str] = []
        for w in user_words:
            if leftover[w] > 0:
                extras.append(w)
                leftover[w] -= 1

        raw_score = hits / len(correct_words)

        return ScoringResult(
            pte_score=to_pte_score(raw_score),
            raw_score=raw_score,
            is_async=False,
            breakdown={
                'hits': hits,
                'total': len(correct_words),
                'word_results': word_results,
                'extras': extras,
            },
        )


# ---------------------------------------------------------------------------
# HIWScorer — listening_hiw (Highlight Incorrect Words)
# ---------------------------------------------------------------------------

class HIWScorer(ScoringStrategy):
    """
    Highlight Incorrect Words scorer.

    evaluation_json:
      correctAnswers.incorrectWords: [str, ...]   (the actual wrong words as strings)
      scoringRules.correctClick: int (default 1)
      scoringRules.incorrectClick: int (default -1)

    answer dict:
      evaluation_json: dict
      highlighted_words: [str, ...]   (words the user highlighted)

    Note: The task spec mentions 'highlighted_indices' but the mobile backend
    uses word strings. This scorer accepts 'highlighted_words' (list of strings)
    to match the mobile backend's actual implementation.
    """

    @property
    def is_async(self) -> bool:
        return False

    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        evaluation_json: dict = answer.get('evaluation_json', {})

        incorrect_words = [
            (w['wrong'] if isinstance(w, dict) else w).lower().strip()
            for w in evaluation_json.get('correctAnswers', {}).get('incorrectWords', [])
        ]
        scoring_rules = evaluation_json.get('scoringRules', {})
        correct_click = scoring_rules.get('correctClick', 1)
        incorrect_click = scoring_rules.get('incorrectClick', -1)

        # Support both 'highlighted_words' and 'highlighted_indices' keys
        # highlighted_indices = word positions (int list)
        # highlighted_words = word strings (str list)
        highlighted_words_raw = answer.get('highlighted_words', answer.get('highlighted_indices', []))

        # If the caller passed indices, treat them as words (legacy compat).
        # Normalize to lowercase strings.
        highlighted_norm = [str(w).lower().strip() for w in highlighted_words_raw]

        correct_set = set(incorrect_words)

        correct_clicks = [w for w in highlighted_norm if w in correct_set]
        incorrect_clicks = [w for w in highlighted_norm if w not in correct_set]
        missed_words = [w for w in incorrect_words if w not in highlighted_norm]

        score = max(0, len(correct_clicks) * correct_click + len(incorrect_clicks) * incorrect_click)
        max_score = len(incorrect_words) * correct_click if incorrect_words else 1

        raw_score = min(1.0, max(0.0, score / max_score)) if max_score > 0 else 0.0

        return ScoringResult(
            pte_score=to_pte_score(raw_score),
            raw_score=raw_score,
            is_async=False,
            breakdown={
                'correct_clicks': correct_clicks,
                'incorrect_clicks': incorrect_clicks,
                'missed_words': missed_words,
                'score': score,
                'max_score': max_score,
            },
        )
