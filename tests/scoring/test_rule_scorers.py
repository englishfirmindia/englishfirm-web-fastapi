"""Tests for rule-based scorers — no DB, no FastAPI, no SQLAlchemy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.scoring.base import to_pte_score
from services.scoring.rule_scorer import FIBScorer, MCQScorer, ReorderScorer, WFDScorer, HIWScorer


# ============================================================
# Helpers
# ============================================================

def _fib_answer(user_answers: dict, correct: dict, marks_per_blank: int = 1,
                case_sensitive: bool = False, trim: bool = True) -> dict:
    return {
        'evaluation_json': {
            'correctAnswers': correct,
            'scoringRules': {
                'marksPerBlank': marks_per_blank,
                'isCaseSensitive': case_sensitive,
                'trimWhitespace': trim,
            },
        },
        'user_answers': user_answers,
    }


def _mcm_answer(selected: list, correct: list,
                marks_per_correct: int = 1, deduct_per_wrong: int = 0) -> dict:
    return {
        'evaluation_json': {
            'correctAnswers': {'correctOptions': correct},
            'scoringRules': {'marksPerCorrect': marks_per_correct, 'deductPerWrong': deduct_per_wrong},
        },
        'selected_options': selected,
    }


# ============================================================
# FIBScorer
# ============================================================

class TestFIBScorer:
    scorer = FIBScorer()

    def test_exact_match_all_correct(self):
        answer = _fib_answer(
            user_answers={'1': 'apple', '2': 'banana'},
            correct={'1': 'apple', '2': 'banana'},
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_all_wrong(self):
        answer = _fib_answer(
            user_answers={'1': 'wrong', '2': 'wrong'},
            correct={'1': 'apple', '2': 'banana'},
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 10

    def test_half_correct(self):
        answer = _fib_answer(
            user_answers={'1': 'apple', '2': 'wrong'},
            correct={'1': 'apple', '2': 'banana'},
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == to_pte_score(0.5)

    def test_case_insensitive_by_default(self):
        answer = _fib_answer(
            user_answers={'1': 'APPLE'},
            correct={'1': 'apple'},
            case_sensitive=False,
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_case_sensitive_mismatch_fails(self):
        answer = _fib_answer(
            user_answers={'1': 'APPLE'},
            correct={'1': 'apple'},
            case_sensitive=True,
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 10

    def test_whitespace_trimming(self):
        answer = _fib_answer(
            user_answers={'1': '  apple  '},
            correct={'1': 'apple'},
            trim=True,
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_missing_blank_treated_as_wrong(self):
        answer = _fib_answer(
            user_answers={},  # no answers provided
            correct={'1': 'apple', '2': 'banana'},
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 10

    def test_is_async_false(self):
        assert FIBScorer().is_async is False

    def test_listening_fib_blanks_format(self):
        """Listening FIB uses correctAnswers.blanks list format."""
        answer = {
            'evaluation_json': {
                'correctAnswers': {
                    'blanks': [
                        {'blankId': 1, 'answer': 'dog'},
                        {'blankId': 2, 'answer': 'cat'},
                    ]
                },
                'scoringRules': {'marksPerCorrect': 1},
            },
            'user_answers': {'1': 'dog', '2': 'cat'},
        }
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_listening_fib_partial(self):
        answer = {
            'evaluation_json': {
                'correctAnswers': {
                    'blanks': [
                        {'blankId': 1, 'answer': 'dog'},
                        {'blankId': 2, 'answer': 'cat'},
                    ]
                },
                'scoringRules': {'marksPerCorrect': 1},
            },
            'user_answers': {'1': 'dog', '2': 'wrong'},
        }
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == to_pte_score(0.5)


# ============================================================
# MCQScorer — single
# ============================================================

class TestMCQScorerSingle:
    scorer = MCQScorer(single=True)

    def test_correct_selection(self):
        answer = {
            'evaluation_json': {
                'correctAnswers': {'correctOption': 'B'},
                'scoringRules': {'marksPerCorrect': 1},
            },
            'selected_option': 'B',
        }
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_wrong_selection(self):
        answer = {
            'evaluation_json': {
                'correctAnswers': {'correctOption': 'B'},
                'scoringRules': {'marksPerCorrect': 1},
            },
            'selected_option': 'A',
        }
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 10

    def test_is_async_false(self):
        assert MCQScorer(single=True).is_async is False

    def test_hcs_options_format(self):
        """HCS/SMW use correctAnswers.options (list) instead of correctOption."""
        answer = {
            'evaluation_json': {
                'correctAnswers': {'options': [3]},
                'scoringRules': {'marksPerCorrect': 1},
            },
            'selected_option': '3',
        }
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90


# ============================================================
# MCQScorer — multi
# ============================================================

class TestMCQScorerMulti:
    scorer = MCQScorer(single=False)

    def test_all_correct_none_wrong(self):
        answer = _mcm_answer(
            selected=['A', 'C'],
            correct=['A', 'C'],
            marks_per_correct=1,
            deduct_per_wrong=1,
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_partial_correct_with_deduction(self):
        # 2 correct options total; user picks 1 correct + 1 wrong
        # score = 1*1 - 1*1 = 0 → raw = 0/2 = 0.0 → pte = 10
        answer = _mcm_answer(
            selected=['A', 'D'],     # A correct, D wrong
            correct=['A', 'C'],
            marks_per_correct=1,
            deduct_per_wrong=1,
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.raw_score == 0.0
        assert result.pte_score == 10

    def test_partial_correct_no_deduction(self):
        # 1 of 2 correct, no deduction → raw = 0.5
        answer = _mcm_answer(
            selected=['A'],
            correct=['A', 'C'],
            marks_per_correct=1,
            deduct_per_wrong=0,
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.raw_score == 0.5
        assert result.pte_score == to_pte_score(0.5)

    def test_all_wrong_clamped_to_floor(self):
        answer = _mcm_answer(
            selected=['D', 'E'],
            correct=['A', 'C'],
            marks_per_correct=1,
            deduct_per_wrong=1,
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.raw_score == 0.0
        assert result.pte_score == 10


# ============================================================
# ReorderScorer
# ============================================================

class TestReorderScorer:
    scorer = ReorderScorer()

    def _answer(self, user_seq: list, correct_seq: list, marks: int = 1) -> dict:
        return {
            'evaluation_json': {
                'correctAnswers': {'correctSequence': correct_seq},
                'scoringRules': {'marksPerAdjacentPair': marks},
            },
            'user_sequence': user_seq,
        }

    def test_perfect_order(self):
        answer = self._answer(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'])
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_zero_correct_pairs(self):
        answer = self._answer(['D', 'C', 'B', 'A'], ['A', 'B', 'C', 'D'])
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 10

    def test_partial_correct(self):
        # correct sequence: A B C D → 3 adjacent pairs: (A,B), (B,C), (C,D)
        # user sequence: A B D C → pairs: (A,B)=correct, (B,D)=wrong, (D,C)=wrong → 1/3
        answer = self._answer(['A', 'B', 'D', 'C'], ['A', 'B', 'C', 'D'])
        result = self.scorer.score(1, 'sess', answer)
        expected_raw = 1 / 3
        assert abs(result.raw_score - expected_raw) < 0.001
        assert result.pte_score == to_pte_score(expected_raw)

    def test_two_element_sequence(self):
        # 1 pair total
        answer = self._answer(['A', 'B'], ['A', 'B'])
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_is_async_false(self):
        assert ReorderScorer().is_async is False


# ============================================================
# WFDScorer
# ============================================================

class TestWFDScorer:
    scorer = WFDScorer()

    def _answer(self, user_text: str, transcript: str) -> dict:
        return {
            'evaluation_json': {
                'correctAnswers': {'transcript': transcript},
            },
            'user_text': user_text,
        }

    def test_exact_match(self):
        answer = self._answer('the quick brown fox', 'the quick brown fox')
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_all_wrong(self):
        answer = self._answer('one two three four', 'the quick brown fox')
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 10

    def test_partial_match(self):
        answer = self._answer('the quick red cat', 'the quick brown fox')
        result = self.scorer.score(1, 'sess', answer)
        assert result.raw_score == 0.5
        assert result.pte_score == to_pte_score(0.5)
        assert result.breakdown['hits'] == 2
        assert result.breakdown['total'] == 4
        assert sorted(result.breakdown['extras']) == ['cat', 'red']

    def test_reorder_full_credit(self):
        answer = self._answer('fox brown quick the', 'the quick brown fox')
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90
        assert result.breakdown['hits'] == 4
        assert result.breakdown['extras'] == []

    def test_extras_do_not_penalise(self):
        answer = self._answer('the quick brown red fox', 'the quick brown fox')
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90
        assert result.breakdown['hits'] == 4
        assert result.breakdown['extras'] == ['red']

    def test_multiset_caps_repeats(self):
        answer = self._answer('the the cat', 'the cat sat')
        result = self.scorer.score(1, 'sess', answer)
        assert result.breakdown['hits'] == 2
        assert result.breakdown['total'] == 3
        assert result.breakdown['extras'] == ['the']

    def test_strict_spelling(self):
        answer = self._answer('developments play', 'development plays')
        result = self.scorer.score(1, 'sess', answer)
        assert result.breakdown['hits'] == 0
        assert sorted(result.breakdown['extras']) == ['developments', 'play']

    def test_report_62_regression(self):
        answer = self._answer(
            'Agricultural developments play plays a vital role in poor and rural areas area.',
            'Agricultural development plays a vital role in poor and rural areas.',
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.breakdown['hits'] == 10
        assert result.breakdown['total'] == 11
        assert result.pte_score == 83
        assert sorted(result.breakdown['extras']) == ['area', 'developments', 'play']

    def test_punctuation_stripped(self):
        answer = self._answer('Hello, world!', 'Hello world')
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_case_insensitive(self):
        answer = self._answer('THE QUICK BROWN FOX', 'the quick brown fox')
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_empty_transcript(self):
        answer = self._answer('some text', '')
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 10

    def test_word_results_shape_unchanged(self):
        answer = self._answer('the quick red cat', 'the quick brown fox')
        wr = self.scorer.score(1, 'sess', answer).breakdown['word_results']
        assert set(wr.keys()) == {'0', '1', '2', '3'}
        assert wr['0'] == {'correct': 'the', 'user': 'the', 'match': True}
        assert wr['1'] == {'correct': 'quick', 'user': 'quick', 'match': True}
        assert wr['2'] == {'correct': 'brown', 'user': '', 'match': False}
        assert wr['3'] == {'correct': 'fox', 'user': '', 'match': False}

    def test_is_async_false(self):
        assert WFDScorer().is_async is False


# ============================================================
# HIWScorer
# ============================================================

class TestHIWScorer:
    scorer = HIWScorer()

    def _answer(self, highlighted: list, incorrect_words: list,
                correct_click: int = 1, incorrect_click: int = -1) -> dict:
        return {
            'evaluation_json': {
                'correctAnswers': {'incorrectWords': incorrect_words},
                'scoringRules': {
                    'correctClick': correct_click,
                    'incorrectClick': incorrect_click,
                },
            },
            'highlighted_words': highlighted,
        }

    def test_all_incorrect_words_highlighted(self):
        answer = self._answer(
            highlighted=['wrong', 'bad'],
            incorrect_words=['wrong', 'bad'],
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90

    def test_none_highlighted(self):
        answer = self._answer(
            highlighted=[],
            incorrect_words=['wrong', 'bad'],
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 10

    def test_wrong_words_highlighted_deducted(self):
        # Incorrect words = ['wrong']; user highlights 'wrong' (correct +1) and 'good' (wrong -1)
        # score = max(0, 1 + (-1)) = 0 → raw = 0 → pte = 10
        answer = self._answer(
            highlighted=['wrong', 'good'],
            incorrect_words=['wrong'],
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.raw_score == 0.0
        assert result.pte_score == 10

    def test_partial_correct(self):
        # 4 incorrect words; user hits 2, misses 2, no wrong clicks
        # score = 2/4 = 0.5
        answer = self._answer(
            highlighted=['a', 'b'],
            incorrect_words=['a', 'b', 'c', 'd'],
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.raw_score == 0.5
        assert result.pte_score == to_pte_score(0.5)

    def test_is_async_false(self):
        assert HIWScorer().is_async is False

    def test_case_insensitive_matching(self):
        answer = self._answer(
            highlighted=['WRONG', 'BAD'],
            incorrect_words=['wrong', 'bad'],
        )
        result = self.scorer.score(1, 'sess', answer)
        assert result.pte_score == 90
