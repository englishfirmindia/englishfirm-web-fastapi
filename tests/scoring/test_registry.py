"""Tests for the scorer registry."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from services.scoring.registry import get_scorer, _REGISTRY
from services.scoring.azure_scorer import AzureSpeakingScorer
from services.scoring.ai_scorer import AIScorer


# All 21 registered question types
ALL_21_TYPES = [
    # Speaking (7)
    'read_aloud',
    'repeat_sentence',
    'describe_image',
    'retell_lecture',
    'summarize_group_discussion',
    'respond_to_situation',
    'answer_short_question',
    # Reading (5)
    'reading_mcs',
    'reading_mcm',
    'reading_fib',
    'reading_fib_drop_down',
    'reorder_paragraphs',
    # Writing (2)
    'summarize_written_text',
    'write_essay',
    # Listening (7)
    'listening_wfd',
    'listening_sst',
    'listening_fib',
    'listening_mcs',
    'listening_mcm',
    'listening_hcs',
    'listening_smw',
    'listening_hiw',
]

SPEAKING_TYPES = [
    'read_aloud', 'repeat_sentence', 'describe_image',
    'retell_lecture', 'summarize_group_discussion',
    'respond_to_situation', 'answer_short_question',
]

# These are async (Azure)
ASYNC_TYPES = SPEAKING_TYPES

# These are sync (rule-based or AI)
SYNC_TYPES = [t for t in ALL_21_TYPES if t not in ASYNC_TYPES]


class TestRegistryCompleteness:

    def test_all_21_types_registered(self):
        for qt in ALL_21_TYPES:
            assert qt in _REGISTRY, f"Question type '{qt}' not registered"

    def test_get_scorer_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="No scorer registered"):
            get_scorer('unknown_type_xyz')

    def test_get_scorer_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            get_scorer('')


class TestScorerTypes:

    def test_speaking_types_are_async(self):
        for qt in SPEAKING_TYPES:
            scorer = get_scorer(qt)
            assert scorer.is_async is True, f"{qt} should be async"

    def test_non_speaking_types_are_sync(self):
        for qt in SYNC_TYPES:
            scorer = get_scorer(qt)
            assert scorer.is_async is False, f"{qt} should be sync (not async)"

    def test_speaking_scorers_are_azure_instances(self):
        for qt in SPEAKING_TYPES:
            scorer = get_scorer(qt)
            assert isinstance(scorer, AzureSpeakingScorer), f"{qt} should be AzureSpeakingScorer"

    def test_sst_is_ai_sync(self):
        scorer = get_scorer('listening_sst')
        assert scorer.is_async is False
        assert isinstance(scorer, AIScorer)

    def test_swt_is_ai_sync(self):
        scorer = get_scorer('summarize_written_text')
        assert scorer.is_async is False
        assert isinstance(scorer, AIScorer)

    def test_write_essay_is_ai_sync(self):
        scorer = get_scorer('write_essay')
        assert scorer.is_async is False
        assert isinstance(scorer, AIScorer)


class TestScorerInstances:

    def test_get_scorer_returns_same_instance(self):
        """Registry should return the same shared instance."""
        s1 = get_scorer('read_aloud')
        s2 = get_scorer('read_aloud')
        assert s1 is s2

    def test_reading_fib_and_drop_down_share_fib_type(self):
        from services.scoring.rule_scorer import FIBScorer
        assert isinstance(get_scorer('reading_fib'), FIBScorer)
        assert isinstance(get_scorer('reading_fib_drop_down'), FIBScorer)

    def test_listening_fib_is_fib_scorer(self):
        from services.scoring.rule_scorer import FIBScorer
        assert isinstance(get_scorer('listening_fib'), FIBScorer)

    def test_reorder_paragraphs_is_reorder_scorer(self):
        from services.scoring.rule_scorer import ReorderScorer
        assert isinstance(get_scorer('reorder_paragraphs'), ReorderScorer)

    def test_wfd_is_wfd_scorer(self):
        from services.scoring.rule_scorer import WFDScorer
        assert isinstance(get_scorer('listening_wfd'), WFDScorer)

    def test_hiw_is_hiw_scorer(self):
        from services.scoring.rule_scorer import HIWScorer
        assert isinstance(get_scorer('listening_hiw'), HIWScorer)
