"""
Scorer registry — maps each PTE question type to its ScoringStrategy.

Usage:
  from services.scoring.registry import get_scorer
  scorer = get_scorer('read_aloud')
  result = scorer.score(question_id, session_id, answer_dict)
"""

from typing import Dict

from .azure_scorer import AzureSpeakingScorer
from .rule_scorer import FIBScorer, HIWScorer, MCQScorer, ReorderScorer, WFDScorer
from .ai_scorer import AIScorer
from .base import ScoringStrategy

_REGISTRY: Dict[str, ScoringStrategy] = {
    # ── Speaking — Azure async ─────────────────────────────────────────────
    'read_aloud':                 AzureSpeakingScorer('read_aloud'),
    'repeat_sentence':            AzureSpeakingScorer('repeat_sentence'),
    'describe_image':             AzureSpeakingScorer('describe_image'),
    'retell_lecture':             AzureSpeakingScorer('retell_lecture'),
    'summarize_group_discussion': AzureSpeakingScorer('summarize_group_discussion'),
    'respond_to_situation':        AzureSpeakingScorer('respond_to_situation'),
    'ptea_respond_situation':      AzureSpeakingScorer('respond_to_situation'),
    'answer_short_question':       AzureSpeakingScorer('answer_short_question'),

    # ── Reading — rule-based sync ──────────────────────────────────────────
    'reading_mcs':           MCQScorer(single=True),
    'reading_mcm':           MCQScorer(single=False),
    'reading_fib':           FIBScorer(),
    'reading_fib_drop_down': FIBScorer(),
    'reorder_paragraphs':    ReorderScorer(),

    # ── Writing — AI sync ─────────────────────────────────────────────────
    'summarize_written_text': AIScorer('swt'),
    'write_essay':            AIScorer('we'),

    # ── Listening — mix ───────────────────────────────────────────────────
    'listening_wfd': WFDScorer(),
    'listening_sst': AIScorer('sst'),
    'listening_fib': FIBScorer(),
    'listening_mcs': MCQScorer(single=True),
    'listening_mcm': MCQScorer(single=False),
    'listening_hcs': MCQScorer(single=True),
    'listening_smw': MCQScorer(single=True),
    'listening_hiw': HIWScorer(),

    # ── Reading sectional cross-module aliases ─────────────────────────────
    'highlight_incorrect_words': HIWScorer(),
}


def get_scorer(question_type: str) -> ScoringStrategy:
    """
    Look up the ScoringStrategy for a given question type.
    Raises ValueError if the type is not registered.
    """
    scorer = _REGISTRY.get(question_type)
    if scorer is None:
        raise ValueError(
            f"No scorer registered for '{question_type}'. "
            "Add it to services/scoring/registry.py."
        )
    return scorer
