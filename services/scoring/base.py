from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import core.config as config


@dataclass
class ScoringResult:
    pte_score: int                          # 10-90 final PTE scale
    raw_score: float                        # 0.0-1.0 normalised percentage
    is_async: bool                          # True = Azure speaking, client must poll
    breakdown: dict = field(default_factory=dict)
    error: Optional[str] = None


def to_pte_score(weighted_pct: float) -> int:
    """PTE formula — floor 10, ceiling 90, scale 80. NEVER inline this."""
    return max(config.PTE_FLOOR, min(config.PTE_CEILING, round(config.PTE_BASE + weighted_pct * config.PTE_SCALE)))


class ScoringStrategy(ABC):
    @property
    @abstractmethod
    def is_async(self) -> bool: ...

    @abstractmethod
    def score(self, question_id: int, session_id: str, answer: dict) -> ScoringResult:
        """
        Score one question answer.
        answer: dict with type-specific fields (see registry.py for shape per type).
        Returns ScoringResult immediately. For async types, pte_score=0 and is_async=True.
        """
        ...
