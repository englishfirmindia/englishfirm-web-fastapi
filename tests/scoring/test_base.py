"""Tests for to_pte_score formula in base.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.scoring.base import to_pte_score


def test_zero_pct_gives_floor():
    assert to_pte_score(0.0) == 10


def test_full_pct_gives_ceiling():
    assert to_pte_score(1.0) == 90


def test_half_pct_gives_50():
    assert to_pte_score(0.5) == 50


def test_negative_pct_clamped_to_floor():
    assert to_pte_score(-0.5) == 10


def test_over_one_pct_clamped_to_ceiling():
    assert to_pte_score(1.5) == 90


def test_quarter_pct_gives_30():
    # 10 + 0.25 * 80 = 10 + 20 = 30
    assert to_pte_score(0.25) == 30


def test_three_quarter_pct_gives_70():
    # 10 + 0.75 * 80 = 10 + 60 = 70
    assert to_pte_score(0.75) == 70
