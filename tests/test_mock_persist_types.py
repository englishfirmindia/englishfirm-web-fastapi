"""Regression tests for `_build_mock_score_args` persist_type values.

History: drag-and-drop submissions used to be persisted under the dropdown
question_type, which made the `fib_drag_drop` weight bucket permanently empty
and silently penalised every mock score. This test pins the persist_type so
that bug cannot regress.
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from services.mock_service import _build_mock_score_args


class _StubQuestion:
    pass


def test_drag_and_drop_persists_under_real_type():
    """_build_mock_score_args must return persist_type='reading_drag_and_drop'
    for drag-drop submissions, even though the scorer key is the dropdown one."""
    scorer_key, _, persist_type, _ = _build_mock_score_args(
        "reading_drag_and_drop",
        {"user_answers": {"1": "a", "2": "b"}},
        {},
        {},
        _StubQuestion(),
    )
    assert persist_type == "reading_drag_and_drop"
    # Scorer key reuse (FIBScorer handles both blank shapes) is intentional
    assert scorer_key == "reading_fib_drop_down"


def test_dropdown_persists_under_its_own_type():
    """Sanity check the symmetric case: dropdown stays as dropdown."""
    _, _, persist_type, _ = _build_mock_score_args(
        "reading_fib_drop_down",
        {"user_answers": {"1": "a"}},
        {},
        {},
        _StubQuestion(),
    )
    assert persist_type == "reading_fib_drop_down"
