"""Regression tests for mcp_server.tools.get_weak_areas
(rewritten 2026-09-05).

Old code had a hand-maintained `_MAX_PTS` dict that had drifted from
the actual DB `attempt_answers.score` semantics on two axes — stale
question-type names AND stale unit assumptions (raw rubric vs PTE
10-90 scale). Result surfaced by Kaspin (u=75) getting "critical, 0%"
Coach responses and Nimisha (u=3) getting NO weak areas flagged
despite 96+ real practice attempts.

These tests pin the new formula (pct = max(0, (avg-10)/80)) and its
core invariants using a pure isolated DB (SQLite in-memory), so a
future refactor can't silently revert to the buggy per-type-max_pts
model.
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, INET
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import JSON


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return compiler.visit_JSON(JSON())


@compiles(ARRAY, "sqlite")
def _compile_array_sqlite(type_, compiler, **kw):
    return "TEXT"


@compiles(INET, "sqlite")
def _compile_inet_sqlite(type_, compiler, **kw):
    return "TEXT"


from db.models import Base, User, PracticeAttempt, AttemptAnswer  # noqa: E402
from mcp_server.tools import get_weak_areas                          # noqa: E402


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()
    engine.dispose()


def _seed_user(db) -> int:
    u = User(
        username="tester",
        email="tester@example.com",
        hashed_password="x",
    )
    db.add(u)
    db.commit()
    return u.id


def _seed_attempt(
    db, user_id: int, question_type: str, scores: list[int],
    filter_type: str = "practice", status: str = "complete",
    scoring_status: str = "complete",
) -> None:
    """Create ONE PracticeAttempt + N AttemptAnswer rows with the given
    per-question scores under `question_type`. All defaulted to
    'complete' so get_weak_areas picks them up."""
    import uuid
    a = PracticeAttempt(
        user_id=user_id,
        session_id=f"sess-{uuid.uuid4()}",
        module="reading",
        question_type=question_type,
        filter_type=filter_type,
        total_questions=len(scores),
        total_score=sum(scores),
        questions_answered=len(scores),
        status=status,
        scoring_status=scoring_status,
    )
    db.add(a)
    db.commit()
    for i, s in enumerate(scores):
        db.add(AttemptAnswer(
            attempt_id=a.id,
            question_id=100 + i,
            question_type=question_type,
            score=s,
            scoring_status=scoring_status,
        ))
    db.commit()


def test_returns_empty_for_user_with_no_attempts(db):
    u = _seed_user(db)
    assert get_weak_areas(u, db) == []


def test_single_attempt_never_flagged(db):
    """N<2 filter is preserved from old code — one bad score shouldn't
    trigger a weak-area flag (too little data to be meaningful)."""
    u = _seed_user(db)
    _seed_attempt(db, u, "mcq_multiple", [15])   # one floor-ish score
    assert get_weak_areas(u, db) == []


def test_pte_scale_formula_is_avg_minus_10_over_80(db):
    """Core new-formula pin. avg=50 → pct = (50-10)/80 = 0.50 exactly."""
    u = _seed_user(db)
    _seed_attempt(db, u, "reading_fib_drop_down", [30, 50, 70])  # avg=50
    weak = get_weak_areas(u, db)
    assert len(weak) == 1
    assert weak[0]["task_type"] == "reading_fib_drop_down"
    assert weak[0]["pct"] == 0.5   # rounded to 3 places


def test_score_at_floor_reports_zero_pct(db):
    """A user averaging the PTE floor (10) must show 0% — the whole
    point of the fix. (Old code would have shown 66-100% depending on
    the type's max_pts entry.)"""
    u = _seed_user(db)
    _seed_attempt(db, u, "mcq_multiple", [10, 10, 10])
    weak = get_weak_areas(u, db)
    assert len(weak) == 1
    assert weak[0]["pct"] == 0.0


def test_below_floor_clamps_to_zero_not_negative(db):
    """Silent-mic / abandoned rows can store score=0 (below the PTE
    floor of 10). Formula must clamp to 0%, not go negative."""
    u = _seed_user(db)
    _seed_attempt(db, u, "listening_wfd", [0, 0])
    weak = get_weak_areas(u, db)
    assert weak[0]["pct"] == 0.0


def test_score_at_ceiling_not_flagged(db):
    """avg=90 → pct=1.0 → NOT below 0.55 threshold → not weak."""
    u = _seed_user(db)
    _seed_attempt(db, u, "read_aloud", [85, 90, 90])   # avg=88.33
    weak = get_weak_areas(u, db)
    assert weak == []


def test_all_live_question_types_use_pte_scale(db):
    """None of the modern question_types should be silently
    misclassified as raw rubric. Seed each with avg=50 (which under
    the new formula is exactly 50% — right at threshold, so it
    should NOT be flagged since threshold is strict `<`)."""
    u = _seed_user(db)
    modern_types = [
        "read_aloud", "repeat_sentence", "describe_image",
        "retell_lecture", "respond_to_situation",
        "summarize_group_discussion", "answer_short_question",
        "summarize_written_text", "write_essay",
        "reading_fib_drop_down", "reading_drag_and_drop",
        "reorder_paragraphs", "mcq_single", "mcq_multiple",
        "listening_wfd", "listening_fib", "listening_sst",
        "listening_mcq_single", "listening_mcq_multiple",
        "listening_hcs", "listening_smw", "highlight_incorrect_words",
    ]
    for qt in modern_types:
        _seed_attempt(db, u, qt, [30, 50, 70])   # avg=50 → 50% exact
    weak = get_weak_areas(u, db)
    # avg=50 → pct=0.5, strictly less than 0.55 threshold → ALL flagged
    got = {w["task_type"] for w in weak}
    assert got == set(modern_types), (
        f"Missing: {set(modern_types) - got}; extra: {got - set(modern_types)}"
    )


def test_deprecated_raw_types_are_skipped(db):
    """The 7 dead legacy types (fill_in_the_blanks, reading_fib, etc.)
    store raw rubric points, not PTE scale. Applying (avg-10)/80 to a
    raw score of 5 would compute a nonsense negative → clamped to 0%
    → false weak-area flag. Better to skip them entirely."""
    u = _seed_user(db)
    _seed_attempt(db, u, "fill_in_the_blanks", [0, 0])
    _seed_attempt(db, u, "reading_fib", [10, 10])
    _seed_attempt(db, u, "write_from_dictation", [0, 5])
    _seed_attempt(db, u, "select_missing_word", [1, 1])
    weak = get_weak_areas(u, db)
    assert weak == [], (
        f"Deprecated types leaked into weak_areas: {[w['task_type'] for w in weak]}"
    )


def test_incomplete_attempts_excluded(db):
    """Only status='complete' PracticeAttempt rows count — an
    abandoned mid-test shouldn't skew the weak-area picture."""
    u = _seed_user(db)
    _seed_attempt(db, u, "mcq_multiple", [15, 20], status="pending")
    _seed_attempt(db, u, "mcq_multiple", [70, 80], status="complete")
    weak = get_weak_areas(u, db)
    # Only the complete attempt (avg=75) counts → NOT weak.
    assert weak == []


def test_uncompleted_scoring_excluded(db):
    """AttemptAnswer.scoring_status must be 'complete' to count.
    An in-progress speaking submission (Azure still working) shouldn't
    poison the average."""
    u = _seed_user(db)
    _seed_attempt(db, u, "mcq_multiple", [15, 20], scoring_status="pending")
    _seed_attempt(db, u, "mcq_multiple", [70, 80], scoring_status="complete")
    weak = get_weak_areas(u, db)
    assert weak == []


def test_results_sorted_worst_first(db):
    """weak_areas.sort(key=pct) — LLM prompt says "worst first" so the
    order matters for the coach's advice."""
    u = _seed_user(db)
    _seed_attempt(db, u, "mcq_multiple",             [10, 10, 10])   # 0%
    _seed_attempt(db, u, "reading_fib_drop_down",    [30, 30, 30])   # 25%
    _seed_attempt(db, u, "listening_hcs",            [40, 40, 40])   # 37.5%
    weak = get_weak_areas(u, db)
    assert [w["task_type"] for w in weak] == [
        "mcq_multiple", "reading_fib_drop_down", "listening_hcs",
    ]


def test_threshold_is_strict_less_than(db):
    """A user exactly at the threshold (55%) should NOT be flagged —
    they're on the boundary of "acceptable". Threshold semantics
    matter for the CTA line the LLM generates."""
    u = _seed_user(db)
    # avg=54 → pct=(54-10)/80=0.55 exact → NOT weak (strict `<`)
    _seed_attempt(db, u, "read_aloud", [54, 54, 54])
    weak = get_weak_areas(u, db)
    assert weak == []


def test_payload_shape_preserved_for_callers(db):
    """_fmt_weak_areas and claude_router both read specific keys off
    each result — pin the payload shape so refactors don't silently
    break the prompt formatting."""
    u = _seed_user(db)
    _seed_attempt(db, u, "mcq_multiple", [20, 30])
    weak = get_weak_areas(u, db)
    assert len(weak) == 1
    w = weak[0]
    assert set(w.keys()) == {"task_type", "avg_score", "max_score",
                              "pct", "attempt_count"}
    assert w["task_type"] == "mcq_multiple"
    assert w["max_score"] == 90    # PTE ceiling, for backwards compat


def test_custom_threshold_is_honoured(db):
    """threshold_pct is a public parameter — some callers might pass
    a tighter or looser cutoff. Verify it flows through."""
    u = _seed_user(db)
    _seed_attempt(db, u, "read_aloud", [50, 60, 70])    # avg=60 → 62.5%
    assert get_weak_areas(u, db, threshold_pct=0.55) == []
    assert len(get_weak_areas(u, db, threshold_pct=0.70)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
