"""Regression tests for question_type alias coverage (2026-06-09).

Background
----------
User reported: SST already-done questions were not showing as "Done" in
the practice list. Audit revealed 7 of 22 question types had a string
mismatch between the submit-side `attempt_answers.question_type` write
and the list-side `practiced_questions_subq` lookup. The badge logic
itself is sound; the lookup was just filtering by a different string
than was written.

Fix: expand the alias table in services/question_list_helper.py so the
helper's IN-clause matches whichever string the submit or list side
passes. The map is now the SINGLE source of truth — routers/user.py
imports from here instead of having its own copy.

These tests pin:
  1. Every previously-broken pair maps both directions to the same set
  2. `practiced_questions_subq` returns the AttemptAnswer row regardless
     of which alias the caller passes
  3. `practiced_question_ids_in` returns the qid regardless of alias
  4. The previously-working types (reading_fib pair, the 15 that were
     never broken) continue to work
  5. routers/user.py and services/question_list_helper.py use the same
     map (no drift can re-emerge)
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import JSON


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return compiler.visit_JSON(JSON())


@compiles(ARRAY, "sqlite")
def _compile_array_sqlite(type_, compiler, **kw):
    return "TEXT"


from db.models import (
    AttemptAnswer,
    PracticeAttempt,
    QuestionFromApeuni,
    User,
    UserQuestionAttempt,
)
from services.question_list_helper import (
    QUESTION_TYPE_ALIASES,
    practiced_questions_subq,
    practiced_question_ids_in,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    for model in (User, QuestionFromApeuni, PracticeAttempt, AttemptAnswer,
                  UserQuestionAttempt):
        model.__table__.create(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    sess = TestingSessionLocal()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


def _make_user(db, uid=1):
    user = User(id=uid, username=f"u{uid}", email=f"u{uid}@ef.com", hashed_password="x")
    db.add(user)
    db.commit()
    return user


def _make_practice_attempt(db, user_id, question_type):
    a = PracticeAttempt(
        user_id=user_id, session_id=f"sess-{user_id}-{question_type}",
        module="speaking", question_type=question_type, filter_type="practice",
        total_questions=1, total_score=0, questions_answered=0,
        status="active", scoring_status="pending",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _make_answer(db, attempt_id, question_id, question_type):
    aa = AttemptAnswer(
        attempt_id=attempt_id, question_id=question_id, question_type=question_type,
        user_answer_json={"x": 1}, result_json={"pte_score": 70}, score=70,
        scoring_status="complete",
    )
    db.add(aa)
    db.commit()


# ── 1. Map structure: every drift pair is symmetric ─────────────────────


# Pairs of (submit-side write, list-side filter) — both must map to the
# same tuple so the IN-clause matches whichever the caller passes.
DRIFT_PAIRS = [
    ("reading_fib", "reading_drag_and_drop"),
    ("respond_to_situation", "ptea_respond_situation"),
    ("reading_mcs", "mcq_single"),
    ("reading_mcm", "mcq_multiple"),
    ("listening_sst", "summarize_spoken_text"),
    ("listening_mcs", "listening_mcq_single"),
    ("listening_mcm", "listening_mcq_multiple"),
    ("listening_hiw", "highlight_incorrect_words"),
]


@pytest.mark.parametrize("a,b", DRIFT_PAIRS)
def test_alias_pair_is_symmetric(a, b):
    """Both directions of every drift pair must resolve to the same set,
    so a caller using either string finds rows written under either."""
    assert a in QUESTION_TYPE_ALIASES, f"Missing alias entry for '{a}'"
    assert b in QUESTION_TYPE_ALIASES, f"Missing alias entry for '{b}'"
    assert QUESTION_TYPE_ALIASES[a] == QUESTION_TYPE_ALIASES[b], (
        f"Aliases for '{a}' and '{b}' must point to the same tuple. "
        f"Got {QUESTION_TYPE_ALIASES[a]} vs {QUESTION_TYPE_ALIASES[b]}."
    )
    assert a in QUESTION_TYPE_ALIASES[a], (
        f"'{a}' must be present in its own alias tuple."
    )
    assert b in QUESTION_TYPE_ALIASES[a], (
        f"'{b}' must be present in '{a}' alias tuple (otherwise the IN-clause "
        f"won't find rows written under either form)."
    )


# ── 2. practiced_questions_subq finds rows under either alias ───────────


@pytest.mark.parametrize("submit_type,list_type", DRIFT_PAIRS)
def test_subq_finds_practiced_row_via_either_alias(
    db_session, submit_type, list_type
):
    """Simulate the production trace: submit writes one string, list
    looks up by the other. Helper must find the row both ways."""
    user = _make_user(db_session)
    # Submit writes AttemptAnswer with `submit_type`.
    attempt = _make_practice_attempt(db_session, user.id, submit_type)
    _make_answer(db_session, attempt.id, question_id=1234,
                 question_type=submit_type)

    # List endpoint queries with `list_type`. Helper should find the row.
    subq = practiced_questions_subq(db_session, user.id, list_type)
    found_ids = {r[0] for r in db_session.query(subq).all()}
    assert 1234 in found_ids, (
        f"Submit wrote question_type='{submit_type}'. List queries with "
        f"question_type='{list_type}'. The alias map must bridge them. "
        f"If this fails, the Done badge for {submit_type}/{list_type} "
        f"questions is broken in production."
    )

    # And the reverse — list queries with `submit_type` should also work.
    subq2 = practiced_questions_subq(db_session, user.id, submit_type)
    found_ids2 = {r[0] for r in db_session.query(subq2).all()}
    assert 1234 in found_ids2


# ── 3. practiced_question_ids_in (per-page enrichment) ──────────────────


@pytest.mark.parametrize("submit_type,list_type", DRIFT_PAIRS)
def test_ids_in_finds_practiced_via_either_alias(
    db_session, submit_type, list_type
):
    user = _make_user(db_session)
    attempt = _make_practice_attempt(db_session, user.id, submit_type)
    _make_answer(db_session, attempt.id, 999, submit_type)

    found_a = practiced_question_ids_in(
        db_session, user.id, list_type, restrict_to=[999, 1000]
    )
    assert found_a == {999}, (
        f"practiced_question_ids_in failed to resolve '{list_type}' → "
        f"'{submit_type}' alias. Done badge for the per-page enrichment "
        f"path is broken."
    )

    found_b = practiced_question_ids_in(
        db_session, user.id, submit_type, restrict_to=[999, 1000]
    )
    assert found_b == {999}


# ── 4. No bleed: questions in OTHER types don't leak across aliases ─────


def test_alias_does_not_bleed_into_unrelated_types(db_session):
    """If a user has a read_aloud row, it must not show up when the
    listening_sst list endpoint queries — the alias map only bridges
    known drift pairs, not unrelated types."""
    user = _make_user(db_session)
    attempt = _make_practice_attempt(db_session, user.id, "read_aloud")
    _make_answer(db_session, attempt.id, 5001, "read_aloud")

    subq = practiced_questions_subq(db_session, user.id, "listening_sst")
    found_ids = {r[0] for r in db_session.query(subq).all()}
    assert found_ids == set(), (
        "listening_sst lookup must not find read_aloud rows — the alias "
        "map only bridges known drift pairs."
    )


# ── 5. Cleared rows still excluded across aliases ───────────────────────


def test_cleared_rows_remain_excluded_even_with_alias_expansion(db_session):
    """Defense check: the scoring_status != 'cleared' filter must still
    apply after the alias expansion. Otherwise a cleared SST attempt
    would re-surface in 'Done' via the alias path."""
    user = _make_user(db_session)
    attempt = _make_practice_attempt(db_session, user.id, "listening_sst")
    aa = AttemptAnswer(
        attempt_id=attempt.id, question_id=7777, question_type="listening_sst",
        user_answer_json={}, result_json={}, score=0,
        scoring_status="cleared",  # explicitly cleared
    )
    db_session.add(aa)
    db_session.commit()

    subq = practiced_questions_subq(db_session, user.id, "summarize_spoken_text")
    found_ids = {r[0] for r in db_session.query(subq).all()}
    assert 7777 not in found_ids, (
        "Cleared rows must stay hidden in Done even through alias expansion."
    )


# ── 6. Single source of truth: routers/user.py imports same map ─────────


def test_routers_user_imports_shared_alias_map():
    """routers/user.py must import the alias table from
    services/question_list_helper.py, not redefine its own copy.
    Prevents drift re-emerging."""
    from routers import user as user_router
    assert hasattr(user_router, "_QUESTION_TYPE_ALIASES")
    assert user_router._QUESTION_TYPE_ALIASES is QUESTION_TYPE_ALIASES, (
        "routers/user.py must use the SAME object as "
        "services/question_list_helper.QUESTION_TYPE_ALIASES (import, "
        "don't copy) — otherwise a future drift between the two could "
        "silently reintroduce the 7-type Done-badge bug."
    )
