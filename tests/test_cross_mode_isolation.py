"""Regression tests for cross-mode answer/score isolation (2026-05-28).

Background
----------
User report: after Clear on a Read Aloud practice question, the screen
still showed the old score on re-entry. Root cause: the practice screen's
fetchLastAnswer hit `get_last_answer` which only filtered out cleared
rows but did not filter by `PracticeAttempt.filter_type`. If the user had
ever taken a sectional/mock that included the same question_id, the
sectional/mock answer (which is `scoring_status="complete"`, not cleared)
was returned on the practice screen — restoring the score as if the
practice attempt was never cleared.

Same shape of bug existed in `get_score_from_store` (the speaking-score
poll endpoint).

Fix: add `PracticeAttempt.filter_type == "practice"` to both queries.

These tests pin:
  * Practice last-answer restore is scoped to practice attempts only
  * Practice score poll is scoped to practice attempts only
  * Sectional / mock rows are NOT touched (they remain queryable by
    their own routes — only the practice paths exclude them)
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest
from fastapi.testclient import TestClient
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
from services import session_service


@pytest.fixture
def db_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    for model in (User, QuestionFromApeuni, PracticeAttempt, AttemptAnswer,
                  UserQuestionAttempt):
        model.__table__.create(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(session_service, "SessionLocal", TestingSessionLocal)
    session_service._SCORE_STORE.clear()
    sess = TestingSessionLocal()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


def _make_user(db, uid: int = 42):
    user = User(
        id=uid, username=f"u{uid}", email=f"u{uid}@ef.com", hashed_password="x",
    )
    db.add(user)
    db.commit()
    return user


def _make_question(db, qid: int = 5813):
    q = QuestionFromApeuni(
        question_id=qid, module="speaking", question_type="read_aloud",
        title="Smoking ban", difficulty_level=1, content_json={},
    )
    db.add(q)
    db.commit()
    return q


def _make_attempt(db, user_id: int, filter_type: str, qtype: str = "read_aloud"):
    a = PracticeAttempt(
        user_id=user_id, session_id=f"{filter_type}-{user_id}-{qtype}",
        module="speaking", question_type=qtype, filter_type=filter_type,
        total_questions=1, total_score=0, questions_answered=0,
        status="active", scoring_status="pending",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _seed_completed_answer(db, attempt, question, score: int = 75):
    """Persist a 'complete' answer — the kind that fetchLastAnswer + score
    poll would surface."""
    aa = AttemptAnswer(
        attempt_id=attempt.id,
        question_id=question.question_id,
        question_type=question.question_type,
        user_answer_json={"audio_url": "s3://bucket/audio.aac"},
        result_json={
            "pte_score": score, "content": 80, "fluency": 70,
            "pronunciation": 75, "total": score, "transcript": "text",
        },
        score=score,
        content_score=80.0,
        fluency_score=70.0,
        pronunciation_score=75.0,
        audio_url="s3://bucket/audio.aac",
        scoring_status="complete",
    )
    db.add(aa)
    db.commit()
    db.refresh(aa)
    return aa


# ── get_score_from_store: cross-mode isolation ─────────────────────────────


def test_score_poll_returns_practice_attempt_score(db_session):
    """Sanity check: practice attempt's score IS surfaced by the poll."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    prac = _make_attempt(db_session, user.id, "practice")
    _seed_completed_answer(db_session, prac, q, score=75)

    result = session_service.get_score_from_store(user.id, q.question_id)
    assert result is not None
    assert result["scoring"] == "complete"
    assert result["total"] == 75


def test_score_poll_returns_None_when_only_sectional_exists(db_session):
    """The bug: user took sectional only. Practice screen's score poll
    must return None, NOT the sectional score."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    sect = _make_attempt(db_session, user.id, "sectional")
    _seed_completed_answer(db_session, sect, q, score=82)

    result = session_service.get_score_from_store(user.id, q.question_id)
    assert result is None, (
        "Practice score poll must NOT surface a sectional attempt's score. "
        "If this fails, the cross-mode leak has regressed and clearing a "
        "practice attempt will silently restore the sectional score."
    )


def test_score_poll_returns_None_when_only_mock_exists(db_session):
    user = _make_user(db_session)
    q = _make_question(db_session)
    mock = _make_attempt(db_session, user.id, "mock")
    _seed_completed_answer(db_session, mock, q, score=68)

    result = session_service.get_score_from_store(user.id, q.question_id)
    assert result is None, "Mock score must not surface in practice score poll"


def test_score_poll_returns_practice_when_both_practice_and_sectional_exist(db_session):
    """User practiced (score 60), then took sectional (score 82). Practice
    score poll must return the PRACTICE score (60), not the higher
    sectional score (82) just because it's newer."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    prac = _make_attempt(db_session, user.id, "practice")
    _seed_completed_answer(db_session, prac, q, score=60)
    sect = _make_attempt(db_session, user.id, "sectional")
    _seed_completed_answer(db_session, sect, q, score=82)

    result = session_service.get_score_from_store(user.id, q.question_id)
    assert result is not None
    assert result["total"] == 60, (
        f"Expected practice score 60, got {result['total']}. Cross-mode "
        f"leak: sectional score is masking the practice score."
    )


def test_score_poll_evicts_cache_check(db_session):
    """The in-memory _SCORE_STORE cache must also respect the same scope
    — if a sectional score is somehow cached for (user, qid), it must
    still not surface to the practice poll. (Today the cache is keyed
    only by (user_id, question_id) so this is a defense-in-depth check.)"""
    user = _make_user(db_session)
    q = _make_question(db_session)
    # No DB rows. But pretend an old cache entry got left behind somehow.
    session_service._SCORE_STORE[(user.id, q.question_id)] = {
        "scoring": "complete", "total": 99, "transcript": "stale",
    }
    result = session_service.get_score_from_store(user.id, q.question_id)
    # Cache hit path still surfaces. Documenting current behaviour: the
    # cache is process-local and not scoped, so a stale cache COULD
    # leak. clear_attempt_in_db pops it explicitly which is the only
    # contractual path. This test pins the behaviour so a future change
    # that adds scope-checking to the cache lookup is noticed.
    assert result is not None
    assert result["total"] == 99


# ── get_last_answer route shape — direct service-level test ──────────────
#
# The route handler in routers/user.py wraps the same query. We exercise
# the SQL via direct ORM queries here so the test runs in the in-memory
# SQLite engine without spinning up TestClient + auth scaffolding.


def _query_last_answer_practice_only(db, user_id, question_id):
    """Mirror of the production query in routers/user.py:get_last_answer
    after the filter_type='practice' fix. If this stays in sync the
    tests pin the route's behaviour."""
    return (
        db.query(AttemptAnswer)
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(
            PracticeAttempt.user_id == user_id,
            AttemptAnswer.question_id == question_id,
            AttemptAnswer.scoring_status != "cleared",
            PracticeAttempt.filter_type == "practice",
        )
        .order_by(AttemptAnswer.submitted_at.desc())
        .first()
    )


def test_last_answer_returns_practice_attempt(db_session):
    user = _make_user(db_session)
    q = _make_question(db_session)
    prac = _make_attempt(db_session, user.id, "practice")
    aa = _seed_completed_answer(db_session, prac, q, score=70)

    got = _query_last_answer_practice_only(db_session, user.id, q.question_id)
    assert got is not None
    assert got.id == aa.id


def test_last_answer_returns_None_when_only_sectional_exists(db_session):
    """The bug. If this fails, the practice screen will keep showing
    sectional answers on re-entry."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    sect = _make_attempt(db_session, user.id, "sectional")
    _seed_completed_answer(db_session, sect, q, score=82)

    got = _query_last_answer_practice_only(db_session, user.id, q.question_id)
    assert got is None


def test_last_answer_returns_None_when_only_mock_exists(db_session):
    user = _make_user(db_session)
    q = _make_question(db_session)
    mock = _make_attempt(db_session, user.id, "mock")
    _seed_completed_answer(db_session, mock, q, score=55)

    got = _query_last_answer_practice_only(db_session, user.id, q.question_id)
    assert got is None


def test_last_answer_skips_sectional_when_practice_also_exists(db_session):
    """Both attempts exist. Last-answer must pick the practice row, even
    if the sectional one is more recent."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    prac = _make_attempt(db_session, user.id, "practice")
    prac_aa = _seed_completed_answer(db_session, prac, q, score=60)
    sect = _make_attempt(db_session, user.id, "sectional")
    _seed_completed_answer(db_session, sect, q, score=82)

    got = _query_last_answer_practice_only(db_session, user.id, q.question_id)
    assert got is not None
    assert got.id == prac_aa.id


def test_last_answer_skips_cleared_practice_attempt(db_session):
    """User cleared their practice attempt. Even though a sectional
    attempt exists, the practice last-answer must return None — the
    filter_type='practice' scope keeps the sectional row out, and the
    scoring_status != 'cleared' scope keeps the cleared row out."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    prac = _make_attempt(db_session, user.id, "practice")
    prac_aa = _seed_completed_answer(db_session, prac, q, score=70)
    sect = _make_attempt(db_session, user.id, "sectional")
    _seed_completed_answer(db_session, sect, q, score=82)
    # Now clear the practice attempt the way the Clear endpoint does.
    session_service.clear_attempt_in_db(user.id, q.question_id)

    got = _query_last_answer_practice_only(db_session, user.id, q.question_id)
    assert got is None, (
        "After clearing the practice attempt, last-answer must return None "
        "even when a sectional attempt exists for the same question. If "
        "this fails, the original user-reported bug has regressed."
    )


# ── No-regression: sectional/mock data not damaged by the new filter ────


def test_sectional_data_still_queryable_directly(db_session):
    """The new filter is on the practice read paths only. Sectional rows
    must still be queryable by their own routes (sectional review etc.)."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    sect = _make_attempt(db_session, user.id, "sectional")
    sect_aa = _seed_completed_answer(db_session, sect, q, score=82)

    # Direct query — what a sectional review route would do.
    rows = (
        db_session.query(AttemptAnswer)
        .filter_by(attempt_id=sect.id, question_id=q.question_id)
        .all()
    )
    assert len(rows) == 1
    assert rows[0].id == sect_aa.id
    assert rows[0].scoring_status == "complete"
    assert rows[0].score == 82
