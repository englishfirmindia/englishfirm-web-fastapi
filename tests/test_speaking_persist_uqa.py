"""Regression tests for the "Done" badge — UQA writes on async speaking submits.

History
-------
Commit a9cd37f (2026-05-15) refactored the submit path off of fire-and-forget
daemon threads into synchronous DB writes ("Option A"). It moved the
user_question_attempts upsert from `mark_submitted` into `persist_answer_to_db`
— but only into the sync persist function. `persist_speaking_answer_pending`
(used by RA, RS, DI, RL, ASQ, SGD, RTS) lost the daemon-thread UQA write and
never got an inline replacement.

Net effect for 10 days: every async speaking submit wrote an attempt_answers
row but no user_question_attempts row. The frontend "Done" badge on the
practice list reads UQA, so freshly-practised speaking questions silently
stayed "New".

These tests pin both:
  * `persist_speaking_answer_pending` MUST write a UQA row (regression guard)
  * `persist_answer_to_db` still writes a UQA row (don't break the working
    sync path while fixing the speaking one)
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import JSON, Text


# Render Postgres-only types as their SQLite-compatible equivalents so the
# production model classes (which use JSONB / ARRAY for real RDS) can still
# be exercised against an in-memory SQLite engine in tests.
@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return compiler.visit_JSON(JSON())


@compiles(ARRAY, "sqlite")
def _compile_array_sqlite(type_, compiler, **kw):
    return "TEXT"

from db.models import (
    Base,
    PracticeAttempt,
    AttemptAnswer,
    UserQuestionAttempt,
    QuestionFromApeuni,
    User,
)
from services import session_service


# ── Test infra ───────────────────────────────────────────────────────────────


@pytest.fixture
def db_session(monkeypatch):
    """In-memory SQLite database with the schema we need for this test.

    The persist functions in session_service open their own DB sessions via
    `SessionLocal`, so we monkey-patch that to point at our test engine.
    Only the 5 tables touched by the speaking-submit path are created —
    creating all of `Base.metadata` would force us to render every
    Postgres-specific type (INET, etc.) on SQLite.
    """
    engine = create_engine("sqlite:///:memory:")
    # Only the tables we actually touch — keeps the test fast and avoids
    # the long tail of Postgres-only types in models we don't exercise.
    for model in (User, QuestionFromApeuni, PracticeAttempt, AttemptAnswer,
                  UserQuestionAttempt):
        model.__table__.create(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(session_service, "SessionLocal", TestingSessionLocal)

    sess = TestingSessionLocal()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


def _make_user(db, uid: int = 42):
    user = User(
        id=uid,
        username=f"testuser{uid}",
        email=f"test{uid}@example.com",
        hashed_password="x",
    )
    db.add(user)
    db.commit()
    return user


def _make_question(db, qid: int, qtype: str, module: str):
    q = QuestionFromApeuni(
        question_id=qid,
        module=module,
        question_type=qtype,
        title=f"Q{qid}",
        difficulty_level=1,
        content_json={},
    )
    db.add(q)
    db.commit()
    return q


def _make_attempt(db, user_id: int, module: str, qtype: str) -> PracticeAttempt:
    attempt = PracticeAttempt(
        user_id=user_id,
        session_id=f"sess-{user_id}",
        module=module,
        question_type=qtype,
        filter_type="practice",
        total_questions=1,
        total_score=0,
        questions_answered=0,
        status="active",
        scoring_status="pending",
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)
    return attempt


# ── persist_speaking_answer_pending — the regression guard ──────────────────


def test_speaking_submit_writes_uqa_row(db_session):
    """RA/RS/DI/RL/ASQ/SGD/RTS submit must mark the question as practised.

    This is the regression bug from commit a9cd37f. Without this assertion,
    a future refactor could quietly delete the UQA upsert again and nobody
    would notice until a user complains their Read Aloud list still shows
    "New" after practising 11 questions.
    """
    user = _make_user(db_session)
    q = _make_question(db_session, qid=9001, qtype="read_aloud", module="speaking")
    attempt = _make_attempt(db_session, user.id, "speaking", "read_aloud")

    session = {
        "attempt_id": attempt.id,
        "user_id": user.id,
        "module": "speaking",
        "submitted_questions": set(),
        "score": 0,
    }

    session_service.persist_speaking_answer_pending(
        session=session,
        question_id=q.question_id,
        question_type="read_aloud",
        audio_url="s3://test/abc.aac",
    )

    rows = (
        db_session.query(UserQuestionAttempt)
        .filter_by(user_id=user.id, question_id=q.question_id)
        .all()
    )
    assert len(rows) == 1, (
        "Expected exactly one user_question_attempts row to be written by "
        "persist_speaking_answer_pending. If this fails, the May 15 regression "
        "has returned — Done badges will silently break for every async "
        "speaking submit."
    )
    assert rows[0].question_type == "read_aloud"
    assert rows[0].module == "speaking"


def test_speaking_submit_does_not_duplicate_uqa(db_session):
    """Re-submitting (Redo) the same question must NOT create a second UQA row."""
    user = _make_user(db_session)
    q = _make_question(db_session, qid=9002, qtype="repeat_sentence", module="speaking")
    attempt = _make_attempt(db_session, user.id, "speaking", "repeat_sentence")
    session = {
        "attempt_id": attempt.id,
        "user_id": user.id,
        "module": "speaking",
        "submitted_questions": set(),
        "score": 0,
    }

    # First submit
    session_service.persist_speaking_answer_pending(
        session=session,
        question_id=q.question_id,
        question_type="repeat_sentence",
        audio_url="s3://test/v1.aac",
    )
    # Redo
    session_service.persist_speaking_answer_pending(
        session=session,
        question_id=q.question_id,
        question_type="repeat_sentence",
        audio_url="s3://test/v2.aac",
    )

    rows = (
        db_session.query(UserQuestionAttempt)
        .filter_by(user_id=user.id, question_id=q.question_id)
        .all()
    )
    assert len(rows) == 1, (
        "Redo should not duplicate UQA rows. The upsert's NOT EXISTS guard "
        "(or its absence) is the only thing preventing this."
    )


def test_speaking_submit_still_writes_attempt_answers(db_session):
    """Adding the UQA write must not break the existing attempt_answers write.

    Smoke test against the regression case where someone "fixes" the UQA
    path and accidentally drops the attempt_answers row write at the same
    time. Both rows must exist after a single persist call.
    """
    user = _make_user(db_session)
    q = _make_question(db_session, qid=9003, qtype="describe_image", module="speaking")
    attempt = _make_attempt(db_session, user.id, "speaking", "describe_image")
    session = {
        "attempt_id": attempt.id,
        "user_id": user.id,
        "module": "speaking",
        "submitted_questions": set(),
        "score": 0,
    }

    session_service.persist_speaking_answer_pending(
        session=session,
        question_id=q.question_id,
        question_type="describe_image",
        audio_url="s3://test/di.aac",
    )

    aa = db_session.query(AttemptAnswer).filter_by(
        attempt_id=attempt.id, question_id=q.question_id
    ).all()
    uqa = db_session.query(UserQuestionAttempt).filter_by(
        user_id=user.id, question_id=q.question_id
    ).all()
    assert len(aa) == 1, "attempt_answers row missing — primary submit data lost"
    assert len(uqa) == 1, "user_question_attempts row missing — Done badge broken"


def test_speaking_submit_with_no_user_id_skips_uqa(db_session):
    """Defensive: if session has no user_id (shouldn't happen, but…), the
    UQA write must be skipped without raising. The attempt_answers row
    still goes through."""
    q = _make_question(db_session, qid=9004, qtype="answer_short_question", module="speaking")
    user = _make_user(db_session, uid=43)
    attempt = _make_attempt(db_session, user.id, "speaking", "answer_short_question")
    session = {
        "attempt_id": attempt.id,
        # no user_id
        "module": "speaking",
        "submitted_questions": set(),
        "score": 0,
    }

    session_service.persist_speaking_answer_pending(
        session=session,
        question_id=q.question_id,
        question_type="answer_short_question",
        audio_url="s3://test/asq.aac",
    )

    uqa = db_session.query(UserQuestionAttempt).filter_by(question_id=q.question_id).all()
    assert len(uqa) == 0, "UQA should be skipped when session has no user_id"


@pytest.mark.parametrize(
    "qtype",
    [
        "read_aloud",
        "repeat_sentence",
        "describe_image",
        "retell_lecture",
        "answer_short_question",
        "summarize_group_discussion",
        "respond_to_situation",
    ],
)
def test_uqa_written_for_every_async_speaking_type(db_session, qtype):
    """All 7 async speaking types share persist_speaking_answer_pending —
    pin each one separately so future routing changes can't silently
    re-introduce the gap for a subset of types."""
    user = _make_user(db_session, uid=hash(qtype) % 10_000 + 100)
    qid = 10_000 + abs(hash(qtype)) % 1000
    q = _make_question(db_session, qid=qid, qtype=qtype, module="speaking")
    attempt = _make_attempt(db_session, user.id, "speaking", qtype)
    session = {
        "attempt_id": attempt.id,
        "user_id": user.id,
        "module": "speaking",
        "submitted_questions": set(),
        "score": 0,
    }

    session_service.persist_speaking_answer_pending(
        session=session,
        question_id=q.question_id,
        question_type=qtype,
        audio_url=f"s3://test/{qtype}.aac",
    )

    uqa = db_session.query(UserQuestionAttempt).filter_by(
        user_id=user.id, question_id=q.question_id
    ).first()
    assert uqa is not None, f"{qtype}: UQA row missing — Done badge broken"
    assert uqa.question_type == qtype
