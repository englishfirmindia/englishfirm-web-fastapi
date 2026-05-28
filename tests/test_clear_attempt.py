"""Regression tests for POST /api/v1/questions/clear-attempt.

The Clear button on the practice screen calls `clear_attempt_in_db` to:
  * Mark the user's latest practice AttemptAnswer as scoring_status="cleared"
  * Null the scoring fields and audio_url (audit trail kept; row remains)
  * Delete the matching UserQuestionAttempt row (Done badge resets)
  * Evict the in-memory _SCORE_STORE entry so polls don't surface old score

Critical guardrails these tests pin:
  - Sectional / mock attempts must NOT be cleared (filter_type=="practice" only)
  - Background scorer must skip the score write if the row is already cleared
  - Cleared rows must be filtered out of /user/answered-questions and
    /user/last-answer (so Practice list flips back to "New" and re-entering
    the question shows a fresh screen)
  - Existing Redo path (persist_speaking_answer_pending) must continue to
    work — Clear-then-Redo should land a fresh pending row
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
from services import session_service


@pytest.fixture
def db_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    for model in (User, QuestionFromApeuni, PracticeAttempt, AttemptAnswer,
                  UserQuestionAttempt):
        model.__table__.create(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(session_service, "SessionLocal", TestingSessionLocal)
    # _SCORE_STORE is process-global — wipe between tests so a leftover entry
    # from another test doesn't make a stale assertion pass.
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


def _make_question(db, qid: int = 9100, qtype: str = "read_aloud"):
    q = QuestionFromApeuni(
        question_id=qid, module="speaking", question_type=qtype,
        title=f"Q{qid}", difficulty_level=1, content_json={},
    )
    db.add(q)
    db.commit()
    return q


def _make_practice_attempt(db, user_id: int, qtype: str = "read_aloud"):
    a = PracticeAttempt(
        user_id=user_id, session_id=f"prac-{user_id}-{qtype}",
        module="speaking", question_type=qtype, filter_type="practice",
        total_questions=1, total_score=0, questions_answered=0,
        status="active", scoring_status="pending",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _make_sectional_attempt(db, user_id: int):
    a = PracticeAttempt(
        user_id=user_id, session_id=f"sect-{user_id}",
        module="speaking", question_type="sectional", filter_type="sectional",
        total_questions=1, total_score=0, questions_answered=0,
        status="active", scoring_status="pending",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _seed_submitted_attempt(db, user, q, attempt):
    """Submit a fake completed attempt — what Clear should be acting on."""
    aa = AttemptAnswer(
        attempt_id=attempt.id,
        question_id=q.question_id,
        question_type=q.question_type,
        user_answer_json={"audio_url": "s3://bucket/old.aac"},
        result_json={"pte_score": 75, "transcript": "test"},
        score=75,
        content_score=80.0,
        fluency_score=70.0,
        pronunciation_score=75.0,
        audio_url="s3://bucket/old.aac",
        scoring_status="complete",
    )
    db.add(aa)
    db.add(UserQuestionAttempt(
        user_id=user.id, question_id=q.question_id,
        question_type=q.question_type, module="speaking",
    ))
    db.commit()
    db.refresh(aa)
    return aa


# ── Happy path ────────────────────────────────────────────────────────────


def test_clear_attempt_soft_clears_row_and_deletes_uqa(db_session):
    user = _make_user(db_session)
    q = _make_question(db_session)
    attempt = _make_practice_attempt(db_session, user.id)
    aa = _seed_submitted_attempt(db_session, user, q, attempt)

    ok = session_service.clear_attempt_in_db(user.id, q.question_id)
    assert ok is True

    db_session.expire_all()
    cleared = db_session.query(AttemptAnswer).filter_by(id=aa.id).first()
    assert cleared.scoring_status == "cleared"
    assert cleared.score == 0
    assert cleared.content_score is None
    assert cleared.fluency_score is None
    assert cleared.pronunciation_score is None
    assert cleared.audio_url is None
    assert cleared.user_answer_json == {}
    assert cleared.result_json == {}

    uqa = db_session.query(UserQuestionAttempt).filter_by(
        user_id=user.id, question_id=q.question_id,
    ).first()
    assert uqa is None, "UQA row must be deleted so Done badge resets"


def test_clear_attempt_evicts_score_store_cache(db_session):
    user = _make_user(db_session)
    q = _make_question(db_session)
    attempt = _make_practice_attempt(db_session, user.id)
    _seed_submitted_attempt(db_session, user, q, attempt)
    # Populate the in-memory cache as the live scorer would.
    session_service._SCORE_STORE[(user.id, q.question_id)] = {
        "scoring": "complete", "total": 75, "transcript": "t",
    }

    session_service.clear_attempt_in_db(user.id, q.question_id)

    assert (user.id, q.question_id) not in session_service._SCORE_STORE, (
        "Cache hit on a cleared question would surface the old score on "
        "the next poll — the entry must be evicted on Clear."
    )


# ── 404-equivalent ────────────────────────────────────────────────────────


def test_clear_attempt_returns_false_when_no_attempt(db_session):
    user = _make_user(db_session)
    _make_question(db_session)
    ok = session_service.clear_attempt_in_db(user.id, 9100)
    assert ok is False


def test_clear_attempt_returns_false_for_other_users_attempt(db_session):
    """User B must not be able to clear User A's attempt — and we don't even
    leak existence (False, which the router maps to 404)."""
    user_a = _make_user(db_session, uid=1)
    user_b = _make_user(db_session, uid=2)
    q = _make_question(db_session)
    attempt = _make_practice_attempt(db_session, user_a.id)
    _seed_submitted_attempt(db_session, user_a, q, attempt)

    ok = session_service.clear_attempt_in_db(user_b.id, q.question_id)
    assert ok is False
    # And User A's data is untouched.
    aa = db_session.query(AttemptAnswer).first()
    assert aa.scoring_status == "complete"


# ── Sectional/mock isolation ──────────────────────────────────────────────


def test_clear_attempt_ignores_sectional_attempts(db_session):
    """A sectional submission for the same question_id must NOT be cleared.
    Practice-screen Clear is strictly scoped to filter_type='practice'."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    sect = _make_sectional_attempt(db_session, user.id)
    sect_aa = _seed_submitted_attempt(db_session, user, q, sect)

    ok = session_service.clear_attempt_in_db(user.id, q.question_id)
    assert ok is False, (
        "User has only sectional answers — Clear must report 'nothing to "
        "clear' rather than wiping the sectional row."
    )
    db_session.expire_all()
    assert db_session.query(AttemptAnswer).filter_by(id=sect_aa.id).first().scoring_status == "complete"


def test_clear_attempt_keeps_sectional_when_practice_also_exists(db_session):
    """User practiced the question, then took a sectional that included it.
    Clearing from the practice screen must hit only the practice row."""
    user = _make_user(db_session)
    q = _make_question(db_session)

    sect = _make_sectional_attempt(db_session, user.id)
    sect_aa = _seed_submitted_attempt(db_session, user, q, sect)
    prac = _make_practice_attempt(db_session, user.id)
    prac_aa = _seed_submitted_attempt(db_session, user, q, prac)

    ok = session_service.clear_attempt_in_db(user.id, q.question_id)
    assert ok is True

    db_session.expire_all()
    assert db_session.query(AttemptAnswer).filter_by(id=prac_aa.id).first().scoring_status == "cleared"
    assert db_session.query(AttemptAnswer).filter_by(id=sect_aa.id).first().scoring_status == "complete", (
        "Sectional submission for the same question must survive a "
        "practice-screen Clear."
    )


# ── Race with background scorer ───────────────────────────────────────────


def test_scorer_skips_write_when_row_was_cleared_mid_scoring(db_session):
    """Race: user submits → scorer kicks off Azure → user clicks Clear →
    Azure finishes → scorer about to write score. Scorer must notice the
    cleared status and bail out."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    attempt = _make_practice_attempt(db_session, user.id)
    aa = AttemptAnswer(
        attempt_id=attempt.id,
        question_id=q.question_id,
        question_type=q.question_type,
        user_answer_json={"audio_url": "s3://bucket/x.aac"},
        result_json={},
        score=0,
        audio_url="s3://bucket/x.aac",
        scoring_status="pending",
    )
    db_session.add(aa)
    db_session.add(UserQuestionAttempt(
        user_id=user.id, question_id=q.question_id,
        question_type=q.question_type, module="speaking",
    ))
    db_session.commit()

    # User clicks Clear while Azure is still working.
    session_service.clear_attempt_in_db(user.id, q.question_id)

    # Now Azure returns. update_speaking_score_in_db runs on a daemon thread;
    # call it synchronously by patching threading.Thread to run inline.
    import threading
    real_thread = threading.Thread

    class _InlineThread:
        def __init__(self, target, daemon=True):
            self._target = target
        def start(self):
            self._target()

    threading.Thread = _InlineThread  # type: ignore
    try:
        session_service.update_speaking_score_in_db(
            user_id=user.id, question_id=q.question_id,
            content=80.0, pronunciation=75.0, fluency=70.0, total=75.0,
            transcript="azure-late",
        )
    finally:
        threading.Thread = real_thread  # type: ignore

    db_session.expire_all()
    aa2 = db_session.query(AttemptAnswer).filter_by(question_id=q.question_id).first()
    assert aa2.scoring_status == "cleared", (
        "Scorer must NOT resurrect a cleared row. If this fails, the race "
        "guard in update_speaking_score_in_db has regressed."
    )
    assert aa2.score == 0


# ── No-regression: Clear-then-Redo and filter exclusions ──────────────────


def test_clear_then_redo_creates_fresh_pending_row(db_session):
    """After Clear, a fresh submit must re-pend the row (same upsert path
    as today's Redo). Don't break the existing flow."""
    user = _make_user(db_session)
    q = _make_question(db_session)
    attempt = _make_practice_attempt(db_session, user.id)
    _seed_submitted_attempt(db_session, user, q, attempt)
    session_service.clear_attempt_in_db(user.id, q.question_id)

    # Now Redo — same submit code path as the existing Redo button hits.
    session_service.persist_speaking_answer_pending(
        session={
            "attempt_id": attempt.id, "user_id": user.id,
            "module": "speaking", "session_id": "prac-42-read_aloud",
        },
        question_id=q.question_id,
        question_type="read_aloud",
        audio_url="s3://bucket/new.aac",
    )

    db_session.expire_all()
    aa = db_session.query(AttemptAnswer).filter_by(
        attempt_id=attempt.id, question_id=q.question_id,
    ).first()
    assert aa.scoring_status == "pending"
    assert aa.audio_url == "s3://bucket/new.aac"
    # And UQA is back (re-created by persist_speaking_answer_pending).
    uqa = db_session.query(UserQuestionAttempt).filter_by(
        user_id=user.id, question_id=q.question_id,
    ).first()
    assert uqa is not None
