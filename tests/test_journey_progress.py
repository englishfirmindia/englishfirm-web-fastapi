"""Regression tests for GET/PATCH /api/v1/user/journey-progress.

The endpoint feeds the v2 practice-hub milestone timeline + goal/date
info card. Contract pinned here:
  - Denominators are hardcoded constants (22 / 4 / 1).
  - Numerators are capped at their denominators.
  - PATCH validates goal ∈ {50, 65, 79} and rejects past dates.
  - PATCH persists to users.score_requirement + users.exam_date.
  - Omitted PATCH fields leave the respective column unchanged.
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import datetime as _dt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
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


from db.models import User, UserQuestionAttempt, PracticeAttempt
from db.database import get_db
from core.dependencies import get_current_user


@pytest.fixture
def client_and_user(monkeypatch):
    """Isolated FastAPI app with only /user/journey-progress mounted.
    Overrides get_current_user to return a fixed test user so JWT
    plumbing isn't in scope. SQLite backs users + attempts tables."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    User.__table__.create(engine)
    UserQuestionAttempt.__table__.create(engine)
    PracticeAttempt.__table__.create(engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    # Seed a single test user we authenticate as.
    sess = TestingSessionLocal()
    test_user = User(
        id=1,
        username="tester",
        email="tester@example.com",
        password_hash="x",
        phone="0400000000",
        score_requirement=None,
        exam_date=None,
    )
    sess.add(test_user)
    sess.commit()
    sess.close()

    def _override_db():
        s = TestingSessionLocal()
        try:
            yield s
        finally:
            s.close()

    def _override_current_user(db=_override_db):
        s = TestingSessionLocal()
        try:
            yield s.query(User).filter(User.id == 1).first()
        finally:
            s.close()

    from routers.user import router as user_router

    app = FastAPI()
    app.include_router(user_router, prefix="/api/v1")
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user

    yield TestClient(app), TestingSessionLocal
    engine.dispose()


# ── GET /journey-progress ────────────────────────────────────────────────

def test_get_default_zero_counts(client_and_user):
    """Brand-new user with no attempts → all completed = 0, totals fixed."""
    client, _ = client_and_user
    r = client.get("/api/v1/user/journey-progress")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["practice"]  == {"completed": 0, "total": 22}
    assert body["sectional"] == {"completed": 0, "total": 4}
    assert body["mock"]      == {"completed": 0, "total": 1}
    assert body["exam_goal"] is None
    assert body["exam_date"] is None


def test_get_caps_practice_at_22(client_and_user):
    """A power user with 30 distinct question types tried still shows 22/22."""
    client, SessionLocal = client_and_user
    sess = SessionLocal()
    for i in range(30):
        sess.add(UserQuestionAttempt(
            user_id=1,
            question_id=i,
            question_type=f"type_{i}",
            module="speaking",
        ))
    sess.commit()
    sess.close()
    r = client.get("/api/v1/user/journey-progress")
    assert r.status_code == 200
    assert r.json()["practice"] == {"completed": 22, "total": 22}


def test_get_caps_sectional_at_4_and_mock_at_1(client_and_user):
    """Insert enough sectional + mock completions to exceed the caps."""
    client, SessionLocal = client_and_user
    sess = SessionLocal()
    # Sectional: one per module, plus an extra duplicate (should still cap at 4).
    for mod in ("speaking", "writing", "reading", "listening", "speaking"):
        sess.add(PracticeAttempt(
            user_id=1, session_id=f"sec-{mod}",
            module=mod, question_type="mixed",
            filter_type="sectional",
            total_questions=10, total_score=70,
            questions_answered=10, status="complete",
        ))
    # Mock: two completed mock sessions (cap = 1)
    for i in range(2):
        sess.add(PracticeAttempt(
            user_id=1, session_id=f"mock-{i}",
            module="all", question_type="mixed",
            filter_type="mock",
            total_questions=100, total_score=800,
            questions_answered=100, status="complete",
        ))
    sess.commit()
    sess.close()
    body = client.get("/api/v1/user/journey-progress").json()
    assert body["sectional"] == {"completed": 4, "total": 4}
    assert body["mock"]      == {"completed": 1, "total": 1}


def test_get_returns_persisted_goal_and_date(client_and_user):
    client, SessionLocal = client_and_user
    sess = SessionLocal()
    u = sess.query(User).filter(User.id == 1).first()
    u.score_requirement = 79
    u.exam_date = _dt.date(2027, 1, 15)
    sess.commit()
    sess.close()
    body = client.get("/api/v1/user/journey-progress").json()
    assert body["exam_goal"] == 79
    assert body["exam_date"] == "2027-01-15"


# ── PATCH /journey-progress ──────────────────────────────────────────────

def test_patch_persists_goal_and_date(client_and_user):
    client, SessionLocal = client_and_user
    future = (_dt.date.today() + _dt.timedelta(days=30)).isoformat()
    r = client.patch("/api/v1/user/journey-progress",
        json={"exam_goal": 65, "exam_date": future})
    assert r.status_code == 204, r.text
    sess = SessionLocal()
    u = sess.query(User).filter(User.id == 1).first()
    assert u.score_requirement == 65
    assert u.exam_date.isoformat() == future
    sess.close()


def test_patch_rejects_invalid_goal(client_and_user):
    client, _ = client_and_user
    r = client.patch("/api/v1/user/journey-progress",
        json={"exam_goal": 55})
    assert r.status_code == 422, r.text


def test_patch_rejects_past_date(client_and_user):
    client, _ = client_and_user
    yesterday = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
    r = client.patch("/api/v1/user/journey-progress",
        json={"exam_date": yesterday})
    assert r.status_code == 422, r.text


def test_patch_null_fields_leave_columns_unchanged(client_and_user):
    """Sending only exam_goal should NOT wipe exam_date and vice versa."""
    client, SessionLocal = client_and_user
    # Seed initial values
    future = (_dt.date.today() + _dt.timedelta(days=60)).isoformat()
    r = client.patch("/api/v1/user/journey-progress",
        json={"exam_goal": 50, "exam_date": future})
    assert r.status_code == 204
    # Now PATCH only the goal — date must stay
    r = client.patch("/api/v1/user/journey-progress",
        json={"exam_goal": 79})
    assert r.status_code == 204
    sess = SessionLocal()
    u = sess.query(User).filter(User.id == 1).first()
    assert u.score_requirement == 79
    assert u.exam_date.isoformat() == future
    sess.close()


def test_patch_empty_body_is_noop(client_and_user):
    client, _ = client_and_user
    r = client.patch("/api/v1/user/journey-progress", json={})
    assert r.status_code == 204
