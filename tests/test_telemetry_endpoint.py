"""Regression tests for the frontend telemetry endpoint (2026-05-30).

Background
----------
The /api/v1/telemetry/event endpoint shipped 2026-05-29 with hard auth
(`Depends(get_current_user)`). 100% of frontend telemetry POSTs were
401'd at the door — including the ones the system was specifically
built to capture (pre-login JS errors, expired-token bugs, sendBeacon
calls on tab close which cannot carry an Authorization header). Nothing
reached CloudWatch.

Fix: switch to a new `try_get_user` dependency that returns Optional[User]
(no 401 raise), and add a per-IP rate limit to prevent the now-public
surface from being abused for CloudWatch ingest spam.

These tests pin the contract:
  1. Anonymous POST → 200 (NOT 401), logs with user=0
  2. POST with valid token → 200, logs with the real user_id
  3. POST with invalid/expired token → 200 (treated as anonymous)
  4. Newlines in event data are stripped (log injection defense)
  5. Per-IP rate limit kicks in at 60/min, returns 429
  6. OTHER routes (using get_current_user) still 401 without auth —
     proves the soft-auth change is scoped to telemetry only
"""

from __future__ import annotations

import os
import importlib

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from jose import jwt
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


from db.models import User
from db.database import get_db
import core.config as config


@pytest.fixture
def app_client(monkeypatch):
    """Spin up a minimal FastAPI app with the telemetry router wired and
    an in-memory SQLite for the user table. Reset the rate-limit bucket
    between tests so the 60/min cap doesn't leak across cases."""
    # check_same_thread=False so the TestClient's request thread can
    # access the connection created on the test thread; StaticPool keeps
    # everyone on the same single in-memory database.
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    User.__table__.create(engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def _override_db():
        sess = TestingSessionLocal()
        try:
            yield sess
        finally:
            sess.close()

    # Seed one user for the authed-path test.
    sess = TestingSessionLocal()
    sess.add(User(
        id=42, username="test", email="t@ef.com", hashed_password="x",
    ))
    sess.commit()
    sess.close()

    # Import fresh so the in-memory rate-limit bucket starts empty.
    import routers.telemetry as telemetry_mod
    importlib.reload(telemetry_mod)
    telemetry_mod._rate_buckets.clear()

    # Also wire in a hard-auth route under the same app, so we can prove
    # other endpoints still 401 unauth'd.
    from core.dependencies import get_current_user

    app = FastAPI()
    app.include_router(telemetry_mod.router, prefix="/api/v1")

    @app.get("/api/v1/_protected")
    def _protected_route(user: User = Depends(get_current_user)):
        return {"user_id": user.id}

    app.dependency_overrides[get_db] = _override_db

    yield TestClient(app), TestingSessionLocal

    engine.dispose()


def _make_token(user_id: int) -> str:
    return jwt.encode(
        {"sub": str(user_id)},
        config.JWT_SECRET_KEY,
        algorithm=config.JWT_ALGORITHM,
    )


# ── 1. Anonymous POST accepted ────────────────────────────────────────────


def test_anonymous_post_returns_200(app_client, caplog):
    client, _ = app_client
    with caplog.at_level("INFO"):
        resp = client.post("/api/v1/telemetry/event", json={
            "client_session": "fe-abc123",
            "events": [{"type": "error", "data": {"msg": "boot crash"}}],
        })
    assert resp.status_code == 200, (
        f"Anonymous telemetry must be accepted (use case: pre-login errors). "
        f"Got {resp.status_code}: {resp.text}"
    )
    assert resp.json() == {"ok": True, "received": 1}
    # The log line carries user=0 for anonymous events.
    assert any("[FE_TELEMETRY]" in r.message and "user=0" in r.message for r in caplog.records), (
        "Log line must include `[FE_TELEMETRY]` and user=0 for anonymous events."
    )


# ── 2. Authed POST logs the real user_id ──────────────────────────────────


def test_authed_post_logs_real_user_id(app_client, caplog):
    client, _ = app_client
    token = _make_token(42)
    with caplog.at_level("INFO"):
        resp = client.post(
            "/api/v1/telemetry/event",
            json={"client_session": "fe-xyz", "events": [{"type": "route"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    assert any("user=42" in r.message for r in caplog.records), (
        "Authed telemetry must log with the real user_id for correlation."
    )


# ── 3. Invalid/expired token treated as anonymous ─────────────────────────


def test_invalid_token_treated_as_anonymous(app_client, caplog):
    client, _ = app_client
    with caplog.at_level("INFO"):
        resp = client.post(
            "/api/v1/telemetry/event",
            json={"client_session": "fe-?", "events": [{"type": "error"}]},
            headers={"Authorization": "Bearer not-a-real-jwt"},
        )
    assert resp.status_code == 200, (
        "An expired or invalid token must NOT 401 the telemetry endpoint — "
        "those are exactly the bugs telemetry is built to catch."
    )
    assert any("user=0" in r.message for r in caplog.records)


# ── 4. Log-injection defense: newlines stripped ───────────────────────────


def test_newlines_in_event_data_are_stripped(app_client, caplog):
    client, _ = app_client
    payload = {
        "client_session": "fe-sess",
        "events": [{
            "type": "error",
            "route": "/practice\n[FE_TELEMETRY] user=999 forged",
            "data": {"msg": "first line\nfake second line"},
        }],
    }
    with caplog.at_level("INFO"):
        resp = client.post("/api/v1/telemetry/event", json=payload)
    assert resp.status_code == 200
    for r in caplog.records:
        if "[FE_TELEMETRY]" in r.message:
            # The literal '\n' character (newline) must not appear in the
            # rendered log line; everything must collapse to spaces.
            assert "\n" not in r.message, (
                "Newlines in telemetry data leaked into the log line — an "
                "attacker could forge fake-looking log entries this way."
            )


# ── 5. Per-IP rate limit kicks in at 60/min ───────────────────────────────


def test_rate_limit_returns_429_after_60_in_a_minute(app_client):
    client, _ = app_client
    payload = {"client_session": "fe-rl", "events": [{"type": "perf"}]}
    # The defaults are 60 requests per 60s window. The 61st should 429.
    for i in range(60):
        resp = client.post("/api/v1/telemetry/event", json=payload)
        assert resp.status_code == 200, (
            f"Request #{i + 1} should still be inside the budget "
            f"(60/min). Got {resp.status_code}."
        )
    resp = client.post("/api/v1/telemetry/event", json=payload)
    assert resp.status_code == 429, (
        "Per-IP rate limit must kick in at 60/min to prevent CloudWatch "
        "ingest spam from an opportunistic attacker."
    )
    assert "rate limit" in resp.json().get("detail", "").lower()


# ── 6. Other routes (using get_current_user) still 401 without auth ───────


def test_other_routes_still_require_hard_auth(app_client):
    """The blast-radius guardrail. `try_get_user` is scoped to telemetry
    only — every OTHER route under the app continues to use
    `get_current_user` and 401s on missing/invalid auth."""
    client, _ = app_client
    resp = client.get("/api/v1/_protected")
    assert resp.status_code == 401, (
        "If this 200s, the soft-auth change has accidentally leaked to "
        "other routes. ONLY /api/v1/telemetry/event should accept "
        "anonymous requests."
    )
    resp = client.get(
        "/api/v1/_protected",
        headers={"Authorization": "Bearer junk"},
    )
    assert resp.status_code == 401
