"""Regression tests for the Google-Ads-only Zapier gate on /signup
(decision date 2026-06-16).

Background
----------
The Zapier webhook had been silently no-op'd for the entire life of the
feature because `ZAPIER_WEBHOOK_URL` was never wired into prod (see
commit a1ad3cb). After fixing the env, every signup started firing the
Zap. Product call: the Zap is wired to a paid-acquisition CRM workflow,
so it should ONLY fire when the signup originated from a Google Ads
click (`from_google_ads=True` set by the frontend after capturing
`?gclid=` on first landing). Organic / direct / social / unknown
signups must NOT trigger it.

These tests pin the gate at the router level:
  1. from_google_ads=True  → webhook task added (Zap fires)
  2. from_google_ads=False → webhook task NOT added (silent skip)
  3. from_google_ads omitted (defaults to False) → NOT added
  4. The user row is still created in all three cases — the gate only
     controls the Zap, not the signup itself.
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

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


from db.models import User
from db.database import get_db


@pytest.fixture
def client_and_calls(monkeypatch):
    """Spin up an isolated FastAPI app with the auth router + SQLite. Capture
    every call to `send_signup_webhook` and `_enrich_user_geoip` so tests can
    assert exactly what fired."""
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

    import routers.auth as auth_mod

    zap_calls: list = []
    def _capture_zap(**kwargs):
        zap_calls.append(kwargs)

    geoip_calls: list = []
    def _capture_geoip(user_id, ip):
        geoip_calls.append({"user_id": user_id, "ip": ip})

    monkeypatch.setattr(auth_mod, "send_signup_webhook", _capture_zap)
    monkeypatch.setattr(auth_mod, "_enrich_user_geoip", _capture_geoip)

    # The /signup route has @limiter.limit("5/minute"). Disabling the limiter
    # globally lets tests fire multiple signups in quick succession without
    # tripping 429. The limit is enforced in prod — tests only verify the
    # gate logic, not the rate limit.
    auth_mod.limiter.enabled = False

    app = FastAPI()
    # The router itself sets prefix="/auth" (see routers/auth.py:27), so
    # mount it under "/api/v1" — anything else double-prefixes.
    app.include_router(auth_mod.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = _override_db

    yield TestClient(app), TestingSessionLocal, zap_calls, geoip_calls
    engine.dispose()
    auth_mod.limiter.enabled = True  # restore for any later test that imports auth_mod


def _signup_payload(**overrides):
    base = {
        "username": "tester",
        "email": "tester@example.com",
        "password": "TestPass!12345",
        "phone": "0400000000",
    }
    base.update(overrides)
    return base


def test_from_google_ads_true_fires_zapier(client_and_calls):
    """The whole point of the gate — a Google-Ads-tagged signup MUST reach
    the Zap with the user's contact details."""
    client, _, zap_calls, _ = client_and_calls
    r = client.post(
        "/api/v1/auth/signup",
        json=_signup_payload(from_google_ads=True),
    )
    assert r.status_code == 201, r.text
    assert len(zap_calls) == 1, "Zap should fire exactly once for a Google-Ads signup"
    payload = zap_calls[0]
    assert payload["student_name"] == "tester"
    # Phone is normalised to E.164 by the SignupRequest validator
    # (2026-06-19): all of 0400000000 / 400000000 / +61400000000 collapse
    # to the same canonical +61... form before persistence.
    assert payload["phone_number"] == "+61400000000"
    # exam_date wasn't provided — must be None, not crashing the call
    assert payload["exam_date"] is None


def test_from_google_ads_false_does_not_fire_zapier(client_and_calls):
    """The defining behaviour change — organic signups must NOT reach the
    Zap. Pre-gate behaviour fired indiscriminately."""
    client, _, zap_calls, _ = client_and_calls
    r = client.post(
        "/api/v1/auth/signup",
        json=_signup_payload(from_google_ads=False),
    )
    assert r.status_code == 201
    assert zap_calls == [], "Zap must stay silent for non-Ads signups"


def test_from_google_ads_omitted_defaults_to_silent(client_and_calls):
    """Older frontends or third-party clients that don't send the field at
    all must default to silent — false-positives in the Zap are worse than
    false-negatives because the Zap drives a CRM action."""
    client, _, zap_calls, _ = client_and_calls
    payload = _signup_payload()
    payload.pop("from_google_ads", None)  # field genuinely missing
    r = client.post("/api/v1/auth/signup", json=payload)
    assert r.status_code == 201
    assert zap_calls == []


def test_user_row_still_persisted_regardless_of_gate(client_and_calls):
    """The gate controls the Zap only — every successful signup must still
    create a User row in the DB."""
    client, TestingSessionLocal, _, _ = client_and_calls
    # Three signups, three from_google_ads states
    cases = [
        ("a@x.com", True),
        ("b@x.com", False),
        ("c@x.com", None),  # omitted
    ]
    for email, flag in cases:
        body = _signup_payload(email=email, username=email.split("@")[0])
        if flag is not None:
            body["from_google_ads"] = flag
        r = client.post("/api/v1/auth/signup", json=body)
        assert r.status_code == 201, f"{email}: {r.text}"

    sess = TestingSessionLocal()
    try:
        rows = sess.query(User).all()
        emails = {u.email for u in rows}
        assert emails == {"a@x.com", "b@x.com", "c@x.com"}
        assert next(u.from_google_ads for u in rows if u.email == "a@x.com") is True
        assert next(u.from_google_ads for u in rows if u.email == "b@x.com") is False
        # Omitted defaults to False per SignupRequest schema
        assert next(u.from_google_ads for u in rows if u.email == "c@x.com") is False
    finally:
        sess.close()


def test_geoip_enrichment_still_runs_for_every_signup(client_and_calls):
    """GeoIP enrichment is independent of the Ads gate — it should fire on
    every signup regardless. Guards against accidentally wrapping both
    background tasks in the gate."""
    client, _, _, geoip_calls = client_and_calls
    # Organic signup
    r = client.post("/api/v1/auth/signup",
                    json=_signup_payload(email="organic@x.com",
                                          username="organic",
                                          from_google_ads=False))
    assert r.status_code == 201
    # Ads signup
    r = client.post("/api/v1/auth/signup",
                    json=_signup_payload(email="ads@x.com",
                                          username="ads",
                                          from_google_ads=True))
    assert r.status_code == 201
    assert len(geoip_calls) == 2, "GeoIP must run for both Ads and organic signups"
