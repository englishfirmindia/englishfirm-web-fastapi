"""Tests for the 2026-06-29 "stop auto-recharge on cards" policy:
  1. /billing/checkout-session always passes
     `subscription_data.cancel_at_period_end=True` so every new Stripe
     subscription ends after the first paid period.
  2. /trainer/grant-vip, when the prior live row is `source='stripe'`,
     calls `stripe.Subscription.modify(..., cancel_at_period_end=True)`
     so the customer's card stops being charged after the current
     period — and persists the result on the audit event.

Both are 1-line code paths but each guards against a real prod incident
(silent $145/mo of unintended bronze charges across 5 customers in
May/June 2026). These tests pin the behaviour.
"""
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import JSON


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ARG001
    return compiler.visit_JSON(JSON())


@compiles(ARRAY, "sqlite")
def _compile_array_sqlite(type_, compiler, **kw):  # noqa: ARG001
    return "TEXT"


@compiles(UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # noqa: ARG001
    return "VARCHAR(36)"


from main import app
from db.database import Base, get_db
from db.models import (
    SubscriptionEvent,
    SubscriptionPlan,
    Trainer,
    User,
    UserSubscription,
)
from core.dependencies import get_current_user
from services.trainer_auth import get_current_trainer


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def db_engine():
    """In-memory SQLite, only the tables the tested endpoints touch.
    Mirrors the partial-unique-index workaround from
    tests/test_trainer_subscriptions.py since SQLite can't honor
    `postgresql_where=` and would degrade to a full unique on user_id."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    us_tbl = UserSubscription.__table__
    detached_indexes = [
        ix for ix in list(us_tbl.indexes)
        if ix.name == "ix_user_subscriptions_one_live"
    ]
    for ix in detached_indexes:
        us_tbl.indexes.discard(ix)

    needed = [
        User.__table__,
        Trainer.__table__,
        SubscriptionPlan.__table__,
        us_tbl,
        SubscriptionEvent.__table__,
    ]
    Base.metadata.create_all(eng, tables=needed)
    yield eng
    Base.metadata.drop_all(eng, tables=needed)
    for ix in detached_indexes:
        us_tbl.indexes.add(ix)
    eng.dispose()


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    s = Session()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def seed_plans(db_session):
    """Bronze plan needed for both flows; include the Stripe price id so
    the checkout-session lookup path resolves."""
    db_session.add_all([
        SubscriptionPlan(
            plan_id="bronze", display_name="Bronze", tier_rank=1,
            monthly_price_aud_cents=2900,
            quarterly_price_aud_cents=7900,
            annual_price_aud_cents=24900,
            stripe_price_id_monthly="price_test_bronze_monthly",
            limits_json={}, is_active=True,
        ),
        SubscriptionPlan(
            plan_id="vip", display_name="VIP", tier_rank=4,
            monthly_price_aud_cents=None,
            quarterly_price_aud_cents=None,
            annual_price_aud_cents=None,
            limits_json={}, is_active=True,
        ),
    ])
    db_session.commit()


@pytest.fixture
def fake_trainer():
    t = MagicMock(spec=Trainer)
    t.id = 99
    t.email = "trainer@test.com"
    return t


@pytest.fixture
def fake_user():
    u = MagicMock(spec=User)
    u.id = 7
    u.email = "buyer@test.com"
    return u


# ── Model 1: /checkout-session sets cancel_at_period_end=True ─────────────

def test_checkout_session_passes_cancel_at_period_end_true(
    db_session, seed_plans, fake_user, monkeypatch
):
    """Every Stripe Checkout session our backend creates must include
    `subscription_data.cancel_at_period_end=True` so the resulting
    Subscription never auto-renews. Regression for the silent-recharge
    pattern fixed on 2026-06-29."""
    import core.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)
    monkeypatch.setattr(cfg_mod, "STRIPE_CHECKOUT_SUCCESS_URL", "https://app/success", raising=False)
    monkeypatch.setattr(cfg_mod, "STRIPE_CHECKOUT_CANCEL_URL", "https://app/cancel", raising=False)
    # /checkout-session calls `_require_configured()` which guards on
    # `config.stripe_configured()` — patch the helper to return True so
    # we don't have to wire every Stripe env var.
    monkeypatch.setattr(cfg_mod, "stripe_configured", lambda: True, raising=False)

    # Add the real user (FK target) — endpoint uses get_current_user mock,
    # but DB-level lookups still need a row.
    db_session.add(User(
        id=fake_user.id, username="buyer", email=fake_user.email,
        hashed_password="x", status="active",
    ))
    db_session.commit()

    # Capture the kwargs Stripe Checkout was called with.
    captured = {}
    fake_session = MagicMock()
    fake_session.id = "cs_test_123"
    fake_session.url = "https://stripe.com/cs_test_123"
    fake_stripe = MagicMock()
    def _create(**kw):
        captured.update(kw)
        return fake_session
    fake_stripe.checkout.Session.create = _create
    # routers/billing.py does `from services.billing.stripe_client import
    # stripe_lib` at module top — that binds the reference into
    # routers.billing.stripe_lib. Patch BOTH so module-local + late-import
    # call sites resolve to the fake.
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )
    monkeypatch.setattr(
        "routers.billing.stripe_lib", lambda: fake_stripe,
    )

    def override_db(): yield db_session
    def override_user(): return fake_user
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    try:
        client = TestClient(app)
        r = client.post(
            "/api/v1/billing/checkout-session",
            json={"plan_id": "bronze", "billing_period": "monthly"},
        )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    assert "subscription_data" in captured, "subscription_data must be passed"
    sd = captured["subscription_data"]
    # The critical assertion — every new sub auto-cancels after period 1.
    assert sd.get("cancel_at_period_end") is True, (
        f"checkout-session must set cancel_at_period_end=True (got {sd})"
    )
    # Metadata still preserved
    assert sd["metadata"]["user_id"] == "7"
    assert sd["metadata"]["plan_id"] == "bronze"


# ── Model 4: grant_vip cancels prior Stripe sub ────────────────────────────

def _seed_user_with_stripe_sub(db_session, *, email, stripe_sub_id):
    user = User(id=42, username="paying", email=email,
                hashed_password="x", status="active")
    db_session.add(user)
    sub = UserSubscription(
        id=uuid.uuid4(),
        user_id=42,
        plan_id="bronze",
        billing_period="monthly",
        status="active",
        started_at=datetime.utcnow() - timedelta(days=2),
        current_period_start=datetime.utcnow() - timedelta(days=2),
        current_period_end=datetime.utcnow() + timedelta(days=28),
        cancel_at_period_end=False,
        auto_renew=True,
        source="stripe",
        external_id=stripe_sub_id,
        stripe_customer_id="cus_paying",
    )
    db_session.add(sub)
    db_session.commit()


def test_grant_vip_cancels_prior_stripe_sub_at_period_end(
    db_session, seed_plans, fake_trainer, monkeypatch
):
    """When the trainer grants VIP to a student whose current row is a
    paying Stripe sub, we must call Stripe to set
    `cancel_at_period_end=True` so the customer's card stops being
    charged. This is the belt+suspenders fix for the 5-customer
    silent-recharge incident from May/June 2026."""
    _seed_user_with_stripe_sub(
        db_session, email="paying@x.com", stripe_sub_id="sub_paying_now",
    )

    modify_calls = []
    fake_stripe = MagicMock()
    def _modify(sub_id, **kw):
        modify_calls.append((sub_id, kw))
        return MagicMock(id=sub_id)
    fake_stripe.Subscription.modify = _modify
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    def override_db(): yield db_session
    def override_trainer(): return fake_trainer
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_trainer] = override_trainer
    try:
        client = TestClient(app)
        r = client.post(
            "/api/v1/trainer/grant-vip",
            json={"student_email": "paying@x.com", "tier": "bronze"},
        )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    body = r.json()
    # Stripe modify was called exactly once, with the prior external_id +
    # cancel_at_period_end=True.
    assert len(modify_calls) == 1, f"expected 1 Stripe modify call, got {len(modify_calls)}"
    sid, kw = modify_calls[0]
    assert sid == "sub_paying_now"
    assert kw.get("cancel_at_period_end") is True
    # Response surfaces the result so the trainer UI can confirm.
    assert body["stripe_cancelled_at_period_end"] is True
    assert body["stripe_cancel_error"] is None
    # Backwards-compat: warning_stripe_active is now False when we
    # successfully cancelled.
    assert body["warning_stripe_active"] is False

    # Audit row in subscription_events has the new fields.
    ev = db_session.query(SubscriptionEvent).filter_by(user_id=42).first()
    assert ev is not None
    md = ev.metadata_ or {}
    assert md.get("stripe_cancelled_at_period_end") is True
    assert md.get("prior_external_id") == "sub_paying_now"


def test_grant_vip_skips_stripe_call_for_manual_admin_prior(
    db_session, seed_plans, fake_trainer, monkeypatch
):
    """If the prior live row is `source='manual_admin'` (a previous VIP
    grant), Stripe has nothing to cancel — don't call the API. Otherwise
    re-granting VIP would spam Stripe with no-op modify calls."""
    user = User(id=42, username="manual", email="manual@x.com",
                hashed_password="x", status="active")
    db_session.add(user)
    db_session.add(UserSubscription(
        id=uuid.uuid4(), user_id=42, plan_id="vip", billing_period="trial",
        status="active",
        started_at=datetime.utcnow(),
        current_period_start=datetime.utcnow(),
        current_period_end=datetime.utcnow() + timedelta(days=30),
        cancel_at_period_end=False, auto_renew=False,
        source="manual_admin",
    ))
    db_session.commit()

    modify_calls = []
    fake_stripe = MagicMock()
    fake_stripe.Subscription.modify = lambda sid, **kw: modify_calls.append((sid, kw)) or MagicMock()
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    def override_db(): yield db_session
    def override_trainer(): return fake_trainer
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_trainer] = override_trainer
    try:
        client = TestClient(app)
        r = client.post(
            "/api/v1/trainer/grant-vip",
            json={"student_email": "manual@x.com", "tier": "bronze"},
        )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200
    assert modify_calls == [], (
        "Stripe modify must NOT be called when prior_source != 'stripe' "
        f"(got {modify_calls})"
    )
    body = r.json()
    assert body["stripe_cancelled_at_period_end"] is False
    assert body["stripe_cancel_error"] is None


def test_grant_vip_proceeds_even_when_stripe_modify_fails(
    db_session, seed_plans, fake_trainer, monkeypatch
):
    """Best-effort policy: if Stripe is down or returns an error, the
    trainer's grant must still complete (their intent is to give the
    student VIP access, not to depend on Stripe availability). The
    failure is surfaced in the audit row + response so the trainer can
    retry the Stripe cancel later via a separate tool."""
    _seed_user_with_stripe_sub(
        db_session, email="flaky@x.com", stripe_sub_id="sub_flaky_stripe",
    )

    def _boom(*_a, **_kw):
        raise RuntimeError("stripe unreachable")
    fake_stripe = MagicMock()
    fake_stripe.Subscription.modify = _boom
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    def override_db(): yield db_session
    def override_trainer(): return fake_trainer
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_trainer] = override_trainer
    try:
        client = TestClient(app)
        r = client.post(
            "/api/v1/trainer/grant-vip",
            json={"student_email": "flaky@x.com", "tier": "bronze"},
        )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    body = r.json()
    # Grant completed.
    assert body["granted_plan"] == "vip"
    # Stripe cancel did NOT succeed — flagged for follow-up.
    assert body["stripe_cancelled_at_period_end"] is False
    assert "stripe unreachable" in (body["stripe_cancel_error"] or "")
    # warning_stripe_active stays True so trainer UI shows the warning.
    assert body["warning_stripe_active"] is True
    # Local cancel still happened — invariant: at most one active row.
    active = db_session.query(UserSubscription).filter_by(
        user_id=42, status="active"
    ).all()
    assert len(active) == 1
    assert active[0].plan_id == "vip"
