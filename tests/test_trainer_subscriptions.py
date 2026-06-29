"""Tests for GET /trainer/subscriptions — the rollup endpoint that powers
the trainer subscription_summary screen.

Strategy: SQLite-backed in-memory DB so we get real SQLAlchemy semantics
(GROUP BY, JOIN, IN-clause) without standing up Postgres. Trainer auth is
overridden with a Mock so each test only exercises the route logic.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
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
    SubscriptionPlan,
    Trainer,
    User,
    UserSubscription,
)
from services.trainer_auth import get_current_trainer


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db_engine():
    """In-memory SQLite, only the tables this endpoint touches.

    Avoids `Base.metadata.create_all` because unrelated tables in the
    project use Postgres-only types (INET on auth_refresh_tokens etc.)
    that SQLite can't render.

    Also strips the partial unique index on user_subscriptions
    (status IN ('active','past_due')) — SQLite doesn't honor the
    postgresql_where dialect clause and degrades it to a full unique
    on user_id, which blocks the legitimate (cancelled, active) pair
    that exists in prod after grant_vip supersedes a Stripe row.
    """
    from sqlalchemy.pool import StaticPool
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    us_tbl = UserSubscription.__table__
    # Temporarily detach the partial unique index so create_all skips it
    # in this SQLite test session. Restore after teardown so we don't
    # affect later test modules that share the same metadata.
    detached_indexes = [
        ix for ix in list(us_tbl.indexes)
        if ix.name == "ix_user_subscriptions_one_live"
    ]
    for ix in detached_indexes:
        us_tbl.indexes.discard(ix)

    needed = [
        User.__table__,
        SubscriptionPlan.__table__,
        us_tbl,
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
    """Mirror prod catalogue so MRR math has real prices to multiply against."""
    plans = [
        SubscriptionPlan(plan_id="free",   display_name="Free",   tier_rank=0,
                         monthly_price_aud_cents=0,
                         quarterly_price_aud_cents=0,
                         annual_price_aud_cents=0,
                         limits_json={}, is_active=True),
        SubscriptionPlan(plan_id="bronze", display_name="Bronze", tier_rank=1,
                         monthly_price_aud_cents=2900,
                         quarterly_price_aud_cents=7900,
                         annual_price_aud_cents=24900,
                         limits_json={}, is_active=True),
        SubscriptionPlan(plan_id="silver", display_name="Silver", tier_rank=2,
                         monthly_price_aud_cents=5900,
                         quarterly_price_aud_cents=15900,
                         annual_price_aud_cents=54900,
                         limits_json={}, is_active=True),
        SubscriptionPlan(plan_id="gold",   display_name="Gold",   tier_rank=3,
                         monthly_price_aud_cents=6900,
                         quarterly_price_aud_cents=18900,
                         annual_price_aud_cents=64900,
                         limits_json={}, is_active=True),
        SubscriptionPlan(plan_id="vip",    display_name="VIP",    tier_rank=4,
                         monthly_price_aud_cents=None,
                         quarterly_price_aud_cents=None,
                         annual_price_aud_cents=None,
                         limits_json={}, is_active=True),
    ]
    db_session.add_all(plans)
    db_session.commit()
    return plans


def _add_user(db, uid, email):
    u = User(id=uid, username=email.split("@")[0], email=email,
             hashed_password="x", status="active")
    db.add(u)
    return u


def _add_sub(db, user_id, plan_id, *, status="active", source="stripe",
             billing_period="monthly", started_days_ago=10,
             external_id=None, stripe_customer_id=None):
    now = datetime.utcnow()
    sub = UserSubscription(
        id=uuid.uuid4(),
        user_id=user_id,
        plan_id=plan_id,
        billing_period=billing_period,
        status=status,
        started_at=now - timedelta(days=started_days_ago),
        current_period_start=now - timedelta(days=started_days_ago),
        current_period_end=now + timedelta(days=20),
        cancel_at_period_end=False,
        auto_renew=True,
        source=source,
        external_id=external_id,
        stripe_customer_id=stripe_customer_id,
        updated_at=now,
    )
    db.add(sub)
    return sub


@pytest.fixture
def fake_trainer():
    t = MagicMock(spec=Trainer)
    t.id = 99
    t.email = "trainer@test.com"
    return t


@pytest.fixture
def client(db_session, fake_trainer):
    def override_db():
        yield db_session

    def override_trainer():
        return fake_trainer

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_trainer] = override_trainer
    yield TestClient(app)
    app.dependency_overrides.clear()


# ── Tests ────────────────────────────────────────────────────────────────────

def test_empty_db_returns_zero_counts(client, seed_plans):
    r = client.get("/api/v1/trainer/subscriptions")
    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["total_active"] == 0
    assert data["summary"]["total_mrr_cents"] == 0
    # All 5 plans listed even with zero subscribers
    by_plan = {p["plan_id"]: p for p in data["summary"]["by_plan"]}
    assert set(by_plan) == {"free", "bronze", "silver", "gold", "vip"}
    for p in by_plan.values():
        assert p["active"] == 0
        assert p["mrr_cents"] == 0
    # Per-plan subscriber lists all empty
    assert all(v == [] for v in data["subscribers_by_plan"].values())
    # Stripe sync block is None when not requested
    assert data["stripe_sync"] is None


def test_counts_and_mrr_per_plan(client, db_session, seed_plans):
    """1 gold monthly + 1 silver quarterly + 1 vip trial = correct counts + MRR."""
    _add_user(db_session, 1, "g@x.com")
    _add_user(db_session, 2, "s@x.com")
    _add_user(db_session, 3, "v@x.com")
    _add_sub(db_session, 1, "gold",   billing_period="monthly",   source="stripe",
             external_id="sub_g1")
    _add_sub(db_session, 2, "silver", billing_period="quarterly", source="stripe",
             external_id="sub_s1")
    _add_sub(db_session, 3, "vip",    billing_period="trial",     source="manual_admin")
    db_session.commit()

    r = client.get("/api/v1/trainer/subscriptions")
    assert r.status_code == 200
    data = r.json()
    by_plan = {p["plan_id"]: p for p in data["summary"]["by_plan"]}
    assert by_plan["gold"]["active"] == 1
    assert by_plan["silver"]["active"] == 1
    assert by_plan["vip"]["active"] == 1
    assert by_plan["vip"]["trial"] == 1
    assert by_plan["gold"]["stripe_count"] == 1
    assert by_plan["vip"]["manual_count"] == 1

    # MRR: gold monthly 6900 + silver quarterly 15900/3 = 5300 → 12200, VIP trial 0
    assert by_plan["gold"]["mrr_cents"] == 6900
    assert by_plan["silver"]["mrr_cents"] == 5300
    assert by_plan["vip"]["mrr_cents"] == 0
    assert data["summary"]["total_active"] == 3
    assert data["summary"]["total_mrr_cents"] == 6900 + 5300


def test_annual_period_amortises_correctly(client, db_session, seed_plans):
    _add_user(db_session, 1, "g@x.com")
    _add_sub(db_session, 1, "gold", billing_period="annual", source="stripe",
             external_id="sub_g_annual")
    db_session.commit()

    r = client.get("/api/v1/trainer/subscriptions")
    by_plan = {p["plan_id"]: p for p in r.json()["summary"]["by_plan"]}
    # 64900 / 12 = 5408.333... → 5408 rounded
    assert by_plan["gold"]["mrr_cents"] == 5408


def test_cancelled_rows_excluded_from_active(client, db_session, seed_plans):
    _add_user(db_session, 1, "g@x.com")
    _add_sub(db_session, 1, "gold", status="cancelled", source="stripe",
             external_id="sub_dead")
    db_session.commit()

    r = client.get("/api/v1/trainer/subscriptions")
    data = r.json()
    by_plan = {p["plan_id"]: p for p in data["summary"]["by_plan"]}
    assert by_plan["gold"]["active"] == 0
    # Recently-cancelled count should still pick it up (within 30 days).
    assert by_plan["gold"]["cancelled_last_30d"] == 1
    assert data["subscribers_by_plan"]["gold"] == []


def test_subscriber_lists_grouped_and_sorted_desc(client, db_session, seed_plans):
    _add_user(db_session, 1, "g_old@x.com")
    _add_user(db_session, 2, "g_new@x.com")
    _add_sub(db_session, 1, "gold", started_days_ago=30, external_id="sub_g_old")
    _add_sub(db_session, 2, "gold", started_days_ago=1,  external_id="sub_g_new")
    db_session.commit()

    r = client.get("/api/v1/trainer/subscriptions")
    rows = r.json()["subscribers_by_plan"]["gold"]
    # Newest first
    assert [row["email"] for row in rows] == ["g_new@x.com", "g_old@x.com"]


def test_stripe_sync_check_false_skips_call(client, db_session, seed_plans):
    """check_stripe omitted → stripe_sync stays None, no Stripe API call."""
    r = client.get("/api/v1/trainer/subscriptions")
    assert r.json()["stripe_sync"] is None


def test_stripe_sync_no_secret_returns_not_configured(client, db_session, seed_plans, monkeypatch):
    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", None, raising=False)
    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    assert r.status_code == 200
    sync = r.json()["stripe_sync"]
    assert sync["checked"] is False
    assert sync["reason"] == "stripe_not_configured"


def test_stripe_sync_true_drift_no_db_row_anywhere(client, db_session, seed_plans, monkeypatch):
    """Real webhook gap: Stripe says paying, DB has NO source='stripe' row
    for this user at any status. The user has only a manual VIP grant,
    never a Stripe row — so the webhook truly missed."""
    _add_user(db_session, 1, "paying@x.com")
    _add_sub(db_session, 1, "vip", source="manual_admin", billing_period="trial")
    db_session.commit()

    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)

    class _FakeIter:
        def __init__(self, items): self._items = items
        def auto_paging_iter(self): return iter(self._items)

    fake_subs = [{
        "id": "sub_live_1",
        "customer": {"id": "cus_X", "email": "paying@x.com"},
        "current_period_end": 9999999999,
    }]

    fake_stripe = MagicMock()
    fake_stripe.Subscription.list = lambda **_kw: _FakeIter(fake_subs)
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    sync = r.json()["stripe_sync"]
    assert sync["checked"] is True
    assert len(sync["stripe_only"]) == 1
    assert sync["stripe_only"][0]["email"] == "paying@x.com"
    overrides = sync["manual_overrides_stripe"]
    assert len(overrides) == 1
    assert overrides[0]["email"] == "paying@x.com"
    # NEW: even when stripe_only also flags this, the overrides row should
    # include the Stripe billing detail so the trainer can decide what to
    # do (cancel on Stripe or accept the double-billing).
    assert len(overrides[0]["stripe_subs"]) == 1
    assert overrides[0]["stripe_subs"][0]["stripe_subscription_id"] == "sub_live_1"
    assert overrides[0]["stripe_subs"][0]["stripe_current_period_end"] == 9999999999


def test_stripe_sync_cancelled_by_grant_vip_is_not_drift(client, db_session, seed_plans, monkeypatch):
    """The Sheetal/sharmanikita pattern, reproduced verbatim.

    Sequence:
      1. User subscribes bronze on Stripe → webhook creates source='stripe'
         row with status='active'.
      2. Trainer calls /grant-vip → that row gets flipped to 'cancelled'
         to satisfy the one-active-sub-per-user partial unique index, and
         a manual_admin VIP row is created on top.

    Result: DB has (cancelled bronze + active VIP), Stripe still lists
    bronze as active. This is intentional, NOT a webhook sync gap.
    The drift detector must:
      - NOT include the bronze sub in `stripe_only` (we have it in DB,
        just superseded)
      - INCLUDE the user in `manual_overrides_stripe` with the Stripe
        billing detail attached.
    Regression: 2026-06-29 — the original implementation only checked
    active+past_due rows, so the cancelled-by-upgrade rows reappeared as
    `stripe_only` drift and confused trainers ("why isn't this picking?").
    """
    _add_user(db_session, 1, "sheetal@x.com")
    # Cancelled bronze row from the original webhook sync.
    _add_sub(db_session, 1, "bronze", status="cancelled", source="stripe",
             external_id="sub_bronze_superseded",
             stripe_customer_id="cus_sheetal")
    # Active VIP grant that superseded it.
    _add_sub(db_session, 1, "vip", source="manual_admin", billing_period="trial")
    db_session.commit()

    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)

    class _FakeIter:
        def __init__(self, items): self._items = items
        def auto_paging_iter(self): return iter(self._items)

    fake_stripe = MagicMock()
    # Stripe still lists the bronze as active.
    fake_stripe.Subscription.list = lambda **_kw: _FakeIter([{
        "id": "sub_bronze_superseded",
        "customer": {"id": "cus_sheetal", "email": "sheetal@x.com"},
        "current_period_end": 9999999999,
    }])
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    sync = r.json()["stripe_sync"]
    assert sync["checked"] is True
    # NOT a webhook gap — we have the DB row (cancelled by VIP grant).
    assert sync["stripe_only"] == [], (
        f"sheetal's bronze was synced; cancelled-by-upgrade is not drift: {sync['stripe_only']}"
    )
    # IS a manual override with the still-charging Stripe detail attached.
    assert len(sync["manual_overrides_stripe"]) == 1
    ov = sync["manual_overrides_stripe"][0]
    assert ov["email"] == "sheetal@x.com"
    assert ov["manual_plan_id"] == "vip"
    assert len(ov["stripe_subs"]) == 1
    assert ov["stripe_subs"][0]["stripe_subscription_id"] == "sub_bronze_superseded"


def test_stripe_sync_reads_period_end_from_item_when_top_level_missing(client, db_session, seed_plans, monkeypatch):
    """Stripe API v2024-09-30+ moved current_period_end from the
    subscription root onto each subscription item. The drift endpoint
    must read item-level first and fall back to top-level so the
    renewal date surfaces on `manual_overrides_stripe.stripe_subs`
    regardless of which API version Stripe ships.
    Regression: 2026-06-29 — first deploy attached every override row
    with stripe_current_period_end=None because the new shape was
    silently swallowed by `.get('current_period_end')` on the root."""
    _add_user(db_session, 1, "u@x.com")
    _add_sub(db_session, 1, "bronze", status="cancelled", source="stripe",
             external_id="sub_item_shape", stripe_customer_id="cus_u")
    _add_sub(db_session, 1, "vip", source="manual_admin", billing_period="trial")
    db_session.commit()

    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)

    class _FakeIter:
        def __init__(self, items): self._items = items
        def auto_paging_iter(self): return iter(self._items)

    fake_stripe = MagicMock()
    # New-shape sub: no top-level current_period_end, only item-level.
    fake_stripe.Subscription.list = lambda **_kw: _FakeIter([{
        "id": "sub_item_shape",
        "customer": {"id": "cus_u", "email": "u@x.com"},
        "items": {"data": [{"current_period_end": 1740000000}]},
    }])
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    sync = r.json()["stripe_sync"]
    assert sync["checked"] is True
    overrides = sync["manual_overrides_stripe"]
    assert len(overrides) == 1
    subs = overrides[0]["stripe_subs"]
    assert len(subs) == 1
    assert subs[0]["stripe_current_period_end"] == 1740000000, (
        f"expected item-level period_end to surface; got {subs[0]}"
    )


def test_stripe_sync_email_match_recognises_renamed_external_id(client, db_session, seed_plans, monkeypatch):
    """If the DB stripe row has a different external_id than Stripe's
    current sub (e.g., a renewed subscription got a new id), the email
    match should still keep it out of `stripe_only`.

    Catches the case where Stripe issues sub_NEW after the user upgrades
    in-portal but our DB still tracks sub_OLD (cancelled). Email anchors
    the relationship even when the id drifts."""
    _add_user(db_session, 1, "renewed@x.com")
    _add_sub(db_session, 1, "bronze", status="cancelled", source="stripe",
             external_id="sub_OLD", stripe_customer_id="cus_r")
    db_session.commit()

    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)

    class _FakeIter:
        def __init__(self, items): self._items = items
        def auto_paging_iter(self): return iter(self._items)

    fake_stripe = MagicMock()
    fake_stripe.Subscription.list = lambda **_kw: _FakeIter([{
        "id": "sub_NEW",
        "customer": {"id": "cus_r", "email": "renewed@x.com"},
        "current_period_end": 9999999999,
    }])
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    sync = r.json()["stripe_sync"]
    # Email match keeps this out of stripe_only even though the id differs.
    assert sync["stripe_only"] == []


def test_stripe_sync_detects_db_only(client, db_session, seed_plans, monkeypatch):
    """DB claims active Stripe sub but Stripe doesn't list it → db_only entry."""
    _add_user(db_session, 1, "ghost@x.com")
    _add_sub(db_session, 1, "gold", source="stripe",
             external_id="sub_doesnt_exist_on_stripe",
             stripe_customer_id="cus_ghost")
    db_session.commit()

    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)

    class _FakeIter:
        def __init__(self, items): self._items = items
        def auto_paging_iter(self): return iter(self._items)

    fake_stripe = MagicMock()
    fake_stripe.Subscription.list = lambda **_kw: _FakeIter([])
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    sync = r.json()["stripe_sync"]
    assert sync["checked"] is True
    db_only = sync["db_only"]
    assert len(db_only) == 1
    assert db_only[0]["email"] == "ghost@x.com"
    assert db_only[0]["external_id"] == "sub_doesnt_exist_on_stripe"


def test_stripe_sync_clean_no_drift(client, db_session, seed_plans, monkeypatch):
    """DB external_id matches Stripe subscription id → no drift entries."""
    _add_user(db_session, 1, "matched@x.com")
    _add_sub(db_session, 1, "gold", source="stripe",
             external_id="sub_matched", stripe_customer_id="cus_matched")
    db_session.commit()

    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)

    class _FakeIter:
        def __init__(self, items): self._items = items
        def auto_paging_iter(self): return iter(self._items)

    fake_stripe = MagicMock()
    fake_stripe.Subscription.list = lambda **_kw: _FakeIter([{
        "id": "sub_matched",
        "customer": {"id": "cus_matched", "email": "matched@x.com"},
        "current_period_end": 9999999999,
    }])
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    sync = r.json()["stripe_sync"]
    assert sync["checked"] is True
    assert sync["stripe_only"] == []
    assert sync["db_only"] == []
    assert sync["manual_overrides_stripe"] == []
    assert sync["stripe_active_count"] == 1
    assert sync["db_live_stripe_count"] == 1


def test_stripe_sync_handles_stripe_object_without_get_method(client, db_session, seed_plans, monkeypatch):
    """Stripe SDK v10+ Subscription objects are NOT dict subclasses and have
    no `.get()` method — only subscript access. Endpoint must use subscript
    (with KeyError fallback) so a real Stripe response doesn't crash.
    Regression: 2026-06-29 prod-deploy 500 on first check_stripe=true call."""
    _add_user(db_session, 1, "ghost@x.com")
    _add_sub(db_session, 1, "gold", source="stripe",
             external_id="sub_dead_in_stripe", stripe_customer_id="cus_g")
    db_session.commit()

    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)

    class _NoGetObject:
        """Mimics stripe v10 StripeObject: subscript works, .get raises."""
        def __init__(self, **kw): self._d = dict(kw)
        def __getitem__(self, k): return self._d[k]
        def __getattr__(self, k):
            try: return self._d[k]
            except KeyError: raise AttributeError(k)

    customer_obj = _NoGetObject(id="cus_live", email="payer@x.com")
    sub_obj = _NoGetObject(id="sub_live_1", customer=customer_obj, current_period_end=9999999999)

    class _FakeIter:
        def __init__(self, items): self._items = items
        def auto_paging_iter(self): return iter(self._items)

    fake_stripe = MagicMock()
    fake_stripe.Subscription.list = lambda **_kw: _FakeIter([sub_obj])
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        lambda: fake_stripe,
    )

    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    assert r.status_code == 200, f"endpoint must not 500 on v10 StripeObject: {r.text[:300]}"
    sync = r.json()["stripe_sync"]
    assert sync["checked"] is True
    # The fake stripe sub doesn't match the DB row (sub_dead_in_stripe vs sub_live_1),
    # so both drift signals should fire.
    assert any(x["email"] == "payer@x.com" for x in sync["stripe_only"])
    assert any(x["email"] == "ghost@x.com" for x in sync["db_only"])


def test_stripe_sync_api_error_returns_gracefully(client, db_session, seed_plans, monkeypatch):
    import core.config as config
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_fake", raising=False)

    def boom():
        raise RuntimeError("stripe down")
    monkeypatch.setattr(
        "services.billing.stripe_client.stripe_lib",
        boom,
    )

    r = client.get("/api/v1/trainer/subscriptions?check_stripe=true")
    assert r.status_code == 200
    sync = r.json()["stripe_sync"]
    assert sync["checked"] is False
    assert sync["reason"].startswith("stripe_error:")
