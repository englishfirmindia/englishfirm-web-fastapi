"""
One-shot test seed: 5 users × 5 tiers.

Creates (or refreshes) five test accounts and grants each one a different
subscription tier so the gate can be exercised end-to-end without Stripe.

Idempotent:
  - User row: UPSERT by email — password is re-hashed every run so it's
    safe to re-run if you forget the password.
  - Subscription row: only inserted if the user has no active/past_due
    row already. Existing test grants are left alone.

Grants use source='manual_admin' (legitimate per CHECK constraint on
user_subscriptions.source) so they're distinct from real Stripe-sourced
rows in analytics. period_end is set to NOW + 30 days for all paid tiers.

Run from repo root with the venv active:
    python scripts/seed_test_users.py

DO NOT REPURPOSE for real account creation — these emails (a@/b@/c@/d@/e@)
are obvious test marks and should be excluded from real product analytics.
"""

import os
import sys
import uuid
from datetime import datetime, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"))

from passlib.context import CryptContext  # noqa: E402

from db.database import SessionLocal  # noqa: E402
from db import models  # noqa: F401, E402
from db.models import (  # noqa: E402
    SubscriptionEvent,
    SubscriptionPlan,
    User,
    UserSubscription,
)

# Same scheme as routers/auth.py — passwords are bcrypt-hashed so the
# /auth/login endpoint can verify them with _pwd.verify().
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


# (email, password, username, plan_id)
# Usernames intentionally do NOT include the tier — they get stale the
# moment the user upgrades / downgrades via Stripe, and the UI surfaces
# the live plan via the SubscriptionBloc snapshot, not the username.
TEST_USERS = [
    ("a@gmail.com", "pass1", "Test A", "free"),
    ("b@gmail.com", "pass2", "Test B", "bronze"),
    ("c@gmail.com", "pass3", "Test C", "silver"),
    ("d@gmail.com", "pass4", "Test D", "gold"),
    ("e@gmail.com", "pass4", "Test E", "vip"),
]


def _upsert_user(db, *, email: str, password: str, username: str) -> User:
    """Insert user or refresh password if already present."""
    existing = db.query(User).filter(User.email == email).first()
    hashed = _pwd.hash(password)
    if existing:
        existing.hashed_password = hashed
        existing.username = username
        existing.status = "active"
        db.commit()
        db.refresh(existing)
        print(f"  [user] refreshed id={existing.id} email={existing.email}")
        return existing

    u = User(
        username=username,
        email=email,
        hashed_password=hashed,
        status="active",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    print(f"  [user] inserted  id={u.id} email={u.email}")
    return u


def _grant_subscription(db, *, user: User, plan_id: str) -> None:
    """Create an active subscription if the user has none. Free tier
    intentionally gets no row — the resolver synthesises Free from
    subscription_plans when no live row exists."""
    if plan_id == "free":
        print(f"  [sub]  skip plan_id=free (synthesised by resolver) user_id={user.id}")
        return

    plan = db.get(SubscriptionPlan, plan_id)
    if plan is None:
        raise SystemExit(
            f"plan_id={plan_id!r} not found — run scripts/init_subscription_tables.py first"
        )

    # Singleton check matches the partial unique index on user_subscriptions.
    existing = (
        db.query(UserSubscription)
        .filter(
            UserSubscription.user_id == user.id,
            UserSubscription.status.in_(("active", "past_due")),
        )
        .first()
    )
    if existing:
        print(
            f"  [sub]  skip — already has live sub user_id={user.id} "
            f"plan_id={existing.plan_id} status={existing.status}"
        )
        return

    now = datetime.utcnow()
    sub = UserSubscription(
        id=uuid.uuid4(),
        user_id=user.id,
        plan_id=plan_id,
        billing_period="monthly",
        status="active",
        started_at=now,
        current_period_start=now,
        current_period_end=now + timedelta(days=30),
        cancel_at_period_end=False,
        auto_renew=False,                 # manual grant, not Stripe-driven
        source="manual_admin",
        external_id=None,
        stripe_customer_id=None,
    )
    db.add(sub)
    # Flush so the subscription row is visible to the FK check on the
    # subsequent SubscriptionEvent insert; both still commit atomically below.
    db.flush()

    # Audit-log the grant so subscription_events isn't empty either.
    db.add(SubscriptionEvent(
        id=uuid.uuid4(),
        user_id=user.id,
        subscription_id=sub.id,
        event_type="grant_admin",
        from_plan_id=None,
        to_plan_id=plan_id,
        actor="seed_test_users.py",
        metadata_={"reason": "test seed", "billing_period": "monthly"},
    ))
    db.commit()
    print(
        f"  [sub]  inserted id={sub.id} user_id={user.id} "
        f"plan_id={plan_id} period_end={sub.current_period_end:%Y-%m-%d}"
    )


def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise SystemExit("DATABASE_URL not set.")
    print("[seed] test users + tier grants")
    db = SessionLocal()
    try:
        for email, password, username, plan_id in TEST_USERS:
            print(f"\n--- {email} → {plan_id} ---")
            u = _upsert_user(db, email=email, password=password, username=username)
            _grant_subscription(db, user=u, plan_id=plan_id)
    finally:
        db.close()
    print("\n[seed] done")


if __name__ == "__main__":
    main()
