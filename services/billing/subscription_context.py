"""
Resolves the active subscription for a user into a `SubscriptionContext`.

Single source of truth for "what tier is this user on, right now, and
what are they allowed to do" — consumed by:
  - GET /api/v1/subscription/me  (returns this to clients)
  - EnforceLimit decorator       (gates per-feature actions on the server)
  - Cron / admin paths           (grants, expiry sweeps, refund decisions)

Resolution order:
  1. Find the user's live row in `user_subscriptions`
     (status IN active/past_due, current_period_end > now)
  2. Join `subscription_plans` to get the catalogue + limits JSON
  3. If no live row exists → synthesize a Free context from the
     `free` plan row

Free is materialised as a real row in `subscription_plans` so the limits
live in one place. The synthetic context just doesn't have a
subscription_id / period_end (Free never expires).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import SubscriptionPlan, UserSubscription

log = logging.getLogger(__name__)

FREE_PLAN_ID = "free"

# Numeric keys inside `limits_json` that represent a per-period cap.
# Each is normalised onto the SubscriptionContext.limits dict so callers
# don't have to know the JSON shape. A value of None means unlimited.
_LIMIT_KEYS = (
    "practice_per_day",
    "sectionals_per_month",
    "mocks_per_month",
    "ef_coach_per_day",
    "trainer_feedback_per_month",
    "study_plan_per_day",
    "sectional_score_per_month",
    "mock_score_per_month",
)


@dataclass(frozen=True)
class SubscriptionContext:
    """Immutable snapshot of a user's billing state.

    `limits[key] is None` means unlimited. `key not in limits` means
    the feature isn't even known to this plan — treat as zero/blocked.
    `features` is a frozenset of boolean feature flags the plan owns.
    """
    user_id: int
    plan_id: str               # free | bronze | silver | gold | vip
    tier_rank: int             # 0..4
    display_name: str
    status: str                # active | past_due | (synthetic) free
    billing_period: Optional[str]   # None for Free
    period_end: Optional[datetime]  # None for Free
    cancel_at_period_end: bool
    auto_renew: bool
    subscription_id: Optional[str]  # None for Free
    limits: dict = field(default_factory=dict)        # {feature_key: int | None}
    features: frozenset = field(default_factory=frozenset)
    mock_review_days: Optional[int] = None  # None = lifetime
    source: Optional[str] = None            # stripe | manual_admin | None (free)

    # ---- Convenience accessors used by EnforceLimit / client code ----

    def is_paid(self) -> bool:
        return self.plan_id != FREE_PLAN_ID

    def has_feature(self, flag: str) -> bool:
        return flag in self.features

    def limit_for(self, feature_key: str) -> Optional[int]:
        """Return the numeric cap for a feature, or None for unlimited.
        Raises KeyError if the feature isn't part of this plan's schema —
        that's a programming error, not a runtime case."""
        return self.limits[feature_key]

    def is_unlimited(self, feature_key: str) -> bool:
        return self.limits.get(feature_key) is None and feature_key in self.limits


def _build_from_plan(
    *,
    user_id: int,
    plan: SubscriptionPlan,
    subscription: Optional[UserSubscription],
) -> SubscriptionContext:
    """Fold a SubscriptionPlan row + (optional) UserSubscription row into
    the immutable context dataclass. Centralises limits_json parsing so
    callers never touch the raw JSON shape."""
    raw = plan.limits_json or {}
    limits = {k: raw.get(k) for k in _LIMIT_KEYS}
    features = frozenset(raw.get("features") or ())
    mock_review_days = raw.get("mock_review_days")

    if subscription is None:
        return SubscriptionContext(
            user_id=user_id,
            plan_id=plan.plan_id,
            tier_rank=plan.tier_rank,
            display_name=plan.display_name,
            status="free",
            billing_period=None,
            period_end=None,
            cancel_at_period_end=False,
            auto_renew=False,
            subscription_id=None,
            limits=limits,
            features=features,
            mock_review_days=mock_review_days,
            source=None,
        )

    return SubscriptionContext(
        user_id=user_id,
        plan_id=plan.plan_id,
        tier_rank=plan.tier_rank,
        display_name=plan.display_name,
        status=subscription.status,
        billing_period=subscription.billing_period,
        period_end=subscription.current_period_end,
        cancel_at_period_end=subscription.cancel_at_period_end,
        auto_renew=subscription.auto_renew,
        subscription_id=str(subscription.id),
        limits=limits,
        features=features,
        mock_review_days=mock_review_days,
        source=subscription.source,
    )


def resolve_subscription_context(db: Session, user_id: int) -> SubscriptionContext:
    """Return the live SubscriptionContext for a user.

    Hot path — called on most authenticated requests. Two indexed queries
    in the worst case (paid user); one query for Free. Both hit
    primary-key / unique-index lookups.
    """
    # Hot query 1: find a live subscription row. Partial unique index
    # `ix_user_subscriptions_one_live` makes this a single index hit.
    live = (
        db.execute(
            select(UserSubscription)
            .where(
                UserSubscription.user_id == user_id,
                UserSubscription.status.in_(("active", "past_due")),
                UserSubscription.current_period_end > datetime.utcnow(),
            )
            .order_by(UserSubscription.started_at.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )

    if live is not None:
        plan = db.get(SubscriptionPlan, live.plan_id)
        if plan is None:
            # Plan row was deleted but a live subscription still references
            # it — should not happen in normal ops. Fall through to Free so
            # the user keeps access at minimum, and surface the inconsistency.
            log.error(
                "[subs] live subscription references missing plan_id=%s sub_id=%s user_id=%d — falling back to Free",
                live.plan_id, live.id, user_id,
            )
        else:
            return _build_from_plan(user_id=user_id, plan=plan, subscription=live)

    # Free path — synthesize from the free plan row. If even that row is
    # missing (seed not run), return a hard-coded zero-permission stub so
    # the API doesn't 500.
    free_plan = db.get(SubscriptionPlan, FREE_PLAN_ID)
    if free_plan is None:
        log.error(
            "[subs] free plan row missing — run scripts/init_subscription_tables.py. "
            "Returning empty stub for user_id=%d", user_id,
        )
        return SubscriptionContext(
            user_id=user_id,
            plan_id=FREE_PLAN_ID,
            tier_rank=0,
            display_name="Free",
            status="free",
            billing_period=None,
            period_end=None,
            cancel_at_period_end=False,
            auto_renew=False,
            subscription_id=None,
            limits={k: 0 for k in _LIMIT_KEYS},
            features=frozenset(),
            mock_review_days=7,
            source=None,
        )

    return _build_from_plan(user_id=user_id, plan=free_plan, subscription=None)
