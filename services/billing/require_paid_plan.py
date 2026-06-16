"""Gate a route behind a paid subscription tier — no usage counter.

EnforceLimit is the right tool when a feature has a per-period quota
(N attempts per day/month) — it talks to usage_counter.try_increment.
A handful of features just want "any paid plan can access this; free
cannot" with no counting at all. Learning Resources is the first such
case: PDFs/slides/study guides are unlimited-access-for-paid, fully
blocked for free, and never tick a counter.

Same 402 shape as EnforceLimit (`code='PLAN_LIMIT_REACHED'`,
`feature_key=...`) so the ApiClient's existing upgrade-sheet
interceptor on the Flutter side picks it up with no extra wiring.
"""
from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException

from core.dependencies import get_current_user, get_subscription_context
from db.models import User
from services.billing.subscription_context import SubscriptionContext


def RequirePaidPlan(feature_label: str) -> Callable:
    """Build a FastAPI dependency that 402s free-tier callers.

    `feature_label` is echoed back in the 402 payload as `feature_key` so
    the client can render a feature-specific upgrade prompt (e.g.
    "Learning resources are available on paid plans").
    """

    def dep(ctx: SubscriptionContext = Depends(get_subscription_context)) -> SubscriptionContext:
        if not ctx.is_paid():
            raise HTTPException(
                status_code=402,
                detail={
                    "code":        "PLAN_LIMIT_REACHED",
                    "feature_key": feature_label,
                    "plan_id":     ctx.plan_id,
                    "message":     f"{feature_label.replace('_', ' ').title()} is available on paid plans.",
                },
            )
        return ctx

    return dep


def RequireLearnAccess() -> Callable:
    """Gate Learning Resources behind EITHER an active paid plan OR the
    perpetual `users.unlimited_learn_access` flag (set by a trainer VIP
    grant with the lifetime-Learn checkbox on).

    The flag lives on `users`, not `user_subscriptions`, so it survives
    the VIP `current_period_end`. See db/models.py:User and the trainer
    grant route in routers/trainer/app.py for the write path.

    On reject: same 402 PLAN_LIMIT_REACHED shape as RequirePaidPlan so
    the Flutter ApiClient interceptor surfaces the upgrade sheet
    consistently.
    """

    def dep(
        user: User = Depends(get_current_user),
        ctx: SubscriptionContext = Depends(get_subscription_context),
    ) -> SubscriptionContext:
        if ctx.is_paid() or bool(getattr(user, "unlimited_learn_access", False)):
            return ctx
        raise HTTPException(
            status_code=402,
            detail={
                "code":        "PLAN_LIMIT_REACHED",
                "feature_key": "resources",
                "plan_id":     ctx.plan_id,
                "message":     "Learning Resources is available on paid plans.",
            },
        )

    return dep
