"""
Subscription read-only API.

Two endpoints:
  - GET /subscription/me     — authenticated; returns the caller's tier,
                                limits, features, period_end. Used by the
                                Flutter SubscriptionBloc to gate UI.
  - GET /subscription/plans  — public; returns the plan catalogue for the
                                pricing page (display_name, price points,
                                limits, features). VIP returns null prices
                                so the page shows "Contact Trainer".

Mutations (checkout, cancel, refund) live in routers/billing.py (Week 4+).
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.dependencies import get_subscription_context
from db.database import get_db
from db.models import SubscriptionPlan
from services.billing.subscription_context import SubscriptionContext

router = APIRouter(prefix="/subscription", tags=["Subscription"])


@router.get("/me")
def get_my_subscription(
    ctx: SubscriptionContext = Depends(get_subscription_context),
):
    """Return the authenticated user's live subscription context.

    Response shape is the canonical wire format consumed by the Flutter
    SubscriptionBloc. Keep this stable — any breaking change forces a
    coordinated client release.
    """
    return {
        "plan_id":              ctx.plan_id,
        "display_name":         ctx.display_name,
        "tier_rank":            ctx.tier_rank,
        "status":               ctx.status,
        "billing_period":       ctx.billing_period,
        "period_end":           ctx.period_end.isoformat() if ctx.period_end else None,
        "cancel_at_period_end": ctx.cancel_at_period_end,
        "auto_renew":           ctx.auto_renew,
        "subscription_id":      ctx.subscription_id,
        "limits":               ctx.limits,           # {feature_key: int | None}
        "features":             sorted(ctx.features), # stable order for client diffing
        "mock_review_days":     ctx.mock_review_days,
        "source":               ctx.source,
    }


@router.get("/plans")
def list_plans(db: Session = Depends(get_db)) -> List[dict]:
    """Public plan catalogue for the pricing page.

    Returns active plans in tier_rank order. VIP returns null prices —
    the client renders "Contact Trainer" for null. Stripe Price IDs are
    intentionally NOT exposed in the public response; the checkout
    endpoint (Week 4) resolves them server-side from plan_id +
    billing_period.
    """
    rows = (
        db.query(SubscriptionPlan)
        .filter(SubscriptionPlan.is_active.is_(True))
        .order_by(SubscriptionPlan.tier_rank.asc())
        .all()
    )
    return [
        {
            "plan_id":      p.plan_id,
            "display_name": p.display_name,
            "tier_rank":    p.tier_rank,
            "pricing": {
                "monthly_aud_cents":   p.monthly_price_aud_cents,
                "quarterly_aud_cents": p.quarterly_price_aud_cents,
                "annual_aud_cents":    p.annual_price_aud_cents,
            },
            "limits_json": p.limits_json or {},
        }
        for p in rows
    ]
