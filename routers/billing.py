"""
Stripe Checkout integration — the entry point users hit to actually pay.

Endpoints:
  POST /billing/checkout-session   — create a Stripe Checkout Session,
                                     return its URL for redirect
  POST /billing/portal-session     — create a Stripe Customer Portal
                                     session for self-serve cancel /
                                     payment-method update
  POST /billing/webhooks/stripe    — receive Stripe events; idempotent;
                                     drives all user_subscriptions /
                                     payment_transactions / subscription_events
                                     state changes

Design notes:
  - Subscriptions are activated EXCLUSIVELY by the webhook. The success
    redirect doesn't trust the URL — the client polls /subscription/me
    until it sees the new tier (Stripe webhook → DB → /me).
  - Idempotency: every event we process records its event_id; replaying
    the same event is safe.
  - We mirror current plan onto users.current_plan / plan_started_at /
    plan_end_at so legacy code paths and dashboards keep working.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy.orm import Session

import core.config as config
from core.dependencies import get_current_user
from db.database import get_db
from db.models import (
    PaymentTransaction,
    SubscriptionEvent,
    SubscriptionPlan,
    User,
    UserSubscription,
)
from services.billing.stripe_client import stripe_lib, verify_webhook_signature

log = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["Billing"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_configured() -> None:
    """Routes that need the SDK must short-circuit when keys are missing
    so we return a clean 503 instead of a stack trace from the SDK."""
    if not config.stripe_configured():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "STRIPE_NOT_CONFIGURED",
                "message": "Payments are not yet available — please try again later.",
            },
        )


def _resolve_price_id(plan: SubscriptionPlan, billing_period: str) -> str:
    """Map (plan, billing_period) → the Stripe Price ID, or raise 400.

    Price IDs live on subscription_plans (one per period). Populated by
    the operator in the Stripe Dashboard, mirrored into Postgres via
    scripts/init_subscription_tables.py or admin tooling. A NULL means
    "not yet configured for this plan+period combo" → the user can't
    buy it yet.
    """
    price_id = {
        "monthly":   plan.stripe_price_id_monthly,
        "quarterly": plan.stripe_price_id_quarterly,
        "annual":    plan.stripe_price_id_annual,
    }.get(billing_period)
    if not price_id:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "PRICE_NOT_CONFIGURED",
                "message": (
                    f"{plan.display_name} ({billing_period}) isn't available "
                    f"for purchase yet. Please pick another option."
                ),
            },
        )
    return price_id


def _find_or_create_active_subscription_row(
    db: Session,
    *,
    user_id: int,
    plan_id: str,
    billing_period: str,
    stripe_subscription_id: str,
    stripe_customer_id: Optional[str],
    current_period_start: datetime,
    current_period_end: datetime,
    cancel_at_period_end: bool,
    status: str,
) -> UserSubscription:
    """Upsert a UserSubscription row keyed by stripe subscription id.

    Re-running the same webhook event lands here a second time and just
    refreshes the period_end / status fields. Concurrent active rows
    for the same user are prevented by the partial unique index on
    (user_id) WHERE status IN ('active','past_due') — we mark any prior
    live row as 'cancelled' before inserting a new one (handles plan
    upgrades / downgrades cleanly).
    """
    # 1) Same Stripe subscription as before? Just sync.
    existing = (
        db.query(UserSubscription)
        .filter(UserSubscription.external_id == stripe_subscription_id)
        .first()
    )
    if existing:
        existing.plan_id = plan_id
        existing.billing_period = billing_period
        existing.status = status
        existing.current_period_start = current_period_start
        existing.current_period_end = current_period_end
        existing.cancel_at_period_end = cancel_at_period_end
        if stripe_customer_id:
            existing.stripe_customer_id = stripe_customer_id
        existing.updated_at = datetime.utcnow()
        db.flush()
        return existing

    # 2) New Stripe subscription — close any other live row for this user
    #    (upgrades / downgrades — Stripe issues a fresh subscription_id).
    prior_live = (
        db.query(UserSubscription)
        .filter(
            UserSubscription.user_id == user_id,
            UserSubscription.status.in_(("active", "past_due")),
        )
        .first()
    )
    if prior_live is not None:
        prior_live.status = "cancelled"
        prior_live.cancel_at_period_end = True
        prior_live.updated_at = datetime.utcnow()
        db.flush()

    row = UserSubscription(
        id=uuid.uuid4(),
        user_id=user_id,
        plan_id=plan_id,
        billing_period=billing_period,
        status=status,
        started_at=current_period_start,
        current_period_start=current_period_start,
        current_period_end=current_period_end,
        cancel_at_period_end=cancel_at_period_end,
        auto_renew=True,
        source="stripe",
        external_id=stripe_subscription_id,
        stripe_customer_id=stripe_customer_id,
    )
    db.add(row)
    db.flush()
    return row


def _log_event(
    db: Session,
    *,
    user_id: int,
    subscription_id: Optional[uuid.UUID],
    event_type: str,
    from_plan_id: Optional[str] = None,
    to_plan_id: Optional[str] = None,
    actor: str = "webhook",
    metadata: Optional[dict] = None,
) -> None:
    """Append-only audit row in subscription_events. Use for activated /
    renewed / cancelled / expired / refunded / payment_failed etc."""
    db.add(SubscriptionEvent(
        id=uuid.uuid4(),
        user_id=user_id,
        subscription_id=subscription_id,
        event_type=event_type,
        from_plan_id=from_plan_id,
        to_plan_id=to_plan_id,
        actor=actor,
        metadata_=metadata,
    ))


def _record_payment(
    db: Session,
    *,
    user_id: int,
    subscription_id: Optional[uuid.UUID],
    provider_transaction_id: str,
    provider_event_id: Optional[str],
    amount_cents: int,
    currency: str,
    status: str,
    raw_payload: dict,
) -> Optional[PaymentTransaction]:
    """Idempotent payment record. (provider, provider_transaction_id) is
    unique — replaying the same charge silently no-ops via an
    ON CONFLICT-style guard implemented by catching IntegrityError."""
    # Up-front check — cheaper than catching the exception.
    existing = (
        db.query(PaymentTransaction)
        .filter(
            PaymentTransaction.provider == "stripe",
            PaymentTransaction.provider_transaction_id == provider_transaction_id,
        )
        .first()
    )
    if existing is not None:
        # Allow status transitions (succeeded → refunded → disputed).
        if existing.status != status:
            existing.status = status
            existing.updated_at = datetime.utcnow()
        return existing

    row = PaymentTransaction(
        id=uuid.uuid4(),
        user_id=user_id,
        subscription_id=subscription_id,
        provider="stripe",
        provider_transaction_id=provider_transaction_id,
        provider_event_id=provider_event_id,
        amount_cents=amount_cents,
        currency=currency.upper(),
        status=status,
        raw_payload_jsonb=raw_payload,
    )
    db.add(row)
    db.flush()
    return row


def _mirror_to_user_cache(
    db: Session,
    *,
    user: User,
    plan_id: str,
    started_at: datetime,
    period_end: datetime,
) -> None:
    """Denormalise current plan onto users.current_plan/.plan_started_at/
    .plan_end_at. Hot reads (dashboards, legacy code) avoid a join."""
    user.current_plan = plan_id
    user.plan_started_at = started_at.date()
    user.plan_end_at = period_end.date()
    user.updated_at = datetime.utcnow()


def _epoch_to_utc(ts: Optional[int]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def _plan_id_from_price(db: Session, price_id: str) -> Optional[str]:
    """Reverse-lookup: Stripe price_id → our plan_id. Each plan has 3
    price IDs (monthly/quarterly/annual); any of them resolves to the
    same plan_id."""
    row = (
        db.query(SubscriptionPlan)
        .filter(
            (SubscriptionPlan.stripe_price_id_monthly == price_id) |
            (SubscriptionPlan.stripe_price_id_quarterly == price_id) |
            (SubscriptionPlan.stripe_price_id_annual == price_id)
        )
        .first()
    )
    return row.plan_id if row else None


def _billing_period_from_price(
    db: Session, price_id: str
) -> Optional[str]:
    row = (
        db.query(SubscriptionPlan)
        .filter(
            (SubscriptionPlan.stripe_price_id_monthly == price_id) |
            (SubscriptionPlan.stripe_price_id_quarterly == price_id) |
            (SubscriptionPlan.stripe_price_id_annual == price_id)
        )
        .first()
    )
    if not row:
        return None
    if row.stripe_price_id_monthly == price_id:   return "monthly"
    if row.stripe_price_id_quarterly == price_id: return "quarterly"
    if row.stripe_price_id_annual == price_id:    return "annual"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Routes — user-initiated
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/checkout-session")
def create_checkout_session(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Build a Stripe Checkout Session for the requested plan+period and
    return its hosted URL. Caller redirects the browser there.

    Request:
      {"plan_id": "gold", "billing_period": "monthly"}
    Response:
      {"checkout_url": "https://checkout.stripe.com/..."}
    """
    _require_configured()
    stripe = stripe_lib()

    plan_id = (payload.get("plan_id") or "").strip()
    billing_period = (payload.get("billing_period") or "monthly").strip()
    if plan_id in ("", "free", "vip"):
        # Free has nothing to charge; VIP is sales-led and not in Stripe.
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_PLAN", "message": "Plan is not purchasable via Checkout."},
        )
    if billing_period not in ("monthly", "quarterly", "annual"):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_BILLING_PERIOD", "message": "Unknown billing period."},
        )

    plan = db.get(SubscriptionPlan, plan_id)
    if plan is None or not plan.is_active:
        raise HTTPException(status_code=404, detail={"code": "PLAN_NOT_FOUND"})
    price_id = _resolve_price_id(plan, billing_period)

    # Re-use an existing Stripe customer if we have one — keeps a single
    # billing identity per user (payment-method reuse, single Portal,
    # consolidated history). Fallback to email-prefill if no customer
    # exists yet; Stripe creates one on first successful Checkout.
    existing_sub = (
        db.query(UserSubscription)
        .filter(
            UserSubscription.user_id == current_user.id,
            UserSubscription.stripe_customer_id.isnot(None),
        )
        .order_by(UserSubscription.started_at.desc())
        .first()
    )
    customer_id = existing_sub.stripe_customer_id if existing_sub else None

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=config.STRIPE_CHECKOUT_SUCCESS_URL,
            cancel_url=config.STRIPE_CHECKOUT_CANCEL_URL,
            # Either pass customer (preferred) OR customer_email — not both.
            **(
                {"customer": customer_id}
                if customer_id
                else {"customer_email": current_user.email}
            ),
            client_reference_id=str(current_user.id),
            metadata={
                "user_id": str(current_user.id),
                "plan_id": plan_id,
                "billing_period": billing_period,
            },
            subscription_data={
                "metadata": {
                    "user_id": str(current_user.id),
                    "plan_id": plan_id,
                    "billing_period": billing_period,
                },
            },
            allow_promotion_codes=True,
        )
    except Exception as exc:
        log.exception("[stripe] checkout.Session.create failed for user_id=%d", current_user.id)
        raise HTTPException(
            status_code=502,
            detail={"code": "STRIPE_API_ERROR", "message": str(exc)[:200]},
        )

    log.info(
        "[stripe] checkout session created user_id=%d plan=%s period=%s session_id=%s",
        current_user.id, plan_id, billing_period, session.id,
    )
    return {"checkout_url": session.url, "session_id": session.id}


@router.post("/portal-session")
def create_portal_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Stripe Customer Portal session for self-serve cancellation +
    payment-method updates. Requires the user to already have a
    Stripe customer record (i.e., they've checked out at least once)."""
    _require_configured()
    stripe = stripe_lib()

    sub = (
        db.query(UserSubscription)
        .filter(
            UserSubscription.user_id == current_user.id,
            UserSubscription.stripe_customer_id.isnot(None),
        )
        .order_by(UserSubscription.started_at.desc())
        .first()
    )
    if sub is None:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "NO_STRIPE_CUSTOMER",
                "message": "You don't have any active subscriptions to manage.",
            },
        )

    try:
        session = stripe.billing_portal.Session.create(
            customer=sub.stripe_customer_id,
            return_url=config.STRIPE_PORTAL_RETURN_URL,
        )
    except Exception as exc:
        log.exception("[stripe] portal create failed user_id=%d", current_user.id)
        raise HTTPException(
            status_code=502,
            detail={"code": "STRIPE_API_ERROR", "message": str(exc)[:200]},
        )

    return {"portal_url": session.url}


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Stripe webhook (no auth — signature verifies authenticity)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Receive + verify + dispatch Stripe events.

    Returns 200 unless the signature is invalid OR a fatal handler error
    occurs. Stripe will retry non-2xx with exponential backoff, so we
    DO return 5xx when our DB is unhealthy — the retry will eventually
    succeed. We return 200 (with a logged error) for unknown event
    types so Stripe doesn't keep retrying them forever.
    """
    _require_configured()

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = verify_webhook_signature(payload, sig)
    except ValueError:
        log.warning("[stripe] webhook bad payload")
        raise HTTPException(status_code=400, detail="bad payload")
    except Exception as exc:
        log.warning("[stripe] webhook signature verification failed: %s", exc)
        raise HTTPException(status_code=400, detail="bad signature")

    # Convert the Stripe `Event` (and its nested `data.object`) into plain
    # Python dicts. stripe-python ≥10 changed StripeObject so that .get()
    # no longer behaves like dict.get — instead it tries to look up an
    # attribute called "get" and raises AttributeError. Round-tripping
    # through to_dict() restores the dict API our handlers
    # were written against.
    event_dict = (
        event.to_dict()
        if hasattr(event, "to_dict")
        else dict(event)
    )

    event_type = event_dict["type"]
    event_id = event_dict["id"]
    log.info("[stripe] webhook received type=%s id=%s", event_type, event_id)

    handler = _EVENT_HANDLERS.get(event_type)
    if handler is None:
        # Unknown event — ack so Stripe doesn't retry indefinitely.
        return {"ok": True, "handled": False, "type": event_type}

    try:
        handler(db, event_dict)
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        log.exception("[stripe] webhook handler failed type=%s id=%s", event_type, event_id)
        # Surface a 500 so Stripe retries — usually a transient DB issue.
        raise HTTPException(status_code=500, detail="handler error")

    return {"ok": True, "handled": True, "type": event_type}


# ─────────────────────────────────────────────────────────────────────────────
# Webhook event handlers (one per supported event_type)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_checkout_session_completed(db: Session, event: dict) -> None:
    """User just completed Checkout. Pull the underlying subscription
    from Stripe to get period bounds, then upsert our row + activate."""
    session = event["data"]["object"]
    stripe_sub_id = session.get("subscription")
    if not stripe_sub_id:
        # One-off mode (we don't use this currently). Ignore.
        return
    user_id = int(session.get("client_reference_id") or session.get("metadata", {}).get("user_id") or 0)
    if user_id == 0:
        log.error("[stripe] checkout.session.completed without user_id session=%s", session.get("id"))
        return

    stripe = stripe_lib()
    sub = stripe.Subscription.retrieve(stripe_sub_id)
    _apply_subscription_state(
        db,
        user_id=user_id,
        stripe_subscription=sub,
        event_type="activated",
        event_id=event["id"],
    )


def _handle_invoice_payment_succeeded(db: Session, event: dict) -> None:
    """Renewal (or first invoice) succeeded. Record the payment + extend
    the period_end if the subscription rolled over."""
    invoice = event["data"]["object"]
    stripe_sub_id = invoice.get("subscription")
    user_id = int(invoice.get("metadata", {}).get("user_id") or 0)
    if user_id == 0 and stripe_sub_id:
        # Fall back to looking up the existing UserSubscription row.
        row = db.query(UserSubscription).filter(
            UserSubscription.external_id == stripe_sub_id
        ).first()
        if row:
            user_id = row.user_id
    if user_id == 0:
        log.error("[stripe] invoice.payment_succeeded without resolvable user_id invoice=%s", invoice.get("id"))
        return

    # Sync subscription state if available (renewal extends period_end).
    if stripe_sub_id:
        stripe = stripe_lib()
        sub = stripe.Subscription.retrieve(stripe_sub_id)
        _apply_subscription_state(
            db,
            user_id=user_id,
            stripe_subscription=sub,
            event_type="renewed",
            event_id=event["id"],
        )

    # Record the payment row regardless of subscription presence.
    sub_row = db.query(UserSubscription).filter(
        UserSubscription.external_id == stripe_sub_id
    ).first()
    _record_payment(
        db,
        user_id=user_id,
        subscription_id=sub_row.id if sub_row else None,
        provider_transaction_id=invoice.get("id"),
        provider_event_id=event["id"],
        amount_cents=int(invoice.get("amount_paid", 0)),
        currency=invoice.get("currency", "aud"),
        status="succeeded",
        raw_payload=invoice,
    )


def _handle_invoice_payment_failed(db: Session, event: dict) -> None:
    invoice = event["data"]["object"]
    stripe_sub_id = invoice.get("subscription")
    if not stripe_sub_id:
        return
    sub_row = db.query(UserSubscription).filter(
        UserSubscription.external_id == stripe_sub_id
    ).first()
    if sub_row is None:
        return
    sub_row.status = "past_due"
    sub_row.updated_at = datetime.utcnow()
    _log_event(
        db,
        user_id=sub_row.user_id,
        subscription_id=sub_row.id,
        event_type="payment_failed",
        to_plan_id=sub_row.plan_id,
        metadata={"invoice_id": invoice.get("id"), "stripe_event_id": event["id"]},
    )
    # Also record the failed attempt for accounting.
    _record_payment(
        db,
        user_id=sub_row.user_id,
        subscription_id=sub_row.id,
        provider_transaction_id=invoice.get("id"),
        provider_event_id=event["id"],
        amount_cents=int(invoice.get("amount_due", 0)),
        currency=invoice.get("currency", "aud"),
        status="failed",
        raw_payload=invoice,
    )


def _handle_subscription_updated(db: Session, event: dict) -> None:
    """Sync subscription state for cancel-at-period-end toggles, plan
    changes, and Stripe-initiated status transitions."""
    sub = event["data"]["object"]
    stripe_sub_id = sub["id"]
    sub_row = db.query(UserSubscription).filter(
        UserSubscription.external_id == stripe_sub_id
    ).first()
    user_id = sub_row.user_id if sub_row else int(sub.get("metadata", {}).get("user_id") or 0)
    if user_id == 0:
        log.error("[stripe] subscription.updated unresolved user_id sub=%s", stripe_sub_id)
        return
    _apply_subscription_state(
        db,
        user_id=user_id,
        stripe_subscription=sub,
        event_type="updated",
        event_id=event["id"],
    )


def _handle_subscription_deleted(db: Session, event: dict) -> None:
    sub = event["data"]["object"]
    stripe_sub_id = sub["id"]
    sub_row = db.query(UserSubscription).filter(
        UserSubscription.external_id == stripe_sub_id
    ).first()
    if sub_row is None:
        return
    sub_row.status = "expired"
    sub_row.updated_at = datetime.utcnow()
    _log_event(
        db,
        user_id=sub_row.user_id,
        subscription_id=sub_row.id,
        event_type="expired",
        from_plan_id=sub_row.plan_id,
        metadata={"stripe_event_id": event["id"]},
    )


def _handle_charge_refunded(db: Session, event: dict) -> None:
    """Mark the matching payment_transaction refunded + optionally the
    subscription. The /billing/refund-request endpoint (Week 5) creates
    the Stripe refund; this webhook lands when Stripe confirms."""
    charge = event["data"]["object"]
    charge_id = charge["id"]
    existing = db.query(PaymentTransaction).filter(
        PaymentTransaction.provider == "stripe",
        PaymentTransaction.provider_transaction_id == charge_id,
    ).first()
    if existing:
        existing.status = "refunded"
        existing.updated_at = datetime.utcnow()
        if existing.subscription_id:
            sub_row = db.query(UserSubscription).filter(
                UserSubscription.id == existing.subscription_id
            ).first()
            if sub_row:
                sub_row.status = "refunded"
                sub_row.updated_at = datetime.utcnow()
                _log_event(
                    db,
                    user_id=sub_row.user_id,
                    subscription_id=sub_row.id,
                    event_type="refunded",
                    from_plan_id=sub_row.plan_id,
                    metadata={"charge_id": charge_id, "stripe_event_id": event["id"]},
                )


def _apply_subscription_state(
    db: Session,
    *,
    user_id: int,
    stripe_subscription: dict,
    event_type: str,
    event_id: str,
) -> None:
    """Common path for activated / renewed / updated: derive plan_id +
    billing_period + period bounds from a Stripe subscription object
    and upsert our row + log the event + mirror to users."""
    # Defensive: callers may pass a stripe StripeObject (returned from
    # SDK retrieve() calls) instead of a plain dict. Normalise so the
    # .get() calls below behave like dict.get().
    if hasattr(stripe_subscription, "to_dict"):
        stripe_subscription = stripe_subscription.to_dict()
    items = stripe_subscription.get("items", {}).get("data") or []
    if not items:
        log.error("[stripe] subscription with no items: %s", stripe_subscription.get("id"))
        return
    item = items[0]
    price_id = item["price"]["id"]
    plan_id = _plan_id_from_price(db, price_id)
    billing_period = _billing_period_from_price(db, price_id)
    if plan_id is None or billing_period is None:
        log.error(
            "[stripe] unknown price_id=%s in subscription=%s — populate subscription_plans first",
            price_id, stripe_subscription.get("id"),
        )
        return

    # Stripe API v2024-09-30 moved current_period_start / current_period_end
    # from the top-level subscription object onto each subscription item,
    # since a subscription can now have multiple items billing on different
    # cycles. Prefer the item-level fields; fall back to top-level for
    # backwards compatibility with older API versions.
    period_start_ts = item.get("current_period_start") or stripe_subscription.get("current_period_start")
    period_end_ts   = item.get("current_period_end")   or stripe_subscription.get("current_period_end")
    period_start = _epoch_to_utc(period_start_ts)
    period_end   = _epoch_to_utc(period_end_ts)
    if period_start is None or period_end is None:
        log.error(
            "[stripe] subscription missing period bounds: %s item=%s",
            stripe_subscription.get("id"), item.get("id"),
        )
        return

    sub_row = _find_or_create_active_subscription_row(
        db,
        user_id=user_id,
        plan_id=plan_id,
        billing_period=billing_period,
        stripe_subscription_id=stripe_subscription["id"],
        stripe_customer_id=stripe_subscription.get("customer"),
        current_period_start=period_start,
        current_period_end=period_end,
        cancel_at_period_end=bool(stripe_subscription.get("cancel_at_period_end")),
        status="active" if stripe_subscription.get("status") in ("active", "trialing") else stripe_subscription.get("status", "active"),
    )

    _log_event(
        db,
        user_id=user_id,
        subscription_id=sub_row.id,
        event_type=event_type,
        to_plan_id=plan_id,
        metadata={"stripe_event_id": event_id, "stripe_subscription_id": stripe_subscription["id"]},
    )

    user = db.get(User, user_id)
    if user is not None:
        _mirror_to_user_cache(
            db,
            user=user,
            plan_id=plan_id,
            started_at=period_start,
            period_end=period_end,
        )


# Event dispatch table — only events we handle land here. Everything else
# is ack'd 200 in stripe_webhook() with handled=False.
_EVENT_HANDLERS = {
    "checkout.session.completed":  _handle_checkout_session_completed,
    "invoice.payment_succeeded":   _handle_invoice_payment_succeeded,
    "invoice.payment_failed":      _handle_invoice_payment_failed,
    "customer.subscription.updated": _handle_subscription_updated,
    "customer.subscription.deleted": _handle_subscription_deleted,
    "charge.refunded":             _handle_charge_refunded,
}
