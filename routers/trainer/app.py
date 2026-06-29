"""
Trainer-facing data endpoints. Auth: trainer JWT (audience='trainer').

  GET    /trainer/shared                  list active shares for me
  GET    /trainer/shared/{share_id}       full review payload + notes
  POST   /trainer/notes                   create note
  PATCH  /trainer/notes/{note_id}         edit own note
  DELETE /trainer/notes/{note_id}         soft-delete own note
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Query
from pydantic import BaseModel, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import (
    PracticeAttempt,
    SubscriptionEvent,
    SubscriptionPlan,
    Trainer,
    TrainerNote,
    TrainerShare,
    User,
    UserSubscription,
)
from services.email import send_student_note_posted
from services.trainer_auth import get_current_trainer
from services.trainer_review import build_trainer_review_payload


router = APIRouter(prefix="/trainer", tags=["Trainer - App"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _module_title(module: str) -> str:
    return (module or "").replace("_", " ").title()


def _test_label(attempt: PracticeAttempt) -> str:
    qtype = (attempt.question_type or "").lower()
    module = (attempt.module or "").lower()
    if qtype == "mock" or module == "mock":
        tn = (attempt.task_breakdown or {}).get("test_number")
        if isinstance(tn, int) and 1 <= tn <= 40:
            return f"Mock {tn}"
        return "Mock"
    if qtype == "sectional":
        return f"{_module_title(module)} Sectional"
    return f"{_module_title(module)} {_module_title(qtype)}".strip()


def _require_my_active_share(
    db: Session, trainer: Trainer, share_id: int
) -> TrainerShare:
    share = db.query(TrainerShare).filter(TrainerShare.id == share_id).first()
    if share is None:
        raise HTTPException(status_code=404, detail="Share not found")
    if share.trainer_id != trainer.id:
        raise HTTPException(status_code=403, detail="Not your share")
    if share.revoked_at is not None:
        raise HTTPException(status_code=410, detail="Share revoked")
    return share


def _require_my_note(
    db: Session, trainer: Trainer, note_id: int
) -> TrainerNote:
    note = db.query(TrainerNote).filter(TrainerNote.id == note_id).first()
    if note is None or note.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Note not found")
    if note.trainer_id != trainer.id:
        raise HTTPException(status_code=403, detail="Not your note")
    return note


# ── Request models ────────────────────────────────────────────────────────────

class NoteCreateBody(BaseModel):
    share_id: int
    question_id: Optional[int] = None
    body: str
    rating: Optional[int] = None

    @field_validator("body")
    @classmethod
    def _check_body(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("body cannot be empty")
        if len(s) > 4000:
            raise ValueError("body too long (max 4000 chars)")
        return s

    @field_validator("rating")
    @classmethod
    def _check_rating(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1 or v > 5:
            raise ValueError("rating must be between 1 and 5")
        return v


class NotePatchBody(BaseModel):
    body: Optional[str] = None
    rating: Optional[int] = None

    @field_validator("body")
    @classmethod
    def _check_body(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("body cannot be empty")
        if len(s) > 4000:
            raise ValueError("body too long (max 4000 chars)")
        return s

    @field_validator("rating")
    @classmethod
    def _check_rating(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1 or v > 5:
            raise ValueError("rating must be between 1 and 5")
        return v


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/shared")
def list_shared(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    """Active shares for this trainer, newest first, paginated."""
    base_query = (
        db.query(TrainerShare, PracticeAttempt, User)
        .join(PracticeAttempt, TrainerShare.attempt_id == PracticeAttempt.id)
        .outerjoin(User, TrainerShare.student_user_id == User.id)
        .filter(
            TrainerShare.trainer_id == trainer.id,
            TrainerShare.revoked_at.is_(None),
        )
    )
    total = base_query.count()
    rows = (
        base_query.order_by(TrainerShare.shared_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    # Notes counts in one query
    share_ids = [s.id for s, _, _ in rows]
    notes_count_by_share = {}
    if share_ids:
        for share_id, cnt in (
            db.query(TrainerNote.share_id, func.count(TrainerNote.id))
            .filter(
                TrainerNote.share_id.in_(share_ids),
                TrainerNote.deleted_at.is_(None),
            )
            .group_by(TrainerNote.share_id)
            .all()
        ):
            notes_count_by_share[share_id] = cnt

    items = []
    for share, attempt, student in rows:
        items.append(
            {
                "share_id": share.id,
                "attempt_id": attempt.id,
                "test_label": _test_label(attempt),
                "module": attempt.module,
                "question_type": attempt.question_type,
                "total_score": attempt.total_score,
                "scoring_status": attempt.scoring_status,
                "completed_at": (
                    attempt.completed_at.isoformat() if attempt.completed_at else None
                ),
                "shared_at": share.shared_at.isoformat() if share.shared_at else None,
                "first_viewed_at": (
                    share.first_viewed_at.isoformat() if share.first_viewed_at else None
                ),
                "last_viewed_at": (
                    share.last_viewed_at.isoformat() if share.last_viewed_at else None
                ),
                "viewed": share.first_viewed_at is not None,
                "notes_count": notes_count_by_share.get(share.id, 0),
                "student_display_name": student.username if student else "[deleted]",
                "student_email": student.email if student else None,
            }
        )

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": page * limit < total,
    }


@router.get("/shared/{share_id}")
def get_shared(
    share_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    """Full review payload for one shared attempt + notes for this share."""
    share = _require_my_active_share(db, trainer, share_id)

    now = datetime.now(timezone.utc)
    if share.first_viewed_at is None:
        share.first_viewed_at = now
    share.last_viewed_at = now
    db.commit()

    return build_trainer_review_payload(db, share)


@router.post("/notes")
def create_note(
    background_tasks: BackgroundTasks,
    body: NoteCreateBody = Body(...),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    """Create a note on a shared attempt (or a specific question within it)."""
    share = _require_my_active_share(db, trainer, body.share_id)

    note = TrainerNote(
        share_id=share.id,
        attempt_id=share.attempt_id,
        question_id=body.question_id,
        trainer_id=trainer.id,
        body=body.body,
        rating=body.rating,
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    # Notify the student (best-effort, opt-in honored at provider level later)
    student = db.query(User).filter(User.id == share.student_user_id).first()
    attempt = db.query(PracticeAttempt).filter(PracticeAttempt.id == share.attempt_id).first()
    if student and attempt and student.email:
        background_tasks.add_task(
            send_student_note_posted,
            to=student.email,
            trainer_name=trainer.display_name,
            test_label=_test_label(attempt),
            student_name=student.username,
        )

    return {
        "note_id": note.id,
        "share_id": note.share_id,
        "attempt_id": note.attempt_id,
        "question_id": note.question_id,
        "body": note.body,
        "rating": note.rating,
        "created_at": note.created_at.isoformat() if note.created_at else None,
    }


@router.patch("/notes/{note_id}")
def update_note(
    note_id: int = Path(..., ge=1),
    body: NotePatchBody = Body(...),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    note = _require_my_note(db, trainer, note_id)

    if body.body is None and body.rating is None:
        raise HTTPException(status_code=422, detail="Nothing to update")

    if body.body is not None:
        note.body = body.body
    if body.rating is not None:
        note.rating = body.rating
    note.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(note)

    return {
        "note_id": note.id,
        "body": note.body,
        "rating": note.rating,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    }


@router.delete("/notes/{note_id}", status_code=204)
def delete_note(
    note_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    note = _require_my_note(db, trainer, note_id)
    note.deleted_at = datetime.now(timezone.utc)
    db.commit()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Trainer-granted VIP
# ─────────────────────────────────────────────────────────────────────────────
#
# Lets a trainer grant a class student a time-boxed VIP plan based on
# what tier that student paid for (out-of-band — to the trainer directly,
# not through our Stripe Checkout). Duration mirrors the billing period
# the trainer's class fee implies:
#
#   bronze paid → 30 days of VIP
#   silver paid → 90 days of VIP
#   gold   paid → 365 days of VIP
#
# Inserts a manual_admin user_subscription with status='active' and a
# computed period_end. Any prior live row (Stripe or otherwise) gets
# cancelled first to satisfy the partial unique index. An audit row in
# subscription_events records which trainer triggered the grant.
#
# Expiry is handled by the Week 5 daily cron — when period_end < now
# the row flips to 'expired' and the user falls back to Free.

_GRANT_DURATION_DAYS = {
    "bronze": 30,
    "silver": 90,
    "gold":   365,
}


class GrantVipRequest(BaseModel):
    student_email: str
    tier: str   # bronze | silver | gold
    # When true, also set the user's lifetime `unlimited_learn_access`
    # flag so Learning Resources stay accessible after the VIP period
    # ends. Default false → behaviour unchanged from prior callers.
    unlimited_learn: bool = False

    @field_validator("student_email")
    @classmethod
    def _normalise_email(cls, v: str) -> str:
        return (v or "").strip().lower()

    @field_validator("tier")
    @classmethod
    def _validate_tier(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in _GRANT_DURATION_DAYS:
            raise ValueError(
                f"tier must be one of {sorted(_GRANT_DURATION_DAYS)}; got {v!r}"
            )
        return v


@router.post("/grant-vip")
def grant_vip(
    body: GrantVipRequest = Body(...),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    """Grant a class student VIP for 30/90/365 days based on tier paid.

    Idempotent-ish: replaying with the same args inserts another VIP
    grant that supersedes the previous (period_end resets from now).
    """
    user = db.query(User).filter(
        func.lower(User.email) == body.student_email.lower()
    ).first()
    if user is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "STUDENT_NOT_FOUND",
                "message": (
                    f"{body.student_email} isn't registered yet. "
                    f"Ask them to sign up first."
                ),
            },
        )

    duration_days = _GRANT_DURATION_DAYS[body.tier]
    now = datetime.utcnow()
    period_end = now + timedelta(days=duration_days)

    # Cancel any current live row. Same pattern the Stripe webhook uses
    # (see routers/billing.py:_find_or_create_active_subscription_row).
    prior_live = (
        db.query(UserSubscription)
        .filter(
            UserSubscription.user_id == user.id,
            UserSubscription.status.in_(("active", "past_due")),
        )
        .first()
    )
    prior_plan_id = prior_live.plan_id if prior_live else None
    prior_source = prior_live.source if prior_live else None
    if prior_live is not None:
        prior_live.status = "cancelled"
        prior_live.cancel_at_period_end = True
        prior_live.updated_at = now
        db.flush()

    new_sub = UserSubscription(
        id=uuid.uuid4(),
        user_id=user.id,
        plan_id="vip",
        billing_period="trial",
        status="active",
        started_at=now,
        current_period_start=now,
        current_period_end=period_end,
        cancel_at_period_end=False,
        auto_renew=False,
        source="manual_admin",
    )
    db.add(new_sub)
    db.flush()

    # Mirror to the denormalised cache so dashboards / legacy reads see
    # VIP immediately without a join.
    user.current_plan = "vip"
    user.plan_started_at = now.date()
    user.plan_end_at = period_end.date()
    user.updated_at = now

    # Lifetime Learn add-on (optional). One-way write — never resets when
    # the VIP period ends. If a future grant turns the box off, we leave
    # the flag alone (additive only); flip via admin tooling if needed.
    if body.unlimited_learn:
        user.unlimited_learn_access = True

    db.add(SubscriptionEvent(
        id=uuid.uuid4(),
        user_id=user.id,
        subscription_id=new_sub.id,
        event_type="grant_admin",
        from_plan_id=prior_plan_id,
        to_plan_id="vip",
        actor=f"trainer:{trainer.id}",
        metadata_={
            "trainer_email": trainer.email,
            "paid_tier": body.tier,
            "duration_days": duration_days,
            "prior_source": prior_source,
            "unlimited_learn": body.unlimited_learn,
        },
    ))
    db.commit()

    return {
        "subscription_id": str(new_sub.id),
        "student_email": user.email,
        "student_username": user.username,
        "paid_tier": body.tier,
        "granted_plan": "vip",
        "duration_days": duration_days,
        "period_end": period_end.isoformat(),
        "prior_plan_id": prior_plan_id,
        "prior_source": prior_source,
        "warning_stripe_active": prior_source == "stripe",
        "unlimited_learn": bool(user.unlimited_learn_access),
    }


@router.get("/granted-vips")
def list_granted_vips(
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Recent VIP grants by THIS trainer — used to render the audit
    list under the grant form so the trainer doesn't double-grant."""
    actor_tag = f"trainer:{trainer.id}"
    events = (
        db.query(SubscriptionEvent, User)
        .join(User, User.id == SubscriptionEvent.user_id)
        .filter(
            SubscriptionEvent.actor == actor_tag,
            SubscriptionEvent.event_type == "grant_admin",
        )
        .order_by(SubscriptionEvent.created_at.desc())
        .limit(limit)
        .all()
    )

    rows = []
    for ev, user in events:
        meta = ev.metadata_ or {}
        # Look up the corresponding subscription row to surface the
        # current period_end (the grant could have been superseded).
        sub = (
            db.query(UserSubscription)
            .filter(UserSubscription.id == ev.subscription_id)
            .first()
            if ev.subscription_id else None
        )
        rows.append({
            "granted_at": ev.created_at.isoformat() if ev.created_at else None,
            "student_email": user.email,
            "student_username": user.username,
            "paid_tier": meta.get("paid_tier"),
            "duration_days": meta.get("duration_days"),
            "period_end": sub.current_period_end.isoformat() if sub else None,
            "current_status": sub.status if sub else None,
        })
    return {"grants": rows}


# ── Subscription summary ──────────────────────────────────────────────────────
#
# Trainer-facing rollup of who's on which plan. Powers the
# /trainer/subscriptions screen (subscription_summary_screen.dart).
#
# Endpoint: GET /trainer/subscriptions
#   ?check_stripe=true   live-sync flag: cross-checks DB stripe rows against
#                        Stripe's active-subscription list. Adds ~1–2s. Default
#                        false so first paint is snappy; UI exposes a "Check
#                        Stripe sync" button that re-fetches with the flag on.
#
# Response shape lets the screen render section headers (counts + MRR) and
# expandable per-user lists without further joins.

_PERIOD_TO_MONTHLY_FACTOR = {
    # MRR normalisation. trial / unknown periods don't contribute.
    "monthly":   1.0,
    "quarterly": 1.0 / 3.0,
    "annual":    1.0 / 12.0,
}


def _row_mrr_cents(plan: SubscriptionPlan, billing_period: str) -> int:
    """Monthly recurring revenue this row contributes, in AUD cents.

    Trial / manual / annual all collapse to a sensible per-month equivalent
    so the trainer-side MRR pill doesn't double-count annual prepayments."""
    factor = _PERIOD_TO_MONTHLY_FACTOR.get(billing_period or "")
    if not factor:
        return 0
    price = None
    if billing_period == "monthly":
        price = plan.monthly_price_aud_cents
    elif billing_period == "quarterly":
        price = plan.quarterly_price_aud_cents
    elif billing_period == "annual":
        price = plan.annual_price_aud_cents
    if not price:
        return 0
    return int(round(price * factor))


@router.get("/subscriptions")
def list_subscriptions(
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
    check_stripe: bool = Query(default=False),
):
    """Active-subscription rollup grouped by plan + optional Stripe-sync diff.

    The default response is DB-only (fast). Setting `check_stripe=true`
    additionally lists Stripe `active` subscriptions and reports two
    classes of drift between Stripe and our DB:

      - stripe_only_emails: paying on Stripe but no `source='stripe' AND
        status IN ('active','past_due')` row in user_subscriptions.
      - db_only_external_ids: DB claims a live Stripe row but the
        subscription is no longer active on Stripe (cancelled / unpaid).

    The grant-vip flow can leave a paying Stripe customer with a
    manual_admin VIP row on top — that's deliberate, so we surface it
    as `manual_overrides_stripe` rather than treating it as drift.
    """
    plans = (
        db.query(SubscriptionPlan)
        .order_by(SubscriptionPlan.tier_rank.asc())
        .all()
    )
    plans_by_id = {p.plan_id: p for p in plans}

    # ── Live rows (active + past_due) for the per-plan summary + lists ──
    live_rows = (
        db.query(UserSubscription, User)
        .join(User, User.id == UserSubscription.user_id)
        .filter(UserSubscription.status.in_(("active", "past_due")))
        .order_by(UserSubscription.started_at.desc())
        .all()
    )

    # Recently-cancelled count per plan (last 30 days) — useful churn pulse
    # in the section header without inflating the subscriber list itself.
    cutoff = datetime.utcnow() - timedelta(days=30)
    cancelled_recent_counts = dict(
        db.query(
            UserSubscription.plan_id,
            func.count(UserSubscription.id),
        )
        .filter(
            UserSubscription.status == "cancelled",
            UserSubscription.updated_at >= cutoff,
        )
        .group_by(UserSubscription.plan_id)
        .all()
    )

    # ── Aggregate per plan ──
    summary_by_plan: dict[str, dict] = {}
    subscribers_by_plan: dict[str, list[dict]] = {}
    for p in plans:
        summary_by_plan[p.plan_id] = {
            "plan_id": p.plan_id,
            "display_name": p.display_name,
            "tier_rank": p.tier_rank,
            "monthly_price_aud_cents": p.monthly_price_aud_cents,
            "active": 0,
            "trial": 0,
            "past_due": 0,
            "stripe_count": 0,
            "manual_count": 0,
            "mrr_cents": 0,
            "cancelled_last_30d": int(cancelled_recent_counts.get(p.plan_id, 0)),
        }
        subscribers_by_plan[p.plan_id] = []

    total_active = 0
    total_mrr_cents = 0
    for sub, user in live_rows:
        bucket = summary_by_plan.get(sub.plan_id)
        if bucket is None:
            # Defensive: a stale plan_id shouldn't crash the page.
            continue
        if sub.status == "active":
            bucket["active"] += 1
            total_active += 1
        elif sub.status == "past_due":
            bucket["past_due"] += 1
        if (sub.billing_period or "") == "trial":
            bucket["trial"] += 1
        if sub.source == "stripe":
            bucket["stripe_count"] += 1
        elif sub.source == "manual_admin":
            bucket["manual_count"] += 1

        plan_obj = plans_by_id.get(sub.plan_id)
        row_mrr = _row_mrr_cents(plan_obj, sub.billing_period) if plan_obj else 0
        bucket["mrr_cents"] += row_mrr
        total_mrr_cents += row_mrr

        subscribers_by_plan[sub.plan_id].append({
            "user_id": user.id,
            "email": user.email,
            "username": user.username,
            "plan_id": sub.plan_id,
            "billing_period": sub.billing_period,
            "status": sub.status,
            "source": sub.source,
            "started_at": sub.started_at.isoformat() if sub.started_at else None,
            "current_period_end": sub.current_period_end.isoformat() if sub.current_period_end else None,
            "cancel_at_period_end": bool(sub.cancel_at_period_end),
            "stripe_customer_id": sub.stripe_customer_id,
            "external_id": sub.external_id,
            "mrr_cents": row_mrr,
        })

    response: dict = {
        "summary": {
            "total_active": total_active,
            "total_mrr_cents": total_mrr_cents,
            "by_plan": [summary_by_plan[p.plan_id] for p in plans],
        },
        "subscribers_by_plan": subscribers_by_plan,
        "stripe_sync": None,
    }

    # ── Optional Stripe live-sync diff ──
    if not check_stripe:
        return response

    import core.config as config
    if not getattr(config, "STRIPE_SECRET_KEY", None):
        response["stripe_sync"] = {
            "checked": False,
            "reason": "stripe_not_configured",
            "stripe_only": [],
            "db_only": [],
            "manual_overrides_stripe": [],
        }
        return response

    try:
        from services.billing.stripe_client import stripe_lib
        stripe = stripe_lib()
        # Pull all active subscriptions (paginate via auto_paging_iter).
        # On a small base this is ~one API call; on a large one it auto-pages.
        stripe_subs = list(
            stripe.Subscription.list(status="active", limit=100, expand=["data.customer"]).auto_paging_iter()
        )
    except Exception as ex:
        response["stripe_sync"] = {
            "checked": False,
            "reason": f"stripe_error: {type(ex).__name__}",
            "stripe_only": [],
            "db_only": [],
            "manual_overrides_stripe": [],
        }
        return response

    # Build lookup: stripe_sub_id → email + customer_id (from Stripe).
    stripe_by_sub_id: dict[str, dict] = {}
    stripe_emails: set[str] = set()
    for s in stripe_subs:
        cust = s.get("customer") if isinstance(s, dict) else None
        # `customer` is expanded so it's a dict; raw API also accepts string id.
        if isinstance(cust, dict):
            email = (cust.get("email") or "").strip().lower()
            cust_id = cust.get("id")
        else:
            email = ""
            cust_id = cust if isinstance(cust, str) else None
        stripe_by_sub_id[s["id"]] = {
            "email": email,
            "customer_id": cust_id,
            "current_period_end": s.get("current_period_end"),
        }
        if email:
            stripe_emails.add(email)

    # DB rows with a stripe linkage that are currently live.
    db_stripe_live = (
        db.query(UserSubscription, User)
        .join(User, User.id == UserSubscription.user_id)
        .filter(
            UserSubscription.source == "stripe",
            UserSubscription.status.in_(("active", "past_due")),
        )
        .all()
    )
    db_stripe_sub_ids = {
        sub.external_id for sub, _u in db_stripe_live if sub.external_id
    }
    db_stripe_emails = {
        (u.email or "").strip().lower() for _sub, u in db_stripe_live if u.email
    }

    # ── 1. stripe_only: live on Stripe, no matching DB row ──
    stripe_only = []
    for sid, meta in stripe_by_sub_id.items():
        if sid in db_stripe_sub_ids:
            continue
        # Email match catches the common case where the DB doesn't yet hold
        # the external_id (webhook missed). Also catches rename mishaps.
        if meta["email"] and meta["email"] in db_stripe_emails:
            continue
        stripe_only.append({
            "stripe_subscription_id": sid,
            "email": meta["email"],
            "stripe_customer_id": meta["customer_id"],
            "current_period_end": meta["current_period_end"],
        })

    # ── 2. db_only: DB claims live Stripe sub but Stripe says it's gone ──
    db_only = []
    for sub, user in db_stripe_live:
        if sub.external_id and sub.external_id in stripe_by_sub_id:
            continue
        db_only.append({
            "user_id": user.id,
            "email": user.email,
            "username": user.username,
            "external_id": sub.external_id,
            "stripe_customer_id": sub.stripe_customer_id,
            "started_at": sub.started_at.isoformat() if sub.started_at else None,
        })

    # ── 3. manual_overrides_stripe: paying Stripe customer also has a live
    # manual_admin row sitting on top (typical: bronze paid → VIP grant).
    # Surface but don't flag as drift — this is deliberate by design.
    manual_overrides = []
    if stripe_emails:
        manual_rows = (
            db.query(UserSubscription, User)
            .join(User, User.id == UserSubscription.user_id)
            .filter(
                UserSubscription.source == "manual_admin",
                UserSubscription.status.in_(("active", "past_due")),
                func.lower(User.email).in_(stripe_emails),
            )
            .all()
        )
        for sub, user in manual_rows:
            manual_overrides.append({
                "user_id": user.id,
                "email": user.email,
                "username": user.username,
                "manual_plan_id": sub.plan_id,
                "manual_billing_period": sub.billing_period,
                "manual_period_end": sub.current_period_end.isoformat() if sub.current_period_end else None,
            })

    response["stripe_sync"] = {
        "checked": True,
        "stripe_active_count": len(stripe_by_sub_id),
        "db_live_stripe_count": len(db_stripe_live),
        "stripe_only": stripe_only,
        "db_only": db_only,
        "manual_overrides_stripe": manual_overrides,
    }
    return response
