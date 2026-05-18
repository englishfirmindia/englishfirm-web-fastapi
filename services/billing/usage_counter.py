"""
Atomic usage counters per (user, feature_key, period_start).

Hot path is `try_increment` — used by EnforceLimit before allowing a
gated action. It runs an atomic `INSERT ... ON CONFLICT DO UPDATE
... WHERE` so two concurrent requests at limit-1 cannot both pass.

Period semantics:
  - daily   features → period_start = today (UTC)
  - monthly features → period_start = first day of this month (UTC)

UTC is used server-side. A future enhancement may shift to user-local
timezone, but the reset window is a single day either way so the UX
impact is minor; not blocking Week 3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db.database import SessionLocal
from db.models import UsageCounter

log = logging.getLogger(__name__)


PeriodType = str  # 'daily' | 'monthly'


def period_start_for(period_type: PeriodType, now: Optional[datetime] = None) -> date:
    """Compute the period_start key for today's bucket."""
    now = now or datetime.utcnow()
    if period_type == "daily":
        return now.date()
    if period_type == "monthly":
        return date(now.year, now.month, 1)
    raise ValueError(f"unknown period_type: {period_type!r}")


@dataclass(frozen=True)
class IncrementResult:
    """Outcome of an attempted increment. `allowed=False` means the cap
    was already reached; `count_after` is the post-increment value when
    allowed, or the existing count when refused."""
    allowed: bool
    count_after: int
    limit: Optional[int]      # None = unlimited
    period_start: date


def try_increment(
    db: Session,  # accepted for API compat; an independent session is used internally
    *,
    user_id: int,
    feature_key: str,
    period_type: PeriodType,
    limit: Optional[int],
) -> IncrementResult:
    """Atomic check-and-increment, fully self-contained.

    Behaviour:
      - limit is None (unlimited) → ALWAYS increment, return allowed=True
      - limit is 0 (feature disabled for this plan) → never allowed
      - else                                       → increment iff count < limit

    Concurrency: the UPDATE has a `count < limit` predicate so two
    parallel requests can't both push it over. The race resolves
    deterministically; the second caller sees count_after=limit and
    allowed=False.

    Transaction lifecycle: opens its OWN `SessionLocal()` and commits the
    UPSERT inline. The original design left commit to the caller's request
    session, but in practice every gated route handler delegated work to
    helpers that opened their own sessions and never committed the request
    session — every counter write was rolled back at request close, leaving
    all per-day/per-month limits unenforced. Mirrors the same independent-
    session pattern used by `_create_practice_attempt_with_retry` in
    session_service. The `db` parameter is accepted but unused.
    """
    ps = period_start_for(period_type)

    if limit == 0:
        # Read-only "what's the current count" path so the response can
        # still surface a meaningful number. Uses its own session.
        s = SessionLocal()
        try:
            current = _read_count(s, user_id=user_id, feature_key=feature_key, period_start=ps)
        finally:
            s.close()
        return IncrementResult(allowed=False, count_after=current, limit=0, period_start=ps)

    s = SessionLocal()
    try:
        if limit is None:
            # Unlimited — still bump the counter for usage analytics, but never
            # block. Same UPSERT as the limited path.
            stmt = pg_insert(UsageCounter).values(
                user_id=user_id,
                feature_key=feature_key,
                period_start=ps,
                period_type=period_type,
                count=1,
            ).on_conflict_do_update(
                index_elements=[
                    UsageCounter.user_id,
                    UsageCounter.feature_key,
                    UsageCounter.period_start,
                ],
                set_={"count": UsageCounter.__table__.c.count + 1},
            ).returning(UsageCounter.count)

            count_after = s.execute(stmt).scalar_one()
            s.commit()
            return IncrementResult(allowed=True, count_after=count_after, limit=None, period_start=ps)

        # Capped path: UPSERT with a `WHERE count < :limit` predicate on the
        # UPDATE branch. Returning the count lets us tell allowed vs blocked
        # in a single round trip.
        stmt = pg_insert(UsageCounter).values(
            user_id=user_id,
            feature_key=feature_key,
            period_start=ps,
            period_type=period_type,
            count=1,
        ).on_conflict_do_update(
            index_elements=[
                UsageCounter.user_id,
                UsageCounter.feature_key,
                UsageCounter.period_start,
            ],
            set_={"count": UsageCounter.__table__.c.count + 1},
            where=UsageCounter.__table__.c.count < limit,
        ).returning(UsageCounter.count)

        row = s.execute(stmt).first()
        if row is not None:
            s.commit()
            return IncrementResult(
                allowed=True,
                count_after=row[0],
                limit=limit,
                period_start=ps,
            )

        # UPSERT predicate failed: the row exists at the cap. Read the
        # current count so the 402 response can surface it. No write, but
        # commit to release any locks the failed UPSERT may hold.
        current = _read_count(s, user_id=user_id, feature_key=feature_key, period_start=ps)
        s.commit()
        return IncrementResult(allowed=False, count_after=current, limit=limit, period_start=ps)
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def read_current_count(
    db: Session,
    *,
    user_id: int,
    feature_key: str,
    period_type: PeriodType,
) -> int:
    """Read-only — used by GET /subscription/usage so clients can show
    remaining quota before the user attempts an action."""
    return _read_count(
        db,
        user_id=user_id,
        feature_key=feature_key,
        period_start=period_start_for(period_type),
    )


def _read_count(
    db: Session,
    *,
    user_id: int,
    feature_key: str,
    period_start: date,
) -> int:
    row = db.execute(
        select(UsageCounter.count).where(
            UsageCounter.user_id == user_id,
            UsageCounter.feature_key == feature_key,
            UsageCounter.period_start == period_start,
        )
    ).scalar_one_or_none()
    return int(row) if row is not None else 0
