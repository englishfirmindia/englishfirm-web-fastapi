"""
Free-tier lifetime gates for sectional and mock exam access.

Replaces the per-month `sectionals_per_month` / `mocks_per_month` counter
model for FREE users. Semantics:

  - Free user starts their FIRST sectional  → users.free_sectional_used
                                              flips TRUE atomically, allowed
  - Free user starts a 2nd sectional        → raises 402 PLAN_LIMIT_REACHED
  - Free user starts their FIRST mock       → users.free_mock_used flips
                                              TRUE atomically, allowed
  - Free user starts a 2nd mock             → raises 402
  - Paid user                               → bypasses the flag entirely

Why atomic UPDATE (not SELECT-then-UPDATE):
  Two parallel /exam requests from the same free user (e.g. double-tap,
  or two open tabs) would both see flag=FALSE if we SELECT first. The
  `WHERE ... AND flag=FALSE RETURNING id` shape resolves the race
  deterministically — exactly one caller flips it, the other sees RETURNING
  empty and gets 402.

Rollback:
  If Option 1D is reverted, delete this file, restore
  `check_and_increment_or_raise(feature_key="sectionals" | "mocks")` on
  the exam-start routes, and re-add the `_score_gate` deps on the
  results routes. The two columns can stay in place — harmless.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.models import User
from services.billing.subscription_context import resolve_subscription_context

log = logging.getLogger(__name__)


_FLAG_COLUMN = {
    "sectional": "free_sectional_used",
    "mock":      "free_mock_used",
}

# Frontend paywall dialog keys on this feature string; keep in sync with
# the SectionalSubmitBloc / MockSubmitBloc 402 handlers.
_FEATURE_KEY = {
    "sectional": "sectional_lifetime",
    "mock":      "mock_lifetime",
}


def _atomic_flip(db: Session, *, user_id: int, column: str) -> bool:
    """Atomic `UPDATE users SET <column>=TRUE WHERE id=? AND <column>=FALSE
    RETURNING id`. Returns True iff this call was the one that flipped
    the flag (i.e. the caller is the first-time user). Otherwise the
    flag was already TRUE.

    Bind params by name — the column name is a fixed lookup, never taken
    from user input, so string-interpolation into the SQL is safe here.
    """
    if column not in ("free_sectional_used", "free_mock_used"):
        # Defence in depth — should be unreachable.
        raise ValueError(f"refusing atomic flip on unknown column={column!r}")
    result = db.execute(
        text(
            f"UPDATE users SET {column} = TRUE "
            f"WHERE id = :uid AND {column} = FALSE "
            f"RETURNING id"
        ),
        {"uid": user_id},
    )
    row = result.first()
    db.commit()
    return row is not None


# Legacy counter feature keys — used for PAID tiers where the plan JSON
# already carries the monthly quota (bronze:5, silver:10, gold/vip:null).
# Free tier still lives on the flag path.
_PAID_COUNTER_KEY = {
    "sectional": "sectionals",
    "mock":      "mocks",
}


def _enforce(
    db: Session,
    *,
    user: User,
    kind: Literal["sectional", "mock"],
) -> None:
    """Shared body — routes to the right gate based on plan tier.

    Paid users: fall through to the existing counter-based gate
    (`sectionals_per_month` / `mocks_per_month` on their plan JSON) so
    bronze:5 / silver:10 / gold+:unlimited semantics stay intact.

    Free users: use the lifetime flag — first attempt flips it TRUE and
    is allowed; any subsequent attempt raises 402 PLAN_LIMIT_REACHED with
    an upgrade CTA payload.
    """
    ctx = resolve_subscription_context(db, user.id)
    if ctx.is_paid():
        # Preserve existing paid-tier monthly counters. Import lazily to
        # avoid a circular between free_trial_gate ↔ enforce_limit.
        from services.billing.enforce_limit import check_and_increment_or_raise
        check_and_increment_or_raise(
            db, user_id=user.id, feature_key=_PAID_COUNTER_KEY[kind],
        )
        return

    column = _FLAG_COLUMN[kind]
    flipped = _atomic_flip(db, user_id=user.id, column=column)
    if flipped:
        log.info(
            "[free_trial_gate] first %s consumed user_id=%d — flag flipped",
            kind, user.id,
        )
        return

    # Flag already TRUE — free user has already consumed their one attempt.
    log.info(
        "[free_trial_gate] %s BLOCKED user_id=%d — free trial exhausted",
        kind, user.id,
    )
    raise HTTPException(
        status_code=402,
        detail={
            "code":         "PLAN_LIMIT_REACHED",
            "feature_key":  _FEATURE_KEY[kind],
            "plan_id":      ctx.plan_id,
            "limit":        1,
            "current":      1,
            "period_type":  "lifetime",
            "message": (
                f"You've used your free {kind}. Upgrade to Bronze for "
                f"unlimited {kind}s."
            ),
        },
    )


def enforce_free_sectional_or_paid(db: Session, user: User) -> None:
    """Call from POST /sectional/{module}/exam BEFORE starting the exam.
    Free user's first sectional: no-op (flag flips). Second: raises 402.
    Paid: no-op."""
    _enforce(db, user=user, kind="sectional")


def enforce_free_mock_or_paid(db: Session, user: User) -> None:
    """Call from POST /mock/start BEFORE starting the mock. Free user's
    first mock: no-op (flag flips). Second: raises 402. Paid: no-op."""
    _enforce(db, user=user, kind="mock")


def submit_is_inside_exam(db: Session, session_id: str | None) -> bool:
    """Return True iff the given practice_attempts.session_id belongs to a
    sectional or mock attempt (as opposed to ad-hoc practice).

    Used by `EnforceLimit("practice", skip_if_in_exam=True)` — when TRUE,
    the per-day practice counter is NOT ticked. Rationale: sectional/mock
    already gate at exam-start via the free_*_used flags (free tier) or
    the sectionals/mocks-per-month limits (paid tier). Ticking practice
    on every question inside them double-gates and kills usability
    (the Roneel bug — free user's 3/day practice cap exhausted by
    question 4 of a 32-question sectional).

    Falls back to False (i.e. treat as practice, apply gate) on any
    lookup failure so we never over-permit. `session_id=None` also → False.
    """
    if not session_id:
        return False
    try:
        row = db.execute(
            text(
                "SELECT filter_type FROM practice_attempts "
                "WHERE session_id = :sid LIMIT 1"
            ),
            {"sid": session_id},
        ).first()
    except Exception as exc:
        log.warning(
            "[free_trial_gate] submit_is_inside_exam lookup failed "
            "sid=%s: %s — treating as practice",
            session_id, exc,
        )
        return False
    if not row:
        return False
    return row[0] in ("sectional", "mock")
