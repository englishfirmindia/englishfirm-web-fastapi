"""
EnforceLimit — FastAPI dependency factory that gates an endpoint by tier.

Usage from a router:

    @router.post("/mock/start")
    def start_mock(
        ctx_and_usage = Depends(EnforceLimit("mocks", period="monthly")),
        db: Session = Depends(get_db),
    ):
        ctx, increment = ctx_and_usage
        # … create mock …

The dependency:
  1. Resolves the caller's SubscriptionContext (via get_subscription_context)
  2. Pulls the limit for `feature_key` from ctx.limits
  3. Calls usage_counter.try_increment atomically — if blocked, raises
     HTTP 402 with a structured payload the Flutter client can render as
     an upgrade prompt (plan_id, current/limit, feature_key, upgrade_url)
  4. On allow, returns (ctx, IncrementResult) so the route can log usage

Limit resolution table — maps the public feature_key to the
limits_json key on SubscriptionPlan:

    feature_key        →  limits_json key                 period
    practice           →  practice_per_day                daily
    sectionals         →  sectionals_per_month            monthly
    mocks              →  mocks_per_month                 monthly
    ef_coach_chat      →  ef_coach_per_day                daily
    trainer_feedback   →  trainer_feedback_per_month      monthly
    study_plan         →  study_plan_per_day              daily

The two namespaces stay separate so the wire/DB shape (`feature_key`) is
short and stable, while the plan-config JSON can rename freely.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Callable, Tuple

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from core.dependencies import get_subscription_context
from db.database import get_db
from services.billing.subscription_context import (
    SubscriptionContext,
    resolve_subscription_context,
)
from services.billing.usage_counter import IncrementResult, try_increment

log = logging.getLogger(__name__)


# (feature_key, period_type, plan_limit_field)
_FEATURE_MAP = {
    "practice":         ("daily",   "practice_per_day"),
    "sectionals":       ("monthly", "sectionals_per_month"),
    "mocks":            ("monthly", "mocks_per_month"),
    "ef_coach_chat":    ("daily",   "ef_coach_per_day"),
    "trainer_feedback": ("monthly", "trainer_feedback_per_month"),
    "study_plan":       ("daily",   "study_plan_per_day"),
    "sectional_score":  ("monthly", "sectional_score_per_month"),
    "mock_score":       ("monthly", "mock_score_per_month"),
}


def check_and_increment_or_raise(
    db,
    *,
    user_id: int,
    feature_key: str,
) -> Tuple[SubscriptionContext, IncrementResult]:
    """Inline-callable equivalent of EnforceLimit's body.

    Use this inside a route handler when the gate decision depends on
    request-time inputs (message content, payload type, …) that aren't
    available to a Depends. The EF Coach /chat path is the canonical
    example — it skips the gate for trivial greetings so a Free user's
    daily allowance isn't burned on a one-word "hi".

    Same behaviour as EnforceLimit: resolves the caller's tier, atomically
    increments the matching usage counter, raises 402 PLAN_LIMIT_REACHED
    when the cap is hit. Counter only ticks when the cap isn't reached.
    """
    if feature_key not in _FEATURE_MAP:
        raise ValueError(
            f"check_and_increment_or_raise: unknown feature_key={feature_key!r}"
        )
    period_type, plan_field = _FEATURE_MAP[feature_key]

    ctx = resolve_subscription_context(db, user_id)
    limit = ctx.limits.get(plan_field, 0)

    result = try_increment(
        db,
        user_id=ctx.user_id,
        feature_key=feature_key,
        period_type=period_type,
        limit=limit,
    )
    if not result.allowed:
        raise HTTPException(
            status_code=402,
            detail={
                "code":         "PLAN_LIMIT_REACHED",
                "feature_key":  feature_key,
                "plan_id":      ctx.plan_id,
                "limit":        result.limit,
                "current":      result.count_after,
                "period_type":  period_type,
                "period_start": result.period_start.isoformat(),
                "message": (
                    f"You've reached your {period_type} question limit. "
                    f"Upgrade to continue."
                ),
            },
        )

    log.info(
        "[enforce] inline allowed user_id=%d feature=%s plan=%s count=%d/%s",
        ctx.user_id, feature_key, ctx.plan_id,
        result.count_after,
        "∞" if result.limit is None else result.limit,
    )
    return ctx, result


# Sentinel returned when a gate is skipped (e.g. practice submit inside an
# exam). Same shape as a real IncrementResult so callers that pattern-match
# on it don't need to special-case; `allowed=True` and `limit=None` mean
# "not counted, no cap enforced here".
_SKIPPED_RESULT = IncrementResult(
    allowed=True, count_after=0, limit=None, period_start=date.today(),
)


async def _payload_session_id(request: Request) -> str | None:
    """Peek at the JSON body without consuming it downstream.

    Starlette caches `await request.body()` inside `request._body`, so
    the route handler's own `Body(...)` binding still deserialises fine
    after we've read the bytes here. Falls back to None on any parse /
    IO error — caller treats None as "not-in-exam" (i.e. apply the gate).
    """
    try:
        raw = await request.body()
        if not raw:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        sid = data.get("session_id")
        return sid if isinstance(sid, str) and sid else None
    except Exception:
        return None


def EnforceLimit(feature_key: str, *, skip_if_in_exam: bool = False) -> Callable:
    """Build a FastAPI dependency that gates a route by `feature_key`.

    Returns a callable suitable for `Depends(...)`. On allow returns the
    tuple `(SubscriptionContext, IncrementResult)` so the route handler
    can log the usage or surface remaining quota in the response.

    `skip_if_in_exam=True` — inspect the request body's `session_id`; if
    it belongs to a sectional or mock `PracticeAttempt`, bypass the
    counter entirely and return a synthetic "skipped" result. Fixes the
    "sectional consumes practice quota" double-gating bug where a free
    user's 3/day practice cap killed sectional/mock questions after Q3.
    Only meaningful on `feature_key="practice"` today, but the parameter
    is generic in case other gates want the same treatment.
    """
    if feature_key not in _FEATURE_MAP:
        raise ValueError(
            f"EnforceLimit: unknown feature_key={feature_key!r}. "
            f"Known: {sorted(_FEATURE_MAP)}"
        )
    period_type, plan_field = _FEATURE_MAP[feature_key]

    async def _dep(
        request: Request,
        ctx: SubscriptionContext = Depends(get_subscription_context),
        db: Session = Depends(get_db),
    ) -> Tuple[SubscriptionContext, IncrementResult]:
        # Exam-embedded submits are gated at exam START (either via
        # `enforce_free_sectional_or_paid`/`_mock_` for free, or
        # sectionals_per_month/mocks_per_month for paid). Ticking the
        # per-question counter as well means a 32-question sectional
        # consumes 32 practice slots — which caps free users at 3
        # scored questions per sectional. Skip if this submit's
        # session_id resolves to a sectional/mock attempt.
        if skip_if_in_exam:
            sid = await _payload_session_id(request)
            if sid:
                # Local import to avoid enforce_limit ↔ free_trial_gate cycle
                from services.billing.free_trial_gate import submit_is_inside_exam
                if submit_is_inside_exam(db, sid):
                    log.info(
                        "[enforce] SKIPPED user_id=%d feature=%s "
                        "reason=in-exam sid=%s",
                        ctx.user_id, feature_key, sid,
                    )
                    return ctx, _SKIPPED_RESULT

        # Pull limit from the plan's limits dict via the canonical field name.
        # Missing key = unknown to this plan = treat as 0 (blocked).
        limit = ctx.limits.get(plan_field, 0)

        result = try_increment(
            db,
            user_id=ctx.user_id,
            feature_key=feature_key,
            period_type=period_type,
            limit=limit,
        )

        if not result.allowed:
            # 402 Payment Required maps cleanly to "you need to upgrade".
            # Body is structured so the Flutter client renders the upgrade
            # modal without parsing prose.
            raise HTTPException(
                status_code=402,
                detail={
                    "code":         "PLAN_LIMIT_REACHED",
                    "feature_key":  feature_key,
                    "plan_id":      ctx.plan_id,
                    "limit":        result.limit,
                    "current":      result.count_after,
                    "period_type":  period_type,
                    "period_start": result.period_start.isoformat(),
                    "message": (
                        f"You've reached your {period_type} question limit. "
                        f"Upgrade to continue."
                    ),
                },
            )

        log.info(
            "[enforce] allowed user_id=%d feature=%s plan=%s count=%d/%s",
            ctx.user_id, feature_key, ctx.plan_id,
            result.count_after,
            "∞" if result.limit is None else result.limit,
        )
        return ctx, result

    return _dep
