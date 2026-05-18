"""
One-shot setup for the EF subscription system.

Idempotent. Run from the repo root with the venv active:
    python scripts/init_subscription_tables.py

Steps:
  1. Loads .env so DATABASE_URL is available
  2. Calls Base.metadata.create_all() — creates the 6 new subscription tables
     if missing (subscription_plans, user_subscriptions, usage_counters,
     payment_transactions, subscription_events, refund_requests)
  3. Upserts the 5 plan rows in PLAN_SEEDS (free / bronze / silver / gold / vip)

Re-running this script is safe: existing tables and plan rows are
updated in-place (price/limits fields refreshed, status preserved).
Stripe Price IDs are NOT touched once set — populate them once after
creating the Products + Prices in the Stripe Dashboard.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"))

from db.database import Base, engine, SessionLocal  # noqa: E402
from db import models  # noqa: F401, E402  — registers all models on Base
from db.models import SubscriptionPlan  # noqa: E402


# Limits convention:
#   - Numeric key with int value     → per-period cap (None = unlimited)
#   - "features" list                → boolean feature flags the plan owns
#   - "mock_review_days"             → null means lifetime retention
PLAN_SEEDS = [
    {
        "plan_id": "free",
        "display_name": "Free",
        "tier_rank": 0,
        "monthly_price_aud_cents":   0,
        "quarterly_price_aud_cents": 0,
        "annual_price_aud_cents":    0,
        "limits_json": {
            "practice_per_day":              3,
            "sectionals_per_month":          1,
            "mocks_per_month":               1,
            "ef_coach_per_day":              1,
            "trainer_feedback_per_month":    0,
            "study_plan_per_day":            0,
            "mock_review_days":              7,
            "sectional_score_per_month":     1,
            "mock_score_per_month":          0,
            "features": [
                "ai_score_reading_only",
                "one_video",
            ],
        },
    },
    {
        "plan_id": "bronze",
        "display_name": "Bronze",
        "tier_rank": 1,
        "monthly_price_aud_cents":    2900,
        "quarterly_price_aud_cents":  7900,
        "annual_price_aud_cents":    24900,
        "limits_json": {
            "practice_per_day":             20,
            "sectionals_per_month":          5,
            "mocks_per_month":               5,
            "ef_coach_per_day":              5,
            "trainer_feedback_per_month":    0,
            "study_plan_per_day":            0,
            "mock_review_days":             30,
            "sectional_score_per_month":     5,
            "mock_score_per_month":          5,
            "features": [
                "ai_score_all_types",
                "weekly_predictions",
                "core_templates",
                "core_videos",
            ],
        },
    },
    {
        "plan_id": "silver",
        "display_name": "Silver",
        "tier_rank": 2,
        "monthly_price_aud_cents":    5900,
        "quarterly_price_aud_cents": 15900,
        "annual_price_aud_cents":    54900,
        "limits_json": {
            "practice_per_day":             35,
            "sectionals_per_month":         10,
            "mocks_per_month":              10,
            "ef_coach_per_day":              8,
            "trainer_feedback_per_month":    0,
            "study_plan_per_day":            0,
            "mock_review_days":             30,
            "sectional_score_per_month":    10,
            "mock_score_per_month":         10,
            "features": [
                "ai_score_all_types",
                "weekly_predictions",
                "full_templates",
                "full_videos",
                "mobile_app",
                "priority_email_support",
            ],
        },
    },
    {
        "plan_id": "gold",
        "display_name": "Gold",
        "tier_rank": 3,
        "monthly_price_aud_cents":    6900,
        "quarterly_price_aud_cents": 18900,
        "annual_price_aud_cents":    64900,
        "limits_json": {
            "practice_per_day":             None,   # unlimited
            "sectionals_per_month":         None,
            "mocks_per_month":              None,
            "ef_coach_per_day":               20,
            "trainer_feedback_per_month":      4,
            "study_plan_per_day":              1,
            "mock_review_days":             None,   # lifetime
            "sectional_score_per_month":    None,
            "mock_score_per_month":         None,
            "features": [
                "ai_score_all_types",
                "skill_gap_analysis",
                "weekly_predictions_alerts",
                "full_templates",
                "scored_breakdown_notes",
                "full_videos",
                "premium_workshops",
                "mobile_app",
                "priority_chat_escalation",
                "satisfaction_guarantee",
                "score_readiness_credit",
            ],
        },
    },
    {
        "plan_id": "vip",
        "display_name": "VIP",
        "tier_rank": 4,
        # VIP is sales-led — no public price. Kept NULL so the public
        # pricing API will not surface a number for it.
        "monthly_price_aud_cents":   None,
        "quarterly_price_aud_cents": None,
        "annual_price_aud_cents":    None,
        "limits_json": {
            "practice_per_day":             None,
            "sectionals_per_month":         None,
            "mocks_per_month":              None,
            "ef_coach_per_day":             None,
            "trainer_feedback_per_month":   None,
            "study_plan_per_day":           None,
            "mock_review_days":             None,
            "sectional_score_per_month":    None,
            "mock_score_per_month":         None,
            "features": [
                "ai_score_all_types",
                "coach_reviewed_scoring",
                "skill_gap_analysis",
                "weekly_predictions_alerts",
                "coach_curated_predictions",
                "full_templates",
                "scored_breakdown_notes",
                "full_videos",
                "premium_workshops",
                "private_workshops",
                "mobile_app",
                "dedicated_coach_line",
                "satisfaction_guarantee",
                "score_readiness_credit",
                "group_classes",
                "one_on_one_sessions",
                "coach_personalised_study_plan",
            ],
        },
    },
]


def ensure_tables() -> None:
    """Create any tables that don't yet exist (no-op for ones that do)."""
    print("[init] creating subscription tables (idempotent) ...")
    Base.metadata.create_all(bind=engine)
    print("[init] done.")


def upsert_plans() -> None:
    """Insert or update plan catalogue rows.

    Refreshes prices, limits, display name, and tier_rank on every run so
    edits to PLAN_SEEDS take effect with one command.

    Does NOT touch Stripe Price IDs (those are populated manually after
    creating Products in the Stripe Dashboard — re-running this script
    must not wipe them).
    """
    db = SessionLocal()
    try:
        for seed in PLAN_SEEDS:
            existing = (
                db.query(SubscriptionPlan)
                .filter(SubscriptionPlan.plan_id == seed["plan_id"])
                .first()
            )
            if existing:
                existing.display_name              = seed["display_name"]
                existing.tier_rank                 = seed["tier_rank"]
                existing.monthly_price_aud_cents   = seed["monthly_price_aud_cents"]
                existing.quarterly_price_aud_cents = seed["quarterly_price_aud_cents"]
                existing.annual_price_aud_cents    = seed["annual_price_aud_cents"]
                existing.limits_json               = seed["limits_json"]
                db.commit()
                print(f"[seed] updated plan plan_id={existing.plan_id}")
                continue

            row = SubscriptionPlan(
                plan_id=seed["plan_id"],
                display_name=seed["display_name"],
                tier_rank=seed["tier_rank"],
                monthly_price_aud_cents=seed["monthly_price_aud_cents"],
                quarterly_price_aud_cents=seed["quarterly_price_aud_cents"],
                annual_price_aud_cents=seed["annual_price_aud_cents"],
                limits_json=seed["limits_json"],
                is_active=True,
            )
            db.add(row)
            db.commit()
            print(f"[seed] inserted plan plan_id={row.plan_id} tier_rank={row.tier_rank}")
    finally:
        db.close()


def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise SystemExit(
            "DATABASE_URL not set. Source .env or export it before running."
        )
    ensure_tables()
    upsert_plans()
    print("[init] all good.")


if __name__ == "__main__":
    main()
