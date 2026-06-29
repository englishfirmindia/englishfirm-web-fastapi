"""One-shot cleanup (2026-06-29):

  1. Mark 5 existing Stripe subscriptions with `cancel_at_period_end=True`
     so they don't auto-renew. Customers finish their already-paid month
     (most fair, no refunds needed). Confirmed empty trialing/past_due/
     unpaid buckets at time of audit.

  2. Flip 3 stale gold rows in user_subscriptions to status='cancelled'
     so the trainer admin "DB claims Stripe sub but Stripe lost it"
     warning clears. These are leftover test accounts (a1/b/c@gmail.com,
     deploy day 2026-05-18) — Stripe has nothing live for any of them.

Usage:
    python3 scripts/cleanup_stripe_auto_recharge_20260629.py            # dry run
    python3 scripts/cleanup_stripe_auto_recharge_20260629.py --apply    # commit
"""
import os
import sys
import json
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import boto3
import psycopg2
from psycopg2.extras import RealDictCursor

APPLY = "--apply" in sys.argv

# ── Stripe subscriptions to cancel-at-period-end ─────────────────────────
STRIPE_SUBS = [
    ("sub_1TYNubGAGzfk0CxlSN3ob3KI", "kaspin.tonus@gmail.com"),
    ("sub_1Tb9tyGAGzfk0CxlcECNNznK", "nishachooramana@gmail.com"),
    ("sub_1TmsFxGAGzfk0CxlTcVsOBiQ", "shirleenprakash97@gmail.com"),
    ("sub_1Tn8OsGAGzfk0CxlOtKCusO7", "thapabhola49@gmail.com"),
    ("sub_1TnAq8GAGzfk0Cxli6TkOavU", "011sharmanikita@gmail.com"),
]

# ── Stale "active" gold rows in DB to mark cancelled ─────────────────────
STALE_GOLD_EMAILS = ["a1@gmail.com", "b@gmail.com", "c@gmail.com"]


def _stripe_key() -> str:
    sm = boto3.session.Session(
        profile_name="englishfirm", region_name="ap-southeast-2",
    ).client("secretsmanager")
    secrets = json.loads(sm.get_secret_value(
        SecretId="englishfirm-web-fastapi/prod"
    )["SecretString"])
    return secrets["STRIPE_SECRET_KEY"]


def _dsn() -> str:
    return os.environ["DATABASE_URL"].replace(
        "postgresql+psycopg2://", "postgresql://"
    )


def main() -> None:
    import stripe
    stripe.api_key = _stripe_key()
    stripe.api_version = "2025-09-30.clover"

    # ── 1. Verify current state of each Stripe sub + the 3 stale DB rows
    print("=== Pre-flight: Stripe sub current state ===")
    pre_stripe = []
    for sub_id, email in STRIPE_SUBS:
        s = stripe.Subscription.retrieve(sub_id).to_dict()
        item = s["items"]["data"][0]
        pe = datetime.datetime.fromtimestamp(item["current_period_end"]).date()
        cap = s.get("cancel_at_period_end")
        amt = item["price"]["unit_amount"] / 100
        print(f"  {sub_id}  {email:<35}  ${amt} {item['price']['currency']}/{item['price']['recurring']['interval']}  "
              f"next={pe}  cancel_at_period_end={cap}")
        pre_stripe.append((sub_id, email, s["status"], cap, pe))

    print("\n=== Pre-flight: stale gold DB rows ===")
    conn = psycopg2.connect(_dsn())
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT us.id::text, u.email, us.plan_id, us.status, us.source,
               us.external_id, us.started_at, us.updated_at
        FROM user_subscriptions us JOIN users u ON u.id = us.user_id
        WHERE u.email = ANY(%s)
          AND us.plan_id = 'gold' AND us.source = 'stripe'
          AND us.status IN ('active', 'past_due')
        """,
        (STALE_GOLD_EMAILS,),
    )
    pre_db = cur.fetchall()
    for row in pre_db:
        print(f"  uuid={row['id']}  {row['email']:<25}  {row['plan_id']}  "
              f"{row['status']}  ext_id={row['external_id']}  started={row['started_at']}")

    if not APPLY:
        print("\n[DRY RUN] No writes. Re-run with --apply to commit:")
        print(f"  Would set cancel_at_period_end=True on {len(STRIPE_SUBS)} Stripe subs")
        print(f"  Would flip {len(pre_db)} user_subscriptions rows to status='cancelled'")
        return

    # ── 2. Apply: Stripe modifications ──
    print("\n[APPLY] Updating Stripe subscriptions...")
    stripe_results = []
    for sub_id, email in STRIPE_SUBS:
        try:
            updated = stripe.Subscription.modify(
                sub_id, cancel_at_period_end=True
            ).to_dict()
            cap = updated.get("cancel_at_period_end")
            stripe_results.append((sub_id, email, "ok", cap, None))
            print(f"  ✓ {sub_id}  {email:<35}  cancel_at_period_end={cap}")
        except Exception as exc:
            stripe_results.append((sub_id, email, "fail", None, str(exc)[:200]))
            print(f"  ✗ {sub_id}  {email:<35}  FAILED: {exc}")

    # ── 3. Apply: DB cleanup of stale gold rows ──
    print("\n[APPLY] Cancelling stale gold DB rows...")
    cur.execute(
        """
        UPDATE user_subscriptions
        SET status = 'cancelled',
            cancel_at_period_end = TRUE,
            updated_at = NOW()
        WHERE id IN (
            SELECT us.id FROM user_subscriptions us
            JOIN users u ON u.id = us.user_id
            WHERE u.email = ANY(%s)
              AND us.plan_id = 'gold' AND us.source = 'stripe'
              AND us.status IN ('active', 'past_due')
        )
        RETURNING id::text
        """,
        (STALE_GOLD_EMAILS,),
    )
    flipped = cur.fetchall()
    print(f"  flipped {len(flipped)} rows")

    # Audit row per flip so /trainer/granted-vips reports stay correct.
    for row in pre_db:
        cur.execute(
            """
            INSERT INTO subscription_events
                (id, user_id, subscription_id, event_type, from_plan_id, to_plan_id,
                 actor, metadata, created_at)
            SELECT gen_random_uuid(), us.user_id, us.id, 'cancelled', us.plan_id, NULL,
                   'cleanup_2026_06_29',
                   %s::jsonb, NOW()
            FROM user_subscriptions us WHERE us.id = %s
            """,
            (
                json.dumps({
                    "reason": "stale db_only row (Stripe had no active sub for this customer)",
                    "operator": "kaspin.tonus + Claude Opus 4.7",
                    "cleanup_script": "cleanup_stripe_auto_recharge_20260629.py",
                }),
                row["id"],
            ),
        )

    conn.commit()
    print("\n=== POST-VERIFY ===")
    # Reverify the Stripe state
    for sub_id, email in STRIPE_SUBS:
        s = stripe.Subscription.retrieve(sub_id).to_dict()
        cap = s.get("cancel_at_period_end")
        print(f"  Stripe {sub_id}  {email:<35}  cancel_at_period_end={cap}")
    cur.execute(
        """
        SELECT u.email, us.status
        FROM user_subscriptions us JOIN users u ON u.id = us.user_id
        WHERE u.email = ANY(%s) AND us.plan_id = 'gold'
        """,
        (STALE_GOLD_EMAILS,),
    )
    for row in cur.fetchall():
        print(f"  DB {row['email']:<25}  status={row['status']}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
