# Stripe Checkout — operator setup

The backend (`routers/billing.py`) and frontend (`features/billing/plans_screen.dart`) are wired. To actually accept payments you need to complete the Stripe Dashboard steps below + populate two env vars + run the seed script.

This runbook is idempotent: re-doing any step is harmless.

## 1. Stripe account

- Sign up at <https://dashboard.stripe.com> (Australia)
- Verify the business (required before live mode)
- **Account settings → Tax** → enable Stripe Tax with **Australia GST** if you want Stripe to compute GST inclusive amounts. (Our prices in `subscription_plans` are already GST-inclusive, so leave "Prices are inclusive of tax" ON.)

You'll have a test mode + live mode toggle at the top of the dashboard. Do all the steps in **test mode** first, validate end-to-end, then duplicate in live mode.

## 2. Products + Prices (one per tier × period — 9 total)

For each of the 3 paid tiers (Bronze / Silver / Gold), create one **Product** with **3 recurring Prices**. VIP is sales-led — don't put it in Stripe.

**Product → Bronze**

| Price | Amount (AUD) | Billing |
|---|---|---|
| Monthly | 29.00 | every 1 month |
| Quarterly | 79.00 | every 3 months |
| Annual | 249.00 | every 1 year |

**Product → Silver**

| Price | Amount (AUD) | Billing |
|---|---|---|
| Monthly | 59.00 | every 1 month |
| Quarterly | 159.00 | every 3 months |
| Annual | 549.00 | every 1 year |

**Product → Gold**

| Price | Amount (AUD) | Billing |
|---|---|---|
| Monthly | 69.00 | every 1 month |
| Quarterly | 189.00 | every 3 months |
| Annual | 649.00 | every 1 year |

For each Price, set "**Include tax in price**" if GST is on. Copy each `price_xxx` ID — you'll paste them into the DB.

## 3. Mirror Price IDs into Postgres

Edit `scripts/init_subscription_tables.py` and paste the IDs onto each plan's `PLAN_SEEDS` entry, then re-run:

```python
{
  "plan_id": "bronze",
  ...
  "stripe_price_id_monthly":   "price_1AB...",
  "stripe_price_id_quarterly": "price_1CD...",
  "stripe_price_id_annual":    "price_1EF...",
  ...
}
```

Or set them directly via SQL:

```sql
UPDATE subscription_plans
SET stripe_price_id_monthly   = 'price_xxx',
    stripe_price_id_quarterly = 'price_xxx',
    stripe_price_id_annual    = 'price_xxx'
WHERE plan_id = 'bronze';
-- repeat for silver, gold
```

`routers/billing.py:_resolve_price_id` raises a 400 PRICE_NOT_CONFIGURED until these are populated, so the user just sees "not available yet" rather than a crash.

## 4. Webhook endpoint

In Stripe Dashboard → **Developers → Webhooks → Add endpoint**:

- Endpoint URL: `https://api.englishfirm.com/api/v1/billing/webhooks/stripe`
- API version: latest (we pin `2025-09-30.clover` server-side, but the endpoint accepts whatever Stripe sends)
- Events to listen to:
  - `checkout.session.completed`
  - `invoice.payment_succeeded`
  - `invoice.payment_failed`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `charge.refunded`

After creating, click "Reveal" on the **Signing secret** (`whsec_xxx`). This goes into `STRIPE_WEBHOOK_SECRET`.

## 5. Customer Portal (self-serve cancel + payment-method update)

Stripe Dashboard → **Settings → Billing → Customer portal**:

- Functionality: enable "Cancel subscriptions" + "Update payment method" + "View invoices"
- Cancellation policy: "Cancel at end of billing period" (preserves the cleanest UX — user keeps access until period_end, then `customer.subscription.deleted` fires)
- Branding: upload the EnglishFirm logo + set the brand color to `#1B6B3A`
- Test mode + live mode each have their own configuration — set both

## 6. Backend env vars (AWS Secrets Manager + ECS task def)

Two secrets to add to AWS:

```
STRIPE_SECRET_KEY    = sk_test_xxx  (or sk_live_xxx in prod)
STRIPE_WEBHOOK_SECRET = whsec_xxx
```

Add them to the secret `englishfirm-web-fastapi/prod` in Secrets Manager, then reference them in the ECS task definition's `secrets` block (matching the existing OPENAI_API_KEY / ANTHROPIC_API_KEY pattern).

After that, redeploy the backend so it picks up the new env vars. `config.stripe_configured()` will start returning `true` and the `/billing/*` routes will leave 503 territory.

## 7. End-to-end verification

In test mode, log in to the app as a test user and:

1. Open `/plans`
2. Click "Choose" on a paid plan
3. Use test card `4242 4242 4242 4242`, any future expiry, any CVC, any name, any postcode
4. Complete Checkout — Stripe redirects to `/billing/success?session_id=cs_test_xxx`
5. The success page polls `/subscription/me`; within ~2-5 seconds you should see "Welcome to <Plan>"
6. Reload `/plans` — the "Current" pill should now sit on the new tier

Verify DB state:
```sql
SELECT user_id, plan_id, status, source, current_period_end
FROM user_subscriptions WHERE source='stripe' ORDER BY started_at DESC LIMIT 5;

SELECT user_id, provider_transaction_id, status, amount_cents
FROM payment_transactions WHERE provider='stripe' ORDER BY created_at DESC LIMIT 5;

SELECT user_id, event_type, to_plan_id, metadata
FROM subscription_events ORDER BY created_at DESC LIMIT 10;
```

Common gotchas:

- **Webhook not firing** → Dashboard → Webhooks → endpoint → "Recent events" tab. Click any event to see request/response. 401/403 means a load balancer / CDN is blocking; verify `api.englishfirm.com/api/v1/billing/webhooks/stripe` is reachable from Stripe IPs.
- **Webhook firing but no DB change** → CloudWatch `/ecs/englishfirm-web-fastapi` → filter `[stripe]`. Look for "unknown price_id" (price not mirrored into Postgres yet) or "unresolved user_id" (metadata missing).
- **Customer Portal fails to load** → Settings → Billing → Customer portal not configured for the current mode.

## 8. Live-mode cutover

Once test mode works end-to-end:

1. Duplicate Products + Prices in live mode (Stripe doesn't share between modes)
2. Update `subscription_plans.stripe_price_id_*` columns with the live-mode IDs
3. Create a new webhook endpoint in live mode → grab its `whsec_xxx`
4. Update Secrets Manager: `STRIPE_SECRET_KEY=sk_live_xxx`, `STRIPE_WEBHOOK_SECRET=whsec_xxx`
5. Redeploy

No code change needed — the SDK picks live keys vs test keys based on the prefix.
