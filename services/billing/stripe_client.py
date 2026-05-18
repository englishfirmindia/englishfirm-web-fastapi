"""
Stripe SDK initialisation + thin wrappers used by routers/billing.py.

Two helpers are exposed:
  - `stripe_lib()` returns the configured stripe module so routes
    can call stripe.checkout.Session.create(...) etc. directly.
  - `verify_webhook_signature(payload, sig_header)` wraps the SDK call
    that authenticates incoming webhook payloads.

Initialisation is lazy + idempotent — module-level state is fine since
the secret never changes during a process lifetime, but we re-set it on
every access in case the env was refreshed (rare in practice; useful in
tests that swap config).
"""
from __future__ import annotations

import logging
from typing import Any

import stripe

import core.config as config

log = logging.getLogger(__name__)

_initialised = False


def _ensure_initialised() -> None:
    """Set the API key + a sane default API version on first use.

    Pinning api_version means our webhook payload shapes don't shift
    under us when Stripe ships a new account-default version. Bump
    deliberately when adopting new event types.
    """
    global _initialised
    if not config.STRIPE_SECRET_KEY:
        raise RuntimeError(
            "Stripe is not configured — STRIPE_SECRET_KEY missing. "
            "Routes should short-circuit via config.stripe_configured() "
            "before reaching this helper."
        )
    stripe.api_key = config.STRIPE_SECRET_KEY
    # Pinned account API version. Update only when intentionally adopting
    # a breaking change in the Stripe API.
    stripe.api_version = "2025-09-30.clover"
    _initialised = True


def stripe_lib() -> Any:
    """Return the configured `stripe` module. Use this from routes
    instead of importing `stripe` directly so initialisation is
    centralised."""
    if not _initialised:
        _ensure_initialised()
    return stripe


def verify_webhook_signature(payload: bytes, sig_header: str) -> Any:
    """Validate the `Stripe-Signature` header against the webhook secret
    and return the parsed `stripe.Event`. Raises ValueError on bad
    payload, SignatureVerificationError on bad signature."""
    if not config.STRIPE_WEBHOOK_SECRET:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET not configured")
    return stripe.Webhook.construct_event(
        payload=payload,
        sig_header=sig_header,
        secret=config.STRIPE_WEBHOOK_SECRET,
    )
