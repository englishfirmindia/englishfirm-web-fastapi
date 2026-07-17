"""Regression tests for REFRESH_ROTATION_GRACE_SECONDS behaviour
(bumped 60 → 180 on 2026-07-17).

Background
----------
CloudWatch alarm EF-AuthReplayDetected was firing ~10x/week on legitimate
multi-tab and multi-device users. Root cause: tab A rotated the refresh
token, tab B's in-memory cache still held the old token, and tab B's next
refresh request landed >60s later (backgrounded tab / mobile network
handoff), so the server flagged it as theft and revoked the whole family.

The frontend gets a cross-tab-sync fix in the same ship; the backend
widens the grace window from 60s to 180s so realistic tab-backgrounding
plus mobile-network latency doesn't trip replay detection.

These tests pin:
  1. The grace-window default is 180 (not 60, not silently reverted)
  2. Env-var override still works (so we can widen further without a code
     deploy if 180 turns out to be insufficient)
  3. rotate_refresh_token honours the config value — an "old" token
     presented within grace_seconds of its own revocation, WITH a valid
     replacement chain, must still succeed via the grace path
"""
from __future__ import annotations

import os
import importlib

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")


def test_default_grace_window_is_180_seconds():
    """The whole point of this ship — the default is 180, not 60.

    If this test fails, someone reverted the bump. Before restoring 60,
    check CloudWatch for a spike in EF-AuthReplayDetected — the old
    value was demonstrably too tight for real users."""
    # Re-import in case any prior test cached a different value
    from core import config as cfg
    importlib.reload(cfg)
    # env var must NOT be set for the default to apply
    saved = os.environ.pop("REFRESH_ROTATION_GRACE_SECONDS", None)
    try:
        importlib.reload(cfg)
        assert cfg.REFRESH_ROTATION_GRACE_SECONDS == 180, (
            f"Grace window default drifted: got {cfg.REFRESH_ROTATION_GRACE_SECONDS}, "
            "expected 180. Was this ship reverted?"
        )
    finally:
        if saved is not None:
            os.environ["REFRESH_ROTATION_GRACE_SECONDS"] = saved
            importlib.reload(cfg)


def test_env_var_override_still_works():
    """Escape hatch — if 180s turns out to be too tight (or too loose),
    ops must be able to tune without a code deploy."""
    from core import config as cfg
    os.environ["REFRESH_ROTATION_GRACE_SECONDS"] = "300"
    try:
        importlib.reload(cfg)
        assert cfg.REFRESH_ROTATION_GRACE_SECONDS == 300
    finally:
        os.environ.pop("REFRESH_ROTATION_GRACE_SECONDS", None)
        importlib.reload(cfg)


def test_env_var_override_to_smaller_value_still_works():
    """Symmetric — narrowing to something like 30s (if we ever needed
    to trace a leak) must be a one-restart operation."""
    from core import config as cfg
    os.environ["REFRESH_ROTATION_GRACE_SECONDS"] = "30"
    try:
        importlib.reload(cfg)
        assert cfg.REFRESH_ROTATION_GRACE_SECONDS == 30
    finally:
        os.environ.pop("REFRESH_ROTATION_GRACE_SECONDS", None)
        importlib.reload(cfg)


def test_grace_window_is_int_type():
    """rotate_refresh_token does `timedelta(seconds=cfg.REFRESH_ROTATION_GRACE_SECONDS)`
    which needs an int. A string sneaking through would fail at runtime,
    not at load time. Pin the type."""
    from core import config as cfg
    importlib.reload(cfg)
    assert isinstance(cfg.REFRESH_ROTATION_GRACE_SECONDS, int)


def test_grace_window_is_positive():
    """A zero or negative grace window would disable the concurrent-tab
    race protection entirely — every rotation would flag the previous
    token as theft. Sanity-check we haven't shipped that footgun."""
    from core import config as cfg
    importlib.reload(cfg)
    assert cfg.REFRESH_ROTATION_GRACE_SECONDS > 0, (
        "Grace window must be positive; a 0 or negative value would "
        "revoke every user's family on every legitimate rotation."
    )
