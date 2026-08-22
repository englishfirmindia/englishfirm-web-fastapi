"""Regression guard for Option 1D — lifetime free-trial flag gating.

Before this change:
  - `sectionals_per_month=1` + `sectional_score_per_month=1` on Free
  - `mocks_per_month=1` + `mock_score_per_month=0` on Free — meant free
    users could TAKE a mock but never see the score. Broken by design.
  - Every question submit inside a sectional/mock ticked the
    `practice_per_day=3` counter, so a 32-Q sectional actually gave the
    free user 3 scored questions and 29 silent failures (the Roneel bug).

After this change:
  - Free user gets ONE fully-scored sectional (all 32 Qs) via
    `users.free_sectional_used` flag flipped atomically on exam start.
  - Free user gets ONE fully-scored mock (all 66 Qs) via
    `users.free_mock_used` flag.
  - Attempts 2+ raise 402 PLAN_LIMIT_REACHED at exam start with an
    upgrade CTA payload.
  - Sectional/mock question submits skip the practice_per_day counter
    (via `EnforceLimit("practice", skip_if_in_exam=True)`).
  - Paid users are unaffected — they still hit sectionals_per_month /
    mocks_per_month counters via the existing paths.

These tests pin the touchpoints that must stay in sync. Any regression
that reverts one of them silently would kill the free-tier experience
again.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# ─── Schema / model ────────────────────────────────────────────────────

def test_user_model_declares_free_sectional_used():
    src = (REPO / "db/models.py").read_text()
    assert "free_sectional_used = Column(" in src, (
        "User model must declare free_sectional_used — the lifetime free "
        "sectional gate reads/writes this column."
    )


def test_user_model_declares_free_mock_used():
    src = (REPO / "db/models.py").read_text()
    assert "free_mock_used = Column(" in src, (
        "User model must declare free_mock_used — the lifetime free "
        "mock gate reads/writes this column."
    )


def test_startup_migration_adds_flag_columns():
    src = (REPO / "main.py").read_text()
    # Both columns must appear inside the ALTER TABLE loop so a fresh
    # deploy backfills them on any DB that lacks them.
    assert "free_sectional_used" in src, "main.py startup must migrate the new flag columns"
    assert "free_mock_used" in src


# ─── SubscriptionContext / /subscription/me wire shape ─────────────────

def test_subscription_context_exposes_flags():
    src = (REPO / "services/billing/subscription_context.py").read_text()
    assert "free_sectional_used: bool" in src
    assert "free_mock_used: bool" in src


def test_subscription_me_endpoint_returns_flags():
    """Client-side (SubscriptionSnapshot) depends on both keys being in
    the /subscription/me response body. Missing = paywall button never
    shows correctly for the second attempt."""
    src = (REPO / "routers/subscription.py").read_text()
    assert '"free_sectional_used"' in src
    assert '"free_mock_used"' in src


# ─── free_trial_gate helper contract ───────────────────────────────────

def test_free_trial_gate_exports_helpers():
    """The two helpers + the submit-inspection helper must exist. Any
    rename would silently miss import errors in the 5 modified routes."""
    src = (REPO / "services/billing/free_trial_gate.py").read_text()
    assert "def enforce_free_sectional_or_paid(" in src
    assert "def enforce_free_mock_or_paid(" in src
    assert "def submit_is_inside_exam(" in src


def test_free_trial_gate_uses_atomic_flip():
    """The flag flip MUST be an atomic UPDATE ... WHERE ... RETURNING id
    so two concurrent /exam calls (double-tap, two tabs) can't both pass
    the flag=false check. A naive SELECT-then-UPDATE would allow the
    race and let a free user consume 2 slots."""
    src = (REPO / "services/billing/free_trial_gate.py").read_text()
    assert "UPDATE users SET" in src
    assert "RETURNING id" in src


def test_free_trial_gate_paid_users_use_legacy_counter():
    """Paid tiers must keep hitting sectionals_per_month / mocks_per_month
    counters — bronze:5 / silver:10 / gold+:unlimited. Removing this
    would give paid users an implicit unlimited free-sectional path."""
    src = (REPO / "services/billing/free_trial_gate.py").read_text()
    assert "check_and_increment_or_raise" in src
    assert '"sectional": "sectionals"' in src
    assert '"mock":      "mocks"' in src


def test_free_trial_gate_402_payload_shape():
    """The 402 body shape is a wire contract with the frontend
    PaywallDialog. Fields must match what the client parses."""
    src = (REPO / "services/billing/free_trial_gate.py").read_text()
    assert '"PLAN_LIMIT_REACHED"' in src
    assert '"feature_key":' in src
    assert '"period_type":  "lifetime"' in src


# ─── Sectional / mock start routes swapped to the new helper ───────────

def test_sectional_speaking_uses_free_trial_gate():
    src = (REPO / "routers/sectional/speaking.py").read_text()
    assert "enforce_free_sectional_or_paid" in src
    # Old counter-based gate for sectionals must be gone from this route.
    assert 'feature_key="sectionals"' not in src


def test_sectional_reading_uses_free_trial_gate():
    src = (REPO / "routers/sectional/reading.py").read_text()
    assert "enforce_free_sectional_or_paid" in src
    assert 'feature_key="sectionals"' not in src


def test_sectional_writing_uses_free_trial_gate():
    src = (REPO / "routers/sectional/writing.py").read_text()
    assert "enforce_free_sectional_or_paid" in src
    assert 'feature_key="sectionals"' not in src


def test_sectional_listening_uses_free_trial_gate():
    src = (REPO / "routers/sectional/listening.py").read_text()
    assert "enforce_free_sectional_or_paid" in src
    assert 'feature_key="sectionals"' not in src


def test_mock_start_uses_free_trial_gate():
    src = (REPO / "routers/mock.py").read_text()
    assert "enforce_free_mock_or_paid" in src
    # Old `EnforceLimit("mocks")` on /mock/start must be gone.
    assert 'EnforceLimit("mocks")' not in src


# ─── Finish routes no longer double-gate scoring ───────────────────────

def test_sectional_finish_score_gates_removed():
    """The `_score_gate=Depends(EnforceLimit("sectional_score"))` params
    are removed from all 4 sectional /finish routes — once an exam is
    started (flag flipped / counter ticked), its scoring is
    unconditional. Keeping the score gate would break paid users on
    the free_score_per_month=1 boundary too."""
    for module in ("speaking", "reading", "writing", "listening"):
        src = (REPO / f"routers/sectional/{module}.py").read_text()
        assert 'EnforceLimit("sectional_score")' not in src, (
            f"routers/sectional/{module}.py still has the sectional_score "
            f"gate on /finish — remove it, gating happens at exam start."
        )


def test_mock_finish_score_gate_removed():
    src = (REPO / "routers/mock.py").read_text()
    assert 'EnforceLimit("mock_score")' not in src, (
        "routers/mock.py still has the mock_score gate on /finish — "
        "the free plan has mock_score_per_month=0, which meant free users "
        "could NEVER see a mock score. Gate must be at /start only."
    )


# ─── 22 practice submit routes carry skip_if_in_exam=True ──────────────

_EXPECTED_PRACTICE_ROUTES = {
    "routers/speaking/read_aloud.py",
    "routers/speaking/repeat_sentence.py",
    "routers/speaking/answer_short_question.py",
    "routers/speaking/describe_image.py",
    "routers/speaking/retell_lecture.py",
    "routers/speaking/summarize_group_discussion.py",
    "routers/speaking/respond_to_situation.py",
    "routers/writing/summarize_written_text.py",
    "routers/writing/write_essay.py",
    "routers/reading/fill_in_blanks.py",
    "routers/reading/mcs.py",
    "routers/reading/mcm.py",
    "routers/reading/reorder_paragraphs.py",
    "routers/reading/fib_drag_drop.py",
    "routers/listening/wfd.py",
    "routers/listening/sst.py",
    "routers/listening/fib.py",
    "routers/listening/mcs.py",
    "routers/listening/mcm.py",
    "routers/listening/hcs.py",
    "routers/listening/smw.py",
    "routers/listening/hiw.py",
}


def test_all_practice_submit_routes_carry_skip_if_in_exam():
    """Every submit route that gates on `practice` MUST pass
    skip_if_in_exam=True. Otherwise a sectional/mock question submit
    ticks the daily practice counter and free users can only get 3
    questions scored per exam (the Roneel bug)."""
    missing = []
    for rel in _EXPECTED_PRACTICE_ROUTES:
        src = (REPO / rel).read_text()
        if 'EnforceLimit("practice")' in src:
            missing.append(rel)
        elif 'EnforceLimit("practice", skip_if_in_exam=True)' not in src:
            # Route no longer gates practice at all — surface this
            # explicitly rather than silently pass.
            missing.append(f"{rel} (practice gate removed entirely?)")
    assert not missing, (
        "These submit routes still use the un-parameterised "
        "`EnforceLimit(\"practice\")` — every sectional/mock question "
        "submitted through them will double-gate against practice_per_day:\n"
        + "\n".join(f"  - {r}" for r in missing)
    )


def test_expected_route_count_matches_grep():
    """Sanity check — the hard-coded list above should still match every
    file in the repo that gates on `practice`. If a new route is added
    with an unparameterised gate, this test flags it."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rln", "EnforceLimit(\"practice\"", str(REPO / "routers")],
        capture_output=True, text=True, check=False,
    )
    found = {
        str(Path(p).relative_to(REPO))
        for p in result.stdout.strip().splitlines()
        if p
    }
    unexpected_new = found - _EXPECTED_PRACTICE_ROUTES
    assert not unexpected_new, (
        "New routes gate on `practice` but aren't in the expected set. "
        "Either add them to _EXPECTED_PRACTICE_ROUTES, or confirm they "
        "pass skip_if_in_exam=True:\n" + "\n".join(f"  - {r}" for r in sorted(unexpected_new))
    )


# ─── EnforceLimit parameterisation ─────────────────────────────────────

def test_enforce_limit_accepts_skip_if_in_exam():
    src = (REPO / "services/billing/enforce_limit.py").read_text()
    assert "def EnforceLimit(feature_key: str, *, skip_if_in_exam: bool = False)" in src, (
        "EnforceLimit signature must accept `skip_if_in_exam` as a keyword-only "
        "argument. Positional would let old call sites break silently."
    )


def test_enforce_limit_defers_free_trial_gate_import():
    """The submit_is_inside_exam import MUST be inside the function body,
    not at module scope. Module-scope (unindented) import creates a
    cycle: free_trial_gate → subscription_context → (transitively via
    helpers) → enforce_limit."""
    src = (REPO / "services/billing/enforce_limit.py").read_text()
    for line in src.splitlines():
        # Only flag TOP-LEVEL imports (no leading indent). Indented imports
        # inside function bodies are exactly what we want (lazy).
        if line.startswith("from services.billing.free_trial_gate"):
            assert False, (
                "free_trial_gate must be imported lazily inside the "
                "EnforceLimit closure, not at module scope, to avoid the "
                "enforce_limit ↔ free_trial_gate circular import."
            )


# ─── CLAUDE.md guardrail: SectionalSubmitBloc no longer swallows 402 ───
# (Frontend side — not checked here; see frontend tests.)
