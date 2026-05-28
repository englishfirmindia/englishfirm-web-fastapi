"""Regression tests for the user-agnostic FIB explanation cache (2026-05-29).

Background
----------
Stefy reported (question_reports id=94, 2026-05-27) that after redoing
a Reading FIB-DD question and submitting again, the AI explanation
panel kept referencing her FIRST attempt's wrong picks. Root cause:
the `question_explanations` cache was keyed by `question_id` only, but
the LLM prompt included the user's specific picks and is_correct flags
— so the cached explanation text was permanently bound to whichever
user submitted the question first.

Fix: strip user-specific context out of the LLM prompt and out of the
cached output. The cached row now contains only
`{blank_id, correct, explanation, scorer}` — strictly user-agnostic.
Routers merge the per-request user_answer + is_correct on top before
returning to the frontend (and persisting result_json).

These tests pin the contract:
  1. The system prompt warns the LLM that output is shared across
     users (no per-student phrasing).
  2. The user block sent to the LLM does NOT include student answers
     or is_correct flags.
  3. `generate_fib_explanations` output does NOT contain user_answer
     or is_correct keys — only blank_id, correct, explanation, scorer.
  4. The two router merge helpers (_merge_user_answers_into_explanations)
     correctly layer per-request user data on top of cached output.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import importlib


# ── 1. System prompt is user-agnostic ─────────────────────────────────────


def test_system_prompt_does_not_reference_student_picks():
    from services import fib_explainer
    prompt = fib_explainer._SYSTEM_PROMPT.lower()
    # Pre-fix the prompt told the LLM "If the student's answer was wrong,
    # focus on why the CORRECT word is right". That sentence is what
    # caused the cached output to reference user picks. The new prompt
    # must explicitly tell the LLM the explanation is shared across users.
    assert "student's answer" not in prompt, (
        "System prompt must not reference 'student's answer' — that's "
        "what caused Stefy's bug. The cached text would bind to whichever "
        "user submitted first."
    )
    assert "shaming" not in prompt, "Stale 'shaming the student's choice' clause must be gone."
    # Positive: prompt must call out the cache-shared nature.
    assert "cached" in prompt or "shared across" in prompt, (
        "Prompt must tell the LLM the explanation is shared/cached so it "
        "doesn't reference any individual student's pick."
    )


# ── 2. User block to the LLM is user-agnostic ─────────────────────────────


def test_user_block_does_not_include_student_pick():
    from services.fib_explainer import _build_user_block
    blanks = [
        {"blank_id": "1", "correct": "however", "user_answer": "moreover", "is_correct": False},
        {"blank_id": "2", "correct": "subsequently", "user_answer": "subsequently", "is_correct": True},
    ]
    block = _build_user_block("PASSAGE TEXT ___1___ MORE TEXT ___2___ END.", blanks)
    # The LLM must NOT see 'student=' or 'correct?=' rows that previously
    # carried per-user data.
    assert "student=" not in block, (
        "User block must not include student picks — that's the data that "
        "got baked into the cached output."
    )
    assert "correct?=" not in block
    assert "moreover" not in block, (
        "Stefy's wrong-pick value must never reach the LLM prompt; if it "
        "did, the cached text could reference it."
    )
    # Positive: the block must still include passage + correct answers.
    assert "however" in block
    assert "subsequently" in block
    assert "PASSAGE TEXT" in block


# ── 3. generate_fib_explanations output is user-agnostic ──────────────────


def test_generate_fib_explanations_returns_only_user_agnostic_fields(monkeypatch):
    """Mock the LLM call and assert the returned dict per blank contains
    only {blank_id, correct, explanation, scorer} — no user_answer or
    is_correct keys. Those are what made cache rows user-specific."""
    from services import fib_explainer

    def _fake_haiku(user_block, timeout):
        return {
            "explanations": [
                {"blank_id": "1", "explanation": "However signals contrast."},
                {"blank_id": "2", "explanation": "Subsequently marks sequence."},
            ]
        }

    monkeypatch.setattr(fib_explainer, "_call_haiku", _fake_haiku)
    monkeypatch.setattr(fib_explainer, "_call_gpt4o", lambda *a, **k: None)

    blanks = [
        {"blank_id": "1", "correct": "however", "user_answer": "moreover", "is_correct": False},
        {"blank_id": "2", "correct": "subsequently", "user_answer": "subsequently", "is_correct": True},
    ]
    out = fib_explainer.generate_fib_explanations(
        passage="PASSAGE TEXT ___1___ MORE TEXT ___2___ END.",
        blanks=blanks,
    )

    assert len(out) == 2
    for entry in out:
        assert set(entry.keys()) == {"blank_id", "correct", "explanation", "scorer"}, (
            f"Each cached entry must have ONLY user-agnostic keys. "
            f"Got: {set(entry.keys())}. If user_answer or is_correct appear, "
            f"the cache row will be user-specific and Stefy's bug will return."
        )


# ── 4. Router merge helpers layer per-request user data on top ────────────


def test_fib_drag_drop_merge_layers_user_data():
    """Mirror of the production helper at
    routers/reading/fib_drag_drop.py:_merge_user_answers_into_explanations."""
    # Re-import the helper from the router (it's module-level).
    mod = importlib.import_module("routers.reading.fib_drag_drop")
    merge = mod._merge_user_answers_into_explanations

    cached = [
        {"blank_id": "1", "correct": "however", "explanation": "However signals contrast."},
        {"blank_id": "2", "correct": "subsequently", "explanation": "Subsequently marks sequence."},
    ]
    # User picks: blank 1 wrong, blank 2 correct.
    user_answers = {"1": "moreover", "2": "subsequently"}
    blank_results = {"1": False, "2": True}

    merged = merge(cached, user_answers, blank_results)
    assert len(merged) == 2
    assert merged[0]["user_answer"] == "moreover"
    assert merged[0]["is_correct"] is False
    assert merged[1]["user_answer"] == "subsequently"
    assert merged[1]["is_correct"] is True
    # The cached fields survive unchanged.
    assert merged[0]["explanation"] == "However signals contrast."


def test_fib_drag_drop_merge_handles_blank_prefix_keys():
    """Some routers use 'blank_1' instead of '1' as the key. The merge
    must accept either form so the per-request user data lands."""
    mod = importlib.import_module("routers.reading.fib_drag_drop")
    merge = mod._merge_user_answers_into_explanations

    cached = [{"blank_id": "1", "correct": "however", "explanation": "X"}]
    merged = merge(cached, {"blank_1": "moreover"}, {"blank_1": False})
    assert merged[0]["user_answer"] == "moreover"
    assert merged[0]["is_correct"] is False


def test_fill_in_blanks_merge_mirrors_fib_dd():
    """The FIB-R (typed) router has an identical merge helper. Pin it
    so a future refactor doesn't drift only one of them."""
    mod = importlib.import_module("routers.reading.fill_in_blanks")
    merge = mod._merge_user_answers_into_explanations

    cached = [{"blank_id": "1", "correct": "however", "explanation": "X"}]
    merged = merge(cached, {"1": "moreover"}, {"1": False})
    assert merged[0]["user_answer"] == "moreover"
    assert merged[0]["is_correct"] is False


# ── 5. End-to-end: Redo scenario — cached row + different picks ───────────


def test_redo_scenario_cached_explanation_per_attempt_user_data(monkeypatch):
    """Simulate Stefy's exact bug: the cache has been populated by
    attempt 1. Attempt 2 picks different answers. The response for
    attempt 2 must show attempt 2's picks (not attempt 1's), even
    though the cached explanation text is identical."""
    from services import fib_explainer

    # Cache HIT — LLM is not called at all on attempt 2. The cached
    # row is user-agnostic (post-fix), so it has no user_answer.
    cached_row = [
        {"blank_id": "1", "correct": "however", "explanation": "However signals contrast.", "scorer": "haiku"},
    ]

    # Attempt 1 picks: "moreover" (wrong)
    # Attempt 2 picks: "yet" (also wrong, different word)
    mod = importlib.import_module("routers.reading.fib_drag_drop")
    merge = mod._merge_user_answers_into_explanations

    attempt_1_response = merge(cached_row, {"1": "moreover"}, {"1": False})
    attempt_2_response = merge(cached_row, {"1": "yet"}, {"1": False})

    # Same explanation text (cached).
    assert attempt_1_response[0]["explanation"] == attempt_2_response[0]["explanation"]
    # Different per-attempt picks.
    assert attempt_1_response[0]["user_answer"] == "moreover"
    assert attempt_2_response[0]["user_answer"] == "yet"
    # Stefy's bug: this assertion would have failed pre-fix (cache held
    # attempt 1's "moreover" forever).
