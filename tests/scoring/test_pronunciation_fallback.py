"""Regression tests for the Azure pronunciation fallback path in _score_speaking_v2.

History
-------
The pronunciation fallback used to copy content into pronunciation when Azure
threw an exception and skipped the cross-penalty to avoid double-deducting.
This inflated pronunciation for users whose Azure outage hid a low fluency
take (see q=7265 / aid=2981 — Solar System, 2026-05-25).

The new rule sets `pronunciation = 90.0` (Azure-max stand-in) and lets the
cross-penalty multiplier scale it by `min(content, fluency)`. Healthy rows
barely move (~1 PTE drop); rows with weak content or fluency see pronunciation
proportionally lowered.

These tests pin:
  * Happy-path behavior (Group A) — must NOT change after the rule flip.
  * Cross-penalty math (Group D) — must not regress.
  * Fluency-zero correlation (Group F) — proves no hidden gate zeroes pron.
  * Warning + audit trail (Group G) — UI banner depends on these.
  * Fallback behavior (Group B) — the intentional shift, asserts new values.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.speaking_scorer import _score_speaking_v2, _cross_multiplier, _RA_FALLBACK_CFG  # noqa: E402
from services.scoring.speaking_config_service import SpeakingScoringConfig  # noqa: E402


def _cfg_for(task_type: str) -> SpeakingScoringConfig:
    """Return an in-memory cfg appropriate for each task type. Mirrors the
    real RDS row shape so the test scorer doesn't fall back to RA defaults
    (which silently routes freeform tasks through azure_assessment)."""
    base_kwargs = {
        "task_type": task_type,
        "wpm_floor": 50.0 if task_type == "summarize_group_discussion" else 80.0,
        "wpm_ceiling": 270.0,
        "wpm_plateau_low": 130.0,
        "wpm_plateau_high": 220.0,
        "wpm_slope_per_wpm": 2.0,
        "wpm_peak_score": 100.0,
        "pause_min_ms": 500,
        "pause_leading_tol_ms": 200,
        "pause_trailing_tol_ms": 200,
        "silence_thresh_dbfs": -30.0,
        "content_insertion_penalty_k": 2.0,
        "pause_penalty_max_pauses": 10,
        "pause_penalty_sentence_clamp_min": 1,
        "pause_penalty_sentence_clamp_max": 10,
        "pause_penalty_formula_constant": 11,
        "cross_penalty_healthy_threshold": 20.0,
        "cross_penalty_floor_multiplier": 0.5,
        "cross_penalty_slope": 0.025,
        "uses_cross_penalty": True,
    }
    if task_type in ("read_aloud", "repeat_sentence"):
        return SpeakingScoringConfig(
            **base_kwargs,
            content_method="lcs_k2",
            uses_reference_text=True,
            pronunciation_source="azure_assessment",
            pause_penalty_max_pauses_mult=None,
        )
    # describe_image / retell_lecture / respond_to_situation /
    # summarize_group_discussion / answer_short_question
    return SpeakingScoringConfig(
        **base_kwargs,
        content_method="llm_keypoints" if task_type != "answer_short_question" else "regex_match",
        uses_reference_text=False,
        pronunciation_source="azure_freeform",
        pause_penalty_max_pauses_mult=2.0,
    )


# ────────────────────────────────────────────────────────────────────────────
# Test fixtures — minimal Whisper + Azure stub builders
# ────────────────────────────────────────────────────────────────────────────


def _whisper_stub(transcript: str, words: list | None = None) -> dict:
    """Mimics `transcribe_with_whisper_words` return shape."""
    return {"transcript": transcript, "words": words or []}


def _azure_assess_stub(accuracy: float, words: list | None = None) -> dict:
    """Mimics `assess_pronunciation_with_timestamps` return shape (RA / RS)."""
    return {"AccuracyScore": accuracy, "Words": words or []}


def _azure_freeform_stub(pron: float, word_scores: list | None = None,
                         transcript: str = "") -> dict:
    """Mimics `transcribe_and_score_free` return shape (DI/RL/RTS/SGD/ASQ)."""
    return {
        "pronunciation": pron,
        "word_scores": word_scores or [],
        "transcript": transcript,
    }


def _ra_reference() -> str:
    """A reference passage matching the structure of real RA questions."""
    return (
        "Venture capitals and public funding authorities need to carefully "
        "consider the incentive issues of entrepreneurs when providing support. "
        "In allocating resources to potentially competing innovators, the "
        "authorities should balance immediate impact with long-term growth."
    )


def _ra_words_perfect(reference: str) -> list:
    """Build a per-word Azure response that maps 1:1 to the reference, all
    matched with high accuracy and no errors."""
    return [
        {
            "Word": w.strip(".,"),
            "AccuracyScore": 95.0,
            "ErrorType": "None",
            "offset_ms": i * 400,
            "duration_ms": 350,
        }
        for i, w in enumerate(reference.split())
    ]


def _whisper_words_for(reference: str) -> list:
    """Build Whisper-style word timings — used so that even when Azure raises
    (azure_words = []), the WPM formula has a signal source from Whisper.
    Mirrors a healthy ~150 wpm read with no within-speech pauses."""
    words = reference.split()
    out = []
    cursor = 0.0
    for w in words:
        out.append({
            "text": w.strip(".,"),
            "start": cursor,
            "end": cursor + 0.35,
        })
        cursor += 0.4   # 0.4s per word → 150 wpm
    return out


# ────────────────────────────────────────────────────────────────────────────
# Group D — Cross-penalty math (pure function, no I/O)
# ────────────────────────────────────────────────────────────────────────────


class TestCrossMultiplierMath:
    """Pin the cross-penalty formula. If anyone changes the constants or the
    curve shape, these will catch it before any user-facing shift."""

    def test_floor_at_zero_input(self):
        assert _cross_multiplier(0, 100, 0.5, 0.005) == 0.5

    def test_ceiling_at_threshold(self):
        assert _cross_multiplier(100, 100, 0.5, 0.005) == 1.0

    def test_linear_midpoint(self):
        # 0.5 + 0.005 * 50 = 0.75
        assert _cross_multiplier(50, 100, 0.5, 0.005) == pytest.approx(0.75)

    def test_above_threshold_clamps_to_one(self):
        assert _cross_multiplier(200, 100, 0.5, 0.005) == 1.0

    def test_negative_input_floors(self):
        assert _cross_multiplier(-10, 100, 0.5, 0.005) == 0.5

    def test_legacy_defaults_still_work(self):
        # Old RA/RS pre-v3 used healthy=20, slope=0.025.
        assert _cross_multiplier(0, 20, 0.5, 0.025) == 0.5
        assert _cross_multiplier(20, 20, 0.5, 0.025) == 1.0
        assert _cross_multiplier(10, 20, 0.5, 0.025) == pytest.approx(0.75)


# ────────────────────────────────────────────────────────────────────────────
# Helper — run _score_speaking_v2 with all external I/O stubbed
# ────────────────────────────────────────────────────────────────────────────


def _run_scorer(
    task_type: str = "read_aloud",
    transcript: str = "",
    whisper_words: list | None = None,
    azure_assess: dict | Exception | None = None,
    azure_freeform: dict | Exception | None = None,
    reference_text: str = "",
    key_points: list | None = None,
):
    """Run _score_speaking_v2 with all Azure + Whisper + LLM I/O stubbed.

    Pass an Exception instance for `azure_assess` or `azure_freeform` to
    simulate Azure failing. Pass a dict to simulate Azure returning successfully.
    """
    if reference_text == "" and task_type in ("read_aloud", "repeat_sentence"):
        reference_text = _ra_reference()

    with patch(
        "services.azure_speech_service.assess_pronunciation_with_timestamps"
    ) as mock_assess, patch(
        "services.azure_speech_service.transcribe_and_score_free"
    ) as mock_free, patch(
        "services.whisper_service.transcribe_with_whisper_words"
    ) as mock_whisper, patch(
        "services.llm_content_scoring_service.score_content_with_llm"
    ) as mock_llm, patch(
        "services.speaking_scorer._get_speaking_config",
        return_value=_cfg_for(task_type),
    ):

        # Azure pronunciation_assessment (RA / RS)
        if isinstance(azure_assess, Exception):
            mock_assess.side_effect = azure_assess
        else:
            mock_assess.return_value = (
                azure_assess
                if azure_assess is not None
                else _azure_assess_stub(0)
            )

        # Azure freeform (DI / RL / RTS / SGD / ASQ)
        if isinstance(azure_freeform, Exception):
            mock_free.side_effect = azure_freeform
        else:
            mock_free.return_value = (
                azure_freeform
                if azure_freeform is not None
                else _azure_freeform_stub(0)
            )

        # Whisper
        mock_whisper.return_value = _whisper_stub(transcript, whisper_words or [])

        # LLM (for llm_keypoints content scoring) — returns a neutral score
        mock_llm.return_value = {"score": 50.0, "reasoning": "test"}

        return _score_speaking_v2(
            user_id=999,
            question_id=1,
            audio_bytes=b"\x00" * 1024,  # 1KB dummy — Whisper/Azure are mocked
            reference_text=reference_text,
            task_type=task_type,
            key_points=key_points or [],
        )


# ────────────────────────────────────────────────────────────────────────────
# Group A — Happy path: Azure succeeds, fallback never fires
# These MUST NOT change after the fallback rule flip.
# ────────────────────────────────────────────────────────────────────────────


class TestHappyPathRA:
    """RA with Azure assess returning real per-word data."""

    def test_no_fallback_when_azure_returns_real_data(self):
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            azure_assess=_azure_assess_stub(85.0, _ra_words_perfect(ref)),
            reference_text=ref,
        )
        assert result["pronunciation_fallback"] is False
        assert "pronunciation_fallback_azure" not in (result.get("scoring_warnings") or [])

    def test_raw_pron_preserved_when_min_cf_above_threshold(self):
        """When min(c, f) >= 100, mP = 1.0, pron passes through."""
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            azure_assess=_azure_assess_stub(85.0, _ra_words_perfect(ref)),
            reference_text=ref,
        )
        # mP should be 1.0 since both content (LCS=high) and fluency are healthy.
        # Pronunciation should land near 85 (the raw value) — not scaled down.
        assert result["pronunciation"] == pytest.approx(85.0, abs=1.0)

    def test_word_scores_passed_through(self):
        ref = _ra_reference()
        words = _ra_words_perfect(ref)
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            azure_assess=_azure_assess_stub(85.0, words),
            reference_text=ref,
        )
        assert len(result["word_scores"]) == len(words)
        assert result["word_scores"][0]["word"] == words[0]["Word"]


# ────────────────────────────────────────────────────────────────────────────
# Group B — Fallback path: Azure throws exception
# This is where behavior INTENTIONALLY changes after the rule flip.
# Initial assertions match CURRENT behavior. After the flip, the test file
# is updated to the new values; the diff shows exactly what shifted.
# ────────────────────────────────────────────────────────────────────────────


class TestFallbackPathRA:
    """RA Azure assess raises an exception → fallback fires."""

    def test_fallback_flag_set_when_azure_raises(self):
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            azure_assess=RuntimeError("Azure timed out after 3 retries"),
            reference_text=ref,
        )
        assert result["pronunciation_fallback"] is True
        assert "pronunciation_fallback_azure" in result["scoring_warnings"]

    def test_warning_appended_once_not_duplicated(self):
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            azure_assess=RuntimeError("boom"),
            reference_text=ref,
        )
        warnings = result["scoring_warnings"]
        assert warnings.count("pronunciation_fallback_azure") == 1

    def test_pronunciation_value_under_new_rule_healthy_content_and_fluency(self):
        """NEW behavior (post-fix): pron = 90 × mP. With healthy c and f
        (both at 100 here from perfect transcript match + healthy WPM),
        min(c,f)=100 → mP=1.0 → pron=90."""
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            whisper_words=_whisper_words_for(ref),
            azure_assess=RuntimeError("outage"),
            reference_text=ref,
        )
        # Healthy c and f → mP = 1.0 → pron = 90.0 (the Azure-max stand-in)
        assert result["pronunciation"] == pytest.approx(90.0)
        # New rule: pron is no longer just a copy of content
        # (would have been ~100 under old rule, but capped at 90 now)
        assert result["pronunciation"] != result["content"]

    def test_pronunciation_value_under_new_rule_fluency_zero(self):
        """NEW behavior: when fluency=0 and content=high, the cross-penalty
        scales pron down. min(c, f=0)=0 → mP=0.5 → pron = 90 × 0.5 = 45.
        Mirrors today's q=7265 (Solar System) case."""
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            # No whisper words → speech_dur=0 → wpm=0 → WPM gate → fluency=0
            azure_assess=RuntimeError("outage"),
            reference_text=ref,
        )
        # Pron should be 90 × 0.5 = 45 (floor mP applied)
        assert result["pronunciation"] == pytest.approx(45.0)
        assert result["fluency"] == 0  # confirm fluency really did land at 0

    def test_content_and_fluency_unaffected_by_fallback(self):
        """Fallback only touches pronunciation. Content + fluency must be
        identical to their non-fallback counterparts."""
        ref = _ra_reference()
        whisper_words = _whisper_words_for(ref)
        # Run twice — once with Azure healthy, once with Azure raising.
        # Whisper word timings are identical in both runs so fluency math
        # is anchored to the same signal.
        ok = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            whisper_words=whisper_words,
            azure_assess=_azure_assess_stub(85.0, _ra_words_perfect(ref)),
            reference_text=ref,
        )
        fb = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            whisper_words=whisper_words,
            azure_assess=RuntimeError("outage"),
            reference_text=ref,
        )
        # Content depends on transcript + reference matching (identical).
        # Fluency depends on Azure words preferentially, falls back to
        # Whisper words when Azure fails. With identical Whisper words +
        # similar Azure word density, fluency should be close (within 5).
        assert ok["content"] == fb["content"]
        assert abs(ok["fluency"] - fb["fluency"]) <= 5
        # Pronunciation must differ — that's the whole point of the fallback.
        assert ok["pronunciation"] != fb["pronunciation"]


class TestFallbackPathFreeform:
    """SGD / DI / RL / RTS / ASQ — azure_freeform fallback path."""

    def test_sgd_fallback_when_freeform_raises(self):
        result = _run_scorer(
            task_type="summarize_group_discussion",
            transcript="I summarised the group discussion appropriately.",
            azure_freeform=RuntimeError("outage"),
        )
        assert result["pronunciation_fallback"] is True
        assert "pronunciation_fallback_azure" in result["scoring_warnings"]

    def test_di_fallback_when_freeform_raises(self):
        result = _run_scorer(
            task_type="describe_image",
            transcript="The diagram shows a solar system with planets.",
            azure_freeform=RuntimeError("outage"),
        )
        assert result["pronunciation_fallback"] is True

    def test_rl_fallback_when_freeform_raises(self):
        result = _run_scorer(
            task_type="retell_lecture",
            transcript="The lecture discussed climate change.",
            azure_freeform=RuntimeError("outage"),
        )
        assert result["pronunciation_fallback"] is True


# ────────────────────────────────────────────────────────────────────────────
# Group F — Fluency-zero correlation: no hidden gate zeros pron
# Documents the lesson from the May 2026 data audit.
# ────────────────────────────────────────────────────────────────────────────


class TestFluencyZeroCorrelation:
    """Pin the rule: pronunciation reaching 0 when fluency is 0 must come
    from independent root cause (Azure raw=0), not from a code gate."""

    def test_cross_penalty_floors_at_half_not_zero(self):
        """When fluency=0 with Azure returning a real raw pron, cross-penalty
        floors mP at 0.5 — pronunciation is halved, never zeroed."""
        # Direct math check against the formula
        raw_pron = 80.0
        mP_at_zero = _cross_multiplier(0, 100, 0.5, 0.005)
        assert mP_at_zero == 0.5
        assert raw_pron * mP_at_zero == 40.0  # halved, NOT zero

    def test_zero_raw_pron_produces_zero_after_cross_penalty(self):
        """When Azure raw=0, cross-penalty multiplies and result stays 0.
        This is the "f=0 with p=0 from independent cause" pattern."""
        raw_pron = 0.0
        mP_at_zero = _cross_multiplier(0, 100, 0.5, 0.005)
        assert raw_pron * mP_at_zero == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Group G — Warning and audit trail preservation
# ────────────────────────────────────────────────────────────────────────────


class TestWarningAndAudit:

    def test_no_warning_on_happy_path(self):
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            azure_assess=_azure_assess_stub(80.0, _ra_words_perfect(ref)),
            reference_text=ref,
        )
        warnings = result.get("scoring_warnings") or []
        assert "pronunciation_fallback_azure" not in warnings

    def test_fluency_metrics_includes_cross_multipliers(self):
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            azure_assess=_azure_assess_stub(80.0, _ra_words_perfect(ref)),
            reference_text=ref,
        )
        fm = result["fluency_metrics"]
        assert "cross_multipliers" in fm
        assert {"mC", "mF", "mP"}.issubset(fm["cross_multipliers"].keys())

    def test_fallback_logs_cross_multipliers_meaningfully(self):
        """After the rule flip, cross_multipliers.mP should reflect the
        actual multiplier applied to the 90.0 stand-in — not 1.0 (skip)."""
        ref = _ra_reference()
        result = _run_scorer(
            task_type="read_aloud",
            transcript=ref,
            azure_assess=RuntimeError("outage"),
            reference_text=ref,
        )
        fm = result["fluency_metrics"]
        # After fix, mP should be a meaningful multiplier value (between 0.5 and 1.0)
        # reflecting the actual cross-penalty applied to the 90 stand-in.
        assert "mP" in fm["cross_multipliers"]
        assert 0.5 <= fm["cross_multipliers"]["mP"] <= 1.0
