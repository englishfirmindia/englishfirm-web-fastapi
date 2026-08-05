"""Regression tests for the SGD Azure‖Whisper parallelization in _score_speaking_v2.

Context
-------
For `summarize_group_discussion` the Azure free-form call
(`transcribe_and_score_free`) and the Whisper word-timestamp call
(`transcribe_with_whisper_words`) are independent — both consume only the raw
audio and neither reads the other's result. They used to run back-to-back, so
the user waited Azure + Whisper. The optimization starts Whisper in a daemon
thread *before* the Azure call so the two overlap; net latency becomes
max(Azure, Whisper). The content LLM still runs afterwards because it depends
on the resolved transcript.

The change is SCOPED TO SGD ONLY. Every other task type (including the other
azure_freeform types like describe_image) keeps the original sequential path.

These tests pin:
  * Both providers are still invoked exactly once for SGD (no dropped call).
  * SGD: Whisper genuinely overlaps Azure (Whisper starts before Azure ends).
  * Non-SGD freeform (describe_image): Azure and Whisper stay SEQUENTIAL —
    proves the parallelization did not leak to other task types.
  * A Whisper failure propagates identically to the synchronous call it
    replaced (exception semantics preserved).
  * The scored result is well-formed and consumes the Whisper transcript.

Numeric scoring equivalence is pinned by the existing
`tests/scoring/test_pronunciation_fallback.py` suite (SGD/freeform paths),
which must continue to pass unchanged alongside this file.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from unittest.mock import patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.speaking_scorer import _score_speaking_v2  # noqa: E402
from services.scoring.speaking_config_service import SpeakingScoringConfig  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Config builder — freeform types (SGD / DI): llm_keypoints + azure_freeform.
# Mirrors the real pte_speaking_scoring_config row so the scorer doesn't fall
# back to RA defaults (which would route freeform tasks through
# azure_assessment and skip the code path under test).
# ────────────────────────────────────────────────────────────────────────────


def _freeform_cfg(task_type: str) -> SpeakingScoringConfig:
    return SpeakingScoringConfig(
        task_type=task_type,
        wpm_floor=50.0 if task_type == "summarize_group_discussion" else 80.0,
        wpm_ceiling=270.0,
        wpm_plateau_low=130.0,
        wpm_plateau_high=220.0,
        wpm_slope_per_wpm=2.0,
        wpm_peak_score=100.0,
        pause_min_ms=500,
        pause_leading_tol_ms=200,
        pause_trailing_tol_ms=200,
        silence_thresh_dbfs=-30.0,
        content_insertion_penalty_k=2.0,
        pause_penalty_max_pauses=10,
        pause_penalty_sentence_clamp_min=1,
        pause_penalty_sentence_clamp_max=10,
        pause_penalty_formula_constant=11,
        cross_penalty_healthy_threshold=20.0,
        cross_penalty_floor_multiplier=0.5,
        cross_penalty_slope=0.025,
        uses_cross_penalty=True,
        content_method="llm_keypoints",
        uses_reference_text=False,
        pronunciation_source="azure_freeform",
        pause_penalty_max_pauses_mult=2.0,
    )


# ────────────────────────────────────────────────────────────────────────────
# Timed stubs — record call order + timestamps so overlap is provable
# deterministically (no wall-clock threshold assertions).
# ────────────────────────────────────────────────────────────────────────────


class _TimedStubs:
    def __init__(self, azure_sleep: float = 0.30, whisper_sleep: float = 0.02,
                 whisper_error: Exception | None = None):
        self.azure_sleep = azure_sleep
        self.whisper_sleep = whisper_sleep
        self.whisper_error = whisper_error
        self.azure_calls = 0
        self.whisper_calls = 0
        self.azure_start = None
        self.azure_end = None
        self.whisper_start = None
        self.whisper_end = None
        self._lock = threading.Lock()

    def azure_free(self, audio_bytes):
        with self._lock:
            self.azure_calls += 1
            self.azure_start = time.monotonic()
        time.sleep(self.azure_sleep)
        with self._lock:
            self.azure_end = time.monotonic()
        return {
            "pronunciation": 70.0,
            "word_scores": [{"word": "the", "accuracy": 90, "error_type": "None"}],
            "transcript": "azure fallback transcript",
        }

    def whisper_words(self, audio_bytes):
        with self._lock:
            self.whisper_calls += 1
            self.whisper_start = time.monotonic()
        time.sleep(self.whisper_sleep)
        with self._lock:
            self.whisper_end = time.monotonic()
        if self.whisper_error is not None:
            raise self.whisper_error
        # ~150 wpm, no within-speech pauses.
        words = "the group discussed the benefits of volunteering in the community".split()
        out = []
        cursor = 0.0
        for w in words:
            out.append({"text": w, "start": cursor, "end": cursor + 0.35})
            cursor += 0.4
        return {"transcript": " ".join(words), "words": out}


def _run(task_type: str, stubs: _TimedStubs, key_points=None):
    with patch(
        "services.azure_speech_service.transcribe_and_score_free",
        side_effect=stubs.azure_free,
    ), patch(
        "services.whisper_service.transcribe_with_whisper_words",
        side_effect=stubs.whisper_words,
    ), patch(
        "services.llm_content_scoring_service.score_content_with_llm",
        return_value={"score": 65.0, "scored": True, "reasoning": "test"},
    ), patch(
        "services.speaking_scorer._get_speaking_config",
        return_value=_freeform_cfg(task_type),
    ):
        return _score_speaking_v2(
            user_id=999,
            question_id=42,
            audio_bytes=b"\x00" * 2048,
            task_type=task_type,
            key_points=key_points or ["volunteering has community benefits",
                                      "the group reached a shared conclusion"],
        )


# ────────────────────────────────────────────────────────────────────────────
# Both providers still invoked exactly once for SGD
# ────────────────────────────────────────────────────────────────────────────


class TestBothProvidersInvoked:

    def test_sgd_calls_azure_and_whisper_once_each(self):
        stubs = _TimedStubs(azure_sleep=0.05)
        _run("summarize_group_discussion", stubs)
        assert stubs.azure_calls == 1
        assert stubs.whisper_calls == 1

    def test_sgd_result_consumes_whisper_transcript(self):
        stubs = _TimedStubs(azure_sleep=0.05)
        result = _run("summarize_group_discussion", stubs)
        # Whisper transcript ("...volunteering...") must win over the Azure
        # fallback transcript — proves the parallel result reached downstream.
        assert result["transcript"].startswith("the group discussed")
        # Well-formed score dict.
        for k in ("content", "fluency", "pronunciation"):
            assert k in result and 0.0 <= float(result[k]) <= 100.0


# ────────────────────────────────────────────────────────────────────────────
# SGD: Azure and Whisper overlap (Whisper starts before Azure finishes)
# ────────────────────────────────────────────────────────────────────────────


class TestSgdParallelOverlap:

    def test_whisper_starts_before_azure_ends(self):
        # Azure sleeps 0.3s; Whisper is kicked off first and is quick, so its
        # start timestamp must precede Azure's end timestamp → they overlapped.
        stubs = _TimedStubs(azure_sleep=0.30, whisper_sleep=0.02)
        _run("summarize_group_discussion", stubs)
        assert stubs.whisper_start is not None and stubs.azure_end is not None
        assert stubs.whisper_start < stubs.azure_end, (
            "Whisper did not overlap Azure — parallelization not active for SGD"
        )


# ────────────────────────────────────────────────────────────────────────────
# Scoping: non-SGD freeform (describe_image) stays SEQUENTIAL
# ────────────────────────────────────────────────────────────────────────────


class TestNonSgdStaysSequential:

    def test_describe_image_runs_azure_then_whisper(self):
        # describe_image is also azure_freeform + llm_keypoints, but must NOT
        # be parallelized. Azure runs to completion on the main thread first,
        # so Whisper cannot start until Azure has ended.
        stubs = _TimedStubs(azure_sleep=0.20, whisper_sleep=0.02)
        _run("describe_image", stubs)
        assert stubs.azure_calls == 1 and stubs.whisper_calls == 1
        assert stubs.whisper_start >= stubs.azure_end, (
            "describe_image overlapped Azure/Whisper — parallelization leaked "
            "beyond SGD"
        )


# ────────────────────────────────────────────────────────────────────────────
# Exception semantics preserved: Whisper failure propagates for SGD
# ────────────────────────────────────────────────────────────────────────────


class TestWhisperExceptionPropagates:

    def test_sgd_whisper_error_raises_verbatim(self):
        boom = RuntimeError("whisper API hung")
        stubs = _TimedStubs(azure_sleep=0.02, whisper_error=boom)
        with pytest.raises(RuntimeError, match="whisper API hung"):
            _run("summarize_group_discussion", stubs)

    def test_azure_failure_still_falls_back_not_aborts(self):
        # Azure freeform raising is caught (pronunciation_fallback), unchanged
        # by the parallelization — scoring still completes off the Whisper
        # transcript.
        def _azure_boom(audio_bytes):
            raise RuntimeError("azure outage")

        stubs = _TimedStubs(azure_sleep=0.02)
        with patch(
            "services.azure_speech_service.transcribe_and_score_free",
            side_effect=_azure_boom,
        ), patch(
            "services.whisper_service.transcribe_with_whisper_words",
            side_effect=stubs.whisper_words,
        ), patch(
            "services.llm_content_scoring_service.score_content_with_llm",
            return_value={"score": 65.0, "scored": True, "reasoning": "test"},
        ), patch(
            "services.speaking_scorer._get_speaking_config",
            return_value=_freeform_cfg("summarize_group_discussion"),
        ):
            result = _score_speaking_v2(
                user_id=999,
                question_id=42,
                audio_bytes=b"\x00" * 2048,
                task_type="summarize_group_discussion",
                key_points=["a", "b"],
            )
        assert result["pronunciation_fallback"] is True
        assert result["transcript"].startswith("the group discussed")
