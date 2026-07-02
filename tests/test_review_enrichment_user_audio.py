"""Regression tests for the 2026-07-03 sectional-review audio fix.

The bug: `enrich_answer_for_review()` had a single `audio_url` field that
carried the STIMULUS URL (question's lecture audio). The frontend
`SpeakingAttemptCard.userAudioUrl` prop played that back as if it were
the student's own recording, so students heard the lecturer's voice
instead of their own on sectional review screens (surfaced when
jishanranjit@gmail.com's listening sectional attempt 8066 played the
biotech lecture in place of her Retell Lecture recording).

The fix splits into two explicit fields:
  - `audio_url`       = STIMULUS  (unchanged for backwards compat)
  - `user_audio_url`  = STUDENT recording  (new)

These tests pin that behaviour so it can't regress silently.
"""
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from services.review_enrichment import (
    _maybe_presigned_audio,
    _maybe_presigned_user_audio,
    enrich_answer_for_review,
)


def _fake_presign(url, expires_in=None):  # noqa: ARG001
    return f"PRESIGNED({url})"


# ── _maybe_presigned_audio (stimulus) ──────────────────────────────────────

@patch("services.review_enrichment.generate_presigned_url", side_effect=_fake_presign)
def test_stimulus_helper_returns_presigned_stimulus(_):
    q = MagicMock()
    q.content_json = {
        "audio_url": "https://apeuni-questions-audio.s3.ap-southeast-2.amazonaws.com/rl/rl_0242.mp3"
    }
    assert _maybe_presigned_audio(q) == (
        "PRESIGNED(https://apeuni-questions-audio.s3.ap-southeast-2.amazonaws.com/rl/rl_0242.mp3)"
    )


@patch("services.review_enrichment.generate_presigned_url", side_effect=_fake_presign)
def test_stimulus_helper_none_when_no_stimulus(_):
    q = MagicMock()
    q.content_json = {}
    assert _maybe_presigned_audio(q) is None


# ── _maybe_presigned_user_audio (student recording) ────────────────────────

@patch("services.review_enrichment.generate_presigned_url", side_effect=_fake_presign)
def test_user_audio_helper_returns_presigned_student_recording(_):
    ans = MagicMock()
    ans.audio_url = (
        "https://apeuni-user-recordings.s3.ap-southeast-2.amazonaws.com/"
        "recordings/123/retell_lecture/8703/e2001d58-ee24-4e04-8aa7-33b3fffaf133.aac"
    )
    got = _maybe_presigned_user_audio(ans)
    assert got == (
        "PRESIGNED(https://apeuni-user-recordings.s3.ap-southeast-2.amazonaws.com/"
        "recordings/123/retell_lecture/8703/e2001d58-ee24-4e04-8aa7-33b3fffaf133.aac)"
    )


@patch("services.review_enrichment.generate_presigned_url", side_effect=_fake_presign)
def test_user_audio_helper_none_when_row_has_no_audio(_):
    ans = MagicMock()
    ans.audio_url = None
    assert _maybe_presigned_user_audio(ans) is None


@patch("services.review_enrichment.generate_presigned_url", side_effect=RuntimeError("bad"))
def test_user_audio_helper_swallows_presign_errors(_):
    ans = MagicMock()
    ans.audio_url = "s3://apeuni-user-recordings/broken/key.aac"
    # Must not raise — reviews should render even if presigning is degraded.
    assert _maybe_presigned_user_audio(ans) is None


# ── enrich_answer_for_review — end-to-end shape ────────────────────────────

@patch("services.review_enrichment.generate_presigned_url", side_effect=_fake_presign)
@patch("services.review_enrichment.enrich_content_json", return_value={"passage": "…"})
def test_enrich_returns_stimulus_and_user_audio_separately(_e, _p):
    """The core assertion: `audio_url` = stimulus, `user_audio_url` = student
    recording. Frontends showing "play my recording" must read the second."""
    q = MagicMock()
    q.content_json = {
        "audio_url": "https://apeuni-questions-audio.s3.ap-southeast-2.amazonaws.com/rl/rl_0242.mp3"
    }
    q.evaluation = None
    q.question_type = "retell_lecture"

    ans = MagicMock()
    ans.question_id = 8703
    ans.question_type = "retell_lecture"
    ans.score = 40
    ans.scoring_status = "complete"
    ans.user_answer_json = {}
    ans.result_json = {"transcript": "The lecture discusses biotech…"}
    ans.audio_url = (
        "https://apeuni-user-recordings.s3.ap-southeast-2.amazonaws.com/"
        "recordings/123/retell_lecture/8703/e2001d58-ee24-4e04-8aa7-33b3fffaf133.aac"
    )

    out = enrich_answer_for_review(q, ans)
    assert "audio_url" in out
    assert "user_audio_url" in out
    assert out["audio_url"] == "PRESIGNED(https://apeuni-questions-audio.s3.ap-southeast-2.amazonaws.com/rl/rl_0242.mp3)", (
        "audio_url must remain the STIMULUS URL for backwards compatibility"
    )
    assert out["user_audio_url"] == (
        "PRESIGNED(https://apeuni-user-recordings.s3.ap-southeast-2.amazonaws.com/"
        "recordings/123/retell_lecture/8703/e2001d58-ee24-4e04-8aa7-33b3fffaf133.aac)"
    ), "user_audio_url must be the STUDENT's recording"
    # The two must NEVER be equal — if a future refactor collapses them
    # this assertion catches the regression that broke jishan's review.
    assert out["audio_url"] != out["user_audio_url"]


@patch("services.review_enrichment.generate_presigned_url", side_effect=_fake_presign)
@patch("services.review_enrichment.enrich_content_json", return_value={})
def test_enrich_user_audio_none_for_non_speaking_row(_e, _p):
    """WFD / listening_fib / MCQ rows have no `attempt_answer.audio_url` —
    ensure user_audio_url is None (not the stimulus by accident)."""
    q = MagicMock()
    q.content_json = {
        "audio_url": "https://apeuni-questions-audio.s3.ap-southeast-2.amazonaws.com/wfd/1.mp3"
    }
    q.evaluation = None
    q.question_type = "listening_wfd"
    ans = MagicMock()
    ans.question_id = 1
    ans.question_type = "listening_wfd"
    ans.score = 50
    ans.scoring_status = "complete"
    ans.user_answer_json = {"text": "…"}
    ans.result_json = {}
    ans.audio_url = None

    out = enrich_answer_for_review(q, ans)
    assert out["user_audio_url"] is None
    assert out["audio_url"] is not None, "WFD still needs stimulus for playback of dictated sentence"


@patch("services.review_enrichment.generate_presigned_url", side_effect=_fake_presign)
@patch("services.review_enrichment.enrich_content_json", return_value={})
def test_enrich_handles_missing_question_row(_e, _p):
    """Defensive: if the questions_from_apeuni row was deleted after the
    student attempted it, enrich still returns a well-formed dict rather
    than raising — student's recording still surfaces from the answer row."""
    ans = MagicMock()
    ans.question_id = 999
    ans.question_type = "retell_lecture"
    ans.score = None
    ans.scoring_status = "complete"
    ans.user_answer_json = {}
    ans.result_json = {}
    ans.audio_url = "https://apeuni-user-recordings.s3.ap-southeast-2.amazonaws.com/recordings/1/rl/1/a.aac"

    out = enrich_answer_for_review(None, ans)
    assert out["audio_url"] is None
    assert out["user_audio_url"] == "PRESIGNED(https://apeuni-user-recordings.s3.ap-southeast-2.amazonaws.com/recordings/1/rl/1/a.aac)"
    assert out["content_json"] == {}
    assert out["correct"] == {}
