"""Regression guard for the 2026-07-10 sectional-speaking audio-playback bug.

The 2026-07-03 refactor split the review payload's audio field into two:
  - `audio_url`      → stimulus (lecturer / prompt voice)
  - `user_audio_url` → student's own recording

The frontend on sectional surfaces now refuses to play `audio_url` as the
student's voice — that was the whole point of the split. But
`speaking_sectional_service.py` and `mock_service.py` build their review
payloads inline and were not migrated, so the student audio was shipped
under the old (now-wrong) `audio_url` key. Reviewers saw transcript +
scores but no play button, which is what triggered this fix.

This test pins the payload shape at the source-code level (no DB /
fixture dependency): both services must reference `user_audio_url` in
their review payloads. Anyone reverting the fix will break this test.
"""
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def test_speaking_sectional_review_ships_user_audio_url():
    src = (REPO / "services/speaking_sectional_service.py").read_text()
    # The build-payload block builds one dict per question inside
    # `get_speaking_sectional_results`. The dict must carry the student's
    # recording under the new key.
    assert '"user_audio_url"' in src, (
        "speaking_sectional_service.py must ship the student's recording "
        "as `user_audio_url` (see services/review_enrichment.py convention). "
        "The frontend on sectional surfaces refuses to play the legacy "
        "`audio_url` key as the student's voice."
    )


def test_mock_review_ships_user_audio_url():
    src = (REPO / "services/mock_service.py").read_text()
    assert '"user_audio_url"' in src, (
        "mock_service.py must ship the student's recording as "
        "`user_audio_url` (matches sectional + review_enrichment.py). "
        "Mock's current frontend also reads `audio_url` for backward "
        "compat, but the primary key should be `user_audio_url`."
    )


def test_review_enrichment_still_emits_both_keys():
    """The canonical shape lives in enrich_answer_for_review — sanity check
    it still emits both `audio_url` (stimulus) and `user_audio_url`
    (student). Reading/writing/listening sectional review depends on this
    function ; if either key disappears those surfaces break too."""
    src = (REPO / "services/review_enrichment.py").read_text()
    assert '"audio_url":' in src, "review_enrichment must emit `audio_url` (stimulus)."
    assert '"user_audio_url":' in src, "review_enrichment must emit `user_audio_url` (student)."
