"""Per-question enrichment for sectional + mock review screens.

The bare answer rows persisted in `attempt_answers` (score, result_json,
user_answer_json) aren't enough for the student feedback / trainer review
screens to show the full question detail. They need the passage / options /
correct answers / source audio URL alongside.

`enrich_answer_for_review(q, attempt_answer)` packs everything the frontend
needs into a single dict that mirrors the rich shape the speaking sectional
already returns. Pure function — does not write to DB.

Output dict shape:
    {
      "question_id":    int,
      "question_type":  str,
      "score":          int | None,
      "scoring_status": str,
      "user_answer_json": dict,
      "result_json":    dict,
      "content_json":   dict,   # passage / options / wordBank / contentBlocks
      "correct":        dict,   # extracted correctAnswers per question type
      "audio_url":      str | None,   # presigned, for listening / RA-like
    }
"""
from typing import Optional

from services.session_service import enrich_content_json
from services.s3_service import generate_presigned_url


def _extract_correct(q) -> dict:
    """Pull correctAnswers out of evaluation_json into a frontend-friendly
    dict. Shape depends on question_type. Returns {} when evaluation is
    absent (e.g. mid-attempt, missing eval row)."""
    if not (q.evaluation and q.evaluation.evaluation_json):
        return {}
    eval_json = q.evaluation.evaluation_json or {}
    ca = eval_json.get("correctAnswers") or {}
    qt = q.question_type
    out: dict = {}

    # FIB family — blank_id → correct_value
    if qt in (
        "reading_fib", "reading_fib_drop_down", "reading_drag_and_drop",
        "listening_fib",
    ):
        # Two backend shapes exist:
        #   1. {"blank_1": "word1", "blank_2": "word2", ...}    (reading)
        #   2. {"blanks": [{"blankId": 1, "answer": "word1"}, ...]}  (listening)
        if isinstance(ca, dict) and ca.get("blanks") is not None:
            out["blanks"] = {
                str(b.get("blankId")): b.get("answer")
                for b in (ca.get("blanks") or [])
            }
        else:
            out["blanks"] = {str(k): v for k, v in ca.items()}
        # Word bank for drag-and-drop variants
        wb = (q.content_json or {}).get("wordBank")
        if wb:
            out["word_bank"] = wb

    # MCQ single (MCS-R/L, HCS, SMW)
    elif qt in ("reading_mcs", "listening_mcs", "listening_hcs",
                "listening_smw", "mcq_single", "listening_mcq_single"):
        out["correct_option"] = ca.get("correctOption") or ca.get("correct")

    # MCQ multiple
    elif qt in ("reading_mcm", "listening_mcm", "mcq_multiple",
                "listening_mcq_multiple"):
        out["correct_options"] = ca.get("correctOptions") or ca.get("correct") or []

    # Reorder Paragraphs
    elif qt == "reorder_paragraphs":
        out["correct_sequence"] = ca.get("correctSequence") or []

    # Highlight Incorrect Words
    elif qt in ("listening_hiw", "highlight_incorrect_words"):
        # Two shapes — list of strings OR list of dicts {wrong, correct, index}
        raw = ca.get("incorrectWords") or []
        if raw and isinstance(raw[0], dict):
            out["incorrect_words"] = [w.get("wrong") for w in raw if isinstance(w, dict)]
            out["corrections"] = {
                str(w.get("index", "")): w.get("correct")
                for w in raw if isinstance(w, dict) and w.get("correct")
            }
        else:
            out["incorrect_words"] = [str(w) for w in raw]

    # SST / SWT / WE — store the source transcript/passage if available
    elif qt in ("summarize_spoken_text", "listening_sst",
                "summarize_written_text", "write_essay"):
        # Source text is usually in content_json.passage / .transcript already
        # (passes through via content_json) so nothing extra here.
        keypoints = ca.get("keyPoints") or ca.get("key_points") or []
        if keypoints:
            out["key_points"] = keypoints

    # WFD — exact target sentence
    elif qt in ("listening_wfd",):
        out["expected_text"] = ca.get("text") or ca.get("expected") or ""

    return out


def _maybe_presigned_audio(q) -> Optional[str]:
    """Best-effort presign for any audio reference on the question. Listening
    + speaking-style questions carry the audio under `audio_url` or
    `audio_s3_key`; returns None silently if presigning fails."""
    content = q.content_json or {}
    raw = (
        content.get("audio_url")
        or content.get("audio_s3_key")
        or content.get("s3_key")
    )
    if not raw:
        return None
    try:
        return generate_presigned_url(raw)
    except Exception:
        return None


def enrich_answer_for_review(q, attempt_answer) -> dict:
    """Build the rich per-question review payload."""
    return {
        "question_id":      attempt_answer.question_id,
        "question_type":    attempt_answer.question_type,
        "score":            attempt_answer.score,
        "scoring_status":   attempt_answer.scoring_status,
        "user_answer_json": attempt_answer.user_answer_json or {},
        "result_json":      attempt_answer.result_json or {},
        "content_json":     enrich_content_json(q) if q else {},
        "correct":          _extract_correct(q) if q else {},
        "audio_url":        _maybe_presigned_audio(q) if q else None,
    }
