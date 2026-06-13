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
import re
from typing import Optional

from services.session_service import enrich_content_json
from services.s3_service import generate_presigned_url


def _hiw_norm(w: str) -> str:
    """HIW word normaliser — mirrors the submit endpoint (routers/listening/hiw.py)
    so review-side index lookup matches what the scorer indexed at submit time."""
    return re.sub(r"[^\w]", "", str(w)).lower()


def _hiw_passage_tokens(q) -> list:
    """Tokenise an HIW passage the same way the submit endpoint does, so
    the indices computed below align with the scorer's incorrect_word_indices."""
    content = q.content_json or {}
    pre = content.get("words")
    if isinstance(pre, list) and pre:
        return [str(w) for w in pre]
    text = (
        content.get("transcript")
        or content.get("passage")
        or content.get("text")
        or ""
    )
    return [w for w in str(text).split() if w]


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
        wrong_to_correct: dict = {}
        if raw and isinstance(raw[0], dict):
            out["incorrect_words"] = [w.get("wrong") for w in raw if isinstance(w, dict)]
            out["corrections"] = {
                str(w.get("index", "")): w.get("correct")
                for w in raw if isinstance(w, dict) and w.get("correct")
            }
            for w in raw:
                if isinstance(w, dict) and w.get("wrong") and w.get("correct"):
                    wrong_to_correct[_hiw_norm(w["wrong"])] = str(w["correct"]).strip()
        else:
            out["incorrect_words"] = [str(w) for w in raw]

        # Positional index path — eliminates string-match over-highlighting
        # when the same word appears multiple times in the passage. Mirrors
        # the submit-time computation so frontend renders match the scorer.
        tokens = _hiw_passage_tokens(q)
        incorrect_norm_set = {_hiw_norm(w) for w in out["incorrect_words"] if w}
        incorrect_indices = [
            i for i, t in enumerate(tokens) if _hiw_norm(t) in incorrect_norm_set
        ]
        out["passage_tokens"] = tokens
        out["incorrect_word_indices"] = incorrect_indices
        if wrong_to_correct:
            out["corrections_by_index"] = {
                str(i): wrong_to_correct[_hiw_norm(t)]
                for i, t in enumerate(tokens)
                if _hiw_norm(t) in wrong_to_correct
            }

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


def compute_time_taken_seconds(attempt, answers_ordered) -> dict:
    """Derive per-question "time spent" from submitted_at deltas.

    Q1 = submitted_at[0] - attempt.started_at
    Qn = submitted_at[n] - submitted_at[n-1]

    No per-Q stopwatch is persisted today, so this is the cheapest fallback
    that works retroactively for every historical attempt. Returns a dict
    keyed by AttemptAnswer.id → int seconds (clamped to [0, 7200] to drop
    pause-the-tab outliers). Answers missing submitted_at map to None.

    Caller is responsible for ordering `answers_ordered` by submitted_at
    (which all the get_*_sectional_results functions already do).
    """
    out: dict = {}
    prev = getattr(attempt, "started_at", None)
    for a in answers_ordered:
        ts = getattr(a, "submitted_at", None)
        if ts is None or prev is None:
            out[a.id] = None
            if ts is not None:
                prev = ts
            continue
        try:
            delta = (ts - prev).total_seconds()
        except Exception:
            out[a.id] = None
            prev = ts
            continue
        if delta < 0:
            delta = 0
        # Cap absurd outliers (paused tab, browser-suspended session). Two
        # hours is well beyond any single PTE question's natural ceiling
        # (WE is the longest at 20 min total). Beyond this, render the cell
        # as `—` rather than misleading the user with a 14h figure.
        if delta > 7200:
            out[a.id] = None
        else:
            out[a.id] = int(delta)
        prev = ts
    return out
