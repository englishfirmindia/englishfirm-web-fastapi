"""
Listening Sectional Service
============================
Selects questions for the listening sectional exam (12 task types).
Includes 4 speaking tasks that contribute to the PTE listening score,
plus 8 pure listening tasks.

Speaking tasks (async Azure scored):
  repeat_sentence:            17%
  retell_lecture:             13%
  summarize_group_discussion: 20%
  answer_short_question:       4%

Pure listening tasks (sync scored at submit):
  summarize_spoken_text:      10%
  listening_wfd:              13%
  listening_fib:               8%
  highlight_incorrect_words:   8%
  listening_hcs:               2%
  listening_mcq_multiple:      3%
  listening_mcq_single:        2%
  listening_smw:               1%

PTE formula (CLAUDE.md guardrail):
  pte_score = max(10, min(90, round(10 + weighted_pct * 80)))
"""

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.orm import Session, joinedload

from db.models import QuestionFromApeuni, UserQuestionAttempt, PracticeAttempt, AttemptAnswer
from services.session_service import ACTIVE_SESSIONS
from services.s3_service import generate_presigned_url
import core.config as config

from core.logging_config import get_logger

log = get_logger(__name__)


# ─── Sectional structure ──────────────────────────────────────────────────────
# Speaking tasks come first (PTE exam ordering).
LISTENING_STRUCTURE = [
    # ── Speaking tasks (contribute to listening score) ────────────────────────
    {"task": "repeat_sentence",            "count": 3, "module": "speaking", "time_seconds":  21, "prep_seconds": 0,  "rec_seconds": 9},
    {"task": "retell_lecture",             "count": 1, "module": "speaking", "time_seconds": 125, "prep_seconds": 10, "rec_seconds": 40},
    {"task": "summarize_group_discussion", "count": 1, "module": "speaking", "time_seconds": 125, "prep_seconds": 3,  "rec_seconds": 120},
    {"task": "answer_short_question",      "count": 2, "module": "speaking", "time_seconds":  19, "prep_seconds": 0,  "rec_seconds": 5},
    # ── Pure listening tasks (sync scored) ────────────────────────────────────
    {"task": "summarize_spoken_text",      "count": 1, "module": "listening", "time_seconds": 675, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_mcq_multiple",     "count": 2, "module": "listening", "time_seconds":  90, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_fib",              "count": 2, "module": "listening", "time_seconds": 105, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_hcs",              "count": 2, "module": "listening", "time_seconds": 135, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_smw",              "count": 1, "module": "listening", "time_seconds":  90, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "highlight_incorrect_words",  "count": 2, "module": "listening", "time_seconds": 120, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_mcq_single",       "count": 1, "module": "listening", "time_seconds":  90, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_wfd",              "count": 3, "module": "listening", "time_seconds":  53, "prep_seconds": 0, "rec_seconds": 0},
]

_LISTENING_WEIGHTS = {
    "repeat_sentence":            17,
    "retell_lecture":             13,
    "summarize_group_discussion": 20,
    "answer_short_question":       4,
    "summarize_spoken_text":      10,
    "listening_wfd":              13,
    "listening_fib":               8,
    "highlight_incorrect_words":   8,
    "listening_hcs":               2,
    "listening_mcq_multiple":      3,
    "listening_mcq_single":        2,
    "listening_smw":               1,
}

_ASYNC_TYPES = {
    "repeat_sentence", "retell_lecture",
    "summarize_group_discussion", "answer_short_question",
}

# Max raw score fallbacks for sync types
_MAX_FALLBACK = {
    "summarize_spoken_text":      10,
    "listening_wfd":               7,
    "listening_fib":               4,
    "highlight_incorrect_words":   3,
    "listening_hcs":               1,
    "listening_mcq_multiple":      2,
    "listening_mcq_single":        1,
    "listening_smw":               1,
}

_DISPLAY_NAMES = {
    "repeat_sentence":            "Repeat Sentence",
    "retell_lecture":             "Re-tell Lecture",
    "summarize_group_discussion": "Summarize Group Discussion",
    "answer_short_question":      "Answer Short Question",
    "summarize_spoken_text":      "Summarize Spoken Text",
    "listening_wfd":              "Write from Dictation",
    "listening_fib":              "Fill in the Blanks",
    "highlight_incorrect_words":  "Highlight Incorrect Words",
    "listening_hcs":              "Highlight Correct Summary",
    "listening_mcq_multiple":     "MCQ – Multiple Answers",
    "listening_mcq_single":       "MCQ – Single Answer",
    "listening_smw":              "Select Missing Word",
}

_ANSWER_TYPES = {
    "repeat_sentence":            "speaking",
    "retell_lecture":             "speaking",
    "summarize_group_discussion": "speaking",
    "answer_short_question":      "speaking",
    "summarize_spoken_text":      "text",
    "listening_wfd":              "text",
    "listening_fib":              "fib",
    "highlight_incorrect_words":  "hiw",
    "listening_hcs":              "mcs",
    "listening_mcq_multiple":     "mcm",
    "listening_mcq_single":       "mcs",
    "listening_smw":              "smw",
}


def _display_name(task_type: str) -> str:
    return _DISPLAY_NAMES.get(task_type, task_type.replace("_", " ").title())


def _question_max(q) -> int:
    """Return max achievable raw score for one sync question."""
    qt = q.question_type
    try:
        ev    = q.evaluation.evaluation_json
        rules = ev.get("scoringRules", {})
        ans   = ev.get("correctAnswers", {})

        if qt == "listening_fib":
            blanks = ans.get("blanks", [])
            return len(blanks) * rules.get("marksPerCorrect", 1)

        if qt == "highlight_incorrect_words":
            incorrect_words = ans.get("incorrectWords", [])
            return len(incorrect_words) * rules.get("correctClick", 1)

        if qt in ("listening_hcs", "listening_smw", "listening_mcq_single"):
            return rules.get("marksPerCorrect", 1)

        if qt == "listening_mcq_multiple":
            return len(ans.get("correctOptions", [])) * rules.get("marksPerCorrect", 1)

        if qt == "listening_wfd":
            transcript = ans.get("transcript", "")
            return len(transcript.split()) if transcript else _MAX_FALLBACK.get(qt, 7)

    except Exception:
        pass

    return _MAX_FALLBACK.get(qt, 1)


def get_listening_sectional_info() -> dict:
    total_questions = sum(t["count"] for t in LISTENING_STRUCTURE)
    total_minutes   = sum(t["count"] * t["time_seconds"] / 60 for t in LISTENING_STRUCTURE)
    sections = [
        {
            "task":         t["task"],
            "display_name": _display_name(t["task"]),
            "count":        t["count"],
            "weight_pct":   _LISTENING_WEIGHTS.get(t["task"], 0),
        }
        for t in LISTENING_STRUCTURE
    ]
    return {
        "total_questions": total_questions,
        "total_minutes":   round(total_minutes, 1),
        "sections":        sections,
    }


def start_listening_sectional_exam(db: Session, user_id: int, test_number: int) -> dict:
    """Select questions, create session, return question list with presigned audio URLs.

    Question set is locked per (module, test_number) in
    `sectional_test_questions` — every user, every redo, same questions.
    """
    from services.sectional_test_catalog import get_locked_question_ids

    locked_ids = get_locked_question_ids(db, "listening", test_number)
    rows_by_id = {
        q.question_id: q
        for q in db.query(QuestionFromApeuni)
        .options(joinedload(QuestionFromApeuni.evaluation))
        .filter(QuestionFromApeuni.question_id.in_(locked_ids))
        .all()
    }
    selected_qs = [rows_by_id[qid] for qid in locked_ids if qid in rows_by_id]
    missing = [qid for qid in locked_ids if qid not in rows_by_id]
    if missing:
        log.warning(
            "[Listening Sectional] %d locked question_ids missing for test %d: %s",
            len(missing), test_number, missing,
        )

    if not selected_qs:
        raise HTTPException(status_code=404, detail="No listening questions available")

    session_id = str(uuid.uuid4())
    task_timing = {t["task"]: t for t in LISTENING_STRUCTURE}

    questions_payload = []
    for q in selected_qs:
        timing = task_timing.get(q.question_type, {"time_seconds": 120, "prep_seconds": 0, "rec_seconds": 0})

        presigned_url: Optional[str] = None
        raw_audio = (
            (q.content_json or {}).get("audio_url")
            or (q.content_json or {}).get("s3_key")
            or (q.content_json or {}).get("audio_s3_key")
        )
        if raw_audio:
            try:
                presigned_url = generate_presigned_url(raw_audio)
            except Exception:
                presigned_url = None

        questions_payload.append({
            "question_id":   q.question_id,
            "task_type":     q.question_type,
            "display_name":  _display_name(q.question_type),
            "answer_type":   _ANSWER_TYPES.get(q.question_type, "text"),
            "time_seconds":  timing["time_seconds"],
            "prep_seconds":  timing["prep_seconds"],
            "rec_seconds":   timing["rec_seconds"],
            "content_json":  q.content_json,
            "presigned_url": presigned_url,
            "session_id":    session_id,
            "is_prediction": q.is_prediction,
        })

    # Remove any incomplete previous attempts so history stays clean
    db.query(PracticeAttempt).filter(
        PracticeAttempt.user_id        == user_id,
        PracticeAttempt.module         == "listening",
        PracticeAttempt.question_type  == "sectional",
        PracticeAttempt.status         != "complete",
    ).delete(synchronize_session=False)
    db.commit()

    # Create PracticeAttempt now so attempt_id is available at every /submit
    attempt = PracticeAttempt(
        user_id               = user_id,
        session_id            = session_id,
        module                = "listening",
        question_type         = "sectional",
        filter_type           = "sectional",
        total_questions       = len(selected_qs),
        total_score           = 0,
        questions_answered    = 0,
        status                = "pending",
        scoring_status        = "pending",
        selected_question_ids = [q.question_id for q in selected_qs],
        task_breakdown        = {"test_number": test_number},
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    ACTIVE_SESSIONS[session_id] = {
        "user_id":              user_id,
        "attempt_id":           attempt.id,
        "test_number":          test_number,
        "start_time":           int(time.time()),
        "questions":            {q.question_id: q for q in selected_qs},
        "submitted_questions":  set(),
        "score":                0,
        "question_scores":      {},
        "question_score_maxes": {},
        "module":               "listening",
        "question_type":        "sectional",
    }

    log.info(f"[Listening Sectional] Started session={session_id} user={user_id} " f"questions={len(selected_qs)}")

    return {
        "session_id":      session_id,
        "attempt_id":      attempt.id,
        "test_number":     test_number,
        "total_questions": len(questions_payload),
        "questions":       questions_payload,
    }


def resume_listening_sectional_exam(session_id: str, user_id: int, db: Session) -> dict:
    """Rebuild session from DB and return questions with fresh presigned URLs + is_submitted flags."""
    from db.models import AttemptAnswer
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="listening",
    ).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="No resumable listening session found")

    qid_order = attempt.selected_question_ids or []
    if not qid_order:
        raise HTTPException(status_code=404, detail="Listening attempt has no stored questions")

    submitted = {
        a.question_id
        for a in db.query(AttemptAnswer).filter_by(attempt_id=attempt.id).all()
    }

    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        qs_by_id = {
            q.question_id: q
            for q in db.query(QuestionFromApeuni)
            .options(joinedload(QuestionFromApeuni.evaluation))
            .filter(QuestionFromApeuni.question_id.in_(qid_order))
            .all()
        }
        selected = [qs_by_id[qid] for qid in qid_order if qid in qs_by_id]
        tb = attempt.task_breakdown or {}
        ACTIVE_SESSIONS[session_id] = {
            "user_id":              user_id,
            "attempt_id":           attempt.id,
            "test_number":          tb.get("test_number", 1),
            "start_time":           int(time.time()),
            "questions":            {q.question_id: q for q in selected},
            "submitted_questions":  submitted,
            "score":                0,
            "question_scores":      {},
            "question_score_maxes": {},
            "module":               "listening",
            "question_type":        "sectional",
        }

    session   = ACTIVE_SESSIONS[session_id]
    task_timing = {t["task"]: t for t in LISTENING_STRUCTURE}

    questions_payload = []
    for qid in qid_order:
        q = session["questions"].get(qid)
        if not q:
            continue
        timing = task_timing.get(q.question_type, {"time_seconds": 120, "prep_seconds": 0, "rec_seconds": 0})
        presigned_url: Optional[str] = None
        raw_audio = (
            (q.content_json or {}).get("audio_url")
            or (q.content_json or {}).get("s3_key")
            or (q.content_json or {}).get("audio_s3_key")
        )
        if raw_audio:
            try:
                presigned_url = generate_presigned_url(raw_audio)
            except Exception:
                presigned_url = None

        questions_payload.append({
            "question_id":   q.question_id,
            "task_type":     q.question_type,
            "display_name":  _display_name(q.question_type),
            "answer_type":   _ANSWER_TYPES.get(q.question_type, "text"),
            "time_seconds":  timing["time_seconds"],
            "prep_seconds":  timing["prep_seconds"],
            "rec_seconds":   timing["rec_seconds"],
            "content_json":  q.content_json,
            "presigned_url": presigned_url,
            "session_id":    session_id,
            "is_prediction": q.is_prediction,
            "is_submitted":  qid in submitted,
        })

    log.info(f"[Listening Sectional] Resumed session={session_id} user={user_id} " f"submitted={len(submitted)}/{len(qid_order)}")
    return {
        "session_id":      session_id,
        "attempt_id":      attempt.id,
        "total_questions": len(qid_order),
        "submitted_count": len(submitted),
        "questions":       questions_payload,
    }


def _earned_pct_from_answer(score: int) -> float:
    """Convert a stored pte_score to a 0.0–1.0 fraction for weighted aggregation.

    Works for both sync and async question types:
      - Sync questions store pte_score directly (10–90).
      - Async (speaking) questions store the Azure pte result (10–90 normally, 0 for no-speech).
    A score of 0 (no-speech error) and a score of 10 (floor) both correctly map to 0.0.
    """
    return max(0.0, (score - config.PTE_BASE) / config.PTE_SCALE)


def _wait_for_speaking_in_rds(
    bg_db,
    attempt_id: int,
    speaking_qids: list,
    timeout: int = 300,
) -> None:
    """Poll attempt_answers until all speaking rows have scoring_status='complete'."""
    if not speaking_qids:
        return
    from db.models import AttemptAnswer
    deadline = time.time() + timeout
    log.info(f"[ListeningBG] Waiting for {len(speaking_qids)} speaking scores in RDS…")
    while time.time() < deadline:
        pending = (
            bg_db.query(AttemptAnswer)
            .filter(
                AttemptAnswer.attempt_id == attempt_id,
                AttemptAnswer.question_id.in_(speaking_qids),
                AttemptAnswer.scoring_status == "pending",
            )
            .count()
        )
        if pending == 0:
            return
        time.sleep(3)
    log.warning(f"[ListeningBG] ⏱ Speaking scoring timed out after {timeout}s for attempt={attempt_id}")


def _aggregate_bg(
    attempt_id: int,
    session_id: str,
    user_id: int,
    all_question_ids: list,
):
    """Background thread: compute weighted score entirely from RDS, write back to DB.

    Reads all answered rows from attempt_answers, waits for async speaking scores to
    land in RDS, then builds task buckets and applies the PTE weighted formula.
    This approach is resilient to server restarts — no in-memory state needed.
    """
    from db.database import SessionLocal
    from db.models import AttemptAnswer, QuestionFromApeuni
    from sqlalchemy.orm.attributes import flag_modified

    bg_db = SessionLocal()
    try:
        # ── 1. Resolve question types for all exam questions ──────────────────
        qs_rows = (
            bg_db.query(QuestionFromApeuni.question_id, QuestionFromApeuni.question_type)
            .filter(QuestionFromApeuni.question_id.in_(all_question_ids))
            .all()
        )
        type_by_qid = {r.question_id: r.question_type for r in qs_rows}

        # ── 2. Wait for speaking answers to be scored in RDS ──────────────────
        speaking_qids = [
            qid for qid in all_question_ids
            if type_by_qid.get(qid) in _ASYNC_TYPES
        ]
        _wait_for_speaking_in_rds(bg_db, attempt_id, speaking_qids)

        # Expire cached objects so subsequent queries see fresh DB state.
        bg_db.expire_all()

        # ── 3. Load all answered rows ──────────────────────────────────────────
        answered_rows = (
            bg_db.query(AttemptAnswer)
            .filter_by(attempt_id=attempt_id)
            .all()
        )
        answered_by_qid = {row.question_id: row for row in answered_rows}

        # ── 4. Build per-task buckets ──────────────────────────────────────────
        task_buckets: dict = {}
        for qid in all_question_ids:
            t = type_by_qid.get(qid)
            if not t:
                continue  # question deleted from DB — skip
            if t not in task_buckets:
                task_buckets[t] = {"earned_pct_sum": 0.0, "total": 0, "answered": 0}
            task_buckets[t]["total"] += 1

            row = answered_by_qid.get(qid)
            if not row:
                continue
            task_buckets[t]["answered"] += 1
            task_buckets[t]["earned_pct_sum"] += _earned_pct_from_answer(row.score or 0)

        # ── 5. Weighted score ──────────────────────────────────────────────────
        weighted_sum   = 0.0
        present_weight = sum(_LISTENING_WEIGHTS.values())
        task_breakdown: dict = {}

        for task_type, bucket in task_buckets.items():
            weight       = _LISTENING_WEIGHTS.get(task_type, 0)
            total        = bucket["total"]
            answered     = bucket["answered"]
            task_pct     = bucket["earned_pct_sum"] / total if total > 0 else 0.0
            contribution = task_pct * weight
            weighted_sum += contribution
            task_breakdown[task_type] = {
                "display_name":       _display_name(task_type),
                "total_questions":    total,
                "questions_answered": answered,
                "task_pct":           round(task_pct * 100, 1),
                "listening_weight":   weight,
                "contribution":       round(contribution, 2),
                "scoring_type":       "async" if task_type in _ASYNC_TYPES else "sync",
            }

        normalised_pct = (weighted_sum / present_weight) if present_weight > 0 else 0.0
        scaled = max(
            config.PTE_FLOOR,
            min(config.PTE_CEILING, round(config.PTE_BASE + normalised_pct * config.PTE_SCALE)),
        )

        # ── 6. Write back to DB ────────────────────────────────────────────────
        attempt = bg_db.query(PracticeAttempt).filter_by(id=attempt_id).first()
        if attempt:
            prior_tb = attempt.task_breakdown or {}
            if "test_number" in prior_tb:
                task_breakdown["test_number"] = prior_tb["test_number"]
            attempt.total_score        = scaled
            attempt.questions_answered = len(answered_by_qid)
            attempt.status             = "complete"
            attempt.scoring_status     = "complete"
            attempt.task_breakdown     = task_breakdown
            attempt.completed_at       = datetime.now(timezone.utc)
            flag_modified(attempt, "task_breakdown")
            bg_db.commit()
            log.info(f"[ListeningBG] ✅ user={user_id} session={session_id} " f"score={scaled} norm_pct={round(normalised_pct * 100, 1)}% " f"answered={len(answered_by_qid)}/{len(all_question_ids)}")
        else:
            log.error(f"[ListeningBG] ❌ attempt_id={attempt_id} not found in DB")

    except Exception as e:
        import traceback
        log.error(f"[ListeningBG] ❌ Failed session={session_id}: {e}")
        traceback.print_exc()
    finally:
        bg_db.close()


def finish_listening_sectional(
    session_id: str,
    user_id: int,
    db: Session,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Kick off RDS-based background scoring and return immediately with scoring_status='pending'.
    The background task reads all answered rows from attempt_answers, waits for speaking
    scores to land in RDS, and computes the final weighted listening score.
    Does not depend on in-memory ACTIVE_SESSIONS — resilient to server restarts.
    """
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="listening",
    ).first()
    if not attempt:
        raise HTTPException(status_code=400, detail="Session not found")

    # Idempotency — if already scored, return existing result
    if attempt.scoring_status == "complete":
        return {
            "attempt_id":      attempt.id,
            "session_id":      session_id,
            "scoring_status":  "complete",
            "listening_score": attempt.total_score,
        }

    all_question_ids = attempt.selected_question_ids or []
    if not all_question_ids:
        raise HTTPException(status_code=400, detail="No questions found for this session")

    log.info(f"[ListeningFinish] session={session_id} user={user_id} " f"attempt={attempt.id} questions={len(all_question_ids)}")

    background_tasks.add_task(
        _aggregate_bg,
        attempt_id       = attempt.id,
        session_id       = session_id,
        user_id          = user_id,
        all_question_ids = all_question_ids,
    )

    return {
        "attempt_id":     attempt.id,
        "session_id":     session_id,
        "scoring_status": "pending",
        "message":        "Scoring in progress. Check the Feedback tab in a moment.",
    }


def get_listening_sectional_results(session_id: str, user_id: int, db: Session) -> dict:
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id
    ).first()
    if not attempt:
        return {"scoring_status": "not_found"}

    answers = (
        db.query(AttemptAnswer)
        .filter_by(attempt_id=attempt.id)
        .order_by(AttemptAnswer.submitted_at)
        .all()
    )
    questions = [
        {
            "question_id":      a.question_id,
            "question_type":    a.question_type,
            "score":            a.score,
            "result_json":      a.result_json or {},
            "user_answer_json": a.user_answer_json or {},
            "scoring_status":   a.scoring_status,
        }
        for a in answers
    ]

    return {
        "attempt_id":         attempt.id,
        "session_id":         session_id,
        "scoring_status":     attempt.scoring_status or "pending",
        "listening_score":    attempt.total_score,
        "task_breakdown":     attempt.task_breakdown or {},
        "total_questions":    attempt.total_questions,
        "questions_answered": attempt.questions_answered,
        "completed_at":       attempt.completed_at.isoformat() if attempt.completed_at else None,
        "questions":          questions,
    }
