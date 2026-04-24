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
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

from db.models import QuestionFromApeuni, UserQuestionAttempt, PracticeAttempt
from services.session_service import ACTIVE_SESSIONS, _SCORE_STORE
from services.s3_service import generate_presigned_url
import core.config as config

# ─── Sectional structure ──────────────────────────────────────────────────────
# Speaking tasks come first (PTE exam ordering).
LISTENING_STRUCTURE = [
    # ── Speaking tasks (contribute to listening score) ────────────────────────
    {"task": "repeat_sentence",            "count": 3, "module": "speaking", "time_seconds":  21, "prep_seconds": 0,  "rec_seconds": 9},
    {"task": "retell_lecture",             "count": 1, "module": "speaking", "time_seconds": 125, "prep_seconds": 10, "rec_seconds": 40},
    {"task": "summarize_group_discussion", "count": 1, "module": "speaking", "time_seconds": 125, "prep_seconds": 3,  "rec_seconds": 120},
    {"task": "answer_short_question",      "count": 2, "module": "speaking", "time_seconds":  19, "prep_seconds": 0,  "rec_seconds": 5},
    # ── Pure listening tasks (sync scored) ────────────────────────────────────
    {"task": "summarize_spoken_text",      "count": 2, "module": "listening", "time_seconds": 675, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_mcq_multiple",     "count": 2, "module": "listening", "time_seconds":  90, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_fib",              "count": 3, "module": "listening", "time_seconds": 105, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_hcs",              "count": 2, "module": "listening", "time_seconds": 135, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_smw",              "count": 2, "module": "listening", "time_seconds":  90, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "highlight_incorrect_words",  "count": 2, "module": "listening", "time_seconds": 120, "prep_seconds": 0, "rec_seconds": 0},
    {"task": "listening_mcq_single",       "count": 2, "module": "listening", "time_seconds":  90, "prep_seconds": 0, "rec_seconds": 0},
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
    """Select questions, create session, return question list with presigned audio URLs."""
    practiced_ids = set(
        row[0]
        for row in db.query(UserQuestionAttempt.question_id)
        .filter(
            UserQuestionAttempt.user_id == user_id,
            UserQuestionAttempt.module.in_(["listening", "speaking"]),
        )
        .all()
    )

    selected_qs: list = []

    for task in LISTENING_STRUCTURE:
        task_type = task["task"]
        db_module = task["module"]
        count     = task["count"]

        opts  = [joinedload(QuestionFromApeuni.evaluation)]
        fresh = (
            db.query(QuestionFromApeuni)
            .options(*opts)
            .filter(
                QuestionFromApeuni.module        == db_module,
                QuestionFromApeuni.question_type == task_type,
                ~QuestionFromApeuni.question_id.in_(practiced_ids) if practiced_ids else True,
            )
            .all()
        )
        pool = fresh
        if len(fresh) < count:
            print(
                f"[Listening Sectional] Not enough fresh questions for {task_type} "
                f"(need {count}, have {len(fresh)}) — falling back to full pool",
                flush=True,
            )
            pool = (
                db.query(QuestionFromApeuni)
                .options(*opts)
                .filter(
                    QuestionFromApeuni.module        == db_module,
                    QuestionFromApeuni.question_type == task_type,
                )
                .all()
            )

        n = min(count, len(pool))
        if n == 0:
            print(f"[Listening Sectional] No questions for {task_type} — skipping", flush=True)
            continue
        selected_qs.extend(random.sample(pool, n))

    if not selected_qs:
        raise HTTPException(status_code=404, detail="No listening questions available")

    # Mark as attempted
    seen = practiced_ids.copy()
    new_attempts = []
    for q in selected_qs:
        if q.question_id not in seen:
            new_attempts.append(UserQuestionAttempt(
                user_id=user_id,
                question_id=q.question_id,
                question_type=q.question_type,
                module=q.module,
            ))
            seen.add(q.question_id)
    if new_attempts:
        db.add_all(new_attempts)
        db.commit()

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

    ACTIVE_SESSIONS[session_id] = {
        "user_id":              user_id,
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

    print(
        f"[Listening Sectional] Started session={session_id} user={user_id} "
        f"questions={len(selected_qs)}",
        flush=True,
    )

    return {
        "session_id":      session_id,
        "test_number":     test_number,
        "total_questions": len(questions_payload),
        "questions":       questions_payload,
    }


def _wait_for_speaking_scores(user_id: int, question_ids: list, timeout: int = 300) -> dict:
    """Poll _SCORE_STORE until all async speaking scores are resolved."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(
            _SCORE_STORE.get((user_id, qid)) is not None
            and _SCORE_STORE.get((user_id, qid), {}).get("scoring") != "pending"
            for qid in question_ids
        ):
            break
        time.sleep(3)
    return {
        qid: _SCORE_STORE.get((user_id, qid), {"scoring": "timeout", "total": config.PTE_FLOOR})
        for qid in question_ids
    }


def _aggregate_bg(
    attempt_id: int,
    session_id: str,
    user_id: int,
    questions: dict,
    submitted: set,
    q_scores: dict,
    q_score_maxes: dict,
):
    """Background thread: wait for Azure speaking scores, aggregate, write to DB."""
    from db.database import SessionLocal
    bg_db = SessionLocal()
    try:
        speaking_qids = [
            qid for qid, q in questions.items()
            if q.question_type in _ASYNC_TYPES and qid in submitted
        ]
        speaking_scores = {}
        if speaking_qids:
            print(f"[ListeningBG] Waiting for {len(speaking_qids)} speaking scores…", flush=True)
            speaking_scores = _wait_for_speaking_scores(user_id, speaking_qids)

        # Build per-task buckets
        task_buckets: dict = {}
        for qid, q in questions.items():
            t = q.question_type
            if t not in task_buckets:
                task_buckets[t] = {"earned_pct_sum": 0.0, "total": 0, "answered": 0,
                                   "earned_raw": 0.0, "max_raw": 0.0}
            task_buckets[t]["total"] += 1
            if qid not in submitted:
                continue
            task_buckets[t]["answered"] += 1

            if t in _ASYNC_TYPES:
                raw = speaking_scores.get(qid, {})
                stored_total = float(raw.get("total") or config.PTE_FLOOR)
                pct = (stored_total - config.PTE_BASE) / config.PTE_SCALE
                task_buckets[t]["earned_pct_sum"] += max(0.0, pct)
                task_buckets[t]["earned_raw"]     += stored_total
                task_buckets[t]["max_raw"]        += config.PTE_CEILING
            else:
                earned = q_scores.get(qid, 0)
                q_max  = q_score_maxes.get(qid) or _question_max(q)
                task_buckets[t]["earned_raw"]     += earned
                task_buckets[t]["max_raw"]        += q_max
                task_buckets[t]["earned_pct_sum"] += (earned / q_max) if q_max > 0 else 0.0

        weighted_sum   = 0.0
        present_weight = 0
        task_breakdown: dict = {}

        for task_type, bucket in task_buckets.items():
            weight   = _LISTENING_WEIGHTS.get(task_type, 0)
            total    = bucket["total"]
            answered = bucket["answered"]
            task_pct = (bucket["earned_pct_sum"] / total) if total > 0 else 0.0
            contribution  = task_pct * weight
            weighted_sum  += contribution
            present_weight += weight
            task_breakdown[task_type] = {
                "display_name":       _display_name(task_type),
                "total_questions":    total,
                "questions_answered": answered,
                "earned_raw":         round(bucket["earned_raw"], 2),
                "max_raw":            round(bucket["max_raw"], 2),
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

        attempt = bg_db.query(PracticeAttempt).filter_by(id=attempt_id).first()
        if attempt:
            attempt.total_score        = scaled
            attempt.questions_answered = len(submitted)
            attempt.status             = "complete"
            attempt.scoring_status     = "complete"
            attempt.task_breakdown     = task_breakdown
            attempt.completed_at       = datetime.now(timezone.utc)
            bg_db.commit()
            print(
                f"[ListeningBG] ✅ user={user_id} session={session_id} "
                f"score={scaled} norm_pct={round(normalised_pct * 100, 1)}%",
                flush=True,
            )
        else:
            print(f"[ListeningBG] ❌ attempt_id={attempt_id} not found in DB", flush=True)

    except Exception as e:
        print(f"[ListeningBG] ❌ Failed session={session_id}: {e}", flush=True)
    finally:
        bg_db.close()


def finish_listening_sectional(session_id: str, user_id: int, db: Session) -> dict:
    """
    Kick off background scoring and return immediately with scoring_status='pending'.
    Background thread waits for Azure speaking scores, aggregates, and writes to DB.
    """
    session_data = ACTIVE_SESSIONS.get(session_id)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found or expired")

    # Idempotency — if already scored, return existing result
    existing = db.query(PracticeAttempt).filter_by(session_id=session_id).first()
    if existing and existing.scoring_status == "complete":
        return {
            "attempt_id":      existing.id,
            "session_id":      session_id,
            "scoring_status":  "complete",
            "listening_score": existing.total_score,
        }

    questions     = session_data.get("questions", {})
    submitted     = set(session_data.get("submitted_questions", set()))
    q_scores      = dict(session_data.get("question_scores", {}))
    q_score_maxes = dict(session_data.get("question_score_maxes", {}))

    print(f"[ListeningFinish] submitted={len(submitted)} q_scores={len(q_scores)}", flush=True)

    if existing:
        attempt_id = existing.id
        existing.questions_answered = len(submitted)
        db.commit()
    else:
        attempt = PracticeAttempt(
            user_id            = user_id,
            session_id         = session_id,
            module             = "listening",
            question_type      = "sectional",
            filter_type        = "sectional",
            total_questions    = len(questions),
            total_score        = 0,
            questions_answered = len(submitted),
            status             = "pending",
            scoring_status     = "pending",
        )
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
        attempt_id = attempt.id

    threading.Thread(
        target=_aggregate_bg,
        args=(attempt_id, session_id, user_id, questions, submitted, q_scores, q_score_maxes),
        daemon=True,
    ).start()
    print(f"[ListeningFinish] ✅ Background scoring started attempt_id={attempt_id}", flush=True)

    return {
        "attempt_id":     attempt_id,
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
    return {
        "attempt_id":         attempt.id,
        "session_id":         session_id,
        "scoring_status":     attempt.scoring_status or "complete",
        "listening_score":    attempt.total_score,
        "task_breakdown":     attempt.task_breakdown or {},
        "total_questions":    attempt.total_questions,
        "questions_answered": attempt.questions_answered,
        "completed_at":       attempt.completed_at.isoformat() if attempt.completed_at else None,
    }
