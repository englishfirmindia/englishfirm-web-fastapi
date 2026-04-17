"""
Listening Sectional Service
============================
Selects questions for the listening sectional exam (8 task types, 18 questions),
stores the session, and handles mixed sync + async scoring at finish.

Task types:
  listening_sst  — 2 questions (AI scored, async-ish)
  listening_mcm  — 2 questions (sync MCQ)
  listening_fib  — 3 questions (sync FIB)
  listening_hcs  — 2 questions (sync MCQ single)
  listening_smw  — 2 questions (sync MCQ single)
  listening_hiw  — 2 questions (sync HIW)
  listening_mcs  — 2 questions (sync MCQ single)
  listening_wfd  — 3 questions (sync WFD)

Scoring weights (from PTE weightage table):
  listening_sst: 10%   listening_wfd: 13%   listening_fib:  8%
  listening_hiw:  8%   listening_hcs:  2%   listening_mcm:  3%
  listening_mcs:  2%   listening_smw:  1%

All questions contribute to the listening band score.

PTE formula (CLAUDE.md guardrail):
  pte_score = max(10, min(90, round(10 + weighted_pct * 80)))
"""

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

from db.models import QuestionFromApeuni, UserQuestionAttempt, PracticeAttempt
from services.session_service import ACTIVE_SESSIONS
from services.s3_service import generate_presigned_url
import core.config as config

# ─── Sectional structure ──────────────────────────────────────────────────────
LISTENING_STRUCTURE = [
    {"task": "summarize_spoken_text",  "count": 2, "module": "listening"},
    {"task": "listening_mcq_multiple", "count": 2, "module": "listening"},
    {"task": "listening_fib",          "count": 3, "module": "listening"},
    {"task": "listening_hcs",          "count": 2, "module": "listening"},
    {"task": "listening_smw",          "count": 2, "module": "listening"},
    {"task": "highlight_incorrect_words", "count": 2, "module": "listening"},
    {"task": "listening_mcq_single",   "count": 2, "module": "listening"},
    {"task": "listening_wfd",          "count": 3, "module": "listening"},
]

_LISTENING_WEIGHTS = {
    "summarize_spoken_text":  10,
    "listening_wfd":          13,
    "listening_fib":           8,
    "highlight_incorrect_words": 8,
    "listening_hcs":           2,
    "listening_mcq_multiple":  3,
    "listening_mcq_single":    2,
    "listening_smw":           1,
}

# Max raw score fallbacks per question type
_MAX_FALLBACK = {
    "summarize_spoken_text":   10,
    "listening_wfd":            7,
    "listening_fib":            4,
    "highlight_incorrect_words": 3,
    "listening_hcs":            1,
    "listening_mcq_multiple":   2,
    "listening_mcq_single":     1,
    "listening_smw":            1,
}

_DISPLAY_NAMES = {
    "summarize_spoken_text":   "Summarize Spoken Text",
    "listening_wfd":           "Write from Dictation",
    "listening_fib":           "Fill in the Blanks",
    "highlight_incorrect_words": "Highlight Incorrect Words",
    "listening_hcs":           "Highlight Correct Summary",
    "listening_mcq_multiple":  "MCQ – Multiple Answers",
    "listening_mcq_single":    "MCQ – Single Answer",
    "listening_smw":           "Select Missing Word",
}


def _display_name(task_type: str) -> str:
    return _DISPLAY_NAMES.get(task_type, task_type.replace("_", " ").title())


def _question_max(q) -> int:
    """Return max achievable raw score for one question."""
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
        "sections":        sections,
    }


def start_listening_sectional_exam(db: Session, user_id: int, test_number: int) -> dict:
    """Select questions, create session, return question list with presigned audio URLs."""
    practiced_ids = set(
        row[0]
        for row in db.query(UserQuestionAttempt.question_id)
        .filter(
            UserQuestionAttempt.user_id == user_id,
            UserQuestionAttempt.module  == "listening",
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

    questions_payload = []
    for q in selected_qs:
        # Presign audio URL for all listening questions
        presigned_url: Optional[str] = None
        raw_audio = (
            q.content_json.get("audio_url")
            or q.content_json.get("s3_key")
            or q.content_json.get("audio_s3_key")
        )
        if raw_audio:
            try:
                presigned_url = generate_presigned_url(raw_audio)
            except Exception:
                presigned_url = None

        questions_payload.append({
            "question_id":   q.question_id,
            "task_type":     q.question_type,
            "content_json":  q.content_json,
            "presigned_url": presigned_url,
            "session_id":    session_id,
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


def finish_listening_sectional(session_id: str, user_id: int, db: Session) -> dict:
    """
    Weighted scoring over 8 listening task types.
    All task types in this web sectional are sync-scored at submit time.
    Formula: max(10, min(90, round(10 + normalised_pct * 80)))
    """
    session_data = ACTIVE_SESSIONS.get(session_id)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found or expired")

    # Idempotency
    existing = db.query(PracticeAttempt).filter_by(session_id=session_id).first()
    if existing:
        return {
            "attempt_id":      existing.id,
            "session_id":      session_id,
            "scoring_status":  existing.scoring_status or "complete",
            "listening_score": existing.total_score,
        }

    questions     = session_data.get("questions", {})
    submitted     = session_data.get("submitted_questions", set())
    q_scores      = session_data.get("question_scores", {})
    q_score_maxes = session_data.get("question_score_maxes", {})

    # Build per-task buckets
    task_buckets: dict = {}
    for qid, q in questions.items():
        t = q.question_type
        if t not in task_buckets:
            task_buckets[t] = {"earned": 0.0, "max": 0.0, "total": 0, "answered": 0}

        q_max = q_score_maxes.get(qid) or _question_max(q)
        task_buckets[t]["max"]   += q_max
        task_buckets[t]["total"] += 1

        if qid in submitted:
            earned = q_scores.get(qid, 0)
            task_buckets[t]["earned"]   += earned
            task_buckets[t]["answered"] += 1

    # Weighted aggregation
    weighted_sum   = 0.0
    present_weight = 0
    task_breakdown: dict = {}

    for task_type, bucket in task_buckets.items():
        weight   = _LISTENING_WEIGHTS.get(task_type, 0)
        q_max    = bucket["max"]
        earned   = bucket["earned"]

        task_pct     = (earned / q_max) if q_max > 0 else 0.0
        contribution = task_pct * weight

        weighted_sum   += contribution
        present_weight += weight

        task_breakdown[task_type] = {
            "display_name":       _display_name(task_type),
            "total_questions":    bucket["total"],
            "questions_answered": bucket["answered"],
            "earned_raw":         round(earned, 2),
            "max_raw":            round(q_max, 2),
            "task_pct":           round(task_pct * 100, 1),
            "listening_weight":   weight,
            "contribution":       round(contribution, 2),
        }

    normalised_pct = (weighted_sum / present_weight) if present_weight > 0 else 0.0
    scaled = max(
        config.PTE_FLOOR,
        min(config.PTE_CEILING, round(config.PTE_BASE + normalised_pct * config.PTE_SCALE)),
    )

    attempt = PracticeAttempt(
        user_id            = user_id,
        session_id         = session_id,
        module             = "listening",
        question_type      = "sectional",
        filter_type        = "sectional",
        total_questions    = len(questions),
        total_score        = scaled,
        questions_answered = len(submitted),
        status             = "complete",
        scoring_status     = "complete",
        task_breakdown     = task_breakdown,
        completed_at       = datetime.now(timezone.utc),
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    print(
        f"[Listening Sectional] Finished session={session_id} score={scaled} "
        f"answered={len(submitted)}/{len(questions)}",
        flush=True,
    )

    return {
        "attempt_id":      attempt.id,
        "session_id":      session_id,
        "scoring_status":  "complete",
        "listening_score": scaled,
        "weighted_pct":    round(normalised_pct * 100, 1),
        "task_breakdown":  task_breakdown,
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
