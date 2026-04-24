"""
Writing Sectional Service
=========================
Selects questions for the writing sectional exam (2 task types, 4 questions),
stores the session, and handles weighted scoring at finish.

Task types:
  summarize_written_text (SWT) — 2 questions, 10 min each
  write_essay             (WE)  — 2 questions, 20 min each

Scoring weights (equal split):
  SWT: 50%   WE: 50%

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
import core.config as config

# ─── Sectional structure ──────────────────────────────────────────────────────
WRITING_STRUCTURE = [
    {"task": "summarize_written_text", "count": 2, "time_seconds": 600},
    {"task": "write_essay",            "count": 2, "time_seconds": 1200},
]

_WRITING_WEIGHTS = {
    "summarize_written_text": 50,
    "write_essay":            50,
}

# Raw rubric max per question type
_SWT_MAX = 10  # PTE SWT rubric max
_WE_MAX  = 15  # PTE WE rubric max

_DISPLAY_NAMES = {
    "summarize_written_text": "Summarize Written Text",
    "write_essay":            "Write Essay",
}


def _display_name(task_type: str) -> str:
    return _DISPLAY_NAMES.get(task_type, task_type.replace("_", " ").title())


def _question_max(q) -> int:
    qt = q.question_type
    if qt == "summarize_written_text":
        return _SWT_MAX
    if qt == "write_essay":
        return _WE_MAX
    return 1


def get_writing_sectional_info() -> dict:
    total_questions = sum(t["count"] for t in WRITING_STRUCTURE)
    total_minutes   = sum(t["count"] * t["time_seconds"] / 60 for t in WRITING_STRUCTURE)
    sections = [
        {
            "task":              t["task"],
            "display_name":      _display_name(t["task"]),
            "count":             t["count"],
            "time_seconds":      t["time_seconds"],
            "time_per_question": t["time_seconds"] / 60,
            "total_minutes":     t["count"] * t["time_seconds"] / 60,
        }
        for t in WRITING_STRUCTURE
    ]
    return {
        "total_questions": total_questions,
        "total_minutes":   round(total_minutes, 1),
        "sections":        sections,
    }


def start_writing_sectional_exam(db: Session, user_id: int, test_number: int) -> dict:
    """Select questions, create session + PracticeAttempt, return question list."""
    practiced_ids = set(
        row[0]
        for row in db.query(UserQuestionAttempt.question_id)
        .filter(
            UserQuestionAttempt.user_id == user_id,
            UserQuestionAttempt.module  == "writing",
        )
        .all()
    )

    selected_qs: list = []

    for task in WRITING_STRUCTURE:
        task_type = task["task"]
        count     = task["count"]

        opts  = [joinedload(QuestionFromApeuni.evaluation)]
        fresh = (
            db.query(QuestionFromApeuni)
            .options(*opts)
            .filter(
                QuestionFromApeuni.module        == "writing",
                QuestionFromApeuni.question_type == task_type,
                ~QuestionFromApeuni.question_id.in_(practiced_ids) if practiced_ids else True,
            )
            .all()
        )
        pool = fresh
        if len(fresh) < count:
            print(
                f"[Writing Sectional] Not enough fresh questions for {task_type} "
                f"(need {count}, have {len(fresh)}) — falling back to full pool",
                flush=True,
            )
            pool = (
                db.query(QuestionFromApeuni)
                .options(*opts)
                .filter(
                    QuestionFromApeuni.module        == "writing",
                    QuestionFromApeuni.question_type == task_type,
                )
                .all()
            )

        n = min(count, len(pool))
        if n == 0:
            print(f"[Writing Sectional] No questions for {task_type} — skipping", flush=True)
            continue
        selected_qs.extend(random.sample(pool, n))

    if not selected_qs:
        raise HTTPException(status_code=404, detail="No writing questions available")

    # Mark as attempted
    seen = practiced_ids.copy()
    new_attempts = []
    for q in selected_qs:
        if q.question_id not in seen:
            new_attempts.append(UserQuestionAttempt(
                user_id=user_id,
                question_id=q.question_id,
                question_type=q.question_type,
                module="writing",
            ))
            seen.add(q.question_id)
    if new_attempts:
        db.add_all(new_attempts)
        db.commit()

    session_id = str(uuid.uuid4())

    attempt = PracticeAttempt(
        user_id         = user_id,
        session_id      = session_id,
        module          = "writing",
        question_type   = "sectional",
        filter_type     = "sectional",
        total_questions = len(selected_qs),
        total_score     = 0,
        status          = "in_progress",
        scoring_status  = "pending",
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    task_timing = {t["task"]: t for t in WRITING_STRUCTURE}
    questions_payload = []
    for q in selected_qs:
        timing = task_timing.get(q.question_type, {"time_seconds": 600})
        questions_payload.append({
            "question_id":  q.question_id,
            "task_type":    q.question_type,
            "time_seconds": timing["time_seconds"],
            "content_json": q.content_json,
            "session_id":   session_id,
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
        "module":               "writing",
        "question_type":        "sectional",
        "attempt_id":           attempt.id,
    }

    print(
        f"[Writing Sectional] Started session={session_id} user={user_id} "
        f"questions={len(selected_qs)}",
        flush=True,
    )

    return {
        "session_id":      session_id,
        "test_number":     test_number,
        "total_questions": len(questions_payload),
        "total_minutes":   round(sum(t["count"] * t["time_seconds"] / 60 for t in WRITING_STRUCTURE), 1),
        "questions":       questions_payload,
    }


def finish_writing_sectional(session_id: str, user_id: int, db: Session) -> dict:
    """
    Weighted scoring: SWT(50%) + WE(50%)
    Scaled with PTE formula: max(10, min(90, round(10 + normalised_pct * 80)))
    """
    session_data = ACTIVE_SESSIONS.get(session_id)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found or expired")

    existing = db.query(PracticeAttempt).filter_by(session_id=session_id).first()
    if existing and existing.scoring_status == "complete":
        return {
            "attempt_id":     existing.id,
            "session_id":     session_id,
            "scoring_status": "complete",
            "writing_score":  existing.total_score,
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
        weight   = _WRITING_WEIGHTS.get(task_type, 0)
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
            "writing_weight":     weight,
            "contribution":       round(contribution, 2),
        }

    normalised_pct = (weighted_sum / present_weight) if present_weight > 0 else 0.0
    scaled = max(
        config.PTE_FLOOR,
        min(config.PTE_CEILING, round(config.PTE_BASE + normalised_pct * config.PTE_SCALE)),
    )

    if existing:
        existing.total_score        = scaled
        existing.questions_answered = len(submitted)
        existing.status             = "complete"
        existing.scoring_status     = "complete"
        existing.task_breakdown     = task_breakdown
        existing.completed_at       = datetime.now(timezone.utc)
        attempt = existing
    else:
        attempt = PracticeAttempt(
            user_id            = user_id,
            session_id         = session_id,
            module             = "writing",
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
        f"[Writing Sectional] Finished session={session_id} score={scaled} "
        f"answered={len(submitted)}/{len(questions)}",
        flush=True,
    )

    return {
        "attempt_id":     attempt.id,
        "session_id":     session_id,
        "scoring_status": "complete",
        "writing_score":  scaled,
        "weighted_pct":   round(normalised_pct * 100, 1),
        "task_breakdown": task_breakdown,
    }


def get_writing_sectional_results(session_id: str, user_id: int, db: Session) -> dict:
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id
    ).first()
    if not attempt:
        return {"scoring_status": "not_found"}
    return {
        "attempt_id":         attempt.id,
        "session_id":         session_id,
        "scoring_status":     attempt.scoring_status or "complete",
        "writing_score":      attempt.total_score,
        "task_breakdown":     attempt.task_breakdown or {},
        "total_questions":    attempt.total_questions,
        "questions_answered": attempt.questions_answered,
        "completed_at":       attempt.completed_at.isoformat() if attempt.completed_at else None,
    }
