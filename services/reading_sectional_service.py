"""
Reading Sectional Service
=========================
Selects questions for the reading sectional exam (5 task types, 15 questions),
stores the session, and handles weighted scoring at finish.

Task types (web variant — uses web question_type names):
  reading_fib         — 5 questions
  reading_mcs         — 2 questions
  reading_mcm         — 2 questions
  reorder_paragraphs  — 2 questions
  reading_fib_drop_down — 4 questions

Scoring weights (from PTE weightage table, normalised to the types present):
  reading_fib:          25%   (FIB Reading & Writing)
  reading_fib_drop_down: 20%  (FIB Reading drag-and-drop)
  reorder_paragraphs:    9%
  reading_mcm:           5%
  reading_mcs:           3%

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
READING_STRUCTURE = [
    {"task": "reading_fib_drop_down", "count": 9, "module": "reading"},
    {"task": "mcq_single",            "count": 2, "module": "reading"},
    {"task": "mcq_multiple",          "count": 2, "module": "reading"},
    {"task": "reorder_paragraphs",    "count": 2, "module": "reading"},
]

_READING_WEIGHTS = {
    "reading_fib_drop_down": 45,
    "reorder_paragraphs":     9,
    "mcq_multiple":           5,
    "mcq_single":             3,
}

_DISPLAY_NAMES = {
    "reading_fib_drop_down": "Fill in the Blanks",
    "mcq_single":            "Multiple Choice (Single)",
    "mcq_multiple":          "Multiple Choice (Multiple)",
    "reorder_paragraphs":    "Re-order Paragraphs",
}

_SWT_MAX = 10


def _display_name(task_type: str) -> str:
    return _DISPLAY_NAMES.get(task_type, task_type.replace("_", " ").title())


def _question_max(q) -> int:
    """Return max achievable raw score for one question using rubric rules."""
    qt = q.question_type
    try:
        ev    = q.evaluation.evaluation_json
        rules = ev.get("scoringRules", {})
        ans   = ev.get("correctAnswers", {})

        if qt == "reading_fib_drop_down":
            return len(ans) * rules.get("marksPerBlank", 1)

        if qt == "mcq_multiple":
            return len(ans.get("correctOptions", [])) * rules.get("marksPerCorrect", 1)

        if qt == "mcq_single":
            return rules.get("marksPerCorrect", 1)

        if qt == "reorder_paragraphs":
            seq = ans.get("correctSequence", [])
            return max(0, len(seq) - 1) * rules.get("marksPerAdjacentPair", 1)

    except Exception:
        pass

    return {
        "reading_fib_drop_down": 4,
        "mcq_multiple":          2,
        "mcq_single":            1,
        "reorder_paragraphs":    3,
    }.get(qt, 1)


def get_reading_sectional_info() -> dict:
    total_questions = sum(t["count"] for t in READING_STRUCTURE)
    sections = [
        {
            "task":         t["task"],
            "display_name": _display_name(t["task"]),
            "count":        t["count"],
            "weight_pct":   _READING_WEIGHTS.get(t["task"], 0),
        }
        for t in READING_STRUCTURE
    ]
    return {
        "total_questions": total_questions,
        "sections":        sections,
    }


def start_reading_sectional_exam(db: Session, user_id: int, test_number: int) -> dict:
    """Select questions, create session, return question list."""
    practiced_ids = set(
        row[0]
        for row in db.query(UserQuestionAttempt.question_id)
        .filter(
            UserQuestionAttempt.user_id == user_id,
            UserQuestionAttempt.module  == "reading",
        )
        .all()
    )

    selected_qs: list = []

    for task in READING_STRUCTURE:
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
                f"[Reading Sectional] Not enough fresh questions for {task_type} "
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
            print(f"[Reading Sectional] No questions for {task_type} — skipping", flush=True)
            continue
        selected_qs.extend(random.sample(pool, n))

    if not selected_qs:
        raise HTTPException(status_code=404, detail="No reading questions available")

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
        questions_payload.append({
            "question_id":  q.question_id,
            "task_type":    q.question_type,
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
        "module":               "reading",
        "question_type":        "sectional",
    }

    print(
        f"[Reading Sectional] Started session={session_id} user={user_id} "
        f"questions={len(selected_qs)}",
        flush=True,
    )

    return {
        "session_id":      session_id,
        "test_number":     test_number,
        "total_questions": len(questions_payload),
        "questions":       questions_payload,
    }


def finish_reading_sectional(session_id: str, user_id: int, db: Session) -> dict:
    """
    Weighted scoring over the 5 reading task types.
    Formula: max(10, min(90, round(10 + normalised_pct * 80)))
    """
    session_data = ACTIVE_SESSIONS.get(session_id)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found or expired")

    # Idempotency
    existing = db.query(PracticeAttempt).filter_by(session_id=session_id).first()
    if existing:
        return {
            "attempt_id":     existing.id,
            "session_id":     session_id,
            "scoring_status": existing.scoring_status or "complete",
            "reading_score":  existing.total_score,
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
        weight   = _READING_WEIGHTS.get(task_type, 0)
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
            "reading_weight":     weight,
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
        module             = "reading",
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
        f"[Reading Sectional] Finished session={session_id} score={scaled} "
        f"answered={len(submitted)}/{len(questions)}",
        flush=True,
    )

    return {
        "attempt_id":     attempt.id,
        "session_id":     session_id,
        "scoring_status": "complete",
        "reading_score":  scaled,
        "weighted_pct":   round(normalised_pct * 100, 1),
        "task_breakdown": task_breakdown,
    }


def get_reading_sectional_results(session_id: str, user_id: int, db: Session) -> dict:
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id
    ).first()
    if not attempt:
        return {"scoring_status": "not_found"}
    return {
        "attempt_id":         attempt.id,
        "session_id":         session_id,
        "scoring_status":     attempt.scoring_status or "complete",
        "reading_score":      attempt.total_score,
        "task_breakdown":     attempt.task_breakdown or {},
        "total_questions":    attempt.total_questions,
        "questions_answered": attempt.questions_answered,
        "completed_at":       attempt.completed_at.isoformat() if attempt.completed_at else None,
    }
