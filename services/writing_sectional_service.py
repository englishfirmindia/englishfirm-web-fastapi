"""
Writing Sectional Service
=========================
Selects questions for the writing sectional exam (4 task types, 7 questions),
stores the session, and handles weighted scoring at finish.

Task types:
  summarize_written_text (SWT) — 2 questions, 10 min each (per-Q timer)
  write_essay             (WE)  — 1 question,  20 min     (per-Q timer)
  summarize_spoken_text   (SST) — 1 question,  10 min     (per-Q timer)
  listening_wfd           (WFD) — 3 questions, 60 s each  (shared block timer)

Scoring weights (from RDS pte_question_weightage.writing_percent):
  SWT: 28  WE: 31  SST: 18  WFD: 23

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


def _maybe_presigned_url(q) -> Optional[str]:
    """Return a presigned audio URL for SST/WFD, None otherwise."""
    cj = q.content_json or {}
    raw_audio = cj.get("audio_url") or cj.get("s3_key") or cj.get("audio_s3_key")
    if not raw_audio:
        return None
    try:
        return generate_presigned_url(raw_audio)
    except Exception:
        return None

# ─── Sectional structure ──────────────────────────────────────────────────────
WRITING_STRUCTURE = [
    {"task": "summarize_written_text", "count": 2, "time_seconds":  600, "module": "writing"},
    {"task": "write_essay",            "count": 1, "time_seconds": 1200, "module": "writing"},
    {"task": "summarize_spoken_text",  "count": 1, "time_seconds":  600, "module": "listening"},
    {"task": "listening_wfd",          "count": 3, "time_seconds":   60, "module": "listening"},
]

# Weights from RDS pte_question_weightage.writing_percent
_WRITING_WEIGHTS = {
    "summarize_written_text": 28,
    "write_essay":            31,
    "summarize_spoken_text":  18,
    "listening_wfd":          23,
}

# Raw rubric max per question type
_SWT_MAX = 10  # PTE SWT rubric max
_WE_MAX  = 15  # PTE WE rubric max
_SST_MAX = 10  # PTE SST rubric max

_DISPLAY_NAMES = {
    "summarize_written_text": "Summarize Written Text",
    "write_essay":            "Write Essay",
    "summarize_spoken_text":  "Summarize Spoken Text",
    "listening_wfd":          "Write From Dictation",
}


def _display_name(task_type: str) -> str:
    return _DISPLAY_NAMES.get(task_type, task_type.replace("_", " ").title())


def _wfd_max(q) -> int:
    """WFD raw max = transcript word count."""
    eval_json = (q.evaluation.evaluation_json if q.evaluation else {}) or {}
    transcript = (eval_json.get("correctAnswers", {}) or {}).get("transcript", "") or ""
    return max(1, len(transcript.split()))


def _question_max(q) -> int:
    qt = q.question_type
    if qt == "summarize_written_text":
        return _SWT_MAX
    if qt == "write_essay":
        return _WE_MAX
    if qt == "summarize_spoken_text":
        return _SST_MAX
    if qt == "listening_wfd":
        return _wfd_max(q)
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
    # Exclude questions already SUBMITTED (attempt_answers row); merely-shown
    # but unanswered questions stay in the pool.
    from db.models import AttemptAnswer as _AA
    practiced_ids = set(
        row[0]
        for row in db.query(_AA.question_id)
        .join(PracticeAttempt, _AA.attempt_id == PracticeAttempt.id)
        .filter(PracticeAttempt.user_id == user_id)
        .all()
    )

    selected_qs: list = []

    for task in WRITING_STRUCTURE:
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
                f"[Writing Sectional] Not enough fresh questions for {task_type} "
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
            print(f"[Writing Sectional] No questions for {task_type} — skipping", flush=True)
            continue
        selected_qs.extend(random.sample(pool, n))

    if not selected_qs:
        raise HTTPException(status_code=404, detail="No writing questions available")

    # Note: pool filter uses attempt_answers (submitted only), so we no longer
    # pre-mark selected questions in user_question_attempts on session start.

    session_id = str(uuid.uuid4())

    # Remove any incomplete previous attempts so history stays clean
    db.query(PracticeAttempt).filter(
        PracticeAttempt.user_id        == user_id,
        PracticeAttempt.module         == "writing",
        PracticeAttempt.question_type  == "sectional",
        PracticeAttempt.status         != "complete",
    ).delete(synchronize_session=False)
    db.commit()

    attempt = PracticeAttempt(
        user_id               = user_id,
        session_id            = session_id,
        module                = "writing",
        question_type         = "sectional",
        filter_type           = "sectional",
        total_questions       = len(selected_qs),
        total_score           = 0,
        status                = "in_progress",
        scoring_status        = "pending",
        selected_question_ids = [q.question_id for q in selected_qs],
        task_breakdown        = {"test_number": test_number},
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    task_timing = {t["task"]: t for t in WRITING_STRUCTURE}
    questions_payload = []
    for q in selected_qs:
        timing = task_timing.get(q.question_type, {"time_seconds": 600})
        questions_payload.append({
            "question_id":   q.question_id,
            "task_type":     q.question_type,
            "time_seconds":  timing["time_seconds"],
            "content_json":  q.content_json,
            "presigned_url": _maybe_presigned_url(q),
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
        "attempt_id":      attempt.id,
        "test_number":     test_number,
        "total_questions": len(questions_payload),
        "total_minutes":   round(sum(t["count"] * t["time_seconds"] / 60 for t in WRITING_STRUCTURE), 1),
        "questions":       questions_payload,
    }


def resume_writing_sectional_exam(session_id: str, user_id: int, db: Session) -> dict:
    """Rebuild session from DB and return remaining questions with is_submitted flags."""
    from db.models import AttemptAnswer
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="writing", status="in_progress",
    ).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="No resumable writing session found")

    qid_order = attempt.selected_question_ids or []
    if not qid_order:
        raise HTTPException(status_code=404, detail="Writing attempt has no stored questions")

    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        qs_by_id = {
            q.question_id: q
            for q in db.query(QuestionFromApeuni)
            .options(joinedload(QuestionFromApeuni.evaluation))
            .filter(QuestionFromApeuni.question_id.in_(qid_order))
            .all()
        }
        submitted = {
            a.question_id
            for a in db.query(AttemptAnswer).filter_by(attempt_id=attempt.id).all()
        }
        selected = [qs_by_id[qid] for qid in qid_order if qid in qs_by_id]
        tb = attempt.task_breakdown or {}
        ACTIVE_SESSIONS[session_id] = {
            "user_id":              user_id,
            "test_number":          tb.get("test_number", 1),
            "start_time":           int(time.time()),
            "questions":            {q.question_id: q for q in selected},
            "submitted_questions":  submitted,
            "score":                0,
            "question_scores":      {},
            "question_score_maxes": {},
            "module":               "writing",
            "question_type":        "sectional",
            "attempt_id":           attempt.id,
        }

    session   = ACTIVE_SESSIONS[session_id]
    submitted = session["submitted_questions"]
    task_timing = {t["task"]: t for t in WRITING_STRUCTURE}

    questions_payload = []
    for qid in qid_order:
        q = session["questions"].get(qid)
        if not q:
            continue
        timing = task_timing.get(q.question_type, {"time_seconds": 600})
        questions_payload.append({
            "question_id":   q.question_id,
            "task_type":     q.question_type,
            "time_seconds":  timing["time_seconds"],
            "content_json":  q.content_json,
            "presigned_url": _maybe_presigned_url(q),
            "session_id":    session_id,
            "is_submitted":  qid in submitted,
        })

    print(
        f"[Writing Sectional] Resumed session={session_id} user={user_id} "
        f"submitted={len(submitted)}/{len(qid_order)}",
        flush=True,
    )
    return {
        "session_id":      session_id,
        "attempt_id":      attempt.id,
        "total_questions": len(qid_order),
        "submitted_count": len(submitted),
        "questions":       questions_payload,
    }


def finish_writing_sectional(session_id: str, user_id: int, db: Session) -> dict:
    """
    Weighted scoring across writing sectional task types.
    task_pct = average of per-Q raw fractions (mirrors listening/reading/speaking).
    final = max(10, min(90, round(10 + normalised_pct * 80)))
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

    # Build per-task buckets — mirrors the listening/reading/speaking pattern:
    # task_pct is the average of per-Q raw fractions, so each question carries
    # equal weight regardless of its rubric max. earned_raw/max_raw are kept
    # for diagnostic display only.
    task_buckets: dict = {}
    for qid, q in questions.items():
        t = q.question_type
        if t not in task_buckets:
            task_buckets[t] = {
                "earned_pct_sum": 0.0,
                "earned_raw":     0.0,
                "max_raw":        0.0,
                "total":          0,
                "answered":       0,
            }

        q_max = q_score_maxes.get(qid) or _question_max(q)
        task_buckets[t]["max_raw"] += q_max
        task_buckets[t]["total"]   += 1

        if qid in submitted:
            earned   = q_scores.get(qid, 0)
            per_q_pct = (earned / q_max) if q_max > 0 else 0.0
            task_buckets[t]["earned_pct_sum"] += per_q_pct
            task_buckets[t]["earned_raw"]     += earned
            task_buckets[t]["answered"]       += 1

    # Weighted aggregation
    weighted_sum   = 0.0
    present_weight = sum(_WRITING_WEIGHTS.values())
    task_breakdown: dict = {}

    for task_type, bucket in task_buckets.items():
        weight = _WRITING_WEIGHTS.get(task_type, 0)
        total  = bucket["total"]

        task_pct     = (bucket["earned_pct_sum"] / total) if total > 0 else 0.0
        contribution = task_pct * weight

        weighted_sum += contribution

        task_breakdown[task_type] = {
            "display_name":       _display_name(task_type),
            "total_questions":    total,
            "questions_answered": bucket["answered"],
            "earned_raw":         round(bucket["earned_raw"], 2),
            "max_raw":            round(bucket["max_raw"], 2),
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
        prior_tb = existing.task_breakdown or {}
        if "test_number" in prior_tb:
            task_breakdown["test_number"] = prior_tb["test_number"]
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
    from db.models import AttemptAnswer
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
        "scoring_status":     attempt.scoring_status or "complete",
        "writing_score":      attempt.total_score,
        "task_breakdown":     attempt.task_breakdown or {},
        "total_questions":    attempt.total_questions,
        "questions_answered": attempt.questions_answered,
        "completed_at":       attempt.completed_at.isoformat() if attempt.completed_at else None,
        "questions":          questions,
    }
