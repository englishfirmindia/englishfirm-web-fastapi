"""
Reading Sectional Service
=========================
Selects questions for the reading sectional exam (8 task types, 17 questions),
stores the session, and handles weighted scoring at finish.

Task types in APEuni Part II order:
  summarize_written_text   — 1 question  (module: writing)
  reading_fib_drop_down    — 5 questions (module: reading)
  mcq_multiple             — 2 questions (module: reading)
  reorder_paragraphs       — 2 questions (module: reading)
  reading_fib              — 4 questions (module: reading)
  mcq_single               — 2 questions (module: reading)
  hcs                      — 2 questions (module: listening)
  highlight_incorrect_words — 1 question (module: listening)

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
from services.scoring.base import to_pte_score
import core.config as config

from core.logging_config import get_logger

log = get_logger(__name__)


def _earned_pct_from_answer(score: int) -> float:
    """Convert a stored pte_score (10–90) to a 0.0–1.0 fraction for weighted aggregation.

    Mirrors listening/speaking sectional aggregation so per-Q contributions are
    averaged uniformly across the three modules.
    """
    return max(0.0, (score - config.PTE_BASE) / config.PTE_SCALE)

_AUDIO_TASKS = {"listening_hcs", "highlight_incorrect_words"}

# ─── Sectional structure ──────────────────────────────────────────────────────
READING_STRUCTURE = [
    {"task": "summarize_written_text",    "count": 1, "module": "writing"},
    {"task": "reading_fib_drop_down",     "count": 6, "module": "reading"},
    {"task": "reading_drag_and_drop",     "count": 4, "module": "reading"},
    {"task": "mcq_multiple",              "count": 3, "module": "reading"},
    {"task": "reorder_paragraphs",        "count": 1, "module": "reading"},
    {"task": "reading_fib",               "count": 4, "module": "reading"},
    {"task": "mcq_single",                "count": 2, "module": "reading"},
    {"task": "listening_hcs",             "count": 2, "module": "listening"},
    {"task": "highlight_incorrect_words", "count": 2, "module": "listening"},
]

_READING_WEIGHTS = {
    "summarize_written_text":    23,
    "reading_fib_drop_down":     25,
    "reading_drag_and_drop":     20,
    "mcq_multiple":               5,
    "reorder_paragraphs":         9,
    "reading_fib":               13,
    "mcq_single":                 3,
    "listening_hcs":              3,
    "highlight_incorrect_words": 13,
}

_DISPLAY_NAMES = {
    "summarize_written_text":    "Summarize Written Text",
    "reading_fib_drop_down":     "Fill in the Blanks (Dropdown)",
    "reading_drag_and_drop":     "Fill in the Blanks (Drag & Drop)",
    "mcq_multiple":              "Multiple Choice (Multiple)",
    "reorder_paragraphs":        "Re-order Paragraphs",
    "reading_fib":               "Fill in the Blanks",
    "mcq_single":                "Multiple Choice (Single)",
    "listening_hcs":             "Highlight Correct Summary",
    "highlight_incorrect_words": "Highlight Incorrect Words",
}


def _display_name(task_type: str) -> str:
    return _DISPLAY_NAMES.get(task_type, task_type.replace("_", " ").title())


def _question_max(q) -> int:
    """Return max achievable raw score for one question using rubric rules."""
    qt = q.question_type

    if qt == "summarize_written_text":
        return 1

    try:
        ev    = q.evaluation.evaluation_json
        rules = ev.get("scoringRules", {})
        ans   = ev.get("correctAnswers", {})

        if qt in ("reading_fib", "reading_fib_drop_down", "reading_drag_and_drop"):
            return len(ans) * rules.get("marksPerBlank", 1)

        if qt == "mcq_multiple":
            return len(ans.get("correctOptions", [])) * rules.get("marksPerCorrect", 1)

        if qt in ("mcq_single", "listening_hcs"):
            return rules.get("marksPerCorrect", 1)

        if qt == "reorder_paragraphs":
            seq = ans.get("correctSequence", [])
            return max(0, len(seq) - 1) * rules.get("marksPerAdjacentPair", 1)

        if qt == "highlight_incorrect_words":
            incorrect = ans.get("incorrectWordIndices", [])
            return len(incorrect) if incorrect else 1

    except Exception:
        pass

    return {
        "summarize_written_text":    1,
        "reading_fib_drop_down":     4,
        "reading_drag_and_drop":     4,
        "reading_fib":               4,
        "mcq_multiple":              2,
        "mcq_single":                1,
        "reorder_paragraphs":        3,
        "listening_hcs":             1,
        "highlight_incorrect_words": 1,
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
    """Select questions, create session, return question list.

    Question set is locked per (module, test_number) in
    `sectional_test_questions` — every user, every redo, same questions.
    """
    from services.sectional_test_catalog import get_locked_question_ids

    locked_ids = get_locked_question_ids(db, "reading", test_number)
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
            "[Reading Sectional] %d locked question_ids missing for test %d: %s",
            len(missing), test_number, missing,
        )

    if not selected_qs:
        raise HTTPException(status_code=404, detail="No reading questions available")

    session_id = str(uuid.uuid4())

    # Remove any incomplete previous attempts so history stays clean
    db.query(PracticeAttempt).filter(
        PracticeAttempt.user_id        == user_id,
        PracticeAttempt.module         == "reading",
        PracticeAttempt.question_type  == "sectional",
        PracticeAttempt.status         != "complete",
    ).delete(synchronize_session=False)
    db.commit()

    attempt = PracticeAttempt(
        user_id               = user_id,
        session_id            = session_id,
        module                = "reading",
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

    questions_payload = []
    for q in selected_qs:
        entry = {
            "question_id":  q.question_id,
            "task_type":    q.question_type,
            "content_json": q.content_json,
            "session_id":   session_id,
        }
        if q.question_type in _AUDIO_TASKS:
            audio_url = (q.content_json or {}).get("audio_url", "")
            if audio_url:
                try:
                    entry["presigned_url"] = generate_presigned_url(audio_url)
                except Exception:
                    entry["presigned_url"] = audio_url
        questions_payload.append(entry)

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
        "attempt_id":           attempt.id,
    }

    log.info(f"[Reading Sectional] Started session={session_id} user={user_id} " f"questions={len(selected_qs)}")

    return {
        "session_id":      session_id,
        "attempt_id":      attempt.id,
        "test_number":     test_number,
        "total_questions": len(questions_payload),
        "questions":       questions_payload,
    }


def resume_reading_sectional_exam(session_id: str, user_id: int, db: Session) -> dict:
    """Rebuild session from DB and return remaining questions with is_submitted flags."""
    from db.models import AttemptAnswer
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="reading", status="in_progress",
    ).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="No resumable reading session found")

    qid_order = attempt.selected_question_ids or []
    if not qid_order:
        raise HTTPException(status_code=404, detail="Reading attempt has no stored questions")

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
            "module":               "reading",
            "question_type":        "sectional",
            "attempt_id":           attempt.id,
        }

    session   = ACTIVE_SESSIONS[session_id]
    submitted = session["submitted_questions"]

    questions_payload = []
    for qid in qid_order:
        q = session["questions"].get(qid)
        if not q:
            continue
        entry = {
            "question_id":  q.question_id,
            "task_type":    q.question_type,
            "content_json": q.content_json,
            "session_id":   session_id,
            "is_submitted": qid in submitted,
        }
        if q.question_type in _AUDIO_TASKS:
            audio_url = (q.content_json or {}).get("audio_url", "")
            if audio_url:
                try:
                    entry["presigned_url"] = generate_presigned_url(audio_url)
                except Exception:
                    entry["presigned_url"] = audio_url
        questions_payload.append(entry)

    log.info(f"[Reading Sectional] Resumed session={session_id} user={user_id} " f"submitted={len(submitted)}/{len(qid_order)}")
    return {
        "session_id":      session_id,
        "attempt_id":      attempt.id,
        "total_questions": len(qid_order),
        "submitted_count": len(submitted),
        "questions":       questions_payload,
    }


def finish_reading_sectional(session_id: str, user_id: int, db: Session) -> dict:
    """
    Weighted scoring over the reading sectional task types.
    Formula: max(10, min(90, round(10 + normalised_pct * 80)))

    Reads entirely from RDS (PracticeAttempt + AttemptAnswer + QuestionFromApeuni)
    so aggregation matches listening/speaking sectional and survives backend
    restarts. Per-Q PTE comes from `attempt_answers.score`, written at submit time.
    """
    from db.models import AttemptAnswer

    existing = db.query(PracticeAttempt).filter_by(session_id=session_id).first()
    if not existing:
        raise HTTPException(status_code=400, detail="Session not found or expired")
    if existing.scoring_status == "complete":
        return {
            "attempt_id":     existing.id,
            "session_id":     session_id,
            "scoring_status": "complete",
            "reading_score":  existing.total_score,
        }

    qid_order = existing.selected_question_ids or []
    if not qid_order:
        raise HTTPException(status_code=400, detail="Reading attempt has no stored questions")

    # 1. Question metadata — needed for type lookup and `max_raw` on unanswered Qs.
    qs_rows = (
        db.query(QuestionFromApeuni)
        .options(joinedload(QuestionFromApeuni.evaluation))
        .filter(QuestionFromApeuni.question_id.in_(qid_order))
        .all()
    )
    questions = {q.question_id: q for q in qs_rows}

    # 2. Per-Q PTE comes from attempt_answers.score — same source as listening/speaking.
    answered_rows = db.query(AttemptAnswer).filter_by(attempt_id=existing.id).all()
    answered_by_qid = {row.question_id: row for row in answered_rows}

    # 3. Per-task buckets — averaged per-Q pct, identical pattern to listening/speaking.
    # earned_raw/max_raw are kept for diagnostic display (mobile feedback page) only.
    task_buckets: dict = {}
    for qid in qid_order:
        q = questions.get(qid)
        if not q:
            continue  # question deleted from DB — skip
        t = q.question_type
        if t not in task_buckets:
            task_buckets[t] = {
                "earned_pct_sum": 0.0,
                "earned_raw":     0.0,
                "max_raw":        0.0,
                "total":          0,
                "answered":       0,
            }

        row = answered_by_qid.get(qid)
        res = (row.result_json or {}) if row else {}
        q_max = res.get("maxScore") or _question_max(q)

        task_buckets[t]["max_raw"] += q_max
        task_buckets[t]["total"]   += 1

        if row:
            task_buckets[t]["earned_raw"]     += res.get("earned_raw", 0)
            task_buckets[t]["answered"]       += 1
            task_buckets[t]["earned_pct_sum"] += _earned_pct_from_answer(row.score or 0)

    # Weighted aggregation: skipped Qs count toward `total` so skipping hurts;
    # task types with zero Qs in the exam stay out of the bucket entirely.
    weighted_sum   = 0.0
    present_weight = sum(_READING_WEIGHTS.values())
    task_breakdown: dict = {}

    for task_type, bucket in task_buckets.items():
        weight = _READING_WEIGHTS.get(task_type, 0)
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
            "reading_weight":     weight,
            "contribution":       round(contribution, 2),
        }

    normalised_pct = (weighted_sum / present_weight) if present_weight > 0 else 0.0
    scaled = max(
        config.PTE_FLOOR,
        min(config.PTE_CEILING, round(config.PTE_BASE + normalised_pct * config.PTE_SCALE)),
    )

    prior_tb = existing.task_breakdown or {}
    if "test_number" in prior_tb:
        task_breakdown["test_number"] = prior_tb["test_number"]
    existing.total_score        = scaled
    existing.questions_answered = len(answered_by_qid)
    existing.status             = "complete"
    existing.scoring_status     = "complete"
    existing.task_breakdown     = task_breakdown
    existing.completed_at       = datetime.now(timezone.utc)
    db.commit()
    db.refresh(existing)

    log.info(f"[Reading Sectional] Finished session={session_id} score={scaled} " f"norm_pct={round(normalised_pct * 100, 1)}% " f"answered={len(answered_by_qid)}/{len(qid_order)}")

    return {
        "attempt_id":     existing.id,
        "session_id":     session_id,
        "scoring_status": "complete",
        "reading_score":  scaled,
        "weighted_pct":   round(normalised_pct * 100, 1),
        "task_breakdown": task_breakdown,
    }


def get_reading_sectional_results(session_id: str, user_id: int, db: Session) -> dict:
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
        "reading_score":      attempt.total_score,
        "task_breakdown":     attempt.task_breakdown or {},
        "total_questions":    attempt.total_questions,
        "questions_answered": attempt.questions_answered,
        "completed_at":       attempt.completed_at.isoformat() if attempt.completed_at else None,
        "questions":          questions,
    }
