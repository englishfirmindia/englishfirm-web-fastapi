"""
Speaking Sectional Service
==========================
Selects questions for the speaking sectional exam (7 task types, 32 questions),
stores the session, persists a PracticeAttempt row, and handles finish/results.

Scoring:
  All 7 task types are async — Azure speech scoring via kick_off_scoring.
  Results are polled from _SCORE_STORE.

PTE formula (CLAUDE.md guardrail):
  pte_score = max(10, min(90, round(10 + weighted_pct * 80)))
"""

import random
import time
import uuid
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

from db.models import QuestionFromApeuni, UserQuestionAttempt, PracticeAttempt
from services.session_service import ACTIVE_SESSIONS, _SCORE_STORE
from services.s3_service import generate_presigned_url
from services.speaking_scorer import kick_off_scoring
from services.scoring.azure_scorer import _compute_question_score
import core.config as config

# ─── Sectional structure ──────────────────────────────────────────────────────
SPEAKING_STRUCTURE = [
    {"task": "read_aloud",                 "count": 6,  "prep_seconds": 35, "rec_seconds": 40},
    {"task": "repeat_sentence",            "count": 10, "prep_seconds": 0,  "rec_seconds": 15},
    {"task": "describe_image",             "count": 3,  "prep_seconds": 25, "rec_seconds": 40},
    {"task": "retell_lecture",             "count": 3,  "prep_seconds": 10, "rec_seconds": 40},
    {"task": "answer_short_question",      "count": 5,  "prep_seconds": 0,  "rec_seconds": 10},
    {"task": "ptea_respond_situation",      "count": 3,  "prep_seconds": 0,  "rec_seconds": 40},
    {"task": "summarize_group_discussion", "count": 2,  "prep_seconds": 10, "rec_seconds": 60},
]

# Weights: speaking contributes equally across all task types (normalised at finish)
_SPEAKING_WEIGHTS = {
    "read_aloud":                 20,
    "repeat_sentence":            17,
    "describe_image":             15,
    "retell_lecture":             13,
    "answer_short_question":       4,
    "ptea_respond_situation":      15,
    "summarize_group_discussion": 16,
}

_DISPLAY_NAMES = {
    "read_aloud":                 "Read Aloud",
    "repeat_sentence":            "Repeat Sentence",
    "describe_image":             "Describe Image",
    "retell_lecture":             "Re-tell Lecture",
    "answer_short_question":      "Answer Short Question",
    "ptea_respond_situation":      "Respond to a Situation",
    "summarize_group_discussion": "Summarize Group Discussion",
}


def _display_name(task_type: str) -> str:
    return _DISPLAY_NAMES.get(task_type, task_type.replace("_", " ").title())


def get_speaking_sectional_info() -> dict:
    total_questions = sum(t["count"] for t in SPEAKING_STRUCTURE)
    total_seconds = sum(
        t["count"] * (t["prep_seconds"] + t["rec_seconds"])
        for t in SPEAKING_STRUCTURE
    )
    sections = [
        {
            "task":              t["task"],
            "display_name":      _display_name(t["task"]),
            "count":             t["count"],
            "prep_seconds":      t["prep_seconds"],
            "rec_seconds":       t["rec_seconds"],
            "time_per_question": t["prep_seconds"] + t["rec_seconds"],
        }
        for t in SPEAKING_STRUCTURE
    ]
    return {
        "total_questions": total_questions,
        "total_minutes":   round(total_seconds / 60, 1),
        "sections":        sections,
    }


def start_speaking_sectional_exam(db: Session, user_id: int, test_number: int) -> dict:
    """Select questions, create session + PracticeAttempt, return question list."""
    # Exclude questions the user has already practiced
    practiced_ids = set(
        row[0]
        for row in db.query(UserQuestionAttempt.question_id)
        .filter(
            UserQuestionAttempt.user_id == user_id,
            UserQuestionAttempt.module  == "speaking",
        )
        .all()
    )

    selected_qs: list = []

    for task in SPEAKING_STRUCTURE:
        task_type = task["task"]
        count     = task["count"]

        fresh = (
            db.query(QuestionFromApeuni)
            .options(joinedload(QuestionFromApeuni.evaluation))
            .filter(
                QuestionFromApeuni.module        == "speaking",
                QuestionFromApeuni.question_type == task_type,
                ~QuestionFromApeuni.question_id.in_(practiced_ids) if practiced_ids else True,
            )
            .all()
        )
        pool = fresh
        if len(fresh) < count:
            print(
                f"[Speaking Sectional] Not enough fresh questions for {task_type} "
                f"(need {count}, have {len(fresh)}) — falling back to full pool",
                flush=True,
            )
            pool = (
                db.query(QuestionFromApeuni)
                .options(joinedload(QuestionFromApeuni.evaluation))
                .filter(
                    QuestionFromApeuni.module        == "speaking",
                    QuestionFromApeuni.question_type == task_type,
                )
                .all()
            )

        n = min(count, len(pool))
        if n == 0:
            print(f"[Speaking Sectional] No questions for {task_type} — skipping", flush=True)
            continue
        selected_qs.extend(random.sample(pool, n))

    if not selected_qs:
        raise HTTPException(status_code=404, detail="No speaking questions available")

    # Mark all selected questions as attempted (deduplication guard)
    seen = practiced_ids.copy()
    new_attempts = []
    for q in selected_qs:
        if q.question_id not in seen:
            new_attempts.append(UserQuestionAttempt(
                user_id=user_id,
                question_id=q.question_id,
                question_type=q.question_type,
                module="speaking",
            ))
            seen.add(q.question_id)
    if new_attempts:
        db.add_all(new_attempts)
        db.commit()

    # Build question payload with presigned URLs
    task_timing = {t["task"]: t for t in SPEAKING_STRUCTURE}
    questions_payload = []

    for q in selected_qs:
        timing = task_timing.get(q.question_type, {"prep_seconds": 25, "rec_seconds": 40})

        presigned_url: Optional[str] = None
        raw_audio = (
            q.content_json.get("audio_url")
            or q.content_json.get("s3_key")
            or q.content_json.get("audio_s3_key")
        )
        if raw_audio:
            if raw_audio.startswith("http"):
                presigned_url = raw_audio  # already a direct URL, no presigning needed
            else:
                try:
                    presigned_url = generate_presigned_url(raw_audio)
                except Exception:
                    presigned_url = None

        presigned_image_url: Optional[str] = None
        raw_image = q.content_json.get("image_url") or q.content_json.get("image_s3_key")
        if raw_image:
            try:
                presigned_image_url = generate_presigned_url(raw_image)
            except Exception:
                presigned_image_url = None

        questions_payload.append({
            "question_id":         q.question_id,
            "task_type":           q.question_type,
            "prep_seconds":        timing["prep_seconds"],
            "rec_seconds":         timing["rec_seconds"],
            "content_json":        q.content_json,
            "presigned_url":       presigned_url,
            "presigned_image_url": presigned_image_url,
            "difficulty_level":    q.difficulty_level,
        })

    # Create PracticeAttempt in DB
    session_id = str(uuid.uuid4())
    attempt = PracticeAttempt(
        user_id               = user_id,
        session_id            = session_id,
        module                = "speaking",
        question_type         = "sectional",
        filter_type           = "sectional",
        total_questions       = len(selected_qs),
        total_score           = 0,
        questions_answered    = 0,
        status                = "pending",
        scoring_status        = "pending",
        selected_question_ids = [q.question_id for q in selected_qs],
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    print(
        f"[Speaking Sectional] Created attempt={attempt.id} session={session_id} "
        f"user={user_id} questions={len(selected_qs)}",
        flush=True,
    )

    # Store session in memory
    ACTIVE_SESSIONS[session_id] = {
        "user_id":             user_id,
        "test_number":         test_number,
        "start_time":          int(time.time()),
        "questions":           {q.question_id: q for q in selected_qs},
        "submitted_questions": set(),
        "submitted_audio":     {},   # {question_id: audio_url}
        "score":               0,
        "question_scores":     {},
        "attempt_id":          attempt.id,
        "module":              "speaking",
        "question_type":       "sectional",
    }

    return {
        "session_id":      session_id,
        "attempt_id":      attempt.id,
        "test_number":     test_number,
        "total_questions": len(questions_payload),
        "questions":       questions_payload,
    }


def finish_speaking_sectional(session_id: str, user_id: int, db: Session) -> dict:
    """
    Kick off Azure scoring for all submitted audio URLs.
    Returns immediately with scoring_status='pending'.
    """
    session_data = ACTIVE_SESSIONS.get(session_id)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found or expired")

    # Idempotency — if already complete, return existing
    attempt = db.query(PracticeAttempt).filter_by(session_id=session_id).first()
    if attempt and attempt.status == "complete":
        return {
            "session_id":      session_id,
            "attempt_id":      attempt.id,
            "scoring_status":  attempt.scoring_status or "pending",
            "speaking_score":  attempt.total_score,
        }

    # Kick off async scoring for every submitted question
    questions    = session_data.get("questions", {})
    submitted_audio = session_data.get("submitted_audio", {})

    kicked = 0
    for qid, audio_url in submitted_audio.items():
        q = questions.get(qid)
        if q and audio_url:
            reference_text = (q.content_json or {}).get("passage", "")
            kick_off_scoring(user_id, qid, q.question_type, audio_url, reference_text)
            kicked += 1

    # Mark attempt as complete (score will be computed in results once polling is done)
    if attempt:
        attempt.status         = "complete"
        attempt.scoring_status = "pending"
        attempt.completed_at   = datetime.now(timezone.utc)
        db.commit()
    else:
        attempt = PracticeAttempt(
            user_id            = user_id,
            session_id         = session_id,
            module             = "speaking",
            question_type      = "sectional",
            filter_type        = "sectional",
            total_questions    = len(questions),
            total_score        = 0,
            questions_answered = len(session_data.get("submitted_questions", set())),
            status             = "complete",
            scoring_status     = "pending",
            completed_at       = datetime.now(timezone.utc),
        )
        db.add(attempt)
        db.commit()
        db.refresh(attempt)

    print(
        f"[Speaking Sectional] Finish session={session_id} "
        f"kicked_scoring={kicked}",
        flush=True,
    )

    return {
        "session_id":     session_id,
        "attempt_id":     attempt.id,
        "status":         "submitted",
        "scoring_status": "pending",
        "kicked_scoring": kicked,
    }


def get_speaking_sectional_results(session_id: str, user_id: int, db: Session) -> dict:
    """
    Poll _SCORE_STORE for all question scores.
    When all async scores are available, compute final PTE score and persist.
    """
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id
    ).first()
    if not attempt:
        return {"scoring_status": "not_found"}

    if attempt.scoring_status == "complete" and attempt.total_score > 0:
        return {
            "attempt_id":      attempt.id,
            "session_id":      session_id,
            "scoring_status":  "complete",
            "speaking_score":  attempt.total_score,
            "task_breakdown":  attempt.task_breakdown or {},
            "total_questions": attempt.total_questions,
        }

    # Gather scores from _SCORE_STORE
    session_data = ACTIVE_SESSIONS.get(session_id)
    submitted_audio = (session_data or {}).get("submitted_audio", {})

    if not submitted_audio:
        # No audio submitted — return pending
        return {
            "attempt_id":     attempt.id,
            "session_id":     session_id,
            "scoring_status": "pending",
            "speaking_score": 0,
        }

    # Check which questions have scores — use rubric-based scoring (matches mock/practice)
    question_scores  = {}   # {qid: float 0-1 from rubric}
    question_details = {}   # {qid: {task_type, content, fluency, pronunciation}}
    pending_count    = 0
    session_questions = (session_data or {}).get("questions", {})

    for qid in submitted_audio:
        result = _SCORE_STORE.get((user_id, qid))
        if result and result.get("scoring") == "complete":
            q         = session_questions.get(qid)
            task_type = q.question_type if q else "read_aloud"
            computed  = _compute_question_score(task_type, result)
            question_scores[qid] = computed["pct"]
            question_details[qid] = {
                "task_type":     task_type,
                "content":       float(result.get("content", 0) or 0),
                "fluency":       float(result.get("fluency", 0) or 0),
                "pronunciation": float(result.get("pronunciation", 0) or 0),
            }
        else:
            pending_count += 1

    if pending_count > 0:
        return {
            "attempt_id":     attempt.id,
            "session_id":     session_id,
            "scoring_status": "pending",
            "scored_count":   len(question_scores),
            "pending_count":  pending_count,
        }

    # All scores are in — compute final PTE score
    task_buckets: dict = {}

    for qid, pct in question_scores.items():
        det       = question_details.get(qid, {})
        task_type = det.get("task_type", "unknown")
        if task_type not in task_buckets:
            task_buckets[task_type] = {
                "earned": 0.0, "count": 0,
                "content_sum": 0.0, "fluency_sum": 0.0, "pronunciation_sum": 0.0,
            }
        task_buckets[task_type]["earned"]           += pct
        task_buckets[task_type]["count"]            += 1
        task_buckets[task_type]["content_sum"]      += det.get("content", 0)
        task_buckets[task_type]["fluency_sum"]      += det.get("fluency", 0)
        task_buckets[task_type]["pronunciation_sum"] += det.get("pronunciation", 0)

    weighted_sum   = 0.0
    present_weight = 0
    task_breakdown: dict = {}

    for task_type, bucket in task_buckets.items():
        weight       = _SPEAKING_WEIGHTS.get(task_type, 0)
        n            = bucket["count"]
        avg_pct      = bucket["earned"] / n if n > 0 else 0.0
        contribution = avg_pct * weight

        weighted_sum   += contribution
        present_weight += weight

        entry = {
            "display_name":  _display_name(task_type),
            "count":         n,
            "avg_pct":       round(avg_pct * 100, 1),
            "weight":        weight,
            "contribution":  round(contribution, 2),
        }
        if n > 0:
            entry["content_avg"]       = round(bucket["content_sum"]       / n, 1)
            entry["fluency_avg"]       = round(bucket["fluency_sum"]       / n, 1)
            entry["pronunciation_avg"] = round(bucket["pronunciation_sum"] / n, 1)
        task_breakdown[task_type] = entry

    normalised_pct = (weighted_sum / present_weight) if present_weight > 0 else 0.0
    scaled = max(config.PTE_FLOOR, min(config.PTE_CEILING, round(config.PTE_BASE + normalised_pct * config.PTE_SCALE)))

    # Persist final score
    attempt.total_score        = scaled
    attempt.scoring_status     = "complete"
    attempt.questions_answered = len(question_scores)
    attempt.task_breakdown     = task_breakdown
    db.commit()

    return {
        "attempt_id":      attempt.id,
        "session_id":      session_id,
        "scoring_status":  "complete",
        "speaking_score":  scaled,
        "weighted_pct":    round(normalised_pct * 100, 1),
        "task_breakdown":  task_breakdown,
        "total_questions": attempt.total_questions,
    }
