"""
Speaking Sectional Service
==========================
Selects questions for the speaking sectional exam (7 task types, 32 questions),
stores the session, persists a PracticeAttempt row, and handles finish/results.

Scoring:
  All 7 task types are async — Azure scoring kicked off at /submit-audio time.
  Results aggregated by _speaking_aggregate_bg thread (RDS-based, restart-safe).

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

from db.models import QuestionFromApeuni, UserQuestionAttempt, PracticeAttempt, AttemptAnswer
from db.database import SessionLocal
from sqlalchemy.orm.attributes import flag_modified
from services.session_service import ACTIVE_SESSIONS
from services.s3_service import generate_presigned_url
from services.speaking_scorer import kick_off_scoring
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

    # Remove any incomplete previous attempts so history stays clean
    db.query(PracticeAttempt).filter(
        PracticeAttempt.user_id        == user_id,
        PracticeAttempt.module         == "speaking",
        PracticeAttempt.question_type  == "sectional",
        PracticeAttempt.status         != "complete",
    ).delete(synchronize_session=False)
    db.commit()

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
        task_breakdown        = {"test_number": test_number},
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


def resume_speaking_sectional_exam(session_id: str, user_id: int, db: Session) -> dict:
    """Rebuild session from DB and return questions with fresh presigned URLs + is_submitted flags."""
    from db.models import AttemptAnswer
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="speaking",
    ).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="No resumable speaking session found")

    qid_order = attempt.selected_question_ids or []
    if not qid_order:
        raise HTTPException(status_code=404, detail="Speaking attempt has no stored questions")

    submitted_answers = {
        a.question_id: a
        for a in db.query(AttemptAnswer).filter_by(attempt_id=attempt.id).all()
    }
    submitted_ids = set(submitted_answers.keys())

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
        ACTIVE_SESSIONS[session_id] = {
            "user_id":             user_id,
            "test_number":         1,
            "start_time":          int(time.time()),
            "questions":           {q.question_id: q for q in selected},
            "submitted_questions": submitted_ids,
            "submitted_audio":     {
                a.question_id: (a.user_answer_json or {}).get("audio_url", "")
                for a in submitted_answers.values()
            },
            "score":               0,
            "question_scores":     {},
            "attempt_id":          attempt.id,
            "module":              "speaking",
            "question_type":       "sectional",
        }

    session = ACTIVE_SESSIONS[session_id]
    task_timing = {t["task"]: t for t in SPEAKING_STRUCTURE}

    questions_payload = []
    for qid in qid_order:
        q = session["questions"].get(qid)
        if not q:
            continue
        timing = task_timing.get(q.question_type, {"prep_seconds": 25, "rec_seconds": 40})

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
            "is_submitted":        qid in submitted_ids,
        })

    print(
        f"[Speaking Sectional] Resumed session={session_id} user={user_id} "
        f"submitted={len(submitted_ids)}/{len(qid_order)}",
        flush=True,
    )
    return {
        "session_id":      session_id,
        "attempt_id":      attempt.id,
        "total_questions": len(qid_order),
        "submitted_count": len(submitted_ids),
        "questions":       questions_payload,
    }


def _earned_pct_from_answer(score: int) -> float:
    """Convert a stored PTE score (10–90, or 0 for no-speech) to 0.0–1.0 fraction."""
    return max(0.0, (score - config.PTE_BASE) / config.PTE_SCALE)


def _speaking_aggregate_bg(
    attempt_id: int,
    session_id: str,
    user_id: int,
    all_question_ids: list,
) -> None:
    """Background thread: compute weighted speaking score entirely from RDS.

    Waits for all AttemptAnswer rows to reach scoring_status='complete', then
    applies the same weighted-average formula as listening sectional.
    Resilient to server restarts — no in-memory state required.
    """
    import traceback

    bg_db = SessionLocal()
    try:
        # 1. Resolve question types from DB
        qs_rows = (
            bg_db.query(QuestionFromApeuni.question_id, QuestionFromApeuni.question_type)
            .filter(QuestionFromApeuni.question_id.in_(all_question_ids))
            .all()
        )
        type_by_qid = {r.question_id: r.question_type for r in qs_rows}

        # 2. Wait for all answers to have scoring_status='complete' (max 300 s)
        deadline = time.time() + 300
        print(
            f"[SpeakingBG] Waiting for {len(all_question_ids)} scores in RDS "
            f"attempt={attempt_id}…",
            flush=True,
        )
        while time.time() < deadline:
            pending = (
                bg_db.query(AttemptAnswer)
                .filter(
                    AttemptAnswer.attempt_id == attempt_id,
                    AttemptAnswer.question_id.in_(all_question_ids),
                    AttemptAnswer.scoring_status == "pending",
                )
                .count()
            )
            if pending == 0:
                break
            time.sleep(3)
        else:
            print(
                f"[SpeakingBG] ⏱ Timed out after 300 s for attempt={attempt_id} — "
                "computing with available scores",
                flush=True,
            )

        bg_db.expire_all()

        # 3. Load all answered rows
        answered_rows = (
            bg_db.query(AttemptAnswer)
            .filter_by(attempt_id=attempt_id)
            .all()
        )
        answered_by_qid = {row.question_id: row for row in answered_rows}

        # 4. Build per-task buckets
        # denominator = total questions asked (unanswered / error = 0, penalised fairly)
        task_buckets: dict = {}
        for qid in all_question_ids:
            t = type_by_qid.get(qid)
            if not t:
                continue
            if t not in task_buckets:
                task_buckets[t] = {"earned_pct_sum": 0.0, "total": 0, "answered": 0}
            task_buckets[t]["total"] += 1

            row = answered_by_qid.get(qid)
            if not row:
                continue
            task_buckets[t]["answered"] += 1
            task_buckets[t]["earned_pct_sum"] += _earned_pct_from_answer(row.score or 0)

        # 5. Weighted PTE score
        weighted_sum   = 0.0
        present_weight = 0
        task_breakdown: dict = {}

        for task_type, bucket in task_buckets.items():
            weight       = _SPEAKING_WEIGHTS.get(task_type, 0)
            total        = bucket["total"]
            answered     = bucket["answered"]
            task_pct     = bucket["earned_pct_sum"] / total if total > 0 else 0.0
            contribution = task_pct * weight
            weighted_sum   += contribution
            present_weight += weight
            task_breakdown[task_type] = {
                "display_name":       _display_name(task_type),
                "total_questions":    total,
                "questions_answered": answered,
                "task_pct":           round(task_pct * 100, 1),
                "speaking_weight":    weight,
                "contribution":       round(contribution, 2),
            }

        normalised_pct = (weighted_sum / present_weight) if present_weight > 0 else 0.0
        scaled = max(
            config.PTE_FLOOR,
            min(config.PTE_CEILING, round(config.PTE_BASE + normalised_pct * config.PTE_SCALE)),
        )

        # 6. Write back to DB
        attempt = bg_db.query(PracticeAttempt).filter_by(id=attempt_id).first()
        if attempt:
            attempt.total_score        = scaled
            attempt.questions_answered = len(answered_by_qid)
            attempt.status             = "complete"
            attempt.scoring_status     = "complete"
            attempt.task_breakdown     = task_breakdown
            attempt.completed_at       = datetime.now(timezone.utc)
            flag_modified(attempt, "task_breakdown")
            bg_db.commit()
            print(
                f"[SpeakingBG] ✅ user={user_id} session={session_id} "
                f"score={scaled} norm_pct={round(normalised_pct * 100, 1)}% "
                f"answered={len(answered_by_qid)}/{len(all_question_ids)}",
                flush=True,
            )
        else:
            print(f"[SpeakingBG] ❌ attempt_id={attempt_id} not found in DB", flush=True)

    except Exception as e:
        print(f"[SpeakingBG] ❌ Failed session={session_id}: {e}", flush=True)
        traceback.print_exc()
    finally:
        bg_db.close()


def finish_speaking_sectional(session_id: str, user_id: int, db: Session) -> dict:
    """
    Start RDS-based background aggregation and return immediately with scoring_status='pending'.
    Azure scoring is already kicked off per-question at /submit-audio time.
    The background thread waits for all scores to land in RDS then computes the
    weighted PTE score. Resilient to server restarts — no in-memory state needed.
    """
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="speaking",
    ).first()
    if not attempt:
        raise HTTPException(status_code=400, detail="Session not found")

    # Idempotency — if already scored, return existing result
    if attempt.scoring_status == "complete":
        return {
            "attempt_id":     attempt.id,
            "session_id":     session_id,
            "scoring_status": "complete",
            "speaking_score": attempt.total_score,
        }

    all_question_ids = attempt.selected_question_ids or []
    if not all_question_ids:
        raise HTTPException(status_code=400, detail="No questions found for this session")

    # Re-kick Azure scoring for any AttemptAnswer rows still pending.
    # Handles the server-restart case where the original kick_off_scoring threads are gone.
    pending_answers = (
        db.query(AttemptAnswer)
        .filter_by(attempt_id=attempt.id, scoring_status="pending")
        .all()
    )
    rekicked = 0
    for ans in pending_answers:
        audio_url = (ans.user_answer_json or {}).get("audio_url", "")
        if not audio_url:
            continue
        q = db.query(QuestionFromApeuni).filter_by(question_id=ans.question_id).first()
        if q:
            cj = q.content_json or {}
            ref = cj.get("transcript", "") if q.question_type == "repeat_sentence" else cj.get("passage", "")
            kick_off_scoring(user_id, ans.question_id, q.question_type, audio_url, ref)
            rekicked += 1

    attempt.status       = "complete"
    attempt.completed_at = datetime.now(timezone.utc)
    db.commit()

    print(
        f"[SpeakingFinish] session={session_id} user={user_id} "
        f"attempt={attempt.id} questions={len(all_question_ids)} rekicked={rekicked}",
        flush=True,
    )

    threading.Thread(
        target=_speaking_aggregate_bg,
        kwargs=dict(
            attempt_id       = attempt.id,
            session_id       = session_id,
            user_id          = user_id,
            all_question_ids = all_question_ids,
        ),
        daemon=True,
    ).start()

    return {
        "attempt_id":     attempt.id,
        "session_id":     session_id,
        "scoring_status": "pending",
        "message":        "Scoring in progress. Check back in a moment.",
    }


def get_speaking_sectional_results(session_id: str, user_id: int, db: Session) -> dict:
    """
    Read scoring status and result entirely from RDS.
    The background thread (_speaking_aggregate_bg) owns the final score computation
    and sets scoring_status='complete' when done.
    """
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
            "question_id":    a.question_id,
            "question_type":  a.question_type,
            "score":          a.score,
            "result_json":    a.result_json or {},
            "scoring_status": a.scoring_status,
        }
        for a in answers
    ]

    return {
        "attempt_id":         attempt.id,
        "session_id":         session_id,
        "scoring_status":     attempt.scoring_status or "pending",
        "speaking_score":     attempt.total_score,
        "task_breakdown":     attempt.task_breakdown or {},
        "total_questions":    attempt.total_questions,
        "questions_answered": attempt.questions_answered,
        "completed_at":       attempt.completed_at.isoformat() if attempt.completed_at else None,
        "questions":          questions,
    }
