"""
MCP Tool implementations — DB query layer.

Tool catalogue:
  get_user_profile        — static profile (score target, exam date)
  get_recent_scores       — last N practice attempt scores across all modules
  get_weak_areas          — task types where avg score < threshold
  get_exam_history        — per-module attempt breakdown
  get_speaking_detail     — per-task-type speaking breakdown from last attempt
  get_trainer_profile     — long-term AI trainer memory for this student
  save_trainer_info       — partial update of trainer profile fields
  get_new_practice_since  — completed attempts since a given timestamp
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

log = logging.getLogger(__name__)


# ── Tool: get_user_profile ────────────────────────────────────────────────────

def get_user_profile(user, db: Session) -> dict:
    days_to_exam: Optional[int] = None
    if user.exam_date:
        try:
            exam = user.exam_date if isinstance(user.exam_date, date) else user.exam_date.date()
            days_to_exam = max(0, (exam - date.today()).days)
        except Exception:
            pass

    return {
        "username":          user.username,
        "score_requirement": user.score_requirement,
        "exam_date":         user.exam_date.isoformat() if user.exam_date else None,
        "days_to_exam":      days_to_exam,
        "current_plan":      getattr(user, "current_plan", None),
    }


# ── Tool: get_recent_scores ───────────────────────────────────────────────────

def get_recent_scores(user_id: int, db: Session, limit: int = 5) -> list[dict]:
    from db.models import PracticeAttempt

    attempts = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.status == "complete",
            PracticeAttempt.filter_type.in_(["sectional", "mock"]),
        )
        .order_by(PracticeAttempt.id.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "attempt_id":      a.id,
            "module":          a.module,
            "score":           a.total_score,
            "total_questions": a.total_questions,
            "completed_at":    a.completed_at.isoformat() if a.completed_at else None,
        }
        for a in attempts
    ]


# ── Tool: get_weak_areas ──────────────────────────────────────────────────────

def get_weak_areas(user_id: int, db: Session, threshold_pct: float = 0.55) -> list[dict]:
    from db.models import AttemptAnswer, PracticeAttempt

    rows = (
        db.query(
            AttemptAnswer.question_type,
            func.avg(AttemptAnswer.score).label("avg_score"),
            func.count(AttemptAnswer.id).label("attempt_count"),
        )
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.status == "complete",
            AttemptAnswer.scoring_status == "complete",
        )
        .group_by(AttemptAnswer.question_type)
        .all()
    )

    _MAX_PTS = {
        "read_aloud": 15, "repeat_sentence": 13, "describe_image": 16,
        "retell_lecture": 16, "respond_to_situation": 16,
        "summarize_group_discussion": 16, "answer_short_question": 1,
        "summarize_written_text": 10, "write_essay": 15,
        "reading_and_writing_fib": 1, "reading_mcm": 1, "reading_mcs": 1,
        "re_order_paragraph": 1, "listening_fib": 1, "write_from_dictation": 1,
        "highlight_correct_summary": 1, "highlight_incorrect_words": 1,
        "select_missing_word": 1, "summarize_spoken_text": 10,
    }

    weak = []
    for qt, avg_score, count in rows:
        if count < 2:
            continue
        max_pts = _MAX_PTS.get(qt, 10)
        pct = (avg_score or 0) / max_pts
        if pct < threshold_pct:
            weak.append({
                "task_type":     qt,
                "avg_score":     round(float(avg_score or 0), 2),
                "max_score":     max_pts,
                "pct":           round(pct, 3),
                "attempt_count": count,
            })

    weak.sort(key=lambda x: x["pct"])
    return weak


# ── Tool: get_exam_history ────────────────────────────────────────────────────

def get_exam_history(user_id: int, db: Session, module: Optional[str] = None) -> list[dict]:
    from db.models import PracticeAttempt

    q = db.query(PracticeAttempt).filter(
        PracticeAttempt.user_id == user_id,
        PracticeAttempt.status == "complete",
    )
    if module:
        q = q.filter(PracticeAttempt.module == module)

    attempts = q.order_by(PracticeAttempt.id.desc()).all()

    by_module: dict[str, list] = {}
    for a in attempts:
        by_module.setdefault(a.module, []).append(a.total_score or 0)

    return [
        {
            "module":     mod,
            "attempts":   len(scores),
            "avg_score":  round(sum(scores) / len(scores), 1),
            "best_score": max(scores),
            "last_score": scores[0],
        }
        for mod, scores in by_module.items()
    ]


# ── Tool: get_speaking_detail ─────────────────────────────────────────────────

def get_speaking_detail(user_id: int, db: Session) -> Optional[dict]:
    from db.models import PracticeAttempt

    last = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.module == "speaking",
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.id.desc())
        .first()
    )
    if not last:
        return None

    breakdown = last.task_breakdown or {}
    summary = {}
    for task, data in breakdown.items():
        summary[task] = {
            "avg_total":  data.get("avg_total", 0),
            "max_points": data.get("max_points", 1),
            "pct":        data.get("pct", 0),
            "count":      data.get("count", 0),
        }

    return {"attempt_id": last.id, "score": last.total_score, "breakdown": summary}


# ── Tool: get_trainer_profile ─────────────────────────────────────────────────

def get_trainer_profile(user_id: int, db: Session) -> dict:
    """
    Reads the trainer profile for this student.
    Creates a default row if none exists yet.
    """
    from db.models import StudentTrainerProfile

    profile = db.query(StudentTrainerProfile).filter(
        StudentTrainerProfile.user_id == user_id
    ).first()

    if profile is None:
        profile = StudentTrainerProfile(user_id=user_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)

    return {
        "phase":                  profile.phase,
        "session_count":          profile.session_count,
        "last_session_at":        profile.last_session_at.isoformat() if profile.last_session_at else None,
        "motivation":             profile.motivation,
        "study_hours_per_day":    profile.study_hours_per_day,
        "study_schedule":         profile.study_schedule,
        "prior_pte_attempts":     profile.prior_pte_attempts,
        "anxiety_level":          profile.anxiety_level,
        "learning_style":         profile.learning_style,
        "biggest_weakness_self":  profile.biggest_weakness_self,
        "plan_text":              profile.plan_text,
        "plan_generated_at":      profile.plan_generated_at.isoformat() if profile.plan_generated_at else None,
        "last_session_summary":   profile.last_session_summary,
    }


# ── Tool: save_trainer_info ───────────────────────────────────────────────────

TRAINER_PROFILE_FIELDS = {
    "motivation", "study_hours_per_day", "study_schedule",
    "prior_pte_attempts", "anxiety_level", "learning_style",
    "biggest_weakness_self", "plan_text", "phase",
}

def save_trainer_info(user_id: int, db: Session, **fields) -> dict:
    """
    Partial update of trainer profile. Only updates fields passed in.
    Called by the tool call handler when Claude extracts student info.
    """
    from db.models import StudentTrainerProfile

    profile = db.query(StudentTrainerProfile).filter(
        StudentTrainerProfile.user_id == user_id
    ).first()

    if profile is None:
        profile = StudentTrainerProfile(user_id=user_id)
        db.add(profile)

    updated = []
    for field, value in fields.items():
        if field in TRAINER_PROFILE_FIELDS and value is not None:
            setattr(profile, field, value)
            updated.append(field)

    if fields.get("plan_text"):
        profile.plan_generated_at = datetime.now(timezone.utc)

    profile.updated_at = datetime.now(timezone.utc)
    db.commit()

    log.info("[TRAINER] saved fields=%s for user_id=%s", updated, user_id)
    return {"saved": updated}


# ── Tool: get_new_practice_since ──────────────────────────────────────────────

def get_new_practice_since(user_id: int, db: Session, since: Optional[datetime]) -> list[dict]:
    """
    Returns completed practice attempts since `since` timestamp.
    Also fetches the previous attempt for the same module to show delta.
    """
    from db.models import PracticeAttempt

    if since is None:
        return []

    new_attempts = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.status == "complete",
            PracticeAttempt.completed_at > since,
        )
        .order_by(PracticeAttempt.completed_at.desc())
        .all()
    )

    if not new_attempts:
        return []

    result = []
    for a in new_attempts:
        # Find previous attempt for same module to compute delta
        prev = (
            db.query(PracticeAttempt)
            .filter(
                PracticeAttempt.user_id == user_id,
                PracticeAttempt.module == a.module,
                PracticeAttempt.status == "complete",
                PracticeAttempt.id < a.id,
            )
            .order_by(PracticeAttempt.id.desc())
            .first()
        )
        result.append({
            "module":        a.module,
            "score":         a.total_score,
            "prev_score":    prev.total_score if prev else None,
            "delta":         (a.total_score - prev.total_score) if prev else None,
            "completed_at":  a.completed_at.isoformat() if a.completed_at else None,
        })

    return result


# ── Public: build_rich_context ────────────────────────────────────────────────

def build_rich_context(user, user_id: int, db: Session) -> dict:
    """
    Calls all MCP tools in parallel and assembles a rich context dict.
    """
    try:
        # Run the 6 independent DB queries in parallel threads
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(get_user_profile,             user, db):       "profile",
                executor.submit(get_recent_scores,            user_id, db, 3): "recent",
                executor.submit(get_weak_areas,               user_id, db):    "weak_areas",
                executor.submit(get_exam_history,             user_id, db):    "history",
                executor.submit(get_speaking_detail,          user_id, db):    "speaking",
                executor.submit(get_speaking_task_breakdowns, user_id, db):    "speaking_tasks",
            }
            results = {}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    log.warning("[MCP] %s fetch failed: %s", key, e)
                    results[key] = None

        profile = results.get("profile") or {}
        return {
            **profile,
            "recent_scores":            results.get("recent") or [],
            "weak_areas":               results.get("weak_areas") or [],
            "exam_history":             results.get("history") or [],
            "speaking_detail":          results.get("speaking"),
            "speaking_task_breakdowns": results.get("speaking_tasks"),
        }

    except Exception as e:
        log.warning("[MCP] Context fetch failed: %s", e)
        return {
            "username":          getattr(user, "username", "Student"),
            "score_requirement": getattr(user, "score_requirement", None),
            "exam_date":         None,
            "days_to_exam":      None,
        }


# ── Tool: get_last_attempt_breakdown ─────────────────────────────────────────

def get_last_attempt_breakdown(user_id: int, db: Session) -> dict:
    """
    Returns the most recent attempt per question_type with task_breakdown JSONB.
    Produces a clean summary dict keyed by question_type.
    """
    from db.models import PracticeAttempt

    attempts = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.id.desc())
        .limit(60)
        .all()
    )

    # Keep only the latest attempt per question_type
    by_type: dict = {}
    for a in attempts:
        qt = a.question_type
        if qt not in by_type:
            by_type[qt] = a

    result = {}
    for qt, a in by_type.items():
        bd = a.task_breakdown or {}
        entry: dict = {
            "total_score":      a.total_score,
            "total_questions":  a.total_questions,
            "pct":              round(a.total_score / a.total_questions, 2) if a.total_questions else 0,
            "completed_at":     a.completed_at.isoformat() if a.completed_at else None,
        }
        # Flatten key sub-task metrics from task_breakdown if present
        for sub_task, data in bd.items():
            if isinstance(data, dict):
                entry[sub_task] = {
                    "pct":        round(data.get("pct", 0), 3),
                    "avg_total":  data.get("avg_total", 0),
                    "max_points": data.get("max_points", 1),
                    "count":      data.get("count", 0),
                }
        result[qt] = entry

    return result

# ── Tool: get_speaking_task_breakdowns ───────────────────────────────────────

SPEAKING_TASK_TYPES = [
    "read_aloud",
    "repeat_sentence",
    "describe_image",
    "retell_lecture",
    "answer_short_question",
    "summarize_group_discussion",
    "respond_to_situation",
]


def get_speaking_task_breakdowns(user_id: int, db: Session) -> Optional[dict]:
    """
    For the user's most recent completed mock or sectional attempt, returns
    per-question speaking scores grouped by task type. Eagerly fetched in
    build_rich_context so the coach can answer "how was my Read Aloud?" type
    questions without a tool call roundtrip.
    """
    from db.models import PracticeAttempt, AttemptAnswer

    umbrella = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.question_type.in_(["mock", "sectional"]),
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.id.desc())
        .first()
    )

    if not umbrella:
        return None

    answers = (
        db.query(AttemptAnswer)
        .filter(
            AttemptAnswer.attempt_id == umbrella.id,
            AttemptAnswer.question_type.in_(SPEAKING_TASK_TYPES),
        )
        .order_by(AttemptAnswer.id.asc())
        .all()
    )

    if not answers:
        return None

    grouped: dict[str, list] = {}
    counters: dict[str, int] = {}
    for a in answers:
        rj = a.result_json or {}
        counters[a.question_type] = counters.get(a.question_type, 0) + 1
        row: dict = {
            "q":              counters[a.question_type],
            "question_id":    a.question_id,
            "score":          a.score,
            "scoring_status": a.scoring_status,
        }
        if rj.get("pte_score") is not None:
            row["pte_score"] = rj["pte_score"]
        if a.content_score is not None:
            row["content"] = round(a.content_score, 2)
        if a.fluency_score is not None:
            row["fluency"] = round(a.fluency_score, 2)
        if a.pronunciation_score is not None:
            row["pronunciation"] = round(a.pronunciation_score, 2)
        if rj.get("maxScore") is not None:
            row["max_score"] = rj["maxScore"]
        grouped.setdefault(a.question_type, []).append(row)

    return {
        "attempt_id":   umbrella.id,
        "source":       umbrella.question_type,
        "completed_at": umbrella.completed_at.isoformat() if umbrella.completed_at else None,
        "total_score":  umbrella.total_score,
        "tasks":        grouped,
    }


# ── Tool: get_attempt_detail ──────────────────────────────────────────────────

def get_attempt_detail(user_id: int, question_type: str, db: Session) -> Optional[dict]:
    """
    Returns per-question breakdown for the student's most recent attempt of the
    given question_type. Checks two paths and returns the more recent result:
      Path 1 — standalone practice attempt (question_type matches directly)
      Path 2 — mock/sectional attempt (umbrella), filtered by question_type in AttemptAnswer
    """
    from db.models import PracticeAttempt, AttemptAnswer

    def _build_answer_rows(answers: list) -> list:
        rows = []
        for i, a in enumerate(answers, 1):
            rj = a.result_json or {}
            row: dict = {
                "q": i,
                "question_id": a.question_id,
                "score": a.score,
                "scoring_status": a.scoring_status,
            }
            if rj.get("pte_score") is not None:
                row["pte_score"] = rj["pte_score"]
            if a.content_score is not None:
                row["content"] = round(a.content_score, 2)
            if a.fluency_score is not None:
                row["fluency"] = round(a.fluency_score, 2)
            if a.pronunciation_score is not None:
                row["pronunciation"] = round(a.pronunciation_score, 2)
            if rj.get("hits") is not None:
                row["hits"] = rj["hits"]
                row["total_words"] = rj.get("total", rj.get("maxScore"))
            if rj.get("maxScore") is not None:
                row["max_score"] = rj["maxScore"]
            rows.append(row)
        return rows

    # ── Path 1: standalone practice attempt ───────────────────────────────────
    standalone = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.question_type == question_type,
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.id.desc())
        .first()
    )

    result_standalone = None
    if standalone:
        answers = (
            db.query(AttemptAnswer)
            .filter(AttemptAnswer.attempt_id == standalone.id)
            .order_by(AttemptAnswer.id.asc())
            .limit(20)
            .all()
        )
        result_standalone = {
            "attempt_id":      standalone.id,
            "source":          "practice",
            "question_type":   question_type,
            "completed_at":    standalone.completed_at.isoformat() if standalone.completed_at else None,
            "total_score":     standalone.total_score,
            "total_questions": standalone.total_questions,
            "answers":         _build_answer_rows(answers),
        }

    # ── Path 2: latest complete mock/sectional, filtered by question_type ─────
    umbrella = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.question_type.in_(["mock", "sectional"]),
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.id.desc())
        .first()
    )

    result_umbrella = None
    if umbrella:
        answers = (
            db.query(AttemptAnswer)
            .filter(
                AttemptAnswer.attempt_id == umbrella.id,
                AttemptAnswer.question_type == question_type,
            )
            .order_by(AttemptAnswer.id.asc())
            .limit(20)
            .all()
        )
        if answers:
            result_umbrella = {
                "attempt_id":      umbrella.id,
                "source":          umbrella.question_type,  # "mock" or "sectional"
                "question_type":   question_type,
                "completed_at":    umbrella.completed_at.isoformat() if umbrella.completed_at else None,
                "total_score":     sum(a.score for a in answers),
                "total_questions": len(answers),
                "answers":         _build_answer_rows(answers),
            }

    # ── Pick the more recent result ───────────────────────────────────────────
    if result_standalone and result_umbrella:
        return result_standalone if standalone.id > umbrella.id else result_umbrella
    return result_standalone or result_umbrella


# ── Tool: get_recent_task_answers ────────────────────────────────────────────

def get_recent_task_answers(
    user_id: int,
    task_type: str,
    db: Session,
    limit: int = 10,
) -> list[dict]:
    """
    Return the student's most recent per-question answers for a given task
    type, across ALL flows (practice, sectional, mock).

    Unlike get_attempt_detail, this tool:
      • does NOT filter on PracticeAttempt.status — includes 'active' /
        in-progress sessions (long practice runs that never reach 'complete')
      • does NOT filter on PracticeAttempt.filter_type — includes practice
        AND sectional AND mock answers
      • returns one row per AttemptAnswer (not per attempt)

    Each row includes 'context' (practice / sectional / mock) and 'module'
    so the coach can disambiguate which flow each answer came from.
    """
    from db.models import AttemptAnswer, PracticeAttempt

    rows = (
        db.query(AttemptAnswer, PracticeAttempt)
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(
            PracticeAttempt.user_id == user_id,
            AttemptAnswer.question_type == task_type,
        )
        .order_by(AttemptAnswer.submitted_at.desc())
        .limit(min(max(int(limit or 10), 1), 50))
        .all()
    )

    out: list[dict] = []
    for ans, pa in rows:
        rj = ans.result_json or {}
        entry: dict = {
            "question_id":    ans.question_id,
            "score":          ans.score,
            "scoring_status": ans.scoring_status,
            "submitted_at":   ans.submitted_at.isoformat() if ans.submitted_at else None,
            "context":        pa.filter_type,   # practice | sectional | mock
            "module":         pa.module,        # speaking | listening | reading | writing | None (mock)
            "attempt_id":     pa.id,
        }
        # Surface the most useful per-Q breakdown fields if present
        for key in ("pte_score", "content", "fluency", "pronunciation",
                    "hits", "total", "maxScore", "earned_raw", "task_type"):
            if rj.get(key) is not None:
                entry[key] = rj[key]
        # Per-Q content/fluency/pronunciation from typed columns, if not in result_json
        if entry.get("content") is None and ans.content_score is not None:
            entry["content"] = round(ans.content_score, 2)
        if entry.get("fluency") is None and ans.fluency_score is not None:
            entry["fluency"] = round(ans.fluency_score, 2)
        if entry.get("pronunciation") is None and ans.pronunciation_score is not None:
            entry["pronunciation"] = round(ans.pronunciation_score, 2)
        out.append(entry)
    return out


# ── Tool: award_milestone ────────────────────────────────────────────────────

# All supported milestone keys and their human-readable labels
MILESTONE_CATALOGUE = {
    # Practice volume
    "first_practice":         "Completed your first practice session 🎉",
    "practice_10":            "Completed 10 practice sessions 🔥",
    "practice_25":            "Completed 25 practice sessions 💪",
    "practice_50":            "Completed 50 practice sessions 🏆",
    "practice_100":           "Completed 100 practice sessions 🌟",
    "all_task_types_tried":   "Tried every PTE task type ✅",
    # Profile / onboarding
    "profile_complete":       "Profile fully set up 📋",
    "plan_generated":         "Study plan created 📅",
    # Coaching engagement
    "session_5":              "5 coaching sessions completed 🎓",
    "session_10":             "10 coaching sessions completed 🎓",
    "session_25":             "25 coaching sessions completed 🎓",
    # Score thresholds (speaking overall)
    "speaking_above_50":      "Speaking score above 50% 📈",
    "speaking_above_65":      "Speaking score above 65% 📈",
    "speaking_above_70":      "Speaking score above 70% 🚀",
    "speaking_above_79":      "Speaking score above 79% — TARGET HIT! 🏆",
    # Task-specific
    "first_perfect_asq":      "Perfect score on Answer Short Question ⭐",
    "streak_7_days":          "7-day practice streak 🗓️",
}


def award_milestone(user_id: int, milestone_key: str, db: Session, metadata: dict = None) -> bool:
    """
    Idempotent — awards milestone only once. Returns True if newly awarded.
    """
    from db.models import StudentMilestone
    from datetime import datetime, timezone

    if milestone_key not in MILESTONE_CATALOGUE:
        log.warning("[MILESTONE] unknown key: %s", milestone_key)
        return False

    exists = db.query(StudentMilestone).filter(
        StudentMilestone.user_id == user_id,
        StudentMilestone.milestone_key == milestone_key,
    ).first()

    if exists:
        return False

    m = StudentMilestone(
        user_id=user_id,
        milestone_key=milestone_key,
        achieved_at=datetime.now(timezone.utc),
        metadata_=metadata or {},
    )
    db.add(m)
    db.commit()
    log.info("[MILESTONE] awarded user_id=%s key=%s", user_id, milestone_key)
    return True


# ── Tool: get_milestones ─────────────────────────────────────────────────────

def get_milestones(user_id: int, db: Session) -> dict:
    """
    Returns achieved milestones + next milestone to unlock + overall progress.
    """
    from db.models import StudentMilestone, PracticeAttempt

    achieved_rows = (
        db.query(StudentMilestone)
        .filter(StudentMilestone.user_id == user_id)
        .order_by(StudentMilestone.achieved_at.asc())
        .all()
    )

    achieved = [
        {
            "key":         r.milestone_key,
            "label":       MILESTONE_CATALOGUE.get(r.milestone_key, r.milestone_key),
            "achieved_at": r.achieved_at.isoformat() if r.achieved_at else None,
        }
        for r in achieved_rows
    ]
    achieved_keys = {r.milestone_key for r in achieved_rows}

    # Compute next practice-volume milestone
    total_attempts = db.query(PracticeAttempt).filter(
        PracticeAttempt.user_id == user_id,
        PracticeAttempt.status == "complete",
    ).count()

    next_milestone = None
    for key, threshold in [("practice_10", 10), ("practice_25", 25), ("practice_50", 50), ("practice_100", 100)]:
        if key not in achieved_keys:
            next_milestone = {
                "key":      key,
                "label":    MILESTONE_CATALOGUE[key],
                "progress": total_attempts,
                "target":   threshold,
            }
            break

    return {
        "achieved":        achieved,
        "achieved_count":  len(achieved),
        "next_milestone":  next_milestone,
        "total_practices": total_attempts,
    }
