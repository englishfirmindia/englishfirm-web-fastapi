from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional

from db.database import get_db
from db.models import (
    User, Milestone, MilestoneTask, MilestoneTaskTier,
    UserMilestoneTaskProgress,
)
from core.dependencies import get_current_user

router = APIRouter(prefix="/milestones", tags=["Milestones"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_progress(
    db: Session, user_id: int, task_id: int
) -> UserMilestoneTaskProgress:
    prog = (
        db.query(UserMilestoneTaskProgress)
        .filter_by(user_id=user_id, task_id=task_id)
        .first()
    )
    if not prog:
        prog = UserMilestoneTaskProgress(user_id=user_id, task_id=task_id)
        db.add(prog)
        db.flush()
    return prog


def _resolve_task_complete(
    task: MilestoneTask,
    prog: UserMilestoneTaskProgress,
    user: User,
) -> bool:
    """Return True if this task is considered complete for the user."""
    if prog.is_complete:
        return True
    # action tasks: infer from user row
    if task.task_type == "action":
        if task.task_code == "SET_EXAM_DATE":
            return user.exam_date is not None
        if task.task_code == "SET_TARGET_SCORE":
            return user.score_requirement is not None
    return prog.current_count >= task.target_count


def _build_task_payload(
    task: MilestoneTask,
    prog: UserMilestoneTaskProgress,
    user: User,
) -> dict:
    is_complete = _resolve_task_complete(task, prog, user)
    current = prog.current_count
    # sync action tasks count
    if task.task_type == "action":
        if task.task_code == "SET_EXAM_DATE":
            current = 1 if user.exam_date is not None else 0
        elif task.task_code == "SET_TARGET_SCORE":
            current = 1 if user.score_requirement is not None else 0
    return {
        "id":            task.id,
        "task_code":     task.task_code,
        "task_label":    task.task_label,
        "task_type":     task.task_type,
        "target_count":  task.target_count,
        "current_count": current,
        "is_complete":   is_complete,
        "points":        task.points,
        "sort_order":    task.sort_order,
        "app_route":     task.app_route,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /milestones/next-steps
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/next-steps")
def get_next_steps(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    milestones = db.query(Milestone).order_by(Milestone.number).all()

    if not milestones:
        return {"milestone": None, "tasks": [], "milestone_complete": False, "total_points_earned": 0}

    active_milestone = None
    for m in milestones:
        all_done = all(
            _resolve_task_complete(
                t,
                _get_or_create_progress(db, current_user.id, t.id),
                current_user,
            )
            for t in m.tasks
        )
        if not all_done:
            active_milestone = m
            break
    db.commit()

    # all milestones complete — show last
    if active_milestone is None:
        active_milestone = milestones[-1]

    task_payloads = []
    points_earned = 0
    for t in active_milestone.tasks:
        prog = _get_or_create_progress(db, current_user.id, t.id)
        payload = _build_task_payload(t, prog, current_user)
        task_payloads.append(payload)
        if payload["is_complete"] and prog.points_earned:
            points_earned += prog.points_earned

    milestone_complete = all(p["is_complete"] for p in task_payloads)

    return {
        "milestone": {
            "id":           active_milestone.id,
            "number":       active_milestone.number,
            "name":         active_milestone.name,
            "emoji":        active_milestone.emoji,
            "category":     active_milestone.category,
            "total_points": active_milestone.total_points,
            "description":  active_milestone.description,
        },
        "tasks":              task_payloads,
        "milestone_complete": milestone_complete,
        "total_points_earned": points_earned,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /milestones/complete-task  — mark a task done (for learn/quiz/practice)
# ─────────────────────────────────────────────────────────────────────────────

class TaskCompletePayload(BaseModel):
    task_code: str
    score: Optional[int] = None   # used for mock_scored / bonus / quiz


@router.post("/complete-task")
def complete_task(
    payload: TaskCompletePayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = db.query(MilestoneTask).filter_by(task_code=payload.task_code).first()
    if not task:
        return {"success": False, "message": "Task not found"}

    prog = _get_or_create_progress(db, current_user.id, task.id)
    if prog.is_complete:
        return {"success": True, "already_complete": True, "points_earned": prog.points_earned}

    user_target = current_user.score_requirement or 65
    points_earned = 0
    is_complete = False

    if task.task_type == "mock_scored":
        score = payload.score or 0
        tiers = sorted(task.tiers, key=lambda t: t.points, reverse=True)
        for tier in tiers:
            if tier.condition == "hit_target" and score >= user_target:
                points_earned = tier.points
                is_complete = True
                break
            elif tier.condition == "near_target" and score >= (user_target - 5):
                points_earned = tier.points
                is_complete = task.complete_on_target
                break
            elif tier.condition == "below_target":
                points_earned = tier.points
                break
        prog.current_count += 1
        if prog.current_count >= task.target_count:
            is_complete = True

    elif task.task_type == "bonus":
        score = payload.score or 0
        if score >= user_target:
            points_earned = task.points or 0
            is_complete = True

    elif task.task_type == "quiz":
        score = payload.score or 0
        threshold = int(task.pass_threshold) if task.pass_threshold and task.pass_threshold.isdigit() else 0
        if score >= threshold:
            points_earned = task.points or 0
            is_complete = True

    else:
        # learn / practice / action
        prog.current_count += 1
        if prog.current_count >= task.target_count:
            points_earned = task.points or 0
            is_complete = True
        if task.complete_on_target and payload.score and payload.score >= user_target:
            points_earned = task.points or 0
            is_complete = True

    if is_complete:
        prog.is_complete = True
        prog.points_earned = points_earned
        prog.completed_at = datetime.now(timezone.utc)

    db.commit()
    return {"success": True, "is_complete": is_complete, "points_earned": points_earned}


# ─────────────────────────────────────────────────────────────────────────────
# Seed data — called once on startup
# ─────────────────────────────────────────────────────────────────────────────

def seed_milestones(db: Session):
    if db.query(Milestone).count() > 0:
        return  # already seeded

    milestones_data = [
        {
            "number": 1, "name": "Goal Setter", "emoji": "🎯",
            "category": "Onboarding", "total_points": 25,
            "description": "You've set your exam goal!",
            "tasks": [
                {"task_code": "SET_EXAM_DATE",    "task_label": "Set Exam Date",    "task_type": "action", "target_count": 1, "points": None, "pass_threshold": None, "complete_on_target": False, "sort_order": 1},
                {"task_code": "SET_TARGET_SCORE", "task_label": "Set Target Score", "task_type": "action", "target_count": 1, "points": None, "pass_threshold": None, "complete_on_target": False, "sort_order": 2},
            ],
        },
        {
            "number": 2, "name": "Foundation Builder", "emoji": "📚",
            "category": "Foundation", "total_points": 20,
            "description": "Master the basics of PTE",
            "tasks": [
                {"task_code": "LEARN_SCORING",    "task_label": "Learn Scoring System",      "task_type": "learn",   "target_count": 1, "points": 5, "pass_threshold": None, "complete_on_target": False, "sort_order": 1},
                {"task_code": "LEARN_TEMPLATES",  "task_label": "Learn Templates",           "task_type": "learn",   "target_count": 1, "points": 5, "pass_threshold": None, "complete_on_target": False, "sort_order": 2},
                {"task_code": "LEARN_QTYPES",     "task_label": "Understand Question Types", "task_type": "learn",   "target_count": 1, "points": 5, "pass_threshold": None, "complete_on_target": False, "sort_order": 3},
                {"task_code": "QUIZ_FOUNDATION",  "task_label": "Foundation Quiz",           "task_type": "quiz",    "target_count": 1, "points": 5, "pass_threshold": "80", "complete_on_target": False, "sort_order": 4},
            ],
        },
        {
            "number": 3, "name": "Sectional Star", "emoji": "⭐",
            "category": "Sectional Tests", "total_points": 25,
            "description": "Complete sectionals & hit your target",
            "tasks": [
                {"task_code": "SECT_SPEAKING",  "task_label": "Speaking Sectional",        "task_type": "practice", "target_count": 3, "points": 5, "pass_threshold": None,          "complete_on_target": True,  "sort_order": 1},
                {"task_code": "SECT_LISTENING", "task_label": "Listening Sectional",       "task_type": "practice", "target_count": 3, "points": 5, "pass_threshold": None,          "complete_on_target": True,  "sort_order": 2},
                {"task_code": "SECT_WRITING",   "task_label": "Writing Sectional",         "task_type": "practice", "target_count": 3, "points": 5, "pass_threshold": None,          "complete_on_target": True,  "sort_order": 3},
                {"task_code": "SECT_READING",   "task_label": "Reading Sectional",         "task_type": "practice", "target_count": 3, "points": 5, "pass_threshold": None,          "complete_on_target": True,  "sort_order": 4},
                {"task_code": "SECT_BONUS",     "task_label": "Hit Target (any sectional)","task_type": "bonus",    "target_count": 1, "points": 5, "pass_threshold": "user_target", "complete_on_target": False, "sort_order": 5},
            ],
        },
        {
            "number": 4, "name": "Mock Champion", "emoji": "🏆",
            "category": "Mock Test", "total_points": 25,
            "description": "Attempt a mock & hit your target",
            "tasks": [
                {"task_code": "MOCK_TEST", "task_label": "Full Mock Test", "task_type": "mock_scored", "target_count": 3, "points": None, "pass_threshold": "user_target", "complete_on_target": True, "sort_order": 1},
            ],
            "tiers": [
                {"task_code": "MOCK_TEST", "tier_label": "Below Target", "condition": "below_target", "points": 5},
                {"task_code": "MOCK_TEST", "tier_label": "Near Target",  "condition": "near_target",  "points": 15},
                {"task_code": "MOCK_TEST", "tier_label": "Hit Target",   "condition": "hit_target",   "points": 25},
            ],
        },
    ]

    for m_data in milestones_data:
        milestone = Milestone(
            number=m_data["number"], name=m_data["name"], emoji=m_data["emoji"],
            category=m_data["category"], total_points=m_data["total_points"],
            description=m_data["description"],
        )
        db.add(milestone)
        db.flush()

        task_map = {}
        for t_data in m_data["tasks"]:
            task = MilestoneTask(
                milestone_id=milestone.id,
                task_code=t_data["task_code"], task_label=t_data["task_label"],
                task_type=t_data["task_type"], target_count=t_data["target_count"],
                points=t_data["points"], pass_threshold=t_data["pass_threshold"],
                complete_on_target=t_data["complete_on_target"], sort_order=t_data["sort_order"],
            )
            db.add(task)
            db.flush()
            task_map[t_data["task_code"]] = task

        for tier_data in m_data.get("tiers", []):
            tier = MilestoneTaskTier(
                task_id=task_map[tier_data["task_code"]].id,
                tier_label=tier_data["tier_label"],
                condition=tier_data["condition"],
                points=tier_data["points"],
            )
            db.add(tier)

    db.commit()
