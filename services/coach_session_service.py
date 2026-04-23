"""
Session service — boundary detection, phase computation, completeness flags.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Gap in hours that triggers a new session
SESSION_GAP_HOURS = 2


# ── Session boundary detection ────────────────────────────────────────────────

def detect_and_handle_session_boundary(user_id: int, db: Session) -> bool:
    """
    Returns True if this is a new session (gap > SESSION_GAP_HOURS or first ever).
    Increments session_count and updates last_session_at in DB.
    """
    from db.models import StudentTrainerProfile

    profile = db.query(StudentTrainerProfile).filter(
        StudentTrainerProfile.user_id == user_id
    ).first()

    if profile is None:
        return False  # get_trainer_profile() creates it — should exist by now

    now = datetime.now(timezone.utc)
    is_new_session = False

    if profile.last_session_at is None:
        is_new_session = True
    else:
        last = profile.last_session_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        gap = now - last
        if gap > timedelta(hours=SESSION_GAP_HOURS):
            is_new_session = True

    if is_new_session:
        profile.session_count  += 1
        profile.last_session_at = now
        db.commit()
        log.info("[SESSION] new session #%s for user_id=%s", profile.session_count, user_id)

    return is_new_session


# ── Phase computation ─────────────────────────────────────────────────────────

def compute_phase(trainer_profile: dict, new_practice: list) -> str:
    """
    Returns: intake | planning | coaching | review
    """
    # Critical fields needed before we can plan
    critical_fields = ["motivation", "study_hours_per_day"]
    missing_critical = any(trainer_profile.get(f) is None for f in critical_fields)

    if missing_critical:
        return "intake"

    if trainer_profile.get("plan_text") is None:
        return "planning"

    if new_practice:
        return "review"

    return "coaching"


# ── Completeness flags ────────────────────────────────────────────────────────

_FIELD_DESCRIPTIONS = {
    "motivation":             "why the student needs PTE (visa, PR, university, job)",
    "study_hours_per_day":    "how many hours per day they can study",
    "study_schedule":         "when they study (morning, evening, lunch break)",
    "prior_pte_attempts":     "whether they have attempted PTE before and how many times",
    "anxiety_level":          "their confidence and anxiety around the exam",
    "biggest_weakness_self":  "what they personally feel is their biggest weakness",
}

def compute_completeness_flags(trainer_profile: dict) -> list[dict]:
    """
    Returns list of missing fields with descriptions.
    Only includes fields that have actual coaching value.
    """
    missing = []
    for field, description in _FIELD_DESCRIPTIONS.items():
        if not trainer_profile.get(field):
            missing.append({"field": field, "description": description})
    return missing
