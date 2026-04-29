"""
Trainer-facing data endpoints. Auth: trainer JWT (audience='trainer').

  GET    /trainer/shared                  list active shares for me
  GET    /trainer/shared/{share_id}       full review payload + notes
  POST   /trainer/notes                   create note
  PATCH  /trainer/notes/{note_id}         edit own note
  DELETE /trainer/notes/{note_id}         soft-delete own note
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Query
from pydantic import BaseModel, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import (
    PracticeAttempt,
    Trainer,
    TrainerNote,
    TrainerShare,
    User,
)
from services.email import send_student_note_posted
from services.trainer_auth import get_current_trainer
from services.trainer_review import build_trainer_review_payload


router = APIRouter(prefix="/trainer", tags=["Trainer - App"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _module_title(module: str) -> str:
    return (module or "").replace("_", " ").title()


def _test_label(attempt: PracticeAttempt) -> str:
    qtype = (attempt.question_type or "").lower()
    module = (attempt.module or "").lower()
    if qtype == "mock" or module == "mock":
        return "Mock"
    if qtype == "sectional":
        return f"{_module_title(module)} Sectional"
    return f"{_module_title(module)} {_module_title(qtype)}".strip()


def _require_my_active_share(
    db: Session, trainer: Trainer, share_id: int
) -> TrainerShare:
    share = db.query(TrainerShare).filter(TrainerShare.id == share_id).first()
    if share is None:
        raise HTTPException(status_code=404, detail="Share not found")
    if share.trainer_id != trainer.id:
        raise HTTPException(status_code=403, detail="Not your share")
    if share.revoked_at is not None:
        raise HTTPException(status_code=410, detail="Share revoked")
    return share


def _require_my_note(
    db: Session, trainer: Trainer, note_id: int
) -> TrainerNote:
    note = db.query(TrainerNote).filter(TrainerNote.id == note_id).first()
    if note is None or note.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Note not found")
    if note.trainer_id != trainer.id:
        raise HTTPException(status_code=403, detail="Not your note")
    return note


# ── Request models ────────────────────────────────────────────────────────────

class NoteCreateBody(BaseModel):
    share_id: int
    question_id: Optional[int] = None
    body: str
    rating: Optional[int] = None

    @field_validator("body")
    @classmethod
    def _check_body(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("body cannot be empty")
        if len(s) > 4000:
            raise ValueError("body too long (max 4000 chars)")
        return s

    @field_validator("rating")
    @classmethod
    def _check_rating(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1 or v > 5:
            raise ValueError("rating must be between 1 and 5")
        return v


class NotePatchBody(BaseModel):
    body: Optional[str] = None
    rating: Optional[int] = None

    @field_validator("body")
    @classmethod
    def _check_body(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("body cannot be empty")
        if len(s) > 4000:
            raise ValueError("body too long (max 4000 chars)")
        return s

    @field_validator("rating")
    @classmethod
    def _check_rating(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1 or v > 5:
            raise ValueError("rating must be between 1 and 5")
        return v


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/shared")
def list_shared(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    """Active shares for this trainer, newest first, paginated."""
    base_query = (
        db.query(TrainerShare, PracticeAttempt, User)
        .join(PracticeAttempt, TrainerShare.attempt_id == PracticeAttempt.id)
        .outerjoin(User, TrainerShare.student_user_id == User.id)
        .filter(
            TrainerShare.trainer_id == trainer.id,
            TrainerShare.revoked_at.is_(None),
        )
    )
    total = base_query.count()
    rows = (
        base_query.order_by(TrainerShare.shared_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    # Notes counts in one query
    share_ids = [s.id for s, _, _ in rows]
    notes_count_by_share = {}
    if share_ids:
        for share_id, cnt in (
            db.query(TrainerNote.share_id, func.count(TrainerNote.id))
            .filter(
                TrainerNote.share_id.in_(share_ids),
                TrainerNote.deleted_at.is_(None),
            )
            .group_by(TrainerNote.share_id)
            .all()
        ):
            notes_count_by_share[share_id] = cnt

    items = []
    for share, attempt, student in rows:
        items.append(
            {
                "share_id": share.id,
                "attempt_id": attempt.id,
                "test_label": _test_label(attempt),
                "module": attempt.module,
                "question_type": attempt.question_type,
                "total_score": attempt.total_score,
                "scoring_status": attempt.scoring_status,
                "completed_at": (
                    attempt.completed_at.isoformat() if attempt.completed_at else None
                ),
                "shared_at": share.shared_at.isoformat() if share.shared_at else None,
                "first_viewed_at": (
                    share.first_viewed_at.isoformat() if share.first_viewed_at else None
                ),
                "last_viewed_at": (
                    share.last_viewed_at.isoformat() if share.last_viewed_at else None
                ),
                "viewed": share.first_viewed_at is not None,
                "notes_count": notes_count_by_share.get(share.id, 0),
                "student_display_name": student.username if student else "[deleted]",
                "student_email": student.email if student else None,
            }
        )

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": page * limit < total,
    }


@router.get("/shared/{share_id}")
def get_shared(
    share_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    """Full review payload for one shared attempt + notes for this share."""
    share = _require_my_active_share(db, trainer, share_id)

    now = datetime.now(timezone.utc)
    if share.first_viewed_at is None:
        share.first_viewed_at = now
    share.last_viewed_at = now
    db.commit()

    return build_trainer_review_payload(db, share)


@router.post("/notes")
def create_note(
    background_tasks: BackgroundTasks,
    body: NoteCreateBody = Body(...),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    """Create a note on a shared attempt (or a specific question within it)."""
    share = _require_my_active_share(db, trainer, body.share_id)

    note = TrainerNote(
        share_id=share.id,
        attempt_id=share.attempt_id,
        question_id=body.question_id,
        trainer_id=trainer.id,
        body=body.body,
        rating=body.rating,
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    # Notify the student (best-effort, opt-in honored at provider level later)
    student = db.query(User).filter(User.id == share.student_user_id).first()
    attempt = db.query(PracticeAttempt).filter(PracticeAttempt.id == share.attempt_id).first()
    if student and attempt and student.email:
        background_tasks.add_task(
            send_student_note_posted,
            to=student.email,
            trainer_name=trainer.display_name,
            test_label=_test_label(attempt),
            student_name=student.username,
        )

    return {
        "note_id": note.id,
        "share_id": note.share_id,
        "attempt_id": note.attempt_id,
        "question_id": note.question_id,
        "body": note.body,
        "rating": note.rating,
        "created_at": note.created_at.isoformat() if note.created_at else None,
    }


@router.patch("/notes/{note_id}")
def update_note(
    note_id: int = Path(..., ge=1),
    body: NotePatchBody = Body(...),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    note = _require_my_note(db, trainer, note_id)

    if body.body is None and body.rating is None:
        raise HTTPException(status_code=422, detail="Nothing to update")

    if body.body is not None:
        note.body = body.body
    if body.rating is not None:
        note.rating = body.rating
    note.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(note)

    return {
        "note_id": note.id,
        "body": note.body,
        "rating": note.rating,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    }


@router.delete("/notes/{note_id}", status_code=204)
def delete_note(
    note_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    trainer: Trainer = Depends(get_current_trainer),
):
    note = _require_my_note(db, trainer, note_id)
    note.deleted_at = datetime.now(timezone.utc)
    db.commit()
    return None
