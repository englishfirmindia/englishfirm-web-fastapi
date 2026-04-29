"""
Student-facing share endpoints.

Lets a student:
  - List active trainers for the share dropdown
  - Share one of their own attempts with a single trainer
  - List who an attempt is currently shared with
  - Revoke a share they previously created
  - Read trainer notes left on their attempts

All endpoints require the regular student JWT (`get_current_user`).
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import (
    PracticeAttempt,
    Trainer,
    TrainerNote,
    TrainerShare,
    User,
)
from core.dependencies import get_current_user
from services.email import send_trainer_share_received, send_trainer_share_revoked


router = APIRouter(prefix="/student/share", tags=["Student - Share with Trainer"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _module_title(module: str) -> str:
    return (module or "").replace("_", " ").title()


def _test_label_for_attempt(attempt: PracticeAttempt) -> str:
    """Human-readable test label used in emails and UI strings."""
    qtype = (attempt.question_type or "").lower()
    module = (attempt.module or "").lower()
    if qtype == "mock" or module == "mock":
        return "Mock"
    if qtype == "sectional":
        return f"{_module_title(module)} Sectional"
    return f"{_module_title(module)} {_module_title(qtype)}".strip()


def _serialize_trainer(t: Trainer) -> dict:
    return {
        "trainer_id": t.id,
        "display_name": t.display_name,
        "email": t.email,
    }


def _serialize_share(share: TrainerShare, trainer: Trainer) -> dict:
    return {
        "share_id": share.id,
        "trainer_id": trainer.id,
        "trainer_display_name": trainer.display_name,
        "trainer_email": trainer.email,
        "shared_at": share.shared_at.isoformat() if share.shared_at else None,
        "first_viewed_at": share.first_viewed_at.isoformat() if share.first_viewed_at else None,
        "last_viewed_at": share.last_viewed_at.isoformat() if share.last_viewed_at else None,
        "viewed": share.first_viewed_at is not None,
    }


def _serialize_note(note: TrainerNote, trainer: Trainer) -> dict:
    return {
        "note_id": note.id,
        "attempt_id": note.attempt_id,
        "question_id": note.question_id,
        "trainer_id": trainer.id,
        "trainer_display_name": trainer.display_name,
        "body": note.body,
        "rating": note.rating,
        "created_at": note.created_at.isoformat() if note.created_at else None,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    }


def _require_attempt_owner(
    db: Session, user: User, attempt_id: int
) -> PracticeAttempt:
    attempt = (
        db.query(PracticeAttempt)
        .filter(PracticeAttempt.id == attempt_id)
        .first()
    )
    if attempt is None:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if attempt.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your attempt")
    return attempt


# ── Request models ────────────────────────────────────────────────────────────

class ShareCreateBody(BaseModel):
    trainer_id: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/trainers")
def list_trainers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """Active trainers for the share dropdown. Ordered by display_name."""
    rows = (
        db.query(Trainer)
        .filter(Trainer.is_active.is_(True))
        .order_by(Trainer.display_name.asc())
        .all()
    )
    return [_serialize_trainer(t) for t in rows]


@router.post("/attempt/{attempt_id}/share")
def share_attempt(
    background_tasks: BackgroundTasks,
    attempt_id: int = Path(..., ge=1),
    body: ShareCreateBody = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create (or surface existing) active share between this attempt and one trainer."""
    attempt = _require_attempt_owner(db, current_user, attempt_id)

    trainer = (
        db.query(Trainer)
        .filter(Trainer.id == body.trainer_id, Trainer.is_active.is_(True))
        .first()
    )
    if trainer is None:
        raise HTTPException(status_code=404, detail="Trainer not found")

    existing = (
        db.query(TrainerShare)
        .filter(
            TrainerShare.attempt_id == attempt_id,
            TrainerShare.trainer_id == trainer.id,
            TrainerShare.revoked_at.is_(None),
        )
        .first()
    )
    if existing:
        return {
            "share": _serialize_share(existing, trainer),
            "already_shared": True,
        }

    share = TrainerShare(
        attempt_id=attempt_id,
        student_user_id=current_user.id,
        trainer_id=trainer.id,
    )
    db.add(share)
    db.commit()
    db.refresh(share)

    background_tasks.add_task(
        send_trainer_share_received,
        to=trainer.email,
        student_name=current_user.username or current_user.email,
        test_label=_test_label_for_attempt(attempt),
        share_id=share.id,
    )

    return {"share": _serialize_share(share, trainer), "already_shared": False}


@router.get("/attempt/{attempt_id}/shares")
def list_attempt_shares(
    attempt_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Active shares for one attempt. Used to render the 'Shared with' panel."""
    _require_attempt_owner(db, current_user, attempt_id)
    rows = (
        db.query(TrainerShare, Trainer)
        .join(Trainer, TrainerShare.trainer_id == Trainer.id)
        .filter(
            TrainerShare.attempt_id == attempt_id,
            TrainerShare.revoked_at.is_(None),
        )
        .order_by(TrainerShare.shared_at.desc())
        .all()
    )
    return [_serialize_share(share, trainer) for share, trainer in rows]


@router.post("/share/{share_id}/revoke")
def revoke_share(
    background_tasks: BackgroundTasks,
    share_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Revoke an active share. Notes left by the trainer remain visible to the student."""
    share = db.query(TrainerShare).filter(TrainerShare.id == share_id).first()
    if share is None:
        raise HTTPException(status_code=404, detail="Share not found")
    if share.student_user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your share")
    if share.revoked_at is not None:
        return {"share_id": share.id, "status": "already_revoked"}

    share.revoked_at = datetime.now(timezone.utc)
    db.commit()

    trainer = db.query(Trainer).filter(Trainer.id == share.trainer_id).first()
    attempt = db.query(PracticeAttempt).filter(PracticeAttempt.id == share.attempt_id).first()
    if trainer and attempt:
        background_tasks.add_task(
            send_trainer_share_revoked,
            to=trainer.email,
            student_name=current_user.username or current_user.email,
            test_label=_test_label_for_attempt(attempt),
        )

    return {"share_id": share.id, "status": "revoked"}


@router.get("/attempt/{attempt_id}/notes")
def list_attempt_notes(
    attempt_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """All non-deleted trainer notes attached to one of the student's attempts."""
    _require_attempt_owner(db, current_user, attempt_id)
    rows = (
        db.query(TrainerNote, Trainer)
        .join(Trainer, TrainerNote.trainer_id == Trainer.id)
        .filter(
            TrainerNote.attempt_id == attempt_id,
            TrainerNote.deleted_at.is_(None),
        )
        .order_by(TrainerNote.created_at.asc())
        .all()
    )
    return [_serialize_note(note, trainer) for note, trainer in rows]


@router.get("/notes")
def list_all_notes(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 100,
):
    """All trainer notes across every attempt this student owns. Newest first."""
    limit = max(1, min(limit, 500))
    rows = (
        db.query(TrainerNote, Trainer, PracticeAttempt)
        .join(Trainer, TrainerNote.trainer_id == Trainer.id)
        .join(PracticeAttempt, TrainerNote.attempt_id == PracticeAttempt.id)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            TrainerNote.deleted_at.is_(None),
        )
        .order_by(TrainerNote.created_at.desc())
        .limit(limit)
        .all()
    )
    out: List[dict] = []
    for note, trainer, attempt in rows:
        item = _serialize_note(note, trainer)
        item["attempt"] = {
            "module": attempt.module,
            "question_type": attempt.question_type,
            "completed_at": (
                attempt.completed_at.isoformat() if attempt.completed_at else None
            ),
            "test_label": _test_label_for_attempt(attempt),
        }
        out.append(item)
    return out


@router.get("/attempt/{attempt_id}/summary")
def attempt_share_summary(
    attempt_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Lightweight summary used to render the badge on Feedback cards."""
    _require_attempt_owner(db, current_user, attempt_id)
    notes_count = (
        db.query(TrainerNote)
        .filter(
            TrainerNote.attempt_id == attempt_id,
            TrainerNote.deleted_at.is_(None),
        )
        .count()
    )
    active_shares = (
        db.query(TrainerShare)
        .filter(
            TrainerShare.attempt_id == attempt_id,
            TrainerShare.revoked_at.is_(None),
        )
        .count()
    )
    return {
        "attempt_id": attempt_id,
        "notes_count": notes_count,
        "active_shares": active_shares,
    }
