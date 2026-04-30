"""
Assembles the read-only review payload trainers see when they open a
shared attempt.

Returns a unified shape so the trainer frontend doesn't need module-
specific branches:

  {
    attempt:  { id, session_id, module, question_type, total_score, ... },
    student:  { id, display_name, email },
    answers:  [
      {
        question_id, question_type, score, audio_url (presigned),
        user_answer_json, result_json, submitted_at,
        question: { id, title, content_json, evaluation_json }
      }, ...
    ],
    notes:    [ ... ]
  }

Audio URLs are presigned with TRAINER_AUDIO_PRESIGN_TTL_SECONDS
(3 days by default) so a trainer can revisit a recording over multiple
sessions without the URL expiring.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

import core.config as config
from db.models import (
    AttemptAnswer,
    PracticeAttempt,
    QuestionFromApeuni,
    QuestionEvaluationApeuni,
    Trainer,
    TrainerNote,
    TrainerShare,
    User,
)
from services.s3_service import generate_presigned_url


def _maybe_presign(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return generate_presigned_url(
            url, expires_in=config.TRAINER_AUDIO_PRESIGN_TTL_SECONDS
        )
    except Exception:
        # If signing fails (e.g. legacy non-S3 URL), fall back to raw URL
        return url


# Keys in content_json/user_answer_json that hold S3 object URLs the trainer
# needs to play/view. Matches what's stored across all 22 question types:
# audio_url for every listening + speaking-stimulus type, image_url for
# describe_image. Recursive walk handles future nesting safely.
_STIMULUS_URL_KEYS = ("audio_url", "image_url")


def _presign_in_place(node: Any) -> Any:
    """Walk a JSON-shaped value and presign any URL value held under
    _STIMULUS_URL_KEYS. Returns a new structure with presigned URLs; safe
    to call on None or primitives."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k in _STIMULUS_URL_KEYS and isinstance(v, str):
                out[k] = _maybe_presign(v)
            else:
                out[k] = _presign_in_place(v)
        return out
    if isinstance(node, list):
        return [_presign_in_place(item) for item in node]
    return node


def _serialize_attempt(a: PracticeAttempt) -> Dict[str, Any]:
    return {
        "id": a.id,
        "session_id": a.session_id,
        "module": a.module,
        "question_type": a.question_type,
        "filter_type": a.filter_type,
        "total_questions": a.total_questions,
        "total_score": a.total_score,
        "questions_answered": a.questions_answered,
        "status": a.status,
        "scoring_status": a.scoring_status,
        "task_breakdown": a.task_breakdown,
        "selected_question_ids": a.selected_question_ids,
        "started_at": a.started_at.isoformat() if a.started_at else None,
        "completed_at": a.completed_at.isoformat() if a.completed_at else None,
    }


def _serialize_student(u: Optional[User]) -> Dict[str, Any]:
    if u is None:
        return {"id": None, "display_name": "[deleted]", "email": None}
    return {
        "id": u.id,
        "display_name": u.username,
        "email": u.email,
    }


def _serialize_question(
    q: Optional[QuestionFromApeuni],
    eval_json: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if q is None:
        return None
    return {
        "id": q.question_id,
        "title": q.title,
        "module": q.module,
        "question_type": q.question_type,
        # Presign any S3 URLs (audio_url / image_url) inside content_json so
        # the trainer can actually hear listening stimuli and see DI images.
        # Without this, the browser hits raw S3 and 403s.
        "content_json": _presign_in_place(q.content_json),
        "evaluation_json": eval_json,
    }


def _serialize_answer(
    ans: AttemptAnswer,
    question: Optional[QuestionFromApeuni],
    eval_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "answer_id": ans.id,
        "question_id": ans.question_id,
        "question_type": ans.question_type,
        "score": ans.score,
        "content_score": ans.content_score,
        "fluency_score": ans.fluency_score,
        "pronunciation_score": ans.pronunciation_score,
        "user_answer_json": ans.user_answer_json,
        "correct_answer_json": ans.correct_answer_json,
        "result_json": ans.result_json,
        "scoring_status": ans.scoring_status,
        "audio_url": _maybe_presign(ans.audio_url),
        "submitted_at": ans.submitted_at.isoformat() if ans.submitted_at else None,
        "question": _serialize_question(question, eval_json),
    }


def _serialize_note(note: TrainerNote, trainer: Trainer) -> Dict[str, Any]:
    return {
        "note_id": note.id,
        "share_id": note.share_id,
        "attempt_id": note.attempt_id,
        "question_id": note.question_id,
        "trainer_id": trainer.id,
        "trainer_display_name": trainer.display_name,
        "body": note.body,
        "rating": note.rating,
        "created_at": note.created_at.isoformat() if note.created_at else None,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    }


def build_trainer_review_payload(
    db: Session,
    share: TrainerShare,
) -> Dict[str, Any]:
    """Bundle everything the trainer review screen needs in one round-trip."""

    attempt = (
        db.query(PracticeAttempt)
        .filter(PracticeAttempt.id == share.attempt_id)
        .first()
    )
    if attempt is None:
        return {
            "attempt": None,
            "student": None,
            "answers": [],
            "notes": [],
        }

    student = (
        db.query(User).filter(User.id == share.student_user_id).first()
    )

    answer_rows = (
        db.query(AttemptAnswer)
        .filter(AttemptAnswer.attempt_id == attempt.id)
        .order_by(AttemptAnswer.submitted_at.asc(), AttemptAnswer.id.asc())
        .all()
    )

    qids = {a.question_id for a in answer_rows}
    questions_by_id: Dict[int, QuestionFromApeuni] = {}
    eval_by_qid: Dict[int, Optional[Dict[str, Any]]] = {}
    if qids:
        for q in (
            db.query(QuestionFromApeuni)
            .filter(QuestionFromApeuni.question_id.in_(qids))
            .all()
        ):
            questions_by_id[q.question_id] = q
        for ev in (
            db.query(QuestionEvaluationApeuni)
            .filter(QuestionEvaluationApeuni.question_id.in_(qids))
            .all()
        ):
            eval_by_qid[ev.question_id] = ev.evaluation_json

    answers: List[Dict[str, Any]] = [
        _serialize_answer(
            a,
            questions_by_id.get(a.question_id),
            eval_by_qid.get(a.question_id),
        )
        for a in answer_rows
    ]

    note_rows = (
        db.query(TrainerNote, Trainer)
        .join(Trainer, TrainerNote.trainer_id == Trainer.id)
        .filter(
            TrainerNote.share_id == share.id,
            TrainerNote.deleted_at.is_(None),
        )
        .order_by(TrainerNote.created_at.asc())
        .all()
    )
    notes = [_serialize_note(note, trainer) for note, trainer in note_rows]

    return {
        "share": {
            "share_id": share.id,
            "shared_at": share.shared_at.isoformat() if share.shared_at else None,
            "first_viewed_at": (
                share.first_viewed_at.isoformat() if share.first_viewed_at else None
            ),
            "last_viewed_at": (
                share.last_viewed_at.isoformat() if share.last_viewed_at else None
            ),
        },
        "attempt": _serialize_attempt(attempt),
        "student": _serialize_student(student),
        "answers": answers,
        "notes": notes,
    }
