"""
In-memory session store. Keys are UUID session_id strings.
Each session dict contains:
  user_id, questions (dict[int -> QuestionFromApeuni]),
  score, submitted_questions (set), question_scores (dict),
  start_time, attempt_id (optional, sectional only)
"""
import uuid
import time
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session, joinedload
from fastapi import HTTPException, status

from db.models import QuestionFromApeuni
import core.config as config

ACTIVE_SESSIONS: Dict[str, dict] = {}
_SCORE_STORE: Dict[tuple, dict] = {}  # keyed by (user_id, question_id)


def start_session(
    db: Session,
    user_id: int,
    module: str,
    question_type: str,
    difficulty_level: Optional[int] = None,
    limit: int = config.SESSION_QUESTION_LIMIT,
) -> dict:
    """
    Load questions from DB, create in-memory session.
    Returns {session_id, total_questions, questions: [...]}.
    Ported from question_service.start_mock_test.
    """
    query = (
        db.query(QuestionFromApeuni)
        .options(joinedload(QuestionFromApeuni.evaluation))
        .filter(
            QuestionFromApeuni.module == module,
            QuestionFromApeuni.question_type == question_type,
        )
    )

    if difficulty_level is not None:
        query = query.filter(
            QuestionFromApeuni.difficulty_level == difficulty_level
        )

    query = query.order_by(QuestionFromApeuni.question_id.asc())

    if limit:
        query = query.limit(limit)

    questions = query.all()

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No questions found",
        )

    session_id = str(uuid.uuid4())

    ACTIVE_SESSIONS[session_id] = {
        "user_id": user_id,
        "start_time": int(time.time()),
        "questions": {q.question_id: q for q in questions},
        "score": 0,
        "submitted_questions": set(),
        "question_scores": {},
    }

    return {
        "session_id": session_id,
        "total_questions": len(questions),
        "questions": [
            {
                "question_id": q.question_id,
                "module": q.module,
                "question_type": q.question_type,
                "difficulty_level": q.difficulty_level,
                "time_limit_seconds": q.time_limit_seconds,
                "content_json": q.content_json,
            }
            for q in questions
        ],
    }


def get_session(session_id: str) -> dict:
    """Get session or raise HTTP 400."""
    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid or expired session")
    return session


def mark_submitted(session_id: str, question_id: int, score: int) -> None:
    """Mark question as submitted and update session score."""
    session = get_session(session_id)
    session["submitted_questions"].add(question_id)
    session["score"] = session.get("score", 0) + score
    session.setdefault("question_scores", {})[question_id] = score


def get_score_from_store(user_id: int, question_id: int) -> Optional[dict]:
    """Poll async score result. Returns None if not ready."""
    return _SCORE_STORE.get((user_id, question_id))


def store_score(user_id: int, question_id: int, result: dict) -> None:
    _SCORE_STORE[(user_id, question_id)] = result
