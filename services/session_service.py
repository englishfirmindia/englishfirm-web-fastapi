"""
In-memory session store + DB persistence for practice answers.
"""
import uuid
import time
import threading
from typing import Optional, Dict

from sqlalchemy.orm import Session, joinedload
from fastapi import HTTPException, status

from db.models import QuestionFromApeuni, PracticeAttempt, AttemptAnswer, UserQuestionAttempt
from db.database import SessionLocal
import core.config as config

ACTIVE_SESSIONS: Dict[str, dict] = {}
_SCORE_STORE: Dict[tuple, dict] = {}


def _enrich_content_json(q) -> dict:
    """Merge evaluation transcript / expected answer into content_json.

    Per question type, `correctAnswers` stores the model answer in different
    fields:
      - RTS  → correctAnswers.transcript (the situation prompt itself; show
               before submit, mapped to `situation_text`)
      - DI / RL / SGD → correctAnswers.transcript (model answer; show after
                        submit, mapped to `sample_answer`)
      - ASQ  → correctAnswers.answer (single expected answer; show after
               submit, mapped to `sample_answer`)
    """
    base = dict(q.content_json or {})
    if not (q.evaluation and q.evaluation.evaluation_json):
        return base
    correct = q.evaluation.evaluation_json.get("correctAnswers", {}) or {}

    if q.question_type == "ptea_respond_situation":
        transcript = correct.get("transcript", "") or ""
        if transcript:
            base.setdefault("situation_text", transcript)
        return base

    if q.question_type == "answer_short_question":
        # ASQ stores the expected answer as a single string in `answer`;
        # `acceptedVariants` lists alternatives but the canonical one is `answer`.
        expected = correct.get("answer", "") or ""
        if expected:
            base.setdefault("sample_answer", expected)
        return base

    transcript = correct.get("transcript", "") or ""
    if transcript:
        base.setdefault("sample_answer", transcript)
    return base


def start_session(
    db: Session,
    user_id: int,
    module: str,
    question_type: str,
    difficulty_level: Optional[int] = None,
    limit: int = config.SESSION_QUESTION_LIMIT,
    question_id: Optional[int] = None,
) -> dict:
    query = (
        db.query(QuestionFromApeuni)
        .options(joinedload(QuestionFromApeuni.evaluation))
        .filter(
            QuestionFromApeuni.module == module,
            QuestionFromApeuni.question_type == question_type,
        )
    )
    start_index = 0
    if question_id is not None:
        from sqlalchemy import func as _func
        _look_back = 2
        position = db.query(_func.count(QuestionFromApeuni.question_id)).filter(
            QuestionFromApeuni.module == module,
            QuestionFromApeuni.question_type == question_type,
            QuestionFromApeuni.question_id < question_id,
        ).scalar() or 0
        start_offset = max(0, position - _look_back)
        start_index = position - start_offset
        query = query.order_by(QuestionFromApeuni.question_id.asc())
        if limit:
            query = query.offset(start_offset).limit(limit)
    else:
        if difficulty_level is not None:
            query = query.filter(QuestionFromApeuni.difficulty_level == difficulty_level)
        query = query.order_by(QuestionFromApeuni.question_id.asc())
        if limit:
            query = query.limit(limit)
    questions = query.all()
    if not questions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No questions found")

    session_id = str(uuid.uuid4())

    # Create PracticeAttempt row in DB for practice sessions too
    try:
        attempt = PracticeAttempt(
            user_id=user_id,
            session_id=session_id,
            module=module,
            question_type=question_type,
            filter_type="practice",
            total_questions=len(questions),
            total_score=0,
            questions_answered=0,
            status="active",
            scoring_status="pending",
        )
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
        attempt_id = attempt.id
    except Exception as e:
        print(f"[SESSION] DB attempt creation failed: {e}", flush=True)
        db.rollback()
        attempt_id = None

    ACTIVE_SESSIONS[session_id] = {
        "session_id": session_id,
        "user_id": user_id,
        "start_time": int(time.time()),
        "questions": {q.question_id: q for q in questions},
        "score": 0,
        "submitted_questions": set(),
        "question_scores": {},
        "attempt_id": attempt_id,
        "module": module,
        "question_type": question_type,
    }

    return {
        "session_id": session_id,
        "total_questions": len(questions),
        "start_index": start_index,
        "questions": [
            {
                "question_id": q.question_id,
                "module": q.module,
                "question_type": q.question_type,
                "difficulty_level": q.difficulty_level,
                "time_limit_seconds": q.time_limit_seconds,
                "content_json": _enrich_content_json(q),
            }
            for q in questions
        ],
    }


class _LazyQuestionsDict(dict):
    """Dict that lazy-loads QuestionFromApeuni rows from the DB on .get(qid).

    Used when ACTIVE_SESSIONS is rebuilt after a backend reload — we don't
    know which questions were originally selected for the practice session,
    so we load them on demand as the user submits each one.
    """
    def get(self, key, default=None):  # type: ignore[override]
        if key in self:
            return self[key]
        db = SessionLocal()
        try:
            from sqlalchemy.orm import joinedload as _joinedload
            q = (
                db.query(QuestionFromApeuni)
                .options(_joinedload(QuestionFromApeuni.evaluation))
                .filter(QuestionFromApeuni.question_id == key)
                .first()
            )
            if q:
                self[key] = q
                return q
        finally:
            db.close()
        return default


def get_session(session_id: str) -> dict:
    session = ACTIVE_SESSIONS.get(session_id)
    if session:
        return session

    # Rebuild from DB — survives uvicorn --reload that wipes in-memory state.
    db = SessionLocal()
    try:
        attempt = (
            db.query(PracticeAttempt)
            .filter_by(session_id=session_id)
            .first()
        )
        if not attempt:
            raise HTTPException(status_code=400, detail="Invalid or expired session")

        submitted_qids = {
            row[0] for row in
            db.query(AttemptAnswer.question_id)
              .filter_by(attempt_id=attempt.id)
              .all()
        }

        rebuilt = {
            "session_id": session_id,
            "user_id": attempt.user_id,
            "start_time": int(time.time()),
            "questions": _LazyQuestionsDict(),
            "score": attempt.total_score or 0,
            "submitted_questions": submitted_qids,
            "module": attempt.module,
            "attempt_id": attempt.id,
        }
        ACTIVE_SESSIONS[session_id] = rebuilt
        return rebuilt
    finally:
        db.close()


def mark_submitted(session_id: str, question_id: int, score: int) -> None:
    session = get_session(session_id)
    session["submitted_questions"].add(question_id)
    session["score"] = session.get("score", 0) + score
    session.setdefault("question_scores", {})[question_id] = score

    # Record attempt in user_question_attempts for deduplication
    def _record():
        last_exc: Exception = RuntimeError("_record: no attempts made")
        for attempt in range(1, 4):
            db = SessionLocal()
            try:
                user_id = session.get("user_id")
                module = session.get("module", "")
                q_type = session.get("question_type", "")
                if user_id:
                    exists = db.query(UserQuestionAttempt).filter_by(
                        user_id=user_id, question_id=question_id
                    ).first()
                    if not exists:
                        db.add(UserQuestionAttempt(
                            user_id=user_id,
                            question_id=question_id,
                            question_type=q_type,
                            module=module,
                        ))
                    # Update PracticeAttempt answered count + total_score
                    # Skip for sectional — bg aggregation thread owns total_score
                    attempt_id = session.get("attempt_id")
                    if attempt_id:
                        pa = db.query(PracticeAttempt).filter_by(id=attempt_id).first()
                        if pa and pa.question_type != "sectional":
                            pa.questions_answered = len(session["submitted_questions"])
                            pa.total_score = session["score"]
                    db.commit()
                return
            except Exception as e:
                last_exc = e
                print(f"[MARK_SUBMITTED] DB error attempt={attempt}/3: {e}", flush=True)
                db.rollback()
                if attempt < 3:
                    time.sleep(attempt)
            finally:
                db.close()
        print(f"[MARK_SUBMITTED] failed after 3 attempts: {last_exc}", flush=True)

    threading.Thread(target=_record, daemon=True).start()


def persist_answer_to_db(
    session: dict,
    question_id: int,
    question_type: str,
    user_answer_json: dict,
    correct_answer_json: dict,
    result_json: dict,
    score: int = 0,
    audio_url: Optional[str] = None,
    scoring_status: str = "complete",
) -> None:
    """Write or upsert an AttemptAnswer row for this question."""
    attempt_id = session.get("attempt_id")
    session_id = session.get("session_id")
    if not attempt_id and not session_id:
        return

    def _write():
        nonlocal attempt_id
        last_exc: Exception = RuntimeError("_write: no attempts made")
        for attempt in range(1, 4):
            db = SessionLocal()
            try:
                # If attempt_id not cached in memory, look it up by session_id
                if not attempt_id and session_id:
                    pa = db.query(PracticeAttempt).filter_by(session_id=session_id).first()
                    if not pa:
                        return
                    attempt_id = pa.id
                existing = db.query(AttemptAnswer).filter_by(
                    attempt_id=attempt_id, question_id=question_id
                ).first()
                if existing:
                    existing.user_answer_json = user_answer_json
                    existing.correct_answer_json = correct_answer_json
                    existing.result_json = result_json
                    existing.score = score
                    existing.scoring_status = scoring_status
                    if audio_url:
                        existing.audio_url = audio_url
                else:
                    db.add(AttemptAnswer(
                        attempt_id=attempt_id,
                        question_id=question_id,
                        question_type=question_type,
                        user_answer_json=user_answer_json,
                        correct_answer_json=correct_answer_json,
                        result_json=result_json,
                        score=score,
                        audio_url=audio_url,
                        scoring_status=scoring_status,
                    ))
                db.commit()
                return
            except Exception as e:
                last_exc = e
                print(f"[PERSIST_ANSWER] DB error attempt={attempt}/3 q={question_id}: {e}", flush=True)
                db.rollback()
                if attempt < 3:
                    time.sleep(attempt)
            finally:
                db.close()
        print(f"[PERSIST_ANSWER] failed after 3 attempts q={question_id}: {last_exc}", flush=True)

    threading.Thread(target=_write, daemon=True).start()


def persist_speaking_answer_pending(
    session: dict,
    question_id: int,
    question_type: str,
    audio_url: str,
) -> None:
    """Write AttemptAnswer row immediately on speaking submit (pending state)."""
    attempt_id = session.get("attempt_id")
    if not attempt_id:
        return

    def _write():
        last_exc: Exception = RuntimeError("_write: no attempts made")
        for attempt in range(1, 4):
            db = SessionLocal()
            try:
                existing = db.query(AttemptAnswer).filter_by(
                    attempt_id=attempt_id, question_id=question_id
                ).first()
                if not existing:
                    db.add(AttemptAnswer(
                        attempt_id=attempt_id,
                        question_id=question_id,
                        question_type=question_type,
                        user_answer_json={"audio_url": audio_url},
                        correct_answer_json={},
                        result_json={},
                        score=0,
                        audio_url=audio_url,
                        scoring_status="pending",
                    ))
                    db.commit()
                return
            except Exception as e:
                last_exc = e
                print(f"[PERSIST_SPEAKING] DB error attempt={attempt}/3 q={question_id}: {e}", flush=True)
                db.rollback()
                if attempt < 3:
                    time.sleep(attempt)
            finally:
                db.close()
        print(f"[PERSIST_SPEAKING] failed after 3 attempts q={question_id}: {last_exc}", flush=True)

    threading.Thread(target=_write, daemon=True).start()


def update_speaking_score_in_db(
    user_id: int,
    question_id: int,
    content: float,
    pronunciation: float,
    fluency: float,
    total: float,
    transcript: str = "",
    word_scores: list = None,
) -> None:
    """Update AttemptAnswer with Azure scores after async scoring completes."""
    def _update():
        from sqlalchemy import func
        last_exc: Exception = RuntimeError("_update: no attempts made")
        for attempt in range(1, 4):
            db = SessionLocal()
            try:
                # Find the most recent pending answer for this user+question
                answer = (
                    db.query(AttemptAnswer)
                    .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
                    .filter(
                        PracticeAttempt.user_id == user_id,
                        AttemptAnswer.question_id == question_id,
                        AttemptAnswer.scoring_status == "pending",
                    )
                    .order_by(AttemptAnswer.submitted_at.desc())
                    .first()
                )
                if answer:
                    answer.content_score = content
                    answer.fluency_score = fluency
                    answer.pronunciation_score = pronunciation
                    answer.score = int(round(total))
                    answer.result_json = {
                        "content": content,
                        "pronunciation": pronunciation,
                        "fluency": fluency,
                        "total": total,
                        "transcript": transcript,
                        "word_scores": word_scores or [],
                    }
                    answer.scoring_status = "complete"
                    db.flush()
                    # For sectional attempts the background aggregation thread owns
                    # total_score and scoring_status — skip the raw-sum accumulation
                    # so the bg thread's weighted PTE score is never overwritten.
                    pa = db.query(PracticeAttempt).filter_by(id=answer.attempt_id).first()
                    if pa and pa.question_type != "sectional":
                        total_score = db.query(func.sum(AttemptAnswer.score)).filter_by(
                            attempt_id=pa.id
                        ).scalar() or 0
                        pa.total_score = int(total_score)
                        pending_count = db.query(AttemptAnswer).filter_by(
                            attempt_id=pa.id, scoring_status="pending"
                        ).count()
                        if pending_count == 0:
                            pa.scoring_status = "complete"
                    db.commit()
                return
            except Exception as e:
                last_exc = e
                print(f"[UPDATE_SCORE] DB error attempt={attempt}/3 q={question_id}: {e}", flush=True)
                db.rollback()
                if attempt < 3:
                    time.sleep(attempt)
            finally:
                db.close()
        print(f"[UPDATE_SCORE] failed after 3 attempts q={question_id}: {last_exc}", flush=True)

    threading.Thread(target=_update, daemon=True).start()


def get_score_from_store(user_id: int, question_id: int) -> Optional[dict]:
    """Look up a speaking score; falls back to AttemptAnswer.result_json if the
    in-memory store has been cleared (e.g. by a backend reload) or the cached
    entry is stale (missing fields added after the entry was cached).
    """
    cached = _SCORE_STORE.get((user_id, question_id))
    # Treat completed-but-missing-transcript cache entries as stale — they were
    # written by an older code path before transcript was included.
    cache_is_fresh = (
        cached is not None
        and (cached.get("scoring") != "complete" or "transcript" in cached)
    )
    if cache_is_fresh:
        return cached

    db = SessionLocal()
    try:
        answer = (
            db.query(AttemptAnswer)
            .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
            .filter(
                PracticeAttempt.user_id == user_id,
                AttemptAnswer.question_id == question_id,
                AttemptAnswer.scoring_status == "complete",
            )
            .order_by(AttemptAnswer.submitted_at.desc())
            .first()
        )
        if not answer or not answer.result_json:
            return None
        rj = answer.result_json
        result = {
            "scoring":       "complete",
            "content":       rj.get("content", 0),
            "fluency":       rj.get("fluency", 0),
            "pronunciation": rj.get("pronunciation", 0),
            "total":         rj.get("total", answer.score),
            "transcript":    rj.get("transcript", ""),
            "word_scores":   rj.get("word_scores", []),
        }
        # Repopulate in-memory store so subsequent polls don't re-hit the DB
        _SCORE_STORE[(user_id, question_id)] = result
        return result
    finally:
        db.close()


def store_score(user_id: int, question_id: int, result: dict) -> None:
    _SCORE_STORE[(user_id, question_id)] = result
