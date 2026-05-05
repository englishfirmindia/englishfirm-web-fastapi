"""
Postgres-backed session store with in-process write-through cache.

Session state is persisted to the ``practice_session_states`` table on each
write so the data survives process restart and (best-effort) is visible across
worker processes. ORM ``QuestionFromApeuni`` instances cannot be JSON-encoded,
so on serialize they are replaced with the list of question IDs and re-fetched
in a single batch query on load.

Existing call sites can mutate the returned dict in place (sets, nested dicts)
as before, but those mutations are only persisted when callers either
re-assign via ``ACTIVE_SESSIONS[sid] = session`` or explicitly call
``ACTIVE_SESSIONS.save(sid)``.
"""
import uuid
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.attributes import flag_modified
from fastapi import HTTPException, status

from db.models import QuestionFromApeuni, PracticeAttempt, AttemptAnswer, UserQuestionAttempt, PracticeSessionState
from db.database import SessionLocal
import core.config as config

from core.logging_config import get_logger

log = get_logger(__name__)


_SESSION_TTL_SECONDS = 24 * 3600


def _serialize_session(value: dict) -> dict:
    """Strip non-JSON-serializable bits (ORM instances, sets) for Postgres storage."""
    out: Dict[str, Any] = {}
    for k, v in value.items():
        if k == "questions":
            if isinstance(v, dict):
                out["__question_ids"] = list(v.keys())
            continue
        if isinstance(v, set):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = {str(kk) if isinstance(kk, int) else kk: vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


_SET_KEYS = {"submitted_questions"}
_INT_KEY_DICTS = {
    "questions",
    "submitted_audio",
    "question_scores",
    "question_score_maxes",
}


def _deserialize_session(payload: dict, db: Session) -> dict:
    out: Dict[str, Any] = dict(payload)
    qids = out.pop("__question_ids", []) or []

    questions: Dict[int, QuestionFromApeuni] = {}
    if qids:
        rows = (
            db.query(QuestionFromApeuni)
            .options(joinedload(QuestionFromApeuni.evaluation))
            .filter(QuestionFromApeuni.question_id.in_(qids))
            .all()
        )
        questions = {q.question_id: q for q in rows}
    out["questions"] = questions

    for key in _SET_KEYS:
        if key in out and isinstance(out[key], list):
            out[key] = set(out[key])

    for key in _INT_KEY_DICTS:
        if key in out and isinstance(out[key], dict):
            out[key] = {
                (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): v
                for k, v in out[key].items()
            }
    return out


class SessionStore:
    """Dict-like Postgres-backed session store with in-process cache."""

    def __init__(self) -> None:
        self._cache: Dict[str, dict] = {}

    def __setitem__(self, session_id: str, value: dict) -> None:
        self._cache[session_id] = value
        self._persist(session_id, value)

    def __getitem__(self, session_id: str) -> dict:
        v = self.get(session_id)
        if v is None:
            raise KeyError(session_id)
        return v

    def __contains__(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    def get(self, session_id: str, default: Any = None) -> Any:
        if session_id in self._cache:
            return self._cache[session_id]
        loaded = self._load(session_id)
        if loaded is None:
            return default
        self._cache[session_id] = loaded
        return loaded

    def save(self, session_id: str) -> None:
        """Re-persist the cached session after callers mutated it in place."""
        if session_id in self._cache:
            self._persist(session_id, self._cache[session_id])

    def pop(self, session_id: str, default: Any = None) -> Any:
        v = self._cache.pop(session_id, default)
        db = SessionLocal()
        try:
            db.query(PracticeSessionState).filter_by(session_id=session_id).delete()
            db.commit()
        except Exception as e:
            log.error(f"[SessionStore] pop failed sid={session_id}: {e}")
            db.rollback()
        finally:
            db.close()
        return v

    def _persist(self, session_id: str, value: dict) -> None:
        payload = _serialize_session(value)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=_SESSION_TTL_SECONDS)
        user_id = value.get("user_id") or 0
        db = SessionLocal()
        try:
            row = db.query(PracticeSessionState).filter_by(session_id=session_id).first()
            if row:
                row.data = payload
                row.expires_at = expires_at
                row.user_id = user_id
                flag_modified(row, "data")
            else:
                db.add(PracticeSessionState(
                    session_id=session_id,
                    user_id=user_id,
                    data=payload,
                    expires_at=expires_at,
                ))
            db.commit()
        except Exception as e:
            log.error(f"[SessionStore] persist failed sid={session_id}: {e}")
            db.rollback()
        finally:
            db.close()

    def _load(self, session_id: str) -> Optional[dict]:
        db = SessionLocal()
        try:
            row = db.query(PracticeSessionState).filter_by(session_id=session_id).first()
            if not row:
                return None
            if row.expires_at and row.expires_at < datetime.now(timezone.utc):
                db.delete(row)
                db.commit()
                return None
            return _deserialize_session(row.data or {}, db)
        except Exception as e:
            log.error(f"[SessionStore] load failed sid={session_id}: {e}")
            return None
        finally:
            db.close()


ACTIVE_SESSIONS = SessionStore()
_SCORE_STORE: Dict[tuple, dict] = {}


def enrich_content_json(q) -> dict:
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
            base["situation_text"] = transcript
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
        log.error(f"[SESSION] DB attempt creation failed: {e}")
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
                "content_json": enrich_content_json(q),
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


def mark_submitted(
    session_id: str,
    question_id: int,
    score: int,
    question_type: Optional[str] = None,
) -> None:
    session = get_session(session_id)
    session["submitted_questions"].add(question_id)
    session["score"] = session.get("score", 0) + score
    session.setdefault("question_scores", {})[question_id] = score
    ACTIVE_SESSIONS.save(session_id)

    # Record attempt in user_question_attempts for deduplication
    def _record():
        last_exc: Exception = RuntimeError("_record: no attempts made")
        for attempt in range(1, 4):
            db = SessionLocal()
            try:
                user_id = session.get("user_id")
                module = session.get("module", "")
                q_type = question_type or session.get("question_type", "")
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
                log.error(f"[MARK_SUBMITTED] DB error attempt={attempt}/3: {e}")
                db.rollback()
                if attempt < 3:
                    time.sleep(attempt)
            finally:
                db.close()
        log.error(f"[MARK_SUBMITTED] failed after 3 attempts: {last_exc}")

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

    # Display floor: when the scorer has produced a final PTE score (10–90),
    # store that in `score` so any view reading the column shows the floored
    # value instead of raw hits/earned (which can be 0). Pending rows (no
    # pte_score yet) keep whatever the caller passed.
    if isinstance(result_json, dict):
        _pte = result_json.get("pte_score")
        if isinstance(_pte, int) and _pte > 0:
            score = _pte

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
                log.error(f"[PERSIST_ANSWER] DB error attempt={attempt}/3 q={question_id}: {e}")
                db.rollback()
                if attempt < 3:
                    time.sleep(attempt)
            finally:
                db.close()
        log.error(f"[PERSIST_ANSWER] failed after 3 attempts q={question_id}: {last_exc}")

    threading.Thread(target=_write, daemon=True).start()


def persist_speaking_answer_pending(
    session: dict,
    question_id: int,
    question_type: str,
    audio_url: str,
) -> None:
    """Write or RESET AttemptAnswer row on speaking submit (pending state).

    Redo path: when a row already exists for (attempt_id, question_id) and
    scoring is complete, we reset it back to pending with the new audio_url
    and a cleared result_json. The background scoring thread then finds the
    pending row and updates it with the new score.

    Without this reset, the previous row stayed `complete` with its first
    score, so update_speaking_score_in_db couldn't find a pending row for
    the new submission and the new score never landed in the DB. The
    in-memory _SCORE_STORE saw it; the DB didn't.
    """
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
                if existing:
                    # Reset for new submission (Redo or any resubmit).
                    existing.user_answer_json    = {"audio_url": audio_url}
                    existing.correct_answer_json = {}
                    existing.result_json         = {}
                    existing.score               = 0
                    existing.audio_url           = audio_url
                    existing.scoring_status      = "pending"
                    existing.content_score       = None
                    existing.fluency_score       = None
                    existing.pronunciation_score = None
                    # Bump submitted_at so /last-answer ordering reflects the
                    # actual latest submission, not the first one.
                    existing.submitted_at        = datetime.now(timezone.utc)
                    log.info(
                        "[PERSIST_SPEAKING] reset existing row for redo "
                        "attempt=%s q=%s answer_id=%s",
                        attempt_id, question_id, existing.id,
                    )
                else:
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
                log.error(f"[PERSIST_SPEAKING] DB error attempt={attempt}/3 q={question_id}: {e}")
                db.rollback()
                if attempt < 3:
                    time.sleep(attempt)
            finally:
                db.close()
        log.error(f"[PERSIST_SPEAKING] failed after 3 attempts q={question_id}: {last_exc}")

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
    fluency_metrics: dict = None,
) -> None:
    """Update AttemptAnswer with Azure scores after async scoring completes.

    fluency_metrics: optional dict produced by _apply_speaking_fluency_formula —
    persisted into result_json under the same key for audit (wpm, silence_pct,
    pause_count, sentence_count, silence_rule_applied, duration_sec, and
    cross_multipliers when in penalty zone). Empty/None is omitted.
    """
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
                    # Display floor: PTE score never shown below PTE_FLOOR (10)
                    # even when scoring failed/returned 0. The raw azure
                    # subscores stay as-is.
                    floored = max(int(round(total)), config.PTE_FLOOR)
                    answer.content_score = content
                    answer.fluency_score = fluency
                    answer.pronunciation_score = pronunciation
                    answer.score = floored
                    rj = {
                        "content": content,
                        "pronunciation": pronunciation,
                        "fluency": fluency,
                        "total": total,
                        "pte_score": floored,
                        "transcript": transcript,
                        "word_scores": word_scores or [],
                    }
                    if fluency_metrics:
                        rj["fluency_metrics"] = fluency_metrics
                    answer.result_json = rj
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
                log.error(f"[UPDATE_SCORE] DB error attempt={attempt}/3 q={question_id}: {e}")
                db.rollback()
                if attempt < 3:
                    time.sleep(attempt)
            finally:
                db.close()
        log.error(f"[UPDATE_SCORE] failed after 3 attempts q={question_id}: {last_exc}")

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
