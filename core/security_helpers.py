"""Boundary validators for speaking submit + S3 upload endpoints.

Three helpers:

  safe_question_id            — coerce + DB-existence check (422 / 404)
  assert_audio_url_owned      — prefix check against user's S3 path (403)
  resolve_question_with_retry — fetch a question row with DB retry, used by
                                every speaking submit handler so a cold
                                session cache + transient DB blip doesn't
                                silently score with an empty reference.

The first two are deliberately small and side-effect-free so each call site
remains obvious. Apply both in every speaking submit; apply only the first
in get-upload-url.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

import core.config as config
from db.models import QuestionFromApeuni

log = logging.getLogger(__name__)


def safe_question_id(payload: dict, db: Session) -> int:
    """Return validated question_id from payload.

    Raises 422 if the value is missing / not coercible to int.
    Raises 404 if no matching row exists in question_from_apeuni.
    """
    raw = payload.get("question_id")
    try:
        qid = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="question_id must be an integer")
    exists = db.query(QuestionFromApeuni.question_id).filter_by(question_id=qid).first()
    if not exists:
        raise HTTPException(status_code=404, detail="question not found")
    return qid


def assert_audio_url_owned(audio_url: str, user_id: int) -> None:
    """Reject audio_url values that don't sit under the user's S3 prefix.

    The prefix is pinned by JWT-derived user_id, so a request can never
    submit another user's recording for scoring. Raises 403 on mismatch.
    """
    if not isinstance(audio_url, str):
        raise HTTPException(status_code=422, detail="audio_url must be a string")
    expected_prefix = (
        f"https://{config.S3_RECORDINGS_BUCKET}.s3.{config.AWS_REGION}.amazonaws.com"
        f"/recordings/{user_id}/"
    )
    if not audio_url.startswith(expected_prefix):
        raise HTTPException(status_code=403, detail="audio_url not owned by user")


def resolve_question_with_retry(
    question_id: int,
    db: Session,
    session: Optional[dict] = None,
    attempts: int = 2,
    sleep_ms: int = 100,
) -> Optional[QuestionFromApeuni]:
    """Resolve a QuestionFromApeuni row, with DB retry on miss.

    Order of attempts:
      1. session["questions"][question_id] — free, in-memory hit
      2. Direct PK query (joinedload evaluation), retried up to `attempts` times

    On DB hit, writes the row back into session["questions"] so subsequent
    calls in the same session don't repeat the query. Returns None only if
    all attempts exhausted — caller should respond with 503 + Retry-After.
    """
    if session is not None:
        cached = session.get("questions", {}).get(question_id)
        if cached:
            return cached

    last_err: Optional[Exception] = None
    for i in range(1, attempts + 1):
        try:
            q = (
                db.query(QuestionFromApeuni)
                .options(joinedload(QuestionFromApeuni.evaluation))
                .filter(QuestionFromApeuni.question_id == question_id)
                .first()
            )
            if q is not None:
                if session is not None:
                    session.setdefault("questions", {})[question_id] = q
                return q
            log.warning("[Q_RESOLVE] attempt=%d/%d q=%d → None", i, attempts, question_id)
        except Exception as e:
            last_err = e
            log.warning(
                "[Q_RESOLVE] attempt=%d/%d q=%d error=%s: %s",
                i, attempts, question_id, type(e).__name__, e,
            )
        if i < attempts:
            time.sleep(sleep_ms / 1000.0)

    log.error(
        "[Q_RESOLVE] q=%d unresolved after %d attempts last_err=%s",
        question_id, attempts, last_err,
    )
    return None
