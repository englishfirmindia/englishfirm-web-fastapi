"""Boundary validators for speaking submit + S3 upload endpoints.

Two helpers share a single responsibility: stop attacker-controlled input
from flowing into S3 keys or being scored under a different user's prefix.

  safe_question_id      — coerce + DB-existence check (422 / 404)
  assert_audio_url_owned — prefix check against user's S3 path (403)

The helpers are deliberately small and side-effect-free so each call site
remains obvious. Apply both in every speaking submit; apply only the first
in get-upload-url.
"""

from fastapi import HTTPException
from sqlalchemy.orm import Session

import core.config as config
from db.models import QuestionFromApeuni


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
