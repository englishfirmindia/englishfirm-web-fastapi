"""
Question reports (flag an issue) — users can report problems with a specific
question. Optional screenshot uploads via presigned S3 URL.

Ported verbatim from englishfirm-app-fastapi/routers/reports.py with only
the required infrastructure path adjustments for this repo (db.database
Base/engine paths).
"""
import os
import uuid
from datetime import datetime, timezone

import boto3
from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import Session

from db.database import get_db, Base, engine
from db.models import User
from core.dependencies import get_current_user

from core.logging_config import get_logger

log = get_logger(__name__)


router = APIRouter(prefix="/reports", tags=["Reports"])


# ── DB model (created on first import) ────────────────────────────────────────
class QuestionReport(Base):
    __tablename__ = "question_reports"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    question_id     = Column(Integer, nullable=False, index=True)
    question_type   = Column(String(50), nullable=True)
    description     = Column(Text, nullable=False)
    screenshot_urls = Column(JSON, nullable=True)
    status          = Column(String(20), nullable=False, default="open")
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# Ensure table exists (safe if already present)
try:
    QuestionReport.__table__.create(bind=engine, checkfirst=True)
except Exception as e:
    log.info(f"[reports] table create skipped: {e}")


# ── S3 upload URL for screenshots ─────────────────────────────────────────────
_REGION  = os.getenv("AWS_S3_REGION", "ap-south-1")
_BUCKET  = os.getenv("S3_REPORTS_BUCKET", os.getenv("S3_RECORDINGS_BUCKET", "apeuni-user-recordings"))
_EXPIRY  = 600


def _upload_client():
    return boto3.client(
        "s3",
        region_name=_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


@router.post("/upload-url")
def get_report_upload_url(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
):
    """Presigned PUT URL for uploading a report screenshot to S3."""
    filename = (payload.get("filename") or "report.jpg").strip()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    content_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")

    key = f"reports/{current_user.id}/{uuid.uuid4().hex}.{ext}"
    try:
        client = _upload_client()
        upload_url = client.generate_presigned_url(
            "put_object",
            Params={"Bucket": _BUCKET, "Key": key, "ContentType": content_type},
            ExpiresIn=_EXPIRY,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create upload URL: {e}")

    s3_url = f"https://{_BUCKET}.s3.{_REGION}.amazonaws.com/{key}"
    return {"upload_url": upload_url, "s3_url": s3_url}


@router.post("/question")
def submit_question_report(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Record a user report against a question."""
    question_id   = payload.get("question_id")
    question_type = payload.get("question_type") or ""
    description   = (payload.get("description") or "").strip()
    screenshots   = payload.get("screenshot_urls") or []

    if not question_id:
        raise HTTPException(status_code=400, detail="question_id required")
    if not description:
        raise HTTPException(status_code=400, detail="description required")

    try:
        qid_int = int(question_id)
    except Exception:
        raise HTTPException(status_code=400, detail="question_id must be an integer")

    report = QuestionReport(
        user_id         = current_user.id,
        question_id     = qid_int,
        question_type   = question_type[:50] if question_type else None,
        description     = description,
        screenshot_urls = screenshots if isinstance(screenshots, list) else [],
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    log.info(f"[reports] user={current_user.id} q={qid_int} type={question_type} → id={report.id}")
    return {"id": report.id, "status": report.status}
