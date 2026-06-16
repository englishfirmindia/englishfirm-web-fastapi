"""Learning videos — list active videos + presigned URLs for playback/thumbnails.

Self-hosted videos live in s3://englishfirm-learning-videos/. Keys are stored
on the `learning_videos` table (videos/*.mp4, thumbnails/*.jpg). Run-time
generates short-lived presigned GETs so the bucket stays private.
"""
import logging
import os
from typing import Optional

import boto3
from botocore.config import Config
from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.dependencies import get_current_user
from db.database import get_db
from db.models import User

log = logging.getLogger(__name__)
router = APIRouter(prefix="/learning", tags=["Learning - Videos"])

_BUCKET = "englishfirm-learning-videos"
_REGION = "ap-southeast-2"
# Long-ish TTL — a 20-min lecture shouldn't expire mid-watch. Reasonable
# upper bound vs the abuse cost of leaked URLs.
_VIDEO_TTL_SEC = 3 * 60 * 60  # 3 hours
_THUMB_TTL_SEC = 24 * 60 * 60  # 24 hours

# Dedicated S3 client pinned to the regional endpoint. The shared
# services.s3_service._S3_CLIENT uses boto's default endpoint resolution
# which generates URLs against the global s3.amazonaws.com host — fine
# for legacy buckets but new buckets (post-2024) issue a 307 redirect
# from the global host to the regional one, and the redirect target
# hostname doesn't match what the signature was computed for → 403.
# Pinning endpoint_url here forces virtual-hosted regional URLs that
# resolve directly with no redirect.
_S3 = boto3.client(
    "s3",
    region_name=_REGION,
    endpoint_url=f"https://s3.{_REGION}.amazonaws.com",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual"},
        connect_timeout=5,
        read_timeout=5,
        retries={"max_attempts": 1, "mode": "standard"},
    ),
)


def _presign(key: str, ttl: int) -> str:
    return _S3.generate_presigned_url(
        "get_object",
        Params={"Bucket": _BUCKET, "Key": key},
        ExpiresIn=ttl,
    )


@router.get("/videos")
def list_videos(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all active learning videos, ordered by order_index.

    Returns each row with a fresh presigned thumbnail URL. The full
    video URL is fetched lazily via /videos/{id}/url to avoid spending
    presign cost on rows the user never opens.
    """
    rows = db.execute(text("""
        SELECT id, title, description, s3_key, thumbnail_s3_key,
               duration_sec, width, height, source_type,
               youtube_video_id, order_index
        FROM learning_videos
        WHERE is_active IS TRUE
        ORDER BY order_index, id
    """)).mappings().all()

    items = []
    for r in rows:
        thumb_url: Optional[str] = None
        if r["thumbnail_s3_key"]:
            try:
                thumb_url = _presign(r["thumbnail_s3_key"], _THUMB_TTL_SEC)
            except Exception as e:
                log.warning("[LEARNING] thumb presign failed id=%s: %s", r["id"], e)
        items.append({
            "id": r["id"],
            "title": r["title"],
            "description": r["description"],
            "duration_sec": r["duration_sec"],
            "width": r["width"],
            "height": r["height"],
            "source_type": r["source_type"],
            "youtube_video_id": r["youtube_video_id"],
            "thumbnail_url": thumb_url,
            "order_index": r["order_index"],
        })
    return {"items": items, "total": len(items)}


@router.get("/videos/{video_id}/url")
def get_video_url(
    video_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Presigned MP4 URL for playback. Short-lived (3h)."""
    row = db.execute(text("""
        SELECT s3_key, source_type, youtube_video_id, is_active
        FROM learning_videos WHERE id = :id
    """), {"id": video_id}).mappings().first()

    if not row or not row["is_active"]:
        raise HTTPException(status_code=404, detail="Video not found")

    if row["source_type"] == "youtube_embed":
        return {
            "video_id": video_id,
            "source_type": "youtube_embed",
            "youtube_video_id": row["youtube_video_id"],
            "url": None,
            "expires_in_sec": None,
        }

    try:
        url = _presign(row["s3_key"], _VIDEO_TTL_SEC)
    except Exception as e:
        log.error("[LEARNING] video presign failed id=%s: %s", video_id, e)
        raise HTTPException(status_code=503, detail="Video URL temporarily unavailable")

    return {
        "video_id": video_id,
        "source_type": "self_hosted",
        "youtube_video_id": None,
        "url": url,
        "expires_in_sec": _VIDEO_TTL_SEC,
    }
