"""Learning videos — list active videos + presigned URLs for playback/thumbnails.

Self-hosted videos live in s3://englishfirm-learning-videos/. Keys are stored
on the `learning_videos` table (videos/*.mp4, thumbnails/*.jpg). Run-time
generates short-lived presigned GETs so the bucket stays private.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.dependencies import get_current_user
from db.database import get_db
from db.models import User
from services.s3_service import generate_presigned_url

log = logging.getLogger(__name__)
router = APIRouter(prefix="/learning", tags=["Learning - Videos"])

_BUCKET = "englishfirm-learning-videos"
_REGION = "ap-southeast-2"
# Long-ish TTL — a 20-min lecture shouldn't expire mid-watch. Reasonable
# upper bound vs the abuse cost of leaked URLs.
_VIDEO_TTL_SEC = 3 * 60 * 60  # 3 hours
_THUMB_TTL_SEC = 24 * 60 * 60  # 24 hours


def _s3_url(key: str) -> str:
    return f"https://{_BUCKET}.s3.{_REGION}.amazonaws.com/{key}"


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
                thumb_url = generate_presigned_url(
                    _s3_url(r["thumbnail_s3_key"]), expires_in=_THUMB_TTL_SEC,
                )
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
        url = generate_presigned_url(_s3_url(row["s3_key"]), expires_in=_VIDEO_TTL_SEC)
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
