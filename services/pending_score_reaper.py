"""Background reaper that marks orphaned `scoring_status='pending'` rows as
failed once they've been stuck longer than the configured threshold.

Why this exists
---------------
Single-replica ECS + in-memory daemon-thread scoring means any task restart
mid-flight leaves attempt rows in `scoring_status='pending'` forever. The
frontend results page then polls indefinitely and shows "Scoring in
progress" forever. The reaper detects these orphans and flips them to
`scoring_status='failed'` so the frontend sees a definite terminal state
and can show a clear error instead of an infinite spinner.

Affected tables:
  - practice_attempts.scoring_status (sectional / mock attempt aggregate)
  - attempt_answers.scoring_status   (per-question rows under an attempt)

Both columns use the same enum: pending | partial | complete | failed.

Trigger:
  - Runs every 60s in a background thread started on FastAPI app boot.
  - Reaps any row where scoring_status='pending' AND (submitted_at or
    started_at) is older than `_STALE_AFTER_SECONDS` (default 300s = 5min).

Idempotent — running twice on the same row is a no-op because the second
pass sees scoring_status='failed' and skips.
"""
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text

from core.logging_config import get_logger
from db.database import SessionLocal

log = get_logger(__name__)

_STALE_AFTER_SECONDS = 300  # 5 minutes
_REAP_INTERVAL_SECONDS = 60

_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _reap_once() -> int:
    """Single sweep. Returns the number of rows flipped from pending → failed."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_STALE_AFTER_SECONDS)
    total = 0
    db = SessionLocal()
    try:
        # attempt_answers: per-question scoring rows.
        res = db.execute(
            text(
                """
                UPDATE attempt_answers
                SET scoring_status = 'failed'
                WHERE scoring_status = 'pending'
                  AND submitted_at < :cutoff
                """
            ),
            {"cutoff": cutoff},
        )
        total += res.rowcount or 0

        # practice_attempts: section-level aggregate row.
        res = db.execute(
            text(
                """
                UPDATE practice_attempts
                SET scoring_status = 'failed'
                WHERE scoring_status = 'pending'
                  AND COALESCE(completed_at, started_at) < :cutoff
                """
            ),
            {"cutoff": cutoff},
        )
        total += res.rowcount or 0
        db.commit()
        if total > 0:
            log.warning("[PENDING-REAPER] flipped %d row(s) pending → failed (older than %ds)",
                        total, _STALE_AFTER_SECONDS)
    except Exception as exc:
        db.rollback()
        log.warning("[PENDING-REAPER] sweep failed: %s: %s", type(exc).__name__, exc)
    finally:
        db.close()
    return total


def _loop():
    log.info("[PENDING-REAPER] starting (interval=%ds, stale_after=%ds)",
             _REAP_INTERVAL_SECONDS, _STALE_AFTER_SECONDS)
    while not _stop_event.is_set():
        try:
            _reap_once()
        except Exception as exc:
            log.warning("[PENDING-REAPER] unexpected error: %s", exc)
        _stop_event.wait(timeout=_REAP_INTERVAL_SECONDS)
    log.info("[PENDING-REAPER] stopped")


def start():
    """Start the reaper background thread. Safe to call multiple times —
    second invocations are no-ops."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_loop, name="pending-score-reaper", daemon=True)
    _thread.start()


def stop():
    """Signal the reaper thread to exit. For test cleanup."""
    _stop_event.set()
