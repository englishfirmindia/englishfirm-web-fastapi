"""
RDS rubric loader for the reading/writing tasks (SWT / WE / SST).

Reads max-point columns from `pte_scoring_reading_writing_rubric` at first
access and refreshes every 5 minutes. Falls back to per-task defaults if the
DB call fails so scoring never breaks just because rubric data is unreachable.

This makes the **max points** data-driven without lifting the scoring logic
(heuristic rules, prompts, formulas) out of code — touch SQL to retune the
denominator, touch Python to retune behaviour. Matches the W7-style "logic in
code, knobs in DB" boundary used elsewhere in the speaking pipeline.
"""
import threading
import time
from typing import Dict, Optional

from sqlalchemy import text as _sql

from db.database import SessionLocal
from core.logging_config import get_logger

log = get_logger(__name__)


_REFRESH_SECONDS = 300  # 5 min

# Defaults match the values that were hardcoded in ai_scorer.py prior to this
# loader landing. Kept in code so a DB outage cannot floor scores to zero.
_DEFAULTS: Dict[str, Dict[str, int]] = {
    "summarize_written_text": {
        "form_max": 1,
        "content_max": 4,
        "grammar_max": 2,
        "vocabulary_max": 2,
    },
}


class _RubricCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, int]] = {}
        self._loaded_at: float = 0.0

    def _refresh(self) -> None:
        """Refresh the rubric cache from RDS. Best-effort: errors are logged
        and the previous cache (or defaults) keeps serving."""
        s = SessionLocal()
        try:
            rows = s.execute(
                _sql(
                    "SELECT task, form_max, content_max, grammar_max, vocabulary_max "
                    "FROM pte_scoring_reading_writing_rubric "
                    "WHERE task = ANY(:tasks)"
                ),
                {"tasks": list(_DEFAULTS.keys())},
            ).fetchall()
            new_data: Dict[str, Dict[str, int]] = {}
            for r in rows:
                new_data[r.task] = {
                    "form_max": r.form_max if r.form_max is not None else _DEFAULTS[r.task]["form_max"],
                    "content_max": r.content_max if r.content_max is not None else _DEFAULTS[r.task]["content_max"],
                    "grammar_max": r.grammar_max if r.grammar_max is not None else _DEFAULTS[r.task]["grammar_max"],
                    "vocabulary_max": r.vocabulary_max if r.vocabulary_max is not None else _DEFAULTS[r.task]["vocabulary_max"],
                }
            with self._lock:
                self._data = new_data
                self._loaded_at = time.time()
            log.info("[RUBRIC] refreshed cache tasks=%s", list(new_data.keys()))
        except Exception as e:
            log.error(f"[RUBRIC] cache refresh failed (will keep using prior cache / defaults): {e}")
        finally:
            s.close()

    def get_max(self, task: str, field: str) -> int:
        """Return the configured max for (task, field) — falls back to the
        hardcoded default if the cache is empty or the task is unknown."""
        now = time.time()
        if now - self._loaded_at > _REFRESH_SECONDS:
            self._refresh()
        with self._lock:
            row = self._data.get(task)
        if row and field in row:
            return row[field]
        defaults = _DEFAULTS.get(task, {})
        return defaults.get(field, 0)


_cache = _RubricCache()


def get_rubric_max(task: str, field: str) -> int:
    """Read the (task, field) max from RDS via cache. Falls back to hardcoded
    defaults if the DB is unreachable or the field is missing."""
    return _cache.get_max(task, field)
