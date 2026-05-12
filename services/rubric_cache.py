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

# Defaults mirror the RDS row content so a DB outage cannot floor scores to
# zero. Update both places if the rubric changes — the loader prefers RDS
# values when present.
_DEFAULTS: Dict[str, Dict[str, int]] = {
    "summarize_written_text": {
        "form_max": 1,
        "content_max": 4,
        "grammar_max": 2,
        "vocabulary_max": 2,
    },
    "write_essay": {
        # Real PTE rubric — 26 max total.
        "content_max": 6,
        "grammar_max": 2,
        "vocabulary_max": 2,
        "coherence_max": 6,   # Development, Structure & Coherence (DSC)
        "glr_max": 6,         # General Linguistic Range
        "spelling_max": 2,
        # Form is determined by word-count band rather than a single max:
        "words_200_300": 2,   # ideal range
        "words_120_199": 1,   # below ideal
        "words_301_380": 1,   # above ideal
        "words_lt_120": 0,    # form-zero — kills the attempt
        "words_gt_380": 0,    # form-zero — kills the attempt
    },
}

# Fields read for each task. Kept separate from defaults so the SQL query
# stays focused and doesn't pull columns that don't apply to the task.
_TASK_FIELDS: Dict[str, list] = {
    "summarize_written_text": [
        "form_max", "content_max", "grammar_max", "vocabulary_max",
    ],
    "write_essay": [
        "content_max", "grammar_max", "vocabulary_max",
        "coherence_max", "glr_max", "spelling_max",
        "words_200_300", "words_120_199", "words_301_380",
        "words_lt_120", "words_gt_380",
    ],
}


class _RubricCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, int]] = {}
        self._loaded_at: float = 0.0

    def _refresh(self) -> None:
        """Refresh the rubric cache from RDS. Best-effort: errors are logged
        and the previous cache (or defaults) keeps serving."""
        # Union of all fields any task wants — keep the SQL one statement.
        all_fields = sorted({
            f for fields in _TASK_FIELDS.values() for f in fields
        })
        select_cols = ", ".join(["task"] + all_fields)
        s = SessionLocal()
        try:
            rows = s.execute(
                _sql(
                    f"SELECT {select_cols} "
                    "FROM pte_scoring_reading_writing_rubric "
                    "WHERE task = ANY(:tasks)"
                ),
                {"tasks": list(_DEFAULTS.keys())},
            ).fetchall()
            new_data: Dict[str, Dict[str, int]] = {}
            for r in rows:
                task = r.task
                wanted = _TASK_FIELDS.get(task, all_fields)
                row_dict: Dict[str, int] = {}
                for field in wanted:
                    db_value = getattr(r, field, None)
                    if db_value is not None:
                        row_dict[field] = db_value
                    else:
                        row_dict[field] = _DEFAULTS.get(task, {}).get(field, 0)
                new_data[task] = row_dict
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


def get_we_form_score(word_count: int) -> int:
    """Map an essay word count to the form sub-score using RDS band values.

    Bands (real PTE):
      200–300 → words_200_300 (default 2)
      120–199 → words_120_199 (default 1)
      301–380 → words_301_380 (default 1)
      < 120   → words_lt_120  (default 0 — form-zero kills the attempt)
      > 380   → words_gt_380  (default 0 — form-zero kills the attempt)
    """
    if 200 <= word_count <= 300:
        return get_rubric_max("write_essay", "words_200_300")
    if 120 <= word_count <= 199:
        return get_rubric_max("write_essay", "words_120_199")
    if 301 <= word_count <= 380:
        return get_rubric_max("write_essay", "words_301_380")
    if word_count < 120:
        return get_rubric_max("write_essay", "words_lt_120")
    # word_count > 380
    return get_rubric_max("write_essay", "words_gt_380")


def get_we_form_max() -> int:
    """Maximum achievable form sub-score for WE (used for the total max)."""
    return get_rubric_max("write_essay", "words_200_300")
