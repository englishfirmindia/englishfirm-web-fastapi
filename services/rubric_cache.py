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
    "summarize_spoken_text": {
        # Honours RDS values exactly — total max = 12 (form 2 + content 4 +
        # grammar 2 + vocab 2 + spelling 2). Spelling is now a real sub-
        # score (was absent in the pre-Claude scorer).
        "content_max": 4,
        "grammar_max": 2,
        "vocabulary_max": 2,
        "spelling_max": 2,
        "form_50_70": 2,    # ideal range
        "form_40_49": 1,    # below ideal
        "form_71_100": 1,   # above ideal
        "form_lt_40": 0,    # form-zero — kills the attempt
        "form_gt_100": 0,   # form-zero — kills the attempt
    },
}

# Fields read for each task. Kept separate from defaults so the SQL query
# stays focused and doesn't pull columns that don't apply to the task.
# Tasks are split by table since SST lives in pte_scoring_listening_rubric
# whereas SWT/WE live in pte_scoring_reading_writing_rubric.
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
    "summarize_spoken_text": [
        "content_max", "grammar_max", "vocabulary_max", "spelling_max",
        "form_50_70", "form_40_49", "form_71_100",
        "form_lt_40", "form_gt_100",
    ],
}

# Which RDS table each task lives in. SWT/WE share the reading/writing
# rubric; SST is a listening-module task and lives in the listening rubric.
_RW_TASKS = {"summarize_written_text", "write_essay"}
_LIS_TASKS = {"summarize_spoken_text"}


class _RubricCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, int]] = {}
        self._loaded_at: float = 0.0

    def _refresh(self) -> None:
        """Refresh the rubric cache from RDS. Best-effort: errors are logged
        and the previous cache (or defaults) keeps serving. Pulls from both
        the reading/writing rubric and the listening rubric in one pass."""
        new_data: Dict[str, Dict[str, int]] = {}

        def _load_from(table: str, tasks: set) -> None:
            if not tasks:
                return
            # Only select the columns the tasks in this table actually use.
            cols = sorted({
                f for t in tasks for f in _TASK_FIELDS.get(t, [])
            })
            select_cols = ", ".join(["task"] + cols)
            try:
                rows = s.execute(
                    _sql(
                        f"SELECT {select_cols} FROM {table} "
                        "WHERE task = ANY(:tasks)"
                    ),
                    {"tasks": list(tasks)},
                ).fetchall()
                for r in rows:
                    task = r.task
                    wanted = _TASK_FIELDS.get(task, cols)
                    row_dict: Dict[str, int] = {}
                    for field in wanted:
                        db_value = getattr(r, field, None)
                        if db_value is not None:
                            row_dict[field] = db_value
                        else:
                            row_dict[field] = _DEFAULTS.get(task, {}).get(field, 0)
                    new_data[task] = row_dict
            except Exception as exc:
                log.error(
                    f"[RUBRIC] {table} query failed (using defaults for {tasks}): {exc}"
                )

        s = SessionLocal()
        try:
            _load_from("pte_scoring_reading_writing_rubric", _RW_TASKS)
            _load_from("pte_scoring_listening_rubric", _LIS_TASKS)
            if new_data:
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


def get_sst_form_score(word_count: int) -> int:
    """Map an SST summary word count to the form sub-score using RDS bands.

    Bands (real PTE):
      50–70  → form_50_70  (default 2)
      40–49  → form_40_49  (default 1)
      71–100 → form_71_100 (default 1)
      < 40   → form_lt_40  (default 0 — form-zero kills the attempt)
      > 100  → form_gt_100 (default 0 — form-zero kills the attempt)

    Note: the RDS row also has a `form_t_40` column (likely a duplicate of
    `form_lt_40`). It is intentionally ignored.
    """
    if 50 <= word_count <= 70:
        return get_rubric_max("summarize_spoken_text", "form_50_70")
    if 40 <= word_count <= 49:
        return get_rubric_max("summarize_spoken_text", "form_40_49")
    if 71 <= word_count <= 100:
        return get_rubric_max("summarize_spoken_text", "form_71_100")
    if word_count < 40:
        return get_rubric_max("summarize_spoken_text", "form_lt_40")
    # word_count > 100
    return get_rubric_max("summarize_spoken_text", "form_gt_100")


def get_sst_form_max() -> int:
    """Maximum achievable form sub-score for SST (used for the total max)."""
    return get_rubric_max("summarize_spoken_text", "form_50_70")


# ── Section weight cache (pte_question_weightage) ─────────────────────────
#
# Separate cache from the rubric one because it lives in a different table
# and has a different shape (one row per task with four section percents).
# Used for cross-section contribution math — e.g. a speaking task can also
# contribute to listening per PTE's enabling-skill rules.

_SECTION_WEIGHT_DEFAULTS: Dict[str, Dict[str, int]] = {
    "read_aloud":                 {"speaking_percent": 9,  "listening_percent": 0,  "reading_percent": 0, "writing_percent": 0},
    "repeat_sentence":            {"speaking_percent": 16, "listening_percent": 17, "reading_percent": 0, "writing_percent": 0},
    "describe_image":             {"speaking_percent": 31, "listening_percent": 0,  "reading_percent": 0, "writing_percent": 0},
    "retell_lecture":             {"speaking_percent": 13, "listening_percent": 13, "reading_percent": 0, "writing_percent": 0},
    "respond_to_situation":       {"speaking_percent": 13, "listening_percent": 0,  "reading_percent": 0, "writing_percent": 0},
    "ptea_respond_situation":     {"speaking_percent": 13, "listening_percent": 0,  "reading_percent": 0, "writing_percent": 0},
    "summarize_group_discussion": {"speaking_percent": 19, "listening_percent": 20, "reading_percent": 0, "writing_percent": 0},
    "answer_short_question":      {"speaking_percent": 0,  "listening_percent": 4,  "reading_percent": 0, "writing_percent": 0},
}


class _SectionWeightCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, int]] = {}
        self._loaded_at: float = 0.0

    def _refresh(self) -> None:
        s = SessionLocal()
        try:
            rows = s.execute(
                _sql(
                    "SELECT task, speaking_percent, listening_percent, "
                    "reading_percent, writing_percent "
                    "FROM pte_question_weightage"
                )
            ).fetchall()
            new_data: Dict[str, Dict[str, int]] = {}
            for r in rows:
                new_data[r.task] = {
                    "speaking_percent": int(r.speaking_percent or 0),
                    "listening_percent": int(r.listening_percent or 0),
                    "reading_percent": int(r.reading_percent or 0),
                    "writing_percent": int(r.writing_percent or 0),
                }
            with self._lock:
                self._data = new_data
                self._loaded_at = time.time()
            log.info(
                "[WEIGHTAGE] refreshed cache tasks=%d", len(new_data)
            )
        except Exception as e:
            log.error(
                f"[WEIGHTAGE] cache refresh failed (using defaults): {e}"
            )
        finally:
            s.close()

    def get_weightage(self, task: str) -> Dict[str, int]:
        """Return the section-weight dict for [task]. Falls back to hardcoded
        defaults when RDS is empty / unreachable."""
        now = time.time()
        if now - self._loaded_at > _REFRESH_SECONDS:
            self._refresh()
        with self._lock:
            row = self._data.get(task)
        if row:
            return row
        return _SECTION_WEIGHT_DEFAULTS.get(task, {
            "speaking_percent": 0, "listening_percent": 0,
            "reading_percent": 0, "writing_percent": 0,
        })


_weightage_cache = _SectionWeightCache()


def get_task_weightage(task: str) -> Dict[str, int]:
    """Read section-weight percentages for [task] from RDS via cache.

    Returns `{speaking_percent, listening_percent, reading_percent, writing_percent}`.
    Falls back to hardcoded defaults on DB outage. Unknown tasks return zeros.
    """
    return _weightage_cache.get_weightage(task)
