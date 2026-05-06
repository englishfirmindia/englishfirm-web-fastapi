"""
TTL-cached loader for `pte_speaking_scoring_config` rows.

The speaking scorer used to keep WPM band / pause-penalty / cross-penalty
constants as Python module globals. Now they live per-task in RDS so a
calibration tweak is `UPDATE pte_speaking_scoring_config ...` instead of
a code deploy.

Fail-open: any DB error or missing row returns the compiled defaults so
scoring never breaks during a database hiccup. The scorer holds the
defaults at the call site so a stale cache + DB outage still produces a
sane score.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from sqlalchemy import text

from core.logging_config import get_logger
from db.database import SessionLocal

log = get_logger(__name__)

_TTL_SECONDS = 300  # 5 min — short enough for live tuning, long enough to dodge per-attempt DB hits
_cache: dict[str, tuple[float, "SpeakingScoringConfig"]] = {}
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class SpeakingScoringConfig:
    task_type: str
    # WPM band
    wpm_floor: float
    wpm_ceiling: float
    wpm_plateau_low: float
    wpm_plateau_high: float
    wpm_slope_per_wpm: float
    wpm_peak_score: float
    # Pause detection
    pause_min_ms: int
    pause_leading_tol_ms: int
    pause_trailing_tol_ms: int
    silence_thresh_dbfs: float
    # Content
    content_insertion_penalty_k: float
    # Pause penalty
    pause_penalty_max_pauses: int
    pause_penalty_sentence_clamp_min: int
    pause_penalty_sentence_clamp_max: int
    pause_penalty_formula_constant: int
    # Cross penalty
    cross_penalty_healthy_threshold: float
    cross_penalty_floor_multiplier: float
    cross_penalty_slope: float


_COLS = (
    "task_type, wpm_floor, wpm_ceiling, wpm_plateau_low, wpm_plateau_high, "
    "wpm_slope_per_wpm, wpm_peak_score, pause_min_ms, pause_leading_tol_ms, "
    "pause_trailing_tol_ms, silence_thresh_dbfs, content_insertion_penalty_k, "
    "pause_penalty_max_pauses, pause_penalty_sentence_clamp_min, "
    "pause_penalty_sentence_clamp_max, pause_penalty_formula_constant, "
    "cross_penalty_healthy_threshold, cross_penalty_floor_multiplier, "
    "cross_penalty_slope"
)


def _row_to_config(row) -> SpeakingScoringConfig:
    return SpeakingScoringConfig(
        task_type=row[0],
        wpm_floor=float(row[1]), wpm_ceiling=float(row[2]),
        wpm_plateau_low=float(row[3]), wpm_plateau_high=float(row[4]),
        wpm_slope_per_wpm=float(row[5]), wpm_peak_score=float(row[6]),
        pause_min_ms=int(row[7]),
        pause_leading_tol_ms=int(row[8]),
        pause_trailing_tol_ms=int(row[9]),
        silence_thresh_dbfs=float(row[10]),
        content_insertion_penalty_k=float(row[11]),
        pause_penalty_max_pauses=int(row[12]),
        pause_penalty_sentence_clamp_min=int(row[13]),
        pause_penalty_sentence_clamp_max=int(row[14]),
        pause_penalty_formula_constant=int(row[15]),
        cross_penalty_healthy_threshold=float(row[16]),
        cross_penalty_floor_multiplier=float(row[17]),
        cross_penalty_slope=float(row[18]),
    )


def get_speaking_config(task_type: str) -> SpeakingScoringConfig | None:
    """
    Return the config row for `task_type`, or `None` if absent / DB error.
    Caller is expected to fall back to a compiled default in either case.
    """
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(task_type)
        if cached and now - cached[0] < _TTL_SECONDS:
            return cached[1]

    try:
        with SessionLocal() as db:
            row = db.execute(
                text(f"SELECT {_COLS} FROM pte_speaking_scoring_config WHERE task_type = :t"),
                {"t": task_type},
            ).fetchone()
    except Exception as e:
        log.warning("[SCORING_CONFIG] DB lookup failed for %s — using fallback: %s", task_type, e)
        return None

    if row is None:
        log.warning("[SCORING_CONFIG] No row for task_type=%s — using fallback", task_type)
        return None

    cfg = _row_to_config(row)
    with _cache_lock:
        _cache[task_type] = (now, cfg)
    return cfg


def clear_cache() -> None:
    """Test/operational hook — drop the in-memory cache."""
    with _cache_lock:
        _cache.clear()
