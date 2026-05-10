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
    # Strategy switches — picked by _score_speaking_v2 per task
    # content_method   : 'lcs_k2' | 'llm_keypoints' | 'regex_match' | 'binary'
    # pronunciation_source : 'azure_assessment' | 'azure_freeform'
    content_method: str = "lcs_k2"
    uses_reference_text: bool = True
    uses_cross_penalty: bool = True
    pronunciation_source: str = "azure_assessment"
    # Conditional pronunciation cross-penalty override:
    #   when fluency >= pronunciation_fluency_gate, mP uses the wider
    #   content-driven curve (threshold/floor/slope below); when fluency
    #   is below the gate, mP falls back to today's symmetric rule.
    # All NULL → override disabled, today's behaviour preserved.
    pronunciation_fluency_gate: float | None = None
    pronunciation_content_threshold: float | None = None
    pronunciation_content_floor: float | None = None
    pronunciation_content_slope: float | None = None
    # Optional softening curve applied to the LLM content score:
    #   final = round(100 · (raw / 100) ** content_curve_exponent)
    # 0.5 (sqrt) lifts the strict-rubric mid-band by ~+20 average; 1.0 or
    # NULL = no transform (today's behaviour). 0 stays 0, 100 stays 100;
    # the curve only changes mid-range scores.
    content_curve_exponent: float | None = None
    # Optional dynamic pause-penalty cliff: when set, max_pauses is
    # computed at scoring time as round(sentence_count × multiplier)
    # instead of using the static pause_penalty_max_pauses column. The
    # formula_constant is then derived as max_pauses + 1 to preserve the
    # "soft start at threshold" semantics. NULL → static behaviour.
    pause_penalty_max_pauses_mult: float | None = None
    # Length-floor cap for LLM-scored content. When the user's transcript
    # is shorter than `length_floor_words` words, content is capped at
    # `length_floor_cap` no matter how the LLM rated each key point. NULL
    # on either column → no cap (today's behaviour for non-LLM tasks).
    length_floor_words: int | None = None
    length_floor_cap: int | None = None


_COLS = (
    "task_type, wpm_floor, wpm_ceiling, wpm_plateau_low, wpm_plateau_high, "
    "wpm_slope_per_wpm, wpm_peak_score, pause_min_ms, pause_leading_tol_ms, "
    "pause_trailing_tol_ms, silence_thresh_dbfs, content_insertion_penalty_k, "
    "pause_penalty_max_pauses, pause_penalty_sentence_clamp_min, "
    "pause_penalty_sentence_clamp_max, pause_penalty_formula_constant, "
    "cross_penalty_healthy_threshold, cross_penalty_floor_multiplier, "
    "cross_penalty_slope, content_method, uses_reference_text, "
    "uses_cross_penalty, pronunciation_source, "
    "pronunciation_fluency_gate, pronunciation_content_threshold, "
    "pronunciation_content_floor, pronunciation_content_slope, "
    "content_curve_exponent, pause_penalty_max_pauses_mult, "
    "length_floor_words, length_floor_cap"
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
        content_method=str(row[19]),
        uses_reference_text=bool(row[20]),
        uses_cross_penalty=bool(row[21]),
        pronunciation_source=str(row[22]),
        pronunciation_fluency_gate=(float(row[23]) if row[23] is not None else None),
        pronunciation_content_threshold=(float(row[24]) if row[24] is not None else None),
        pronunciation_content_floor=(float(row[25]) if row[25] is not None else None),
        pronunciation_content_slope=(float(row[26]) if row[26] is not None else None),
        content_curve_exponent=(float(row[27]) if row[27] is not None else None),
        pause_penalty_max_pauses_mult=(float(row[28]) if row[28] is not None else None),
        length_floor_words=(int(row[29]) if row[29] is not None else None),
        length_floor_cap=(int(row[30]) if row[30] is not None else None),
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
