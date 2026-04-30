"""
Mock Test Service — Full PTE Academic exam simulation.

Structure:
  Intro → Tips → Overview → Mic Check → Headset Check → Instructions → Personal Intro
  Part 1 (Speaking + Writing) — per-Q timers, 76 min budget
  Break (optional 10 min)
  Part 2 (Reading) — block timer 25 min
  Part 3 (Listening) — SST per-Q 10 min + block timer 31 min

One PracticeAttempt row (module="mock") holds all 65 answers. Each submit
routes through the existing per-type submit endpoints which write an
AttemptAnswer row using the attempt_id resolved from ACTIVE_SESSIONS.

Scoring (at finish):
  Speaking  — weighted by pte_question_weightage.speaking_percent
  Writing   — weighted by pte_question_weightage.writing_percent
  Reading   — weighted by pte_question_weightage.reading_percent
  Listening — weighted by pte_question_weightage.listening_percent
  Overall   — weighted by pte_question_weightage.overall_percent across all 65 Qs
  All use:  max(10, min(90, round(10 + weighted_pct * 80)))
"""

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.attributes import set_committed_value

from db.models import QuestionFromApeuni, UserQuestionAttempt, PracticeAttempt, AttemptAnswer
from services.session_service import ACTIVE_SESSIONS, mark_submitted, persist_answer_to_db, enrich_content_json
from services.s3_service import generate_presigned_url
from services.scoring import get_scorer
from services.scoring.base import to_pte_score

# ─── Task metadata ────────────────────────────────────────────────────────────
# Each task tagged with section (for scoring) and part (for runner ordering).
# `module` = DB column in question_from_apeuni used to fetch the question row.
# `scoring` = async (speaking) vs sync (everything else).
# `prep_seconds` / `rec_seconds` = per-Q timer for speaking/SST.
# `time_seconds` = per-Q time budget for Writing + SST (no block timer).
MOCK_STRUCTURE = [
    # ── Part 1: Speaking ──────────────────────────────────────────────────────
    {"task_type": "read_aloud",                 "module": "speaking",  "section": "speaking", "part": 1, "count": 6,  "scoring": "async", "prep_seconds": 40, "rec_seconds": 35},
    {"task_type": "repeat_sentence",            "module": "speaking",  "section": "speaking", "part": 1, "count": 10, "scoring": "async", "prep_seconds": 3,  "rec_seconds": 9},
    {"task_type": "describe_image",             "module": "speaking",  "section": "speaking", "part": 1, "count": 5,  "scoring": "async", "prep_seconds": 25, "rec_seconds": 40},
    {"task_type": "retell_lecture",             "module": "speaking",  "section": "speaking", "part": 1, "count": 2,  "scoring": "async", "prep_seconds": 10, "rec_seconds": 40},
    {"task_type": "answer_short_question",      "module": "speaking",  "section": "speaking", "part": 1, "count": 5,  "scoring": "async", "prep_seconds": 3,  "rec_seconds": 5},
    {"task_type": "respond_to_situation",       "module": "speaking",  "section": "speaking", "part": 1, "count": 2,  "scoring": "async", "prep_seconds": 10, "rec_seconds": 40},
    {"task_type": "summarize_group_discussion", "module": "speaking",  "section": "speaking", "part": 1, "count": 2,  "scoring": "async", "prep_seconds": 3,  "rec_seconds": 120},
    # ── Part 1: Writing ───────────────────────────────────────────────────────
    {"task_type": "summarize_written_text",     "module": "writing",   "section": "writing",  "part": 1, "count": 2,  "scoring": "sync",  "time_seconds": 600},
    {"task_type": "write_essay",                "module": "writing",   "section": "writing",  "part": 1, "count": 1,  "scoring": "sync",  "time_seconds": 1200},
    # ── Part 2: Reading (block timer 25 min) ──────────────────────────────────
    {"task_type": "reading_fib_drop_down",      "module": "reading",   "section": "reading",  "part": 2, "count": 6,  "scoring": "sync"},
    {"task_type": "mcq_multiple",               "module": "reading",   "section": "reading",  "part": 2, "count": 3,  "scoring": "sync"},
    {"task_type": "reorder_paragraphs",         "module": "reading",   "section": "reading",  "part": 2, "count": 1,  "scoring": "sync"},
    {"task_type": "reading_drag_and_drop",      "module": "reading",   "section": "reading",  "part": 2, "count": 4,  "scoring": "sync"},
    {"task_type": "mcq_single",                 "module": "reading",   "section": "reading",  "part": 2, "count": 2,  "scoring": "sync"},
    # ── Part 3: Listening ─────────────────────────────────────────────────────
    # SST is per-Q (10 min each, shown separately before block timer starts)
    {"task_type": "summarize_spoken_text",      "module": "listening", "section": "listening","part": 3, "count": 1,  "scoring": "sync",  "time_seconds": 600},
    # Rest share 31-min block timer
    {"task_type": "listening_mcq_multiple",     "module": "listening", "section": "listening","part": 3, "count": 2,  "scoring": "sync"},
    {"task_type": "listening_fib",              "module": "listening", "section": "listening","part": 3, "count": 2,  "scoring": "sync"},
    {"task_type": "listening_hcs",              "module": "listening", "section": "listening","part": 3, "count": 2,  "scoring": "sync"},
    {"task_type": "listening_mcq_single",       "module": "listening", "section": "listening","part": 3, "count": 1,  "scoring": "sync"},
    {"task_type": "listening_smw",              "module": "listening", "section": "listening","part": 3, "count": 1,  "scoring": "sync"},
    {"task_type": "highlight_incorrect_words",  "module": "listening", "section": "listening","part": 3, "count": 2,  "scoring": "sync"},
    {"task_type": "listening_wfd",              "module": "listening", "section": "listening","part": 3, "count": 3,  "scoring": "sync"},
]

# Block timer duration for parts (seconds). Per-Q timed tasks have their own timers.
PART_BLOCK_TIMERS = {
    1: None,           # per-Q timers throughout
    2: 25 * 60,        # Reading block: 25 min
    3: 31 * 60,        # Listening block (SST has its own 10-min timer on top)
}

# Optional break between Part 2 and Part 3
BREAK_SECONDS = 10 * 60

# Submit paths — reuse existing per-type submit endpoints.
_SUBMIT_PATHS = {
    "read_aloud":                 "/questions/speaking/read-aloud/submit",
    "repeat_sentence":            "/questions/speaking/repeat-sentence/submit",
    "describe_image":             "/questions/speaking/describe-image/submit",
    "retell_lecture":             "/questions/speaking/retell-lecture/submit",
    "answer_short_question":      "/questions/speaking/answer-short-question/submit",
    "respond_to_situation":       "/questions/speaking/respond-to-situation/submit",
    "summarize_group_discussion": "/questions/speaking/summarize-group-discussion/submit",
    "summarize_written_text":     "/questions/writing/summarize-written-text/submit",
    "write_essay":                "/questions/writing/write-essay/submit",
    "reading_fib_drop_down":      "/questions/reading/fill-in-blanks/submit",
    "mcq_multiple":               "/questions/reading/mcm/submit",
    "reorder_paragraphs":         "/questions/reading/reorder-paragraphs/submit",
    "reading_drag_and_drop":      "/questions/reading/fib-drag-drop/submit",
    "mcq_single":                 "/questions/reading/mcs/submit",
    "summarize_spoken_text":      "/questions/listening/sst/submit",
    "listening_mcq_multiple":     "/questions/listening/mcm/submit",
    "listening_fib":              "/questions/listening/fib/submit",
    "listening_hcs":              "/questions/listening/hcs/submit",
    "listening_mcq_single":       "/questions/listening/mcs/submit",
    "listening_smw":              "/questions/listening/smw/submit",
    "highlight_incorrect_words":  "/questions/listening/hiw/submit",
    "listening_wfd":              "/questions/listening/wfd/submit",
}

_ASYNC_TYPES = {"read_aloud", "repeat_sentence", "describe_image", "retell_lecture",
                "answer_short_question", "respond_to_situation", "summarize_group_discussion"}

# ─── Weightage cache (loaded from pte_question_weightage RDS table) ───────────
_WEIGHTS_CACHE: dict = {}

def _load_weights(db: Session) -> dict:
    """Returns {task: {overall, listening, reading, speaking, writing}} with None → 0."""
    if _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE
    rows = db.execute(text(
        "SELECT task, overall_percent, listening_percent, reading_percent, speaking_percent, writing_percent "
        "FROM pte_question_weightage"
    )).fetchall()
    for r in rows:
        _WEIGHTS_CACHE[r[0]] = {
            "overall":   float(r[1] or 0),
            "listening": float(r[2] or 0),
            "reading":   float(r[3] or 0),
            "speaking":  float(r[4] or 0),
            "writing":   float(r[5] or 0),
        }
    return _WEIGHTS_CACHE


# Map code task_type → RDS task name (pte_question_weightage.task column)
_RDS_TASK = {
    "highlight_incorrect_words":  "listening_hiw",
    "reading_fib_drop_down":      "reading_fib_dropdown",
    "reading_drag_and_drop":      "fib_drag_drop",
    "reorder_paragraphs":         "reorder_paragraph",
    "listening_sst":              "summarize_spoken_text",
}

# Alias map — questions_from_apeuni sometimes stores legacy APEUni-prefixed
# names. We filter on the full set when fetching a pool, then normalize
# q.question_type back to the canonical name on the way out.
_DB_TYPE_ALIASES = {
    "respond_to_situation": ["respond_to_situation", "ptea_respond_situation"],
}
_NORMALIZE_TYPE = {"ptea_respond_situation": "respond_to_situation"}

def _rds_key(task_type: str) -> str:
    return _RDS_TASK.get(task_type, task_type)


# ─── Load counts from pte_mock_question_count ─────────────────────────────────

def _load_mock_counts(db: Session) -> dict:
    rows = db.execute(text(
        "SELECT task, section, count FROM pte_mock_question_count"
    )).fetchall()
    return {(r[0], r[1]): int(r[2]) for r in rows}


def _apply_mock_counts(db: Session) -> list:
    counts = _load_mock_counts(db)
    out = []
    for entry in MOCK_STRUCTURE:
        item = dict(entry)
        key = (item["task_type"], item["section"])
        if key in counts:
            item["count"] = counts[key]
        out.append(item)
    return out


# ─── Info (for intro screen) ──────────────────────────────────────────────────

def get_mock_info(db: Session) -> dict:
    structure = _apply_mock_counts(db)
    part_minutes = {1: 76, 2: 25, 3: 31}
    parts = [
        {"part": 1, "name": "Speaking and Writing", "minutes": part_minutes[1]},
        {"part": 2, "name": "Reading",              "minutes": part_minutes[2]},
        {"part": 3, "name": "Listening",            "minutes": part_minutes[3]},
    ]
    total_minutes = sum(part_minutes.values())
    return {
        "total_questions": sum(t["count"] for t in structure),
        "total_minutes":   total_minutes,
        "parts":           parts,
        "break_seconds":   BREAK_SECONDS,
    }


# ─── Start ────────────────────────────────────────────────────────────────────

def start_mock_test(db: Session, user_id: int, test_number: int = 1) -> dict:
    """Pick 65 questions (one set per task/section), create single PracticeAttempt."""
    # Exclude questions already SUBMITTED (attempt_answers row); merely-shown
    # but unanswered questions stay in the pool.
    practiced_ids = set(
        row[0] for row in db.query(AttemptAnswer.question_id)
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(PracticeAttempt.user_id == user_id)
        .all()
    ) if test_number != 0 else set()

    structure = _apply_mock_counts(db)

    selected: list = []
    for t in structure:
        task_type = t["task_type"]
        module    = t["module"]
        count     = t["count"]
        if count == 0:
            continue

        opts = [joinedload(QuestionFromApeuni.evaluation)]
        # Accept canonical + legacy names from the DB (e.g. ptea_respond_situation)
        db_type_candidates = _DB_TYPE_ALIASES.get(task_type, [task_type])
        base_filter = [
            QuestionFromApeuni.module == module,
            QuestionFromApeuni.question_type.in_(db_type_candidates),
        ]

        if test_number != 0 and practiced_ids:
            fresh = db.query(QuestionFromApeuni).options(*opts).filter(
                *base_filter,
                ~QuestionFromApeuni.question_id.in_(practiced_ids),
            ).all()
            pool = fresh if len(fresh) >= count else (
                db.query(QuestionFromApeuni).options(*opts).filter(*base_filter).all()
            )
        else:
            pool = db.query(QuestionFromApeuni).options(*opts).filter(*base_filter).all()

        # Normalize legacy question_type → canonical so the rest of the pipeline
        # (submit routes, Flutter widget switch, scoring) stays consistent.
        # Use set_committed_value so the ORM doesn't mark the row as dirty and
        # try to rename the DB row on the next commit.
        for q in pool:
            if q.question_type in _NORMALIZE_TYPE:
                set_committed_value(q, "question_type", _NORMALIZE_TYPE[q.question_type])

        # HIW: filter to questions with passage + incorrectWords populated
        if task_type == "highlight_incorrect_words":
            pool = [
                q for q in pool
                if (q.content_json or {}).get("passage")
                and (q.evaluation.evaluation_json if q.evaluation else {}).get("correctAnswers", {}).get("incorrectWords")
            ]

        n = min(count, len(pool))
        if n == 0:
            print(f"[Mock] WARNING: no questions available for {task_type}", flush=True)
            continue
        selected.extend(random.sample(pool, n))

    if not selected:
        raise HTTPException(status_code=404, detail="No questions available for mock test")

    # Note: pool filter uses attempt_answers (submitted only), so we no longer
    # pre-mark selected questions in user_question_attempts on session start.

    session_id = str(uuid.uuid4())
    attempt = PracticeAttempt(
        user_id               = user_id,
        session_id            = session_id,
        module                = "mock",
        question_type         = "mock",
        filter_type           = "mock",
        total_questions       = len(selected),
        total_score           = 0,
        questions_answered    = 0,
        status                = "pending",
        scoring_status        = "pending",
        selected_question_ids = [q.question_id for q in selected],
        task_breakdown        = {"current_part": 1, "part_timer_remaining": {}},
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    # db.commit() above expires all ORM attributes — next q.question_type access
    # re-queries the DB and would reset any earlier normalize. Re-apply the
    # legacy → canonical mapping now so ACTIVE_SESSIONS stores canonical names.
    for q in selected:
        if q.question_type in _NORMALIZE_TYPE:
            set_committed_value(q, "question_type", _NORMALIZE_TYPE[q.question_type])

    # Order questions per MOCK_STRUCTURE order (speaking → writing → reading → listening)
    order_map = {t["task_type"]: i for i, t in enumerate(structure)}
    selected.sort(key=lambda q: order_map.get(q.question_type, 999))

    ACTIVE_SESSIONS[session_id] = {
        "user_id":              user_id,
        "test_number":          test_number,
        "start_time":           int(time.time()),
        "questions":            {q.question_id: q for q in selected},
        "question_order":       [q.question_id for q in selected],
        "submitted_questions":  set(),
        "score":                0,
        "question_scores":      {},
        "question_score_maxes": {},
        "attempt_id":           attempt.id,
        "current_part":         1,
        "part_timer_remaining": {1: None, 2: PART_BLOCK_TIMERS[2], 3: PART_BLOCK_TIMERS[3]},
    }

    print(f"[Mock] Started session={session_id} attempt={attempt.id} total_q={len(selected)}", flush=True)

    return {
        "session_id":      session_id,
        "attempt_id":      attempt.id,
        "total_questions": len(selected),
    }


# ─── Get part (returns questions + timing for that part) ──────────────────────

def get_mock_part(db: Session, session_id: str, part: int) -> dict:
    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Mock session not found")
    if part not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Invalid part")

    structure = _apply_mock_counts(db)
    # Belt-and-braces: build task_meta keyed by both canonical + legacy names
    # so that if a q.question_type is still legacy (e.g. ptea_respond_situation
    # from a re-fetch after session expiry) the lookup still resolves.
    task_meta = {}
    for t in structure:
        if t["part"] != part: continue
        task_meta[t["task_type"]] = t
        for legacy, canonical in _NORMALIZE_TYPE.items():
            if canonical == t["task_type"]:
                task_meta[legacy] = t

    questions_payload = []
    qorder = session.get("question_order", [])
    for qid in qorder:
        q = session["questions"].get(qid)
        if not q:
            continue
        # Always serialize the canonical task_type to Flutter
        canonical_type = _NORMALIZE_TYPE.get(q.question_type, q.question_type)
        meta = task_meta.get(q.question_type) or task_meta.get(canonical_type)
        if not meta:
            continue  # not in this part

        presigned_url: Optional[str] = None
        if q.module in ("speaking", "listening") or canonical_type in ("summarize_spoken_text", "listening_wfd"):
            raw_audio = (q.content_json or {}).get("audio_url") or (q.content_json or {}).get("s3_key")
            if raw_audio:
                try:
                    presigned_url = generate_presigned_url(raw_audio)
                except Exception:
                    presigned_url = None

        # Pre-sign image URL for describe_image (ported from speaking_sectional_service.py:183-190)
        presigned_image_url: Optional[str] = None
        raw_image = (q.content_json or {}).get("image_url") or (q.content_json or {}).get("image_s3_key")
        if raw_image:
            try:
                presigned_image_url = generate_presigned_url(raw_image)
            except Exception:
                presigned_image_url = None

        questions_payload.append({
            "question_id":         q.question_id,
            "task_type":           canonical_type,
            "section":             meta["section"],
            "part":                meta["part"],
            "scoring":             meta.get("scoring", "sync"),
            "prep_seconds":        meta.get("prep_seconds", 0),
            "rec_seconds":         meta.get("rec_seconds",  0),
            "time_seconds":        meta.get("time_seconds", 0),
            "content_json":        enrich_content_json(q),
            "submit_path":         _SUBMIT_PATHS.get(canonical_type, ""),
            "session_id":          session_id,
            "is_prediction":       q.is_prediction,
            "presigned_url":       presigned_url,
            "presigned_image_url": presigned_image_url,
            "is_submitted":        q.question_id in session.get("submitted_questions", set()),
        })

    return {
        "session_id":            session_id,
        "part":                  part,
        "block_timer_seconds":   PART_BLOCK_TIMERS.get(part),
        "timer_remaining":       session["part_timer_remaining"].get(part),
        "questions":             questions_payload,
    }


# ─── Save progress (called by Flutter on timer tick / Next) ───────────────────

def update_mock_progress(db: Session, session_id: str, current_part: int, timer_remaining: Optional[int]) -> dict:
    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Mock session not found")

    session["current_part"] = current_part
    if timer_remaining is not None:
        session["part_timer_remaining"][current_part] = timer_remaining

    # Persist snapshot to PracticeAttempt.task_breakdown for crash recovery / resume
    attempt = db.query(PracticeAttempt).filter_by(session_id=session_id).first()
    if attempt:
        tb = attempt.task_breakdown or {}
        tb["current_part"] = current_part
        ptr = dict(tb.get("part_timer_remaining", {}))
        if timer_remaining is not None:
            ptr[str(current_part)] = timer_remaining
        tb["part_timer_remaining"] = ptr
        attempt.task_breakdown = tb
        db.commit()

    return {"ok": True, "current_part": current_part, "timer_remaining": timer_remaining}


# ─── Finish (compute all 5 scores) ────────────────────────────────────────────

def _compute_section_score(section: str, answers: list, weights: dict, max_pts_map: dict) -> dict:
    """Section score = sum(task_pct × section_%) / present_weight → CLAUDE.md formula."""
    buckets: dict = {}
    questions: list = []

    for a in answers:
        qt = a.question_type
        rds = _rds_key(qt)
        w = weights.get(rds, {}).get(section, 0)
        if w <= 0:
            continue
        if rds not in buckets:
            buckets[rds] = {"earned": 0.0, "max": 0.0, "weight": w,
                            "display": qt.replace("_", " ").title(), "count": 0, "answered": 0}
        raw_score, max_pts = _resolve_score_max(a, max_pts_map)
        buckets[rds]["earned"] += raw_score
        buckets[rds]["max"]    += float(max_pts or 0)
        buckets[rds]["count"]  += 1
        if a.score is not None:
            buckets[rds]["answered"] += 1

        q_max  = float(max_pts or 0)
        q_pct  = round(raw_score / q_max * 100, 1) if q_max > 0 else 0.0
        q_pte  = max(10, min(90, round(10 + (raw_score / q_max) * 80))) if q_max > 0 else 10
        questions.append({
            "question_id":   a.question_id,
            "question_type": qt,
            "task":          rds,
            "score":         round(raw_score, 2),
            "max_score":     round(q_max, 2),
            "pct":           q_pct,
            "pte_equivalent": q_pte,
            "result_detail": a.result_json or {},
        })

    # Full section weight = sum of ALL task weights for this section from RDS,
    # regardless of whether the user answered them. This prevents an inflated score
    # when a task type has zero answers (e.g. unanswered HIW, missing drag-drop pool).
    full_section_weight = sum(
        v.get(section, 0) for v in weights.values() if (v.get(section, 0) or 0) > 0
    )

    weighted_sum   = 0.0
    answered_weight = 0.0
    breakdown = {}
    for rds, b in buckets.items():
        task_pct = (b["earned"] / b["max"]) if b["max"] > 0 else 0.0
        contrib  = task_pct * b["weight"]
        weighted_sum    += contrib
        answered_weight += b["weight"]
        breakdown[rds] = {
            "display_name":    b["display"],
            "count":           b["count"],
            "answered":        b["answered"],
            "earned_raw":      round(b["earned"], 2),
            "max_raw":         round(b["max"], 2),
            "task_pct":        round(task_pct * 100, 1),
            "section_weight":  b["weight"],
            "contribution":    round(contrib, 2),
            "task_pte":        max(10, min(90, round(10 + task_pct * 80))),
        }

    normalised_pct = (weighted_sum / full_section_weight) if full_section_weight > 0 else 0.0
    score = max(10, min(90, round(10 + normalised_pct * 80)))
    return {
        "score":              score,
        "weighted_pct":       round(normalised_pct * 100, 1),
        "weighted_sum":       round(weighted_sum, 2),
        "full_section_weight": round(full_section_weight, 2),
        "answered_weight":    round(answered_weight, 2),
        "breakdown":          breakdown,
        "questions":          questions,
    }


def _compute_overall_score(answers: list, weights: dict, max_pts_map: dict) -> int:
    """Overall uses overall_percent column from pte_question_weightage."""
    full_overall_weight = sum(
        v.get("overall", 0) for v in weights.values() if (v.get("overall", 0) or 0) > 0
    )

    buckets: dict = {}
    for a in answers:
        qt = a.question_type
        rds = _rds_key(qt)
        w = weights.get(rds, {}).get("overall", 0)
        if w <= 0:
            continue
        if rds not in buckets:
            buckets[rds] = {"earned": 0.0, "max": 0.0, "weight": w}
        raw_score, max_pts = _resolve_score_max(a, max_pts_map)
        buckets[rds]["earned"] += raw_score
        buckets[rds]["max"]    += float(max_pts or 0)

    weighted_sum = 0.0
    for b in buckets.values():
        task_pct = (b["earned"] / b["max"]) if b["max"] > 0 else 0.0
        weighted_sum += task_pct * b["weight"]

    normalised_pct = (weighted_sum / full_overall_weight) if full_overall_weight > 0 else 0.0
    return max(10, min(90, round(10 + normalised_pct * 80)))


# Fallback max scores — used only when scorer breakdown is unavailable (e.g. async speaking).
_MAX_FALLBACK = {
    "read_aloud": 15, "repeat_sentence": 13, "describe_image": 16,
    "retell_lecture": 16, "respond_to_situation": 16,
    "summarize_group_discussion": 16, "answer_short_question": 1,
    "summarize_written_text": 10, "write_essay": 13,
    "reading_fib_drop_down": 4, "mcq_multiple": 2, "reorder_paragraphs": 3,
    "reading_drag_and_drop": 4, "mcq_single": 1,
    "summarize_spoken_text": 12, "listening_wfd": 7, "listening_fib": 4,
    "highlight_incorrect_words": 3, "listening_hcs": 1,
    "listening_mcq_multiple": 2, "listening_mcq_single": 1, "listening_smw": 1,
}


def _extract_score_and_max(breakdown: dict, persist_type: str) -> tuple:
    """Return (actual_score, actual_max) from scorer breakdown.

    Priority order matches mobile: use real counts from breakdown, fall back to
    _MAX_FALLBACK only when breakdown is absent (async speaking types).
    """
    bd = breakdown or {}
    # FIBScorer (reading/listening FIB, drag-drop), WFDScorer, ReorderScorer
    if 'hits' in bd:
        total = float(bd.get('total') or bd.get('total_pairs') or 0)
        if total > 0:
            return float(bd['hits']), total
    # MCQScorer multi — has explicit score + max_possible
    if 'max_possible' in bd:
        return float(bd.get('score', 0)), float(bd['max_possible'])
    # HIWScorer — has score + max_score
    if 'max_score' in bd:
        return float(bd.get('score', 0)), float(bd['max_score'])
    # MCQScorer single — is_correct bool, max is always 1
    if 'is_correct' in bd:
        return (1.0 if bd['is_correct'] else 0.0), 1.0
    # Async speaking or missing breakdown — keep old behaviour
    return 0.0, float(_MAX_FALLBACK.get(persist_type, 1))


_RULE_SCORED_KEYS = frozenset({'hits', 'max_possible', 'max_score', 'is_correct'})

def _resolve_score_max(a, max_pts_map: dict) -> tuple:
    """Return (raw_score, max_pts) for any AttemptAnswer type.

    Three cases:
    - Async speaking (PTE 10-90 stored): de-normalise back to rubric scale.
    - Rule-scored (FIB/MCQ/HIW/Reorder/WFD): read real counts from result_json breakdown.
    - AI-scored sync (SWT/WE/SST): use stored score + maxScore directly.
    """
    qt = a.question_type
    rj = a.result_json or {}
    if qt in _ASYNC_TYPES:
        max_pts = float(rj.get("maxScore") or max_pts_map.get(qt, 1))
        return ((float(a.score or 0) - 10.0) / 80.0) * max_pts, max_pts
    if _RULE_SCORED_KEYS & rj.keys():
        return _extract_score_and_max(rj, qt)
    # AI-scored sync: trust stored score and maxScore
    max_pts = float(rj.get("maxScore") or max_pts_map.get(qt, 1))
    return float(a.score or 0), max_pts


def finish_mock_test(db: Session, session_id: str, user_id: int) -> dict:
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="mock"
    ).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Mock attempt not found")

    # Idempotent: return cached result if already complete
    if attempt.status == "complete" and attempt.scoring_status == "complete":
        return _format_results(attempt)

    answers = db.query(AttemptAnswer).filter_by(attempt_id=attempt.id).all()
    weights = _load_weights(db)

    speaking  = _compute_section_score("speaking",  answers, weights, _MAX_FALLBACK)
    writing   = _compute_section_score("writing",   answers, weights, _MAX_FALLBACK)
    reading   = _compute_section_score("reading",   answers, weights, _MAX_FALLBACK)
    listening = _compute_section_score("listening", answers, weights, _MAX_FALLBACK)
    overall   = _compute_overall_score(answers, weights, _MAX_FALLBACK)

    task_breakdown = {
        "overall_score":      overall,
        "speaking":           speaking,
        "writing":            writing,
        "reading":            reading,
        "listening":          listening,
        "current_part":       3,
        "part_timer_remaining": {},
    }

    attempt.total_score         = overall
    attempt.questions_answered  = len(answers)
    attempt.status              = "complete"
    # If any async speaking is still pending, mark overall scoring_status accordingly
    pending_speaking = any(
        (a.question_type in _ASYNC_TYPES) and (a.scoring_status not in ("complete", None))
        for a in answers
    )
    attempt.scoring_status      = "pending" if pending_speaking else "complete"
    attempt.task_breakdown      = task_breakdown
    attempt.completed_at        = datetime.now(timezone.utc)
    db.commit()
    db.refresh(attempt)

    print(f"[Mock] Finished session={session_id} scores S={speaking['score']} W={writing['score']} R={reading['score']} L={listening['score']} O={overall}", flush=True)

    return _format_results(attempt)


def _format_results(attempt: PracticeAttempt) -> dict:
    tb = attempt.task_breakdown or {}
    return {
        "attempt_id":         attempt.id,
        "session_id":         attempt.session_id,
        "scoring_status":     attempt.scoring_status or "complete",
        "overall_score":      tb.get("overall_score", attempt.total_score),
        "speaking_score":     (tb.get("speaking") or {}).get("score"),
        "writing_score":      (tb.get("writing")  or {}).get("score"),
        "reading_score":      (tb.get("reading")  or {}).get("score"),
        "listening_score":    (tb.get("listening") or {}).get("score"),
        "section_breakdown":  {
            "speaking":  tb.get("speaking"),
            "writing":   tb.get("writing"),
            "reading":   tb.get("reading"),
            "listening": tb.get("listening"),
        },
        "total_questions":    attempt.total_questions,
        "questions_answered": attempt.questions_answered,
        "completed_at":       attempt.completed_at.isoformat() if attempt.completed_at else None,
    }


def get_mock_results(session_id: str, user_id: int, db: Session) -> dict:
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="mock"
    ).first()
    if not attempt:
        return {"scoring_status": "not_found"}
    # Re-run finish if still pending (async speaking may now be complete)
    if attempt.status != "complete" or attempt.scoring_status != "complete":
        try:
            return finish_mock_test(db, session_id, user_id)
        except Exception:
            pass
    return _format_results(attempt)


# ─── Review (reuses same schema as sectional reviews) ─────────────────────────

def get_mock_review(session_id: str, user_id: int, db: Session) -> dict:
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="mock"
    ).first()
    if not attempt:
        return {"found": False, "items": []}

    answers = (
        db.query(AttemptAnswer)
        .filter_by(attempt_id=attempt.id)
        .order_by(AttemptAnswer.submitted_at)
        .all()
    )
    if not answers:
        return {"found": True, "items": []}

    qids = [a.question_id for a in answers]
    qs = {
        q.question_id: q
        for q in db.query(QuestionFromApeuni)
        .filter(QuestionFromApeuni.question_id.in_(qids))
        .all()
    }

    # Map task_type → section for grouping
    task_section = {t["task_type"]: t["section"] for t in MOCK_STRUCTURE}

    items = []
    for i, a in enumerate(answers, 1):
        q  = qs.get(a.question_id)
        cj = (q.content_json or {}) if q else {}
        qt = a.question_type
        section = task_section.get(qt, "other")

        context: dict = {}
        # Passage / options / image / audio — same pattern as sectional review endpoints
        if qt == "read_aloud":
            context["passage"] = (cj.get("passage", "") or "")[:500]
        elif qt == "repeat_sentence":
            context["passage"] = (cj.get("passage", "") or cj.get("text", "") or "")[:300]
        elif qt == "describe_image":
            context["image_url"] = cj.get("image_url", "") or cj.get("img_url", "") or ""
        elif qt == "answer_short_question":
            context["correct_answer"] = cj.get("correct_answer", "") or ""
        elif qt == "summarize_written_text":
            context["passage"] = (cj.get("passage", "") or "")[:800]
        elif qt == "write_essay":
            context["prompt"] = (cj.get("prompt", "") or "")[:500]
        elif qt in ("reading_fib_drop_down", "reading_drag_and_drop"):
            context["contentBlocks"] = cj.get("contentBlocks", [])
            context["wordBank"]      = cj.get("wordBank", [])
        elif qt in ("mcq_single", "mcq_multiple", "highlight_correct_summary"):
            context["passage"] = (cj.get("passage", "") or "")[:800]
            context["options"] = cj.get("options", [])
        elif qt == "reorder_paragraphs":
            context["paragraphs"] = cj.get("paragraphs", [])
        elif qt in ("fill_in_the_blanks", "listening_fib", "highlight_incorrect_words"):
            context["passage"] = (cj.get("passage", "") or "")[:1000]
        elif qt in ("listening_mcq_single", "listening_mcq_multiple",
                    "listening_hcs", "listening_smw",
                    "multiple_choice_single", "multiple_choice_multiple",
                    "select_missing_word"):
            context["options"] = cj.get("options", [])

        # Audio stimulus for any task with audio
        if qt not in ("read_aloud", "describe_image"):
            raw_audio = cj.get("audio_url") or cj.get("s3_key")
            if raw_audio:
                try:
                    context["presigned_url"] = generate_presigned_url(raw_audio)
                except Exception:
                    pass

        # Student recording for speaking tasks
        if a.audio_url:
            try:
                context["recording_url"] = generate_presigned_url(a.audio_url)
            except Exception:
                pass

        items.append({
            "question_number":     i,
            "question_id":         a.question_id,
            "question_type":       qt,
            "section":             section,
            "display_name":        qt.replace("_", " ").title(),
            "score":               float(a.score or 0),
            "max_score":           (a.result_json or {}).get("maxScore") or _MAX_FALLBACK.get(qt, 1),
            "content_score":       float(a.content_score or 0),
            "fluency_score":       float(a.fluency_score or 0),
            "pronunciation_score": float(a.pronunciation_score or 0),
            "scoring_status":      a.scoring_status or "complete",
            "user_answer":         a.user_answer_json or {},
            "correct_answer":      a.correct_answer_json or {},
            "result_detail":       a.result_json or {},
            "context":             context,
        })

    return {"found": True, "items": items}


# ─── Resume (generous mode — returns saved timer snapshot) ────────────────────

def resume_mock_test(session_id: str, user_id: int, db: Session) -> dict:
    attempt = db.query(PracticeAttempt).filter_by(
        session_id=session_id, user_id=user_id, module="mock", status="pending",
    ).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="No resumable mock session found")

    qid_order = attempt.selected_question_ids or []
    if not qid_order:
        raise HTTPException(status_code=404, detail="Mock attempt has no stored questions")

    # Rebuild in-memory session if needed
    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        qs_by_id = {
            q.question_id: q
            for q in db.query(QuestionFromApeuni)
            .options(joinedload(QuestionFromApeuni.evaluation))
            .filter(QuestionFromApeuni.question_id.in_(qid_order))
            .all()
        }
        structure = _apply_mock_counts(db)
        order_map = {t["task_type"]: i for i, t in enumerate(structure)}
        selected = [qs_by_id[qid] for qid in qid_order if qid in qs_by_id]
        selected.sort(key=lambda q: order_map.get(q.question_type, 999))

        # Recover submitted set from existing AttemptAnswer rows
        submitted = {a.question_id for a in db.query(AttemptAnswer).filter_by(attempt_id=attempt.id).all()}

        tb = attempt.task_breakdown or {}
        ptr = {int(k): v for k, v in (tb.get("part_timer_remaining") or {}).items()}
        ptr.setdefault(1, None)
        ptr.setdefault(2, PART_BLOCK_TIMERS[2])
        ptr.setdefault(3, PART_BLOCK_TIMERS[3])

        ACTIVE_SESSIONS[session_id] = {
            "user_id":              user_id,
            "test_number":          1,
            "start_time":           int(time.time()),
            "questions":            {q.question_id: q for q in selected},
            "question_order":       [q.question_id for q in selected],
            "submitted_questions":  submitted,
            "score":                0,
            "question_scores":      {},
            "question_score_maxes": {},
            "attempt_id":           attempt.id,
            "current_part":         tb.get("current_part", 1),
            "part_timer_remaining": ptr,
        }

    session = ACTIVE_SESSIONS[session_id]
    return {
        "session_id":           session_id,
        "attempt_id":           attempt.id,
        "current_part":         session["current_part"],
        "part_timer_remaining": session["part_timer_remaining"],
        "total_questions":      len(qid_order),
        "submitted_count":      len(session["submitted_questions"]),
    }


# ─── Unified non-speaking submit ──────────────────────────────────────────────

def submit_mock_answer(session_id: str, question_id: int, payload: dict) -> dict:
    """Score and persist a single non-speaking mock answer."""
    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Mock session not found")

    qid = int(question_id)
    question = session["questions"].get(qid)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found in mock session")

    task_type = _NORMALIZE_TYPE.get(question.question_type, question.question_type)
    content_json = question.content_json or {}
    eval_json = (question.evaluation.evaluation_json if question.evaluation else {}) or {}

    scorer_key, answer_dict, persist_type, user_answer_json = _build_mock_score_args(
        task_type, payload, content_json, eval_json, question
    )

    scorer = get_scorer(scorer_key)
    result = scorer.score(question_id=qid, session_id=session_id, answer=answer_dict)

    actual_score, actual_max = _extract_score_and_max(result.breakdown, persist_type)

    mark_submitted(session_id, qid, result.pte_score)
    persist_answer_to_db(
        session=session,
        question_id=qid,
        question_type=persist_type,
        user_answer_json=user_answer_json,
        correct_answer_json={},
        result_json={**(result.breakdown or {}), "maxScore": actual_max},
        score=actual_score,
    )
    return {"pte_score": to_pte_score(result.raw_score), "ok": True}


def _build_mock_score_args(task_type, payload, content_json, eval_json, question):
    """Returns (scorer_key, answer_dict, persist_type, user_answer_json)."""
    if task_type in ("summarize_written_text", "write_essay"):
        text = payload.get("user_text", "")
        prompt = content_json.get("passage") or content_json.get("text") or content_json.get("prompt", "")
        return task_type, {"text": text, "prompt": prompt}, task_type, {"text": text}

    if task_type == "summarize_spoken_text":
        text = payload.get("user_text", "")
        prompt = content_json.get("transcript") or content_json.get("audio_url", "")
        return "listening_sst", {"text": text, "prompt": prompt}, "summarize_spoken_text", {"text": text}

    if task_type == "listening_wfd":
        text = payload.get("user_text", "")
        return "listening_wfd", {"user_text": text, "evaluation_json": eval_json}, task_type, {"user_text": text}

    if task_type == "reading_fib_drop_down":
        raw = payload.get("user_answers", {})
        ua = {str(i + 1): v for i, v in enumerate(raw)} if isinstance(raw, list) else raw
        return "reading_fib", {"user_answers": ua, "evaluation_json": eval_json}, task_type, {"user_answers": ua}

    if task_type == "reading_drag_and_drop":
        raw = payload.get("user_answers", {})
        ua = {str(i + 1): v for i, v in enumerate(raw)} if isinstance(raw, list) else raw
        return "reading_fib_drop_down", {"user_answers": ua, "evaluation_json": eval_json}, "reading_fib_drop_down", {"user_answers": ua}

    if task_type in ("mcq_single", "listening_mcq_single", "listening_hcs", "listening_smw"):
        ids = payload.get("selected_option_ids", [])
        sel = payload.get("selected_option") or (ids[0] if ids else "")
        scorer_map = {"mcq_single": "reading_mcs", "listening_mcq_single": "listening_mcs",
                      "listening_hcs": "listening_hcs", "listening_smw": "listening_smw"}
        return scorer_map[task_type], {"selected_option": sel, "evaluation_json": eval_json}, task_type, {"selected_option": sel}

    if task_type in ("mcq_multiple", "listening_mcq_multiple"):
        opts = payload.get("selected_options") or payload.get("selected_option_ids", [])
        scorer_map = {"mcq_multiple": "reading_mcm", "listening_mcq_multiple": "listening_mcm"}
        return scorer_map[task_type], {"selected_options": opts, "evaluation_json": eval_json}, task_type, {"selected_options": opts}

    if task_type == "reorder_paragraphs":
        seq = payload.get("user_sequence") or payload.get("paragraphs", [])
        return "reorder_paragraphs", {"user_sequence": seq, "evaluation_json": eval_json}, task_type, {"user_sequence": seq}

    if task_type == "listening_fib":
        raw = payload.get("user_answers", {})
        ua = {str(i + 1): v for i, v in enumerate(raw)} if isinstance(raw, list) else raw
        return "listening_fib", {"user_answers": ua, "evaluation_json": eval_json}, task_type, {"user_answers": ua}

    if task_type == "highlight_incorrect_words":
        hw = payload.get("highlighted_words")
        if hw is None:
            indices = payload.get("highlighted_indices", [])
            words = content_json.get("words") or content_json.get("transcript", "").split()
            hw = [words[i] for i in indices if isinstance(i, int) and i < len(words)]
        return "listening_hiw", {"highlighted_words": hw, "evaluation_json": eval_json}, task_type, {"highlighted_words": list(hw or [])}

    raise HTTPException(status_code=400, detail=f"Unsupported task_type for mock submit: {task_type}")
