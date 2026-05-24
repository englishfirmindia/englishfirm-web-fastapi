"""
Retrofill practice_attempts.task_breakdown for completed mock attempts.

Why:
  - The WE persist fix earlier today corrected attempt_answers.score and
    result_json.maxScore, but task_breakdown is a cached aggregate
    written ONCE at /finish time and never recomputed.
  - Stale rows still hold per-task entries like
    {earned_raw: 25, max_raw: 13, task_pct: 192.3} — which the mock
    feedback screen reads verbatim and displays as "Write Essay 192.3%".
  - Same issue affects SWT (was 8.5/9 stored as legacy 10 maxScore,
    similar drift) and any other AI-scored sync task with vintage drift.

This script walks every completed mock attempt and re-aggregates by
calling the same `_compute_section_score` / `_compute_overall_score`
helpers the finish endpoint uses. Idempotent: rows already in correct
shape are no-ops.

Run with prod .env in place:
    PYTHONPATH=. python scripts/migrations/2026-05-23_retrofill_mock_task_breakdown.py [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from sqlalchemy.orm.attributes import flag_modified  # noqa: E402

from db.database import SessionLocal  # noqa: E402
from db.models import PracticeAttempt, AttemptAnswer  # noqa: E402
from services.mock_service import (  # noqa: E402
    _compute_section_score,
    _compute_overall_score,
    _MAX_FALLBACK,
)


def _load_weights(db) -> dict:
    """Section weights, same shape as finish_mock_test reads."""
    from sqlalchemy import text as _sql
    rows = db.execute(_sql(
        "SELECT task, speaking_percent, writing_percent, reading_percent, "
        "listening_percent, overall_percent FROM pte_question_weightage"
    )).all()
    return {
        r.task: {
            "speaking":  float(r.speaking_percent or 0),
            "writing":   float(r.writing_percent or 0),
            "reading":   float(r.reading_percent or 0),
            "listening": float(r.listening_percent or 0),
            "overall":   float(r.overall_percent or 0),
        }
        for r in rows
    }


def _looks_stale(tb: dict) -> bool:
    """A task_breakdown is stale if any task_pct > 100 — that's a clear
    signature of the legacy maxScore-vs-max_pts mismatch."""
    if not isinstance(tb, dict):
        return False
    for section_name in ("speaking", "writing", "reading", "listening"):
        section = tb.get(section_name) or {}
        bd = section.get("breakdown") or {}
        if isinstance(bd, dict):
            for task in bd.values():
                if isinstance(task, dict):
                    pct = task.get("task_pct")
                    if isinstance(pct, (int, float)) and pct > 100.5:
                        return True
    return False


def retrofill(dry: bool) -> None:
    db = SessionLocal()
    try:
        weights = _load_weights(db)
        attempts = (
            db.query(PracticeAttempt)
            .filter(PracticeAttempt.module == "mock")
            .filter(PracticeAttempt.status == "complete")
            .order_by(PracticeAttempt.id)
            .all()
        )
        total = len(attempts)
        stale_count = 0
        updated = 0
        unchanged = 0
        examples = []
        for pa in attempts:
            answers = db.query(AttemptAnswer).filter_by(attempt_id=pa.id).all()
            if not answers:
                unchanged += 1
                continue
            tb = pa.task_breakdown or {}
            was_stale = _looks_stale(tb)
            if was_stale:
                stale_count += 1
            speaking  = _compute_section_score("speaking",  answers, weights, _MAX_FALLBACK)
            writing   = _compute_section_score("writing",   answers, weights, _MAX_FALLBACK)
            reading   = _compute_section_score("reading",   answers, weights, _MAX_FALLBACK)
            listening = _compute_section_score("listening", answers, weights, _MAX_FALLBACK)
            overall   = _compute_overall_score(answers, weights, _MAX_FALLBACK)
            new_tb = dict(tb)
            new_tb["speaking"]  = speaking
            new_tb["writing"]   = writing
            new_tb["reading"]   = reading
            new_tb["listening"] = listening
            new_tb["overall_score"] = overall
            if new_tb == tb and pa.total_score == overall:
                unchanged += 1
                continue
            updated += 1
            if len(examples) < 8:
                we_pct = (
                    (tb.get("writing", {}).get("breakdown", {}) or {})
                    .get("write_essay", {}).get("task_pct")
                )
                we_pct_new = (
                    (new_tb.get("writing", {}).get("breakdown", {}) or {})
                    .get("write_essay", {}).get("task_pct")
                )
                examples.append({
                    "attempt_id": pa.id,
                    "user_id": pa.user_id,
                    "old_total": pa.total_score,
                    "new_total": overall,
                    "old_we_pct": we_pct,
                    "new_we_pct": we_pct_new,
                    "was_stale": was_stale,
                })
            if not dry:
                pa.task_breakdown = new_tb
                flag_modified(pa, "task_breakdown")
                pa.total_score = overall
        print(f"Total completed mocks scanned: {total}")
        print(f"  rows with stale task_pct > 100 (clear bug signature): {stale_count}")
        print(f"  rows where task_breakdown changed:                    {updated}")
        print(f"  rows already in correct shape:                        {unchanged}")
        if examples:
            print()
            print("Example transitions (first 8):")
            for e in examples:
                tag = " (stale!)" if e["was_stale"] else ""
                print(f"  attempt={e['attempt_id']:5d} user={e['user_id']}  "
                      f"total {e['old_total']} → {e['new_total']}  "
                      f"WE pct {e['old_we_pct']} → {e['new_we_pct']}{tag}")
        if dry:
            db.rollback()
            print("\nDry-run — rolled back.")
        else:
            db.commit()
            print("\nCommitted.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    retrofill(args.dry_run)
