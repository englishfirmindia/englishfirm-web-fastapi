"""
Retrofill historical Write Essay rows in attempt_answers.

Background — two bugs in the legacy persist code at
routers/writing/write_essay.py (fixed today):
  1. attempt_answers.score was set to result.breakdown.earned (raw rubric
     points, 0-26 scale) instead of result.pte_score (10-90 PTE scale).
     Effect: a near-perfect essay (25/26) showed as PTE 25 instead of 87.
  2. result_json.maxScore was set from a stale fallback (15 or 13 from
     the pre-DSC rubric) instead of the new max_pts (26). Effect: the
     review screen computed earned / maxScore = 25 / 13 = 192% instead
     of earned / max_pts = 25 / 26 = 96%.

This script walks every WE row and:
  - Computes the canonical PTE score from result_json.earned / max_pts
    using the standard PTE formula:
        pte = max(10, min(90, round(10 + (earned/max_pts) * 80)))
  - Updates attempt_answers.score with the canonical PTE
  - Sets result_json.maxScore = max_pts so the review-screen pct math
    no longer over-counts
  - Leaves everything else untouched
  - Idempotent: rows already in correct shape are no-ops

Run with prod .env in place:
    PYTHONPATH=. python scripts/migrations/2026-05-23_retrofill_we_score.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from sqlalchemy import text as _sql  # noqa: E402
from sqlalchemy.orm.attributes import flag_modified  # noqa: E402

from db.database import SessionLocal  # noqa: E402
from db.models import AttemptAnswer  # noqa: E402


def _pte(earned: float, max_pts: float) -> int:
    """Canonical PTE 10-90 conversion. Mirrors services/scoring/base.to_pte_score."""
    if max_pts <= 0:
        return 10
    pct = max(0.0, min(1.0, earned / max_pts))
    return max(10, min(90, round(10 + pct * 80)))


def retrofill(dry: bool) -> None:
    db = SessionLocal()
    try:
        # Pull every WE row that has both earned + max_pts (i.e. is scoreable).
        rows = (
            db.query(AttemptAnswer)
            .filter(AttemptAnswer.question_type == "write_essay")
            .filter(AttemptAnswer.scoring_status == "complete")
            .order_by(AttemptAnswer.id)
            .all()
        )
        total = len(rows)
        updated_score = 0
        updated_maxscore = 0
        skipped_no_data = 0
        skipped_clean = 0
        examples = []
        for a in rows:
            rj = dict(a.result_json or {})
            # Try flat shape first, fall back to nested-breakdown (legacy)
            earned = rj.get("earned")
            max_pts = rj.get("max_pts")
            if earned is None or max_pts is None:
                bd = rj.get("breakdown") or {}
                if isinstance(bd, dict):
                    earned = earned if earned is not None else bd.get("earned")
                    max_pts = max_pts if max_pts is not None else bd.get("max_pts")
            try:
                earned_f = float(earned) if earned is not None else None
                max_pts_f = float(max_pts) if max_pts is not None else None
            except (TypeError, ValueError):
                earned_f = max_pts_f = None
            if earned_f is None or max_pts_f is None or max_pts_f <= 0:
                skipped_no_data += 1
                continue

            canonical_pte = _pte(earned_f, max_pts_f)
            current_score = a.score
            current_maxscore = rj.get("maxScore")

            changed = False
            if current_score != canonical_pte:
                if not dry:
                    a.score = canonical_pte
                changed = True
                updated_score += 1
            if current_maxscore != max_pts_f and current_maxscore != int(max_pts_f):
                rj["maxScore"] = max_pts_f
                if not dry:
                    a.result_json = rj
                    flag_modified(a, "result_json")
                changed = True
                updated_maxscore += 1
            if not changed:
                skipped_clean += 1
                continue
            if len(examples) < 8:
                examples.append({
                    "id": a.id,
                    "attempt_id": a.attempt_id,
                    "question_id": a.question_id,
                    "earned": earned_f,
                    "max_pts": max_pts_f,
                    "old_score": current_score,
                    "new_score": canonical_pte,
                    "old_maxScore": current_maxscore,
                    "new_maxScore": max_pts_f,
                })

        print(f"Total WE rows scanned: {total}")
        print(f"  rows with no earned/max_pts data:    {skipped_no_data}")
        print(f"  rows already in correct shape:       {skipped_clean}")
        print(f"  rows where aa.score updated:         {updated_score}")
        print(f"  rows where maxScore field updated:   {updated_maxscore}")
        if examples:
            print()
            print("Example transitions (first 8 updated rows):")
            for e in examples:
                print(f"  id={e['id']:5d} attempt={e['attempt_id']:5d} qid={e['question_id']:6d} "
                      f"earned={e['earned']:.1f}/{e['max_pts']:.0f} "
                      f"PTE {e['old_score']} → {e['new_score']}  "
                      f"maxScore {e['old_maxScore']} → {e['new_maxScore']}")
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
