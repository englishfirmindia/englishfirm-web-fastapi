"""
Expand each of the 40 listening sectional templates from 21 → 33 questions.

LISTENING_STRUCTURE counts were bumped to match pte_mock_question_count:
  repeat_sentence            3 → 10  (+7)
  retell_lecture             1 →  2  (+1)
  summarize_group_discussion 1 →  2  (+1)
  answer_short_question      2 →  5  (+3)
  (everything else unchanged — pure listening tasks already match)

This script appends the +12 new questions per template into the
appropriate task blocks so the structure order is preserved
(speaking tasks first, then pure listening). For each template:

  walk question_ids in order
  → identify the boundary of each task type's existing block
  → insert N new same-type qids right after that block ends
  → continue to the next task

Idempotent — re-running on an already-expanded template (where the
template already has the target count for that task) is a no-op for
that task slot.

Run with prod .env in place:
    PYTHONPATH=. python scripts/migrations/2026-05-21_backfill_listening_sectional_expansion.py [--dry-run]
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from sqlalchemy import text as _sql  # noqa: E402

from db.database import SessionLocal  # noqa: E402


# Target counts per task (matching the new LISTENING_STRUCTURE).
TARGET_COUNTS = [
    ("repeat_sentence", 10, "speaking"),
    ("retell_lecture", 2, "speaking"),
    ("summarize_group_discussion", 2, "speaking"),
    ("answer_short_question", 5, "speaking"),
    ("summarize_spoken_text", 1, "listening"),
    ("listening_mcq_multiple", 2, "listening"),
    ("listening_fib", 2, "listening"),
    ("listening_hcs", 2, "listening"),
    ("listening_smw", 1, "listening"),
    ("highlight_incorrect_words", 2, "listening"),
    ("listening_mcq_single", 1, "listening"),
    ("listening_wfd", 3, "listening"),
]


def _qtype_map(db) -> dict:
    """qid → question_type for everything in questions_from_apeuni."""
    return {
        r.question_id: r.question_type
        for r in db.execute(_sql(
            "SELECT question_id, question_type FROM questions_from_apeuni"
        )).all()
    }


def _pool_by_type(db) -> dict:
    """task_type → list of healthy qids."""
    pool = {}
    for r in db.execute(_sql(
        "SELECT question_id, question_type FROM questions_from_apeuni"
    )).all():
        pool.setdefault(r.question_type, []).append(r.question_id)
    return pool


def _expand_template(qids: list, qtype_map: dict, pool: dict, used_overall: set) -> list:
    """Group existing question_ids by task type (regardless of original
    position), then emit them in TARGET_COUNTS order, padding with fresh
    healthy qids when a task block is below target. Side benefit: also
    reorders any qids that prior cleanup left in non-canonical positions.
    """
    qids = list(qids)
    used_in_template = set(qids)
    by_type: dict = {}
    for q in qids:
        t = qtype_map.get(q)
        if t is None:
            print(f"  ⚠ qid {q} not in questions_from_apeuni — dropping")
            continue
        by_type.setdefault(t, []).append(q)

    out = []
    for task, target, _module in TARGET_COUNTS:
        block = list(by_type.get(task, []))[:target]
        deficit = target - len(block)
        if deficit > 0:
            candidates = [
                q for q in pool.get(task, [])
                if q not in used_in_template
            ]
            random.shuffle(candidates)
            picked = candidates[:deficit]
            if len(picked) < deficit:
                print(f"  ⚠ task={task}: needed {deficit} extra, pool only gave {len(picked)}")
            block.extend(picked)
            used_in_template.update(picked)
        # If existing block had MORE than target (shouldn't happen for listening,
        # but defend), the slice above already truncated.
        out.extend(block)
    return out


def main(dry: bool) -> None:
    db = SessionLocal()
    try:
        qtype_map = _qtype_map(db)
        pool = _pool_by_type(db)

        rows = db.execute(_sql(
            "SELECT test_number, question_ids FROM sectional_test_questions "
            "WHERE module = 'listening' ORDER BY test_number"
        )).all()

        used_overall = set()
        changed = 0
        for r in rows:
            new_qids = _expand_template(r.question_ids, qtype_map, pool, used_overall)
            if new_qids == list(r.question_ids):
                print(f"  ↷ listening/test {r.test_number}: already at target shape, no change")
                continue
            changed += 1
            old_count = len(r.question_ids)
            new_count = len(new_qids)
            added = [q for q in new_qids if q not in r.question_ids]
            print(f"  ✓ listening/test {r.test_number}: {old_count} → {new_count} qids (+{len(added)} new: {added[:6]}{'...' if len(added) > 6 else ''})")
            if not dry:
                db.execute(_sql(
                    "UPDATE sectional_test_questions "
                    "SET question_ids = CAST(:qids AS int[]), seeded_at = NOW() "
                    "WHERE module = 'listening' AND test_number = :tn"
                ), {"qids": new_qids, "tn": r.test_number})

        print()
        print(f"Templates updated: {changed} of {len(rows)}")
        if dry:
            db.rollback()
            print("Dry-run — rolled back.")
        else:
            db.commit()
            print("Committed.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.dry_run)
