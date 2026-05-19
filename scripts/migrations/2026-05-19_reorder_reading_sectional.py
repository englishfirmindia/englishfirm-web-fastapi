"""
Reorder existing reading sectional question lists to match the new
READING_STRUCTURE order.

What changed
------------
READING_STRUCTURE was edited so:
  - The orphan `reading_fib` (typed FIB Reading) is dropped — the
    question pool is empty in production so it was always silently
    skipped, but it sat in `_READING_WEIGHTS` with weight 13,
    capping every user at PTE 81.
  - The remaining 8 tasks are reordered to:
      SWT → FIB Dropdown → MCM → Reorder → FIB Drag & Drop → MCS → HCS → HIW
    (was: SWT → FIB Dropdown → FIB Drag & Drop → MCM → Reorder → MCS → HCS → HIW)

The 40 rows in `sectional_test_questions` (module='reading',
test_number 1..40) store `question_ids` as a flat ordered array.
Without this migration, existing tests would still deliver questions
in the OLD task order until each row is re-emitted.

This script regroups `question_ids` per row by joining each id to
its `question_type` and re-emitting in the new task order. Within a
task, the existing order is preserved. NO questions are added or
removed — same 21 questions per test, different order.

Idempotent: re-running compares before/after and only writes when
they differ.

Run
---
    cd englishfirm-web-fastapi
    ./venv/bin/python scripts/migrations/2026-05-19_reorder_reading_sectional.py
"""
from __future__ import annotations

import os
import sys

from dotenv import dotenv_values

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
_env = dotenv_values(".env")
for _k, _v in _env.items():
    os.environ.setdefault(_k, _v)

import psycopg2  # noqa: E402

# Task order as of 2026-05-19 (must mirror READING_STRUCTURE).
_NEW_ORDER = [
    "summarize_written_text",
    "reading_fib_drop_down",
    "mcq_multiple",
    "reorder_paragraphs",
    "reading_drag_and_drop",
    "mcq_single",
    "listening_hcs",
    "highlight_incorrect_words",
]

# Stable rank for sort. Unknown task types go to the end in their original
# relative order (rank = len + appearance index).
def _rank_for(task: str) -> int:
    try:
        return _NEW_ORDER.index(task)
    except ValueError:
        return len(_NEW_ORDER) + 99


def main() -> None:
    dsn = _env["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT s.id, s.test_number, s.question_ids
        FROM sectional_test_questions s
        WHERE s.module = 'reading'
        ORDER BY s.test_number
        """
    )
    rows = cur.fetchall()
    print(f"loaded {len(rows)} reading sectional rows")

    if not rows:
        print("nothing to do")
        return

    # Pre-fetch task types for every question_id that appears across all rows.
    all_ids: set[int] = set()
    for _, _, ids in rows:
        all_ids.update(ids or [])
    cur.execute(
        "SELECT question_id, question_type FROM questions_from_apeuni WHERE question_id = ANY(%s)",
        (list(all_ids),),
    )
    qtype = {qid: t for qid, t in cur.fetchall()}

    updated = 0
    unchanged = 0
    for row_id, test_number, old_ids in rows:
        old_ids = list(old_ids or [])
        if not old_ids:
            unchanged += 1
            continue
        # Stable sort by task rank — within a task the relative order is preserved.
        new_ids = sorted(old_ids, key=lambda qid: _rank_for(qtype.get(qid, "")))
        if new_ids == old_ids:
            unchanged += 1
            continue
        cur.execute(
            "UPDATE sectional_test_questions SET question_ids = %s WHERE id = %s",
            (new_ids, row_id),
        )
        updated += 1
        # Compact summary of the new order: task abbreviation list
        order_summary = " → ".join(
            _abbrev(qtype.get(qid, "?")) for qid in new_ids
        )
        print(f"  test {test_number:>2}: reordered ({len(new_ids)} q) {order_summary}")

    conn.commit()
    print(f"\ndone — updated {updated}, unchanged {unchanged}, total {len(rows)}")


def _abbrev(task: str) -> str:
    return {
        "summarize_written_text":    "SWT",
        "reading_fib_drop_down":     "FIBd",
        "reading_drag_and_drop":     "FIBdd",
        "mcq_multiple":              "MCM",
        "reorder_paragraphs":        "RO",
        "mcq_single":                "MCS",
        "listening_hcs":             "HCS",
        "highlight_incorrect_words": "HIW",
    }.get(task, task[:5])


if __name__ == "__main__":
    main()
