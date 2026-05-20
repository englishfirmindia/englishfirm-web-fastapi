"""
Hard-delete the 150 HIW questions whose answer-key has broken `wrong`
fields (legacy / data-quality issue from apeuni import).

Audit logic — a row is "broken" iff its evaluation_json uses the new
{wrong, correct} dict shape AND at least one `wrong` value does not
appear as a word in the printed passage. The 90 legacy-shape rows
(plain string lists) are healthy by definition.

Pipeline (single transaction):
  1. Recompute broken_ids + healthy_ids from current RDS state.
  2. Replace any broken qid inside sectional_test_questions.question_ids
     with a healthy HIW not already in that template. Listening templates
     get listening-module replacements; reading templates also get
     listening-module replacements (HIW is always module='listening',
     READING_STRUCTURE pulls it cross-module — same as today).
  3. DELETE the paired rows from question_evaluation_apeuni (FK blocker
     is NO ACTION, so this must precede the question delete).
  4. DELETE the 150 rows from questions_from_apeuni.

Mock tests select HIW dynamically at start time, so they auto-exclude
deleted qids — no mock template cleanup needed.

Historical attempt_answers / user_question_attempts / question_reports
rows referencing deleted qids will orphan (no FK), which is intentional
— preserves the headline scores but the question lookup in review
screens will surface "question not found" for those past attempts.

Run with prod .env in place:

    PYTHONPATH=. python scripts/migrations/2026-05-20_purge_broken_hiw.py [--dry-run]

--dry-run prints the plan without committing.
"""
import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from sqlalchemy import text as _sql  # noqa: E402

from db.database import SessionLocal  # noqa: E402


def _in_passage(passage_words: set, w: str) -> bool:
    if not w:
        return False
    return w.strip(".,!?;:\"'()").lower() in passage_words


def _classify_hiw(db) -> tuple:
    """Returns (broken_ids: list[int], healthy_ids: list[int])."""
    rows = db.execute(_sql("""
      SELECT q.question_id,
             q.content_json->>'passage' AS passage,
             qe.evaluation_json -> 'correctAnswers' -> 'incorrectWords' AS iw
      FROM questions_from_apeuni q
      JOIN question_evaluation_apeuni qe ON qe.question_id = q.question_id
      WHERE q.question_type = 'highlight_incorrect_words'
    """)).all()
    broken, healthy = [], []
    for r in rows:
        passage_words = set(re.findall(r"[A-Za-z'-]+", (r.passage or "").lower()))
        iw = r.iw or []
        if not iw:
            continue
        if all(isinstance(x, str) for x in iw):
            healthy.append(r.question_id)
            continue
        n, wp = 0, 0
        for e in iw:
            if not isinstance(e, dict):
                continue
            w = (e.get("wrong") or "").strip()
            if not w:
                continue
            n += 1
            if _in_passage(passage_words, w):
                wp += 1
        if n == 0:
            continue
        if wp == n:
            healthy.append(r.question_id)
        else:
            broken.append(r.question_id)
    return broken, healthy


def _swap_in_templates(db, broken_set: set, healthy_pool: list, dry: bool) -> int:
    """For each sectional row whose question_ids contains a broken qid,
    replace with a random healthy HIW not already in that template."""
    rows = db.execute(_sql("""
      SELECT module, test_number, question_ids
      FROM sectional_test_questions
      WHERE question_ids && CAST(:broken AS int[])
      ORDER BY module, test_number
    """), {"broken": list(broken_set)}).all()
    updates = 0
    for r in rows:
        qids = list(r.question_ids)
        original = list(qids)
        candidates = [h for h in healthy_pool if h not in qids]
        random.shuffle(candidates)
        cand_iter = iter(candidates)
        for i, q in enumerate(qids):
            if q in broken_set:
                try:
                    replacement = next(cand_iter)
                except StopIteration:
                    print(f"  ✗ {r.module}/test {r.test_number}: ran out of healthy candidates; leaving broken qid {q}")
                    continue
                qids[i] = replacement
        if qids != original:
            updates += 1
            removed = [q for q in original if q in broken_set]
            added = [q for q in qids if q not in original]
            print(f"  ✓ {r.module}/test {r.test_number}: swapped {removed} → {added}")
            if not dry:
                db.execute(
                    _sql("UPDATE sectional_test_questions "
                         "SET question_ids = CAST(:qids AS int[]), seeded_at = NOW() "
                         "WHERE module = :m AND test_number = :tn"),
                    {"qids": qids, "m": r.module, "tn": r.test_number},
                )
    return updates


def main(dry: bool) -> None:
    db = SessionLocal()
    try:
        broken, healthy = _classify_hiw(db)
        broken_set = set(broken)
        print(f"Broken HIW qids:  {len(broken)}")
        print(f"Healthy HIW qids: {len(healthy)}")
        if not broken:
            print("Nothing to do.")
            return

        # 1. Sectional template cleanup
        print()
        print(f"Step 1 — sectional template cleanup ({'dry-run' if dry else 'live'}):")
        templates_updated = _swap_in_templates(db, broken_set, healthy, dry)
        print(f"  Templates updated: {templates_updated}")

        # 2. Delete eval rows
        print()
        print(f"Step 2 — delete question_evaluation_apeuni rows ({len(broken)}):")
        if dry:
            print("  (dry-run, no delete)")
        else:
            res = db.execute(
                _sql("DELETE FROM question_evaluation_apeuni WHERE question_id = ANY(CAST(:ids AS int[]))"),
                {"ids": broken},
            )
            print(f"  rowcount: {res.rowcount}")

        # 3. Delete question rows
        print()
        print(f"Step 3 — delete questions_from_apeuni rows ({len(broken)}):")
        if dry:
            print("  (dry-run, no delete)")
        else:
            res = db.execute(
                _sql("DELETE FROM questions_from_apeuni WHERE question_id = ANY(CAST(:ids AS int[]))"),
                {"ids": broken},
            )
            print(f"  rowcount: {res.rowcount}")

        if dry:
            db.rollback()
            print("\nDry-run — rolled back.")
        else:
            db.commit()
            print("\nDone — committed.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.dry_run)
