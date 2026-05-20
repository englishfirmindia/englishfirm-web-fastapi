"""
Backfill pre-existing dangling qid references in sectional_test_questions.

Some sectional templates still reference qids that no longer exist in
either `questions_from_apeuni` or the native `questions` table — left
behind by earlier cleanups. Today's HIW purge did not introduce these;
they predate it.

Strategy:
  For each dangling qid in a template:
    1. Try `attempt_answers` history for question_type.
    2. If no history, infer from the same template — look at the
       non-dangling slots and pick the type of the nearest same-type
       cluster (the seed script groups questions by type, so dangling
       slots inherit the surrounding cluster's type).
    3. Pick a random replacement question of that type from the same
       module that isn't already in the template.

Idempotent: re-running on a clean template is a no-op.

Run with prod .env in place:
    PYTHONPATH=. python scripts/migrations/2026-05-20_purge_dangling_template_refs.py [--dry-run]
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from sqlalchemy import text as _sql  # noqa: E402

from db.database import SessionLocal  # noqa: E402


def _live_qid_types(db) -> dict:
    """qid → question_type for everything still in either table."""
    out = {}
    for r in db.execute(_sql(
        "SELECT question_id, question_type FROM questions_from_apeuni"
    )).all():
        out[r.question_id] = r.question_type
    for r in db.execute(_sql(
        "SELECT question_id, question_type FROM questions"
    )).all():
        out[r.question_id] = r.question_type
    return out


def _history_type_map(db, dangling_ids: list) -> dict:
    """qid → question_type recovered from attempt_answers history."""
    out = {}
    rows = db.execute(_sql("""
        SELECT question_id, question_type, COUNT(*) AS n
        FROM attempt_answers
        WHERE question_id = ANY(CAST(:ids AS int[]))
        GROUP BY question_id, question_type
        ORDER BY question_id, n DESC
    """), {"ids": dangling_ids}).all()
    seen = set()
    for r in rows:
        if r.question_id in seen:
            continue
        out[r.question_id] = r.question_type
        seen.add(r.question_id)
    return out


def _normalise_type(t: str) -> str:
    """Some legacy attempt_answers rows use shorthand types. Normalise to
    questions_from_apeuni values."""
    return {
        "highlight_correct_summary": "listening_hcs",
        "listening_hiw": "highlight_incorrect_words",
    }.get(t, t)


def _build_replacement_pool(live_types: dict) -> dict:
    """question_type → list[qid] of healthy live questions, for sampling."""
    pool = {}
    for qid, t in live_types.items():
        pool.setdefault(t, []).append(qid)
    return pool


def _resolve_dangling_type(
    qid: int,
    pos: int,
    template_qids: list,
    history_type: dict,
    live_types: dict,
) -> str:
    """Best-effort question_type recovery for a dangling qid.

    1. attempt_answers history wins if present.
    2. Otherwise inspect the same template: find the nearest non-dangling
       slot (closer-first) and inherit that type. The seed script groups
       questions by type, so adjacent slots are usually the same type.
    """
    if qid in history_type:
        return _normalise_type(history_type[qid])
    # Walk outward from pos
    n = len(template_qids)
    for off in range(1, n):
        for delta in (-off, off):
            j = pos + delta
            if 0 <= j < n:
                other = template_qids[j]
                if other in live_types:
                    return live_types[other]
    return ""


def main(dry: bool) -> None:
    db = SessionLocal()
    try:
        live_types = _live_qid_types(db)
        all_template_rows = db.execute(_sql(
            "SELECT module, test_number, question_ids "
            "FROM sectional_test_questions ORDER BY module, test_number"
        )).all()

        # Find all dangling qids
        dangling_ids = set()
        for r in all_template_rows:
            for qid in r.question_ids:
                if qid not in live_types:
                    dangling_ids.add(qid)
        dangling_ids = sorted(dangling_ids)
        print(f"Dangling qids found across templates: {len(dangling_ids)}")
        if not dangling_ids:
            print("Nothing to do.")
            return

        history = _history_type_map(db, dangling_ids)
        history = {qid: _normalise_type(t) for qid, t in history.items()}
        print(f"  classified by attempt history: {len(history)}")
        print(f"  to be inferred from template:  {len(dangling_ids) - len(history)}")

        replacement_pool = _build_replacement_pool(live_types)

        templates_changed = 0
        unresolved_qids = []
        type_swap_counts = Counter()

        for r in all_template_rows:
            qids = list(r.question_ids)
            original = list(qids)
            in_template = set(qids)
            for i, qid in enumerate(qids):
                if qid in live_types:
                    continue  # healthy
                qt = _resolve_dangling_type(qid, i, qids, history, live_types)
                if not qt:
                    unresolved_qids.append((r.module, r.test_number, i, qid))
                    continue
                pool = replacement_pool.get(qt) or []
                candidates = [q for q in pool if q not in in_template]
                if not candidates:
                    unresolved_qids.append((r.module, r.test_number, i, qid))
                    continue
                replacement = random.choice(candidates)
                qids[i] = replacement
                in_template.discard(qid)
                in_template.add(replacement)
                type_swap_counts[qt] += 1
            if qids != original:
                templates_changed += 1
                if not dry:
                    db.execute(_sql(
                        "UPDATE sectional_test_questions "
                        "SET question_ids = CAST(:qids AS int[]), seeded_at = NOW() "
                        "WHERE module = :m AND test_number = :tn"
                    ), {"qids": qids, "m": r.module, "tn": r.test_number})
                # Show diff
                removed = [q for q in original if q not in qids]
                added = [q for q in qids if q not in original]
                print(f"  ✓ {r.module}/test {r.test_number}: swapped {removed} → {added}")

        print()
        print(f"Templates updated: {templates_changed}")
        print()
        print("Swap-by-type:")
        for t, n in type_swap_counts.most_common():
            print(f"  {t:35s} {n}")
        if unresolved_qids:
            print()
            print(f"⚠️  Unresolved (still dangling): {len(unresolved_qids)}")
            for m, tn, pos, qid in unresolved_qids[:20]:
                print(f"  {m}/test {tn} pos {pos}: qid {qid} — could not infer type")

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
