"""One-shot retrofill (2026-07-04):

The mock exam's HIW render tokeniser used `(?<=\\s)|(?=\\s)` (lookaround
split, keeps whitespace as tokens), while the submit tokeniser used `\\s+`
(strips whitespace). After 2026-06-23 (frontend commit 2a89cf1 fixed the
submit path but not the render path), every mock HIW answer stored the
WRONG `highlighted_words` — the render index landed on a whitespace-adjacent
regular word instead of the wrong word the student actually tapped.

Real-world example: jishanranjit@gmail.com's mock aid=7063 recorded
[`Obviously,`, `professional`, `based`, `important`] for a Classical Music
passage where she actually clicked [`disposers`, `gastronomical`,
`mizzenmasts`, `inventions`]. The stored index 2N maps to the correctly-
clicked word at position N in the submit tokenisation (i.e. `passage.split()`).

Frontend fix shipped in commit <TBD> — future mock HIW writes use `\\s+`
on both sides. This script retrofills existing rows.

Scope: mock HIW rows with `submitted_at >= 2026-06-23 10:21:56` and at least
one recorded click. Confirmed 25 rows across 10 users at the time of writing.

Usage:
    python3 scripts/retrofill_mock_hiw_tokenizer_20260704.py            # dry run
    python3 scripts/retrofill_mock_hiw_tokenizer_20260704.py --apply    # commit
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from services.scoring.rule_scorer import HIWScorer
from services.scoring.base import to_pte_score

APPLY = "--apply" in sys.argv
FIX_TIME = "2026-06-23 10:21:56+00"


def _dsn() -> str:
    return os.environ["DATABASE_URL"].replace(
        "postgresql+psycopg2://", "postgresql://"
    )


def main() -> None:
    conn = psycopg2.connect(_dsn())
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)
    scorer = HIWScorer()

    # Fetch every affected mock HIW row + its passage + eval JSON. We
    # left-join against the current questions/evaluation rows so deleted
    # questions surface with NULL passage/eval — we skip those since we
    # can't rescore without the eval key.
    cur.execute(
        """
        SELECT
            aa.id AS aid,
            aa.attempt_id,
            aa.question_id,
            aa.score AS old_pte,
            aa.user_answer_json,
            aa.result_json,
            q.content_json,
            qe.evaluation_json,
            pa.user_id
        FROM attempt_answers aa
        JOIN practice_attempts pa ON pa.id = aa.attempt_id
        LEFT JOIN questions_from_apeuni q ON q.question_id = aa.question_id
        LEFT JOIN question_evaluation_apeuni qe ON qe.question_id = aa.question_id
        WHERE pa.module = 'mock'
          AND aa.question_type IN ('listening_hiw','highlight_incorrect_words')
          AND aa.submitted_at >= TIMESTAMP %s
          AND jsonb_array_length(COALESCE(aa.user_answer_json->'highlighted_indices','[]'::jsonb)) > 0
        ORDER BY aa.submitted_at
        """,
        (FIX_TIME,),
    )
    rows = cur.fetchall()
    print(f"Found {len(rows)} mock HIW row(s) to retrofill.\n")

    updates = []            # (aid, new_ua_json, new_rj, new_pte, old_pte)
    skipped = []            # (aid, reason)
    attempt_ids_touched = set()

    for r in rows:
        aid = r["aid"]
        uaj = r["user_answer_json"] or {}
        old_indices = uaj.get("highlighted_indices") or []
        old_words = uaj.get("highlighted_words") or []

        cj = r["content_json"] or {}
        passage = cj.get("passage") or cj.get("transcript") or ""
        ej = r["evaluation_json"] or {}

        if not passage:
            # Fall back to backup snapshot when the question row is gone
            # (we may have deleted the question after the student attempted).
            cur.execute(
                """
                SELECT row_data FROM hiw_bulk_delete_backup_20260626
                WHERE question_id=%s AND source_table='questions_from_apeuni'
                LIMIT 1
                """,
                (r["question_id"],),
            )
            b = cur.fetchone()
            if b:
                passage = ((b["row_data"] or {}).get("content_json") or {}).get("passage", "") or ""
            if not passage:
                skipped.append((aid, "no passage available"))
                continue

        if not ej:
            cur.execute(
                """
                SELECT row_data FROM hiw_bulk_delete_backup_20260626
                WHERE question_id=%s AND source_table='question_evaluation_apeuni'
                LIMIT 1
                """,
                (r["question_id"],),
            )
            b = cur.fetchone()
            if b:
                ej = (b["row_data"] or {}).get("evaluation_json") or {}
            if not ej:
                skipped.append((aid, "no evaluation_json available"))
                continue

        # Halve each index — the render tokeniser was exactly 2× the submit.
        halved_indices = [i // 2 for i in old_indices if isinstance(i, int)]

        # Rebuild the `highlighted_words` list from the SUBMIT tokeniser
        # (which matches sectional + practice + review + scorer).
        submit_tokens = [w for w in passage.split() if w]
        new_words = [
            submit_tokens[i] for i in halved_indices
            if 0 <= i < len(submit_tokens)
        ]

        # Compose the corrected user_answer_json.
        new_ua = {
            **uaj,
            "highlighted_indices": halved_indices,
            "highlighted_words":   new_words,
        }

        # Re-run the scorer end-to-end against the corrected input.
        result = scorer.score(
            question_id=r["question_id"],
            session_id="retrofill_20260704",
            answer={
                "highlighted_words": new_words,
                "evaluation_json":   ej,
            },
        )
        new_pte = result.pte_score

        # Rebuild result_json: preserve any non-scorer fields (like
        # `is_correct`, per-index corrections) but replace scorer output.
        old_rj = r["result_json"] or {}
        preserved = {
            k: v for k, v in old_rj.items()
            if k not in ("score", "max_score", "maxScore", "correct_clicks",
                         "incorrect_clicks", "missed_words", "is_correct",
                         "pte_score")
        }
        new_rj = {
            **preserved,
            **(result.breakdown or {}),
            "maxScore": (result.breakdown or {}).get("max_score"),
            "pte_score": new_pte,
            "is_correct": (
                len((result.breakdown or {}).get("incorrect_clicks", [])) == 0
                and len((result.breakdown or {}).get("missed_words", [])) == 0
            ),
            "retrofill_2026_07_04": {
                "reason": "mock HIW render tokeniser (lookaround) was 2x the "
                          "submit tokeniser (\\s+); indices halved + words "
                          "re-derived from submit tokens.",
                "old_indices": old_indices,
                "old_words":   old_words,
                "old_pte":     r["old_pte"],
            },
        }

        updates.append((aid, new_ua, new_rj, new_pte, r["old_pte"], r["attempt_id"],
                        new_words, old_words))
        attempt_ids_touched.add(r["attempt_id"])

    # ── Report ──
    print(f"── SUMMARY: {len(updates)} to update, {len(skipped)} skipped ──\n")
    if skipped:
        print("Skipped:")
        for aid, reason in skipped:
            print(f"  aid={aid}: {reason}")
        print()

    print(f"{'aid':>6} {'uid':>4} {'qid':>6}  old_pte→new_pte  old_words → new_words")
    print("─" * 130)
    for aid, new_ua, new_rj, new_pte, old_pte, att_id, new_words, old_words in updates:
        cur.execute(
            "SELECT pa.user_id, aa.question_id FROM attempt_answers aa "
            "JOIN practice_attempts pa ON pa.id=aa.attempt_id WHERE aa.id=%s",
            (aid,),
        )
        rr = cur.fetchone()
        uid = rr["user_id"] if rr else '?'
        qid = rr["question_id"] if rr else '?'
        delta = new_pte - old_pte
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"{aid:>6} {uid:>4} {qid:>6}  {old_pte:>3} → {new_pte:>3}  "
              f"[{marker}{delta:+3}]  {str(old_words)[:40]:<42} → {str(new_words)[:40]}")

    if not APPLY:
        print("\n[DRY RUN] Re-run with --apply to commit.")
        return

    # ── Apply ──
    print(f"\n[APPLY] Writing {len(updates)} row(s)...")
    for aid, new_ua, new_rj, new_pte, _old_pte, _att, _nw, _ow in updates:
        cur.execute(
            """
            UPDATE attempt_answers
            SET user_answer_json = %s,
                result_json      = %s,
                score            = %s
            WHERE id = %s
            """,
            (Json(new_ua), Json(new_rj), new_pte, aid),
        )
    print(f"  ...{len(updates)} attempt_answers rows updated.")

    # Recompute total_score for every touched attempt.
    print(f"\n[APPLY] Recomputing practice_attempts.total_score for "
          f"{len(attempt_ids_touched)} attempt(s)...")
    for attempt_id in attempt_ids_touched:
        cur.execute(
            """
            UPDATE practice_attempts
            SET total_score = (
                SELECT COALESCE(SUM(score), 0)
                FROM attempt_answers WHERE attempt_id=%s
            )
            WHERE id=%s
            RETURNING total_score
            """,
            (attempt_id, attempt_id),
        )
        rr = cur.fetchone()
        new_total = rr["total_score"] if rr else None
        print(f"  attempt {attempt_id} → total_score={new_total}")

    conn.commit()
    print("\nDONE.")


if __name__ == "__main__":
    main()
