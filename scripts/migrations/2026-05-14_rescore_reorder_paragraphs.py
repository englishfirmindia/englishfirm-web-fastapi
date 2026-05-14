"""
Backfill: rescore every Reorder Paragraphs attempt_answer row.

Why
---
The frontend ships `user_sequence` as a list of paragraph TEXTS, but the
rule scorer compares against `correctAnswers.correctSequence` which is a
list of paragraph IDs (e.g. ["2", "4", "3", "1", "5"]). The router used
to pass the texts straight through, so the scorer's element-by-element
adjacency check failed on every pair — and 100% of historic RP attempts
(39/39) sat at PTE floor 10.

Forward fix shipped in the same backend commit maps text → id at the
router boundary before scoring.

What this script does
---------------------
1. Loads every completed RP attempt_answer + its question's paragraphs +
   its evaluation_json.
2. Re-runs ReorderScorer with the text → id mapping applied.
3. If the new PTE score differs from the stored one, updates
   `score` + `result_json` (preserving `time_on_question_seconds` and
   `user_answer_json` — only the result_json + score columns change).

Idempotent: re-running compares old vs new and only writes when they
differ.

Run
---
    cd englishfirm-web-fastapi
    ./venv/bin/python scripts/migrations/2026-05-14_rescore_reorder_paragraphs.py
"""
from __future__ import annotations

import json
from dotenv import dotenv_values
import os
import sys

# Bootstrap repo path + env
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
_env = dotenv_values(".env")
for _k, _v in _env.items():
    os.environ.setdefault(_k, _v)

import psycopg2  # noqa: E402
from psycopg2.extras import Json  # noqa: E402

from services.scoring.rule_scorer import ReorderScorer  # noqa: E402


def _text_to_id_map(content_json: dict) -> dict[str, str]:
    paragraphs = (content_json or {}).get("paragraphs") or []
    out: dict[str, str] = {}
    for p in paragraphs:
        if isinstance(p, dict):
            pid = p.get("id") or p.get("paragraphId")
            ptext = (p.get("text") or "").strip()
            if pid is not None and ptext:
                out[ptext] = str(pid)
    return out


def main() -> None:
    dsn = _env["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT aa.id, aa.question_id, aa.user_answer_json, aa.result_json, aa.score,
               q.content_json, qe.evaluation_json
        FROM attempt_answers aa
        JOIN questions_from_apeuni q ON q.question_id = aa.question_id
        LEFT JOIN question_evaluation_apeuni qe ON qe.question_id = aa.question_id
        WHERE aa.question_type = 'reorder_paragraphs'
          AND aa.scoring_status = 'complete'
        ORDER BY aa.id
        """
    )
    rows = cur.fetchall()
    print(f"loaded {len(rows)} reorder_paragraphs rows")

    scorer = ReorderScorer()
    updated = 0
    unchanged = 0
    for aa_id, qid, ua, rj, old_score, content_json, eval_json in rows:
        ua = ua or {}
        user_seq_texts = ua.get("user_sequence") or []
        text_to_id = _text_to_id_map(content_json)
        user_seq_ids = [text_to_id.get((s or "").strip(), s) for s in user_seq_texts]

        result = scorer.score(
            question_id=qid,
            session_id="backfill",
            answer={
                "user_sequence": user_seq_ids,
                "evaluation_json": eval_json or {},
            },
        )
        new_pte = int(result.pte_score)
        bd = result.breakdown or {}
        pair_results = list(bd.get("pair_results") or [])
        is_correct = bool(pair_results) and all(pair_results)

        if new_pte == int(old_score or 0):
            unchanged += 1
            continue

        # Build new result_json preserving fields the row already had
        # (notably time_on_question_seconds added by the stopwatch
        # feature) but replacing the scoring-derived fields.
        rj = dict(rj or {})
        rj.update(
            {
                "pte_score": new_pte,
                "hits": bd.get("hits"),
                "total_pairs": bd.get("total_pairs"),
                "max_score": bd.get("max_score"),
                "pair_results": pair_results,
                "correct_sequence": list(
                    (eval_json or {}).get("correctAnswers", {}).get("correctSequence", [])
                    or []
                ),
                "is_correct": is_correct,
                "rescored_by": "2026-05-14_rescore_reorder_paragraphs",
            }
        )

        cur.execute(
            "UPDATE attempt_answers SET score = %s, result_json = %s WHERE id = %s",
            (new_pte, Json(rj), aa_id),
        )
        updated += 1
        print(f"  aa_id={aa_id} qid={qid} {old_score} → {new_pte} "
              f"({bd.get('hits')}/{bd.get('total_pairs')} pairs)")

    conn.commit()
    print(f"\ndone — updated {updated}, unchanged {unchanged}, total {len(rows)}")


if __name__ == "__main__":
    main()
