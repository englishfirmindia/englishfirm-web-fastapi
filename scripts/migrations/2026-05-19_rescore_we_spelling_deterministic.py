"""
Backfill: rescore WE attempts after dropping the min(LLM, deterministic)
spelling hybrid.

Bug
---
WE's gpt-4o/claude split scorer has no spelling sub-call. The pipeline
synthesised a placeholder `{"score": 0.0, "reasoning": None}` and then
applied `final_spelling = min(llm_spell_score, spell_remaining)` — which
is always `min(0, X) = 0`. Every clean essay silently lost 2 raw points
(~6 PTE) on the Spelling row regardless of whether the deterministic
spell-check found any typos.

Forward fix shipped in same backend commit: trust the deterministic
spell-check (pyspell + passage filter + Claude judge) directly.

What this script does
---------------------
1. Loads every completed WE attempt_answer.
2. For each, reads the stored breakdown's `spelling_check.mistakes` and
   recomputes `final_spelling = max(0, 2 - len(mistakes))`.
3. If the new spelling differs from the stored sub-score, recomputes
   `earned` and rerouts through `to_pte_score` for the final pte_score.
4. Updates `score`, `result_json.breakdown.spelling`, `earned`, and the
   top-level `pte_score` fields. Preserves everything else.

Idempotent. Tags rescored rows with `rescored_by` so we can audit.

Run
---
    cd englishfirm-web-fastapi
    ./venv/bin/python scripts/migrations/2026-05-19_rescore_we_spelling_deterministic.py
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
from psycopg2.extras import Json  # noqa: E402

from services.scoring.base import to_pte_score  # noqa: E402


_TAG = "2026-05-19_rescore_we_spelling_deterministic"


def main() -> None:
    dsn = _env["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, question_id, result_json, score
        FROM attempt_answers
        WHERE question_type = 'write_essay'
          AND scoring_status = 'complete'
        ORDER BY id
        """
    )
    rows = cur.fetchall()
    print(f"loaded {len(rows)} write_essay rows")

    updated = 0
    unchanged = 0
    for aa_id, qid, rj, old_score in rows:
        rj = dict(rj or {})
        bd = dict(rj.get("breakdown") or {})
        sp = dict(bd.get("spelling") or {})
        sp_check = sp.get("spelling_check") or {}
        mistakes = sp_check.get("mistakes") or []
        new_spelling = float(max(0, 2 - len(mistakes)))
        old_spelling = float(sp.get("score", 0) or 0)

        if abs(new_spelling - old_spelling) < 1e-6:
            unchanged += 1
            continue

        # Recompute earned and pte_score.
        max_pts = bd.get("max_pts") or 26
        try:
            form     = float(bd.get("form", 0) or 0)
            content  = float((bd.get("content") or {}).get("score", 0) or 0)
            dsc      = float((bd.get("dsc") or {}).get("score", 0) or 0)
            grammar  = float((bd.get("grammar") or {}).get("score", 0) or 0)
            glr      = float((bd.get("glr") or {}).get("score", 0) or 0)
            vocab    = float((bd.get("vocabulary") or {}).get("score", 0) or 0)
        except (TypeError, ValueError):
            unchanged += 1
            continue

        new_earned = form + content + dsc + grammar + glr + vocab + new_spelling
        new_earned = max(0.0, min(float(max_pts), new_earned))
        pct = new_earned / max_pts if max_pts > 0 else 0.0
        new_pte = int(to_pte_score(pct))

        sp["score"] = new_spelling
        sp["hybrid_remaining"] = new_spelling
        sp.pop("llm_score", None)  # placeholder is no longer meaningful
        bd["spelling"] = sp
        bd["earned"] = new_earned
        rj["breakdown"] = bd
        rj["pte_score"] = new_pte
        rj["rescored_by"] = _TAG

        cur.execute(
            "UPDATE attempt_answers SET score = %s, result_json = %s WHERE id = %s",
            (new_pte, Json(rj), aa_id),
        )
        updated += 1
        print(
            f"  aa_id={aa_id} q={qid}  "
            f"spelling {old_spelling:.0f}→{new_spelling:.0f}  "
            f"pte {int(old_score or 0)}→{new_pte}"
        )

    conn.commit()
    print(f"\ndone — updated {updated}, unchanged {unchanged}, total {len(rows)}")


if __name__ == "__main__":
    main()
