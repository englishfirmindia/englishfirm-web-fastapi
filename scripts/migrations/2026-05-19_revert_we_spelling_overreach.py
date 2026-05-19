"""
Corrective backfill: revert the spelling-rescore on attempts where it
shouldn't have applied.

The 2026-05-19_rescore_we_spelling_deterministic migration unconditionally
recomputed `earned + pte_score` from the stored breakdown. Two classes of
attempts got bumped that shouldn't have:

  1. Form-zero attempts (form == 0, scorer == "form-gate-floor"). These
     should stay pinned at PTE 10 — form-zero is a hard floor, no
     sub-score can lift it.
  2. Legacy attempts with sparse breakdown (no form/content/dsc/grammar/
     glr/vocabulary fields). The migration treated missing fields as 0
     and computed earned from spelling alone, producing PTE 16/21 from a
     previous 0 or 10. Those numbers are spurious — there isn't enough
     data to assign a meaningful score.

This script reverts both classes to their original PTE values. The
spelling sub-block stays as-is (deterministic check is authoritative
either way) — only the top-level `score`, `breakdown.earned`,
`breakdown.pte_score`, and the `rescored_by` tag are touched.

Run
---
    cd englishfirm-web-fastapi
    ./venv/bin/python scripts/migrations/2026-05-19_revert_we_spelling_overreach.py
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

_BAD_TAG = "2026-05-19_rescore_we_spelling_deterministic"
_FIXED_TAG = "2026-05-19_revert_we_spelling_overreach"

# Pre-migration scores captured from the original migration's stdout.
# Used to revert. Keys are attempt_answers.id, values are the score the
# row had BEFORE the bad migration ran.
_PRE_SCORES = {
    # Incomplete-breakdown attempts (no form/content fields):
    402: 0, 458: 0, 532: 0, 577: 0, 622: 0, 930: 0, 1512: 0, 1668: 0, 2033: 0,
    686: 10, 721: 10, 1396: 10, 1403: 10, 1410: 10, 1417: 10, 1424: 10,
    1447: 10, 1448: 10, 1449: 10, 1545: 10, 1629: 10, 1775: 10, 1891: 10,
    1952: 10, 2524: 10,
    # Form-zero attempts (form==0, scorer=='form-gate-floor'):
    2327: 10, 2328: 10, 2453: 10,
}


def main() -> None:
    dsn = _env["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, score, result_json
        FROM attempt_answers
        WHERE question_type = 'write_essay'
          AND result_json->>'rescored_by' = %s
        """,
        (_BAD_TAG,),
    )
    rows = cur.fetchall()
    print(f"loaded {len(rows)} rows tagged by {_BAD_TAG}")

    reverted = 0
    kept = 0
    for aa_id, current_score, rj in rows:
        rj = dict(rj or {})
        bd = dict(rj.get("breakdown") or {})
        form_val = bd.get("form")
        content_val = (bd.get("content") or {}).get("score") if isinstance(bd.get("content"), dict) else None
        is_form_zero = form_val == 0
        is_incomplete = form_val is None and content_val is None

        if not (is_form_zero or is_incomplete):
            kept += 1
            continue

        original_score = _PRE_SCORES.get(aa_id)
        if original_score is None:
            # Defensive: if we don't have a recorded pre-score, leave it.
            print(f"  aa_id={aa_id} no pre-score recorded — skipping")
            kept += 1
            continue

        # Restore score; clear the bad tag; mark with the fix tag.
        bd["pte_score"] = original_score
        # earned: form-zero already had earned=0 in the form-gate-floor
        # branch; legacy attempts had earned=2.0 already (spelling-only).
        # For form-zero we restore earned=0; for legacy we leave as-is.
        if is_form_zero:
            bd["earned"] = 0
            sp = dict(bd.get("spelling") or {})
            sp["score"] = 0.0
            sp["reasoning"] = "Not scored — form-zero."
            bd["spelling"] = sp
        rj["breakdown"] = bd
        rj["pte_score"] = original_score
        rj["rescored_by"] = _FIXED_TAG

        cur.execute(
            "UPDATE attempt_answers SET score = %s, result_json = %s WHERE id = %s",
            (original_score, Json(rj), aa_id),
        )
        reverted += 1
        kind = "form-zero" if is_form_zero else "incomplete"
        print(f"  aa_id={aa_id} ({kind}) {current_score} → {original_score}")

    conn.commit()
    print(f"\ndone — reverted {reverted}, kept {kept}, total {len(rows)}")


if __name__ == "__main__":
    main()
