"""
One-shot retrofill: re-score Nimisha's mock 3782 speaking answers using the
new loudness-normalized pause detection. Supports any speaking task_type by
calling the production scorer (_score_speaking_v2) + rubric weight conversion
(_compute_question_score) so PTE totals stay consistent with live scoring.

Usage:
    python3 scripts/retrofill_nimisha_mock_3782_speaking.py read_aloud           # dry run
    python3 scripts/retrofill_nimisha_mock_3782_speaking.py summarize_group_discussion --apply
"""
import os
import sys
import urllib.parse
import boto3
import psycopg2
from psycopg2.extras import Json, RealDictCursor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from services.speaking_scorer import _score_speaking_v2  # noqa: E402
from services.scoring.azure_scorer import _compute_question_score  # noqa: E402

ATTEMPT_ID = 3782
USER_ID = 3
AWS_PROFILE = "englishfirm"
AWS_REGION = "ap-southeast-2"
PTE_FLOOR, PTE_CEILING = 10, 90

if len(sys.argv) < 2:
    print("Usage: <task_type> [--apply]"); sys.exit(1)
TASK_TYPE = sys.argv[1]
APPLY = "--apply" in sys.argv


def _dsn():
    return os.environ["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://")


def _s3_bytes(audio_url: str) -> bytes:
    parsed = urllib.parse.urlparse(audio_url)
    bucket = parsed.netloc.split(".")[0]
    key = parsed.path.lstrip("/")
    s3 = boto3.session.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION).client("s3")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _passage_for(qid, cur):
    cur.execute("SELECT content_json FROM questions_from_apeuni WHERE question_id=%s", (qid,))
    row = cur.fetchone()
    cj = row["content_json"] if row else {}
    # RA/RS use 'passage'; SGD/RL/DI/RTS use stimulus audio (no reference text needed)
    return cj.get("passage", "") if cj else ""


def _pte_from_pct(pct: float) -> int:
    return max(PTE_FLOOR, min(PTE_CEILING, int(round(10 + (pct or 0.0) * 80))))


def main():
    conn = psycopg2.connect(_dsn())
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute(
        """
        SELECT aa.id, aa.question_id, aa.question_type, aa.audio_url,
               aa.score, aa.content_score, aa.fluency_score, aa.pronunciation_score,
               aa.result_json
        FROM attempt_answers aa
        WHERE aa.attempt_id = %s
          AND aa.question_type = %s
        ORDER BY aa.id
        """,
        (ATTEMPT_ID, TASK_TYPE),
    )
    rows = cur.fetchall()
    print(f"Found {len(rows)} {TASK_TYPE} answers in attempt {ATTEMPT_ID}")

    updates = []
    for r in rows:
        aid = r["id"]; qid = r["question_id"]
        passage = _passage_for(qid, cur)
        print(f"\n=== q={qid} aid={aid} type={TASK_TYPE} ===")
        print(f"  OLD: pte={r['score']} c={r['content_score']} f={r['fluency_score']} p={r['pronunciation_score']}")

        audio_bytes = _s3_bytes(r["audio_url"])
        result = _score_speaking_v2(
            user_id=USER_ID,
            question_id=qid,
            audio_bytes=audio_bytes,
            reference_text=passage,
            task_type=TASK_TYPE,
        )

        new_c = float(result.get("content", 0))
        new_f = float(result.get("fluency", 0))
        new_p = float(result.get("pronunciation", 0))

        # Rubric-weighted total via production helper. Pass content_llm_scored so
        # the SGD/DI/RL/RTS content-zero rule fires correctly.
        raw = {
            "content": new_c,
            "fluency": new_f,
            "pronunciation": new_p,
            "scoring": "complete",
            "content_llm_scored": (result.get("fluency_metrics", {}) or {}).get("content_method") == "llm_keypoints",
        }
        q = _compute_question_score(TASK_TYPE, raw)
        new_pte = _pte_from_pct(q["pct"])

        old_pauses = (r["result_json"] or {}).get("fluency_metrics", {}).get("gap_pauses")
        new_pauses = (result.get("fluency_metrics", {}) or {}).get("gap_pauses")
        print(f"  NEW: pte={new_pte} c={new_c:.1f} f={new_f:.1f} p={new_p:.1f}  (pauses {old_pauses}→{new_pauses})")

        new_rj = {
            **(r["result_json"] or {}),
            "content": new_c, "fluency": new_f, "pronunciation": new_p,
            "total": new_pte, "pte_score": new_pte,
            "transcript": result.get("transcript", ""),
            "word_scores": result.get("word_scores", []),
            "fluency_metrics": result.get("fluency_metrics", {}),
            "retrofill_2026_05_24": {
                "reason": "Loudness-normalized pause detection (A+C deploy)",
                "old": {
                    "score": int(r["score"]),
                    "content": float(r["content_score"]),
                    "fluency": float(r["fluency_score"]),
                    "pronunciation": float(r["pronunciation_score"]),
                    "old_pauses": old_pauses,
                },
            },
        }
        updates.append((aid, new_c, new_f, new_p, new_pte, new_rj))

    print("\n=== SUMMARY ===")
    print(f"{'aid':>5} {'qid':>5} | {'old_pte':>7} {'old_c':>6} {'old_f':>6} {'old_p':>6} → {'new_pte':>7} {'new_c':>6} {'new_f':>6} {'new_p':>6}")
    for aid, nc, nf, np_, npte, _ in updates:
        old = next(r for r in rows if r["id"] == aid)
        print(
            f"{aid:>5} {old['question_id']:>5} | "
            f"{old['score']:>7} {float(old['content_score']):>6.1f} {float(old['fluency_score']):>6.1f} {float(old['pronunciation_score']):>6.1f} → "
            f"{npte:>7} {nc:>6.1f} {nf:>6.1f} {np_:>6.1f}"
        )

    if not APPLY:
        print("\n[DRY RUN] No DB updates. Re-run with --apply to commit.")
        return

    print("\n[APPLY] Writing to RDS...")
    for aid, nc, nf, np_, npte, nrj in updates:
        cur.execute(
            """
            UPDATE attempt_answers
            SET content_score = %s, fluency_score = %s, pronunciation_score = %s,
                score = %s, result_json = %s
            WHERE id = %s
            """,
            (nc, nf, np_, npte, Json(nrj), aid),
        )

    cur.execute(
        "SELECT COALESCE(SUM(score), 0) AS s FROM attempt_answers WHERE attempt_id=%s",
        (ATTEMPT_ID,),
    )
    new_attempt_total = cur.fetchone()["s"]
    cur.execute(
        "UPDATE practice_attempts SET total_score=%s WHERE id=%s",
        (new_attempt_total, ATTEMPT_ID),
    )
    print(f"  attempt {ATTEMPT_ID} total_score → {new_attempt_total}")

    conn.commit()
    print("\nDONE.")


if __name__ == "__main__":
    main()
