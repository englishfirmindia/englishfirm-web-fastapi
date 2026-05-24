"""
One-shot retrofill: re-score Nimisha's mock 3782 RA answers with the new
loudness-normalized pause detection.

Usage:
    DRY_RUN=1 python3 scripts/retrofill_nimisha_mock_3782_ra.py     # default — prints diff
    python3 scripts/retrofill_nimisha_mock_3782_ra.py --apply        # writes to RDS

Each RA answer in attempt 3782 is re-scored by calling the production scorer
(_score_speaking_v2) which now applies +N dB gain to recordings below -20 dBFS
before pydub.detect_silence. Azure pronunciation_assessment + Whisper are
called again — small cost, ~6 audios. The attempt's total_score is then
recomputed as the sum of all answer scores.
"""
import os
import sys
import json
import urllib.parse
import boto3
import psycopg2
from psycopg2.extras import Json, RealDictCursor

# Make the project importable so we get the deployed scorer code.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from services.speaking_scorer import _score_speaking_v2  # noqa: E402

APPLY = "--apply" in sys.argv
ATTEMPT_ID = 3782
USER_ID = 3
AWS_PROFILE = "englishfirm"
AWS_REGION = "ap-southeast-2"

def _dsn():
    return os.environ["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://")


def _s3_bytes(audio_url: str) -> bytes:
    """Download audio bytes from a public S3 URL via boto3 (uses ~/.aws creds)."""
    parsed = urllib.parse.urlparse(audio_url)
    bucket = parsed.netloc.split(".")[0]
    key = parsed.path.lstrip("/")
    session = boto3.session.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    s3 = session.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def main():
    conn = psycopg2.connect(_dsn())
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # All RA answers for this attempt
    cur.execute(
        """
        SELECT aa.id, aa.question_id, aa.audio_url,
               aa.score, aa.content_score, aa.fluency_score, aa.pronunciation_score,
               aa.result_json
        FROM attempt_answers aa
        WHERE aa.attempt_id = %s
          AND aa.question_type = 'read_aloud'
        ORDER BY aa.id
        """,
        (ATTEMPT_ID,),
    )
    rows = cur.fetchall()
    print(f"Found {len(rows)} RA answers in attempt {ATTEMPT_ID}")

    updates = []      # (answer_id, new_payload, old_summary, new_summary)
    for r in rows:
        aid = r["id"]
        qid = r["question_id"]
        # Fetch reference passage
        cur.execute(
            "SELECT content_json FROM questions_from_apeuni WHERE question_id=%s",
            (qid,),
        )
        cj = cur.fetchone()["content_json"]
        passage = cj.get("passage", "")
        if not passage:
            print(f"  [SKIP] q={qid} — no passage in content_json")
            continue

        print(f"\n=== q={qid} answer_id={aid} ===")
        print(f"  audio_url: {r['audio_url']}")
        print(f"  OLD: total={r['score']} c={r['content_score']} f={r['fluency_score']} p={r['pronunciation_score']}")

        # Download audio
        audio_bytes = _s3_bytes(r["audio_url"])
        print(f"  audio bytes: {len(audio_bytes)}")

        # Re-score with new normalization
        result = _score_speaking_v2(
            user_id=USER_ID,
            question_id=qid,
            audio_bytes=audio_bytes,
            reference_text=passage,
            task_type="read_aloud",
        )

        new_c = float(result.get("content", 0))
        new_f = float(result.get("fluency", 0))
        new_p = float(result.get("pronunciation", 0))
        new_total = (new_c + new_f + new_p) / 3.0  # equal-weight RA — matches azure_scorer weighting
        new_pte = max(int(round(new_total)), 10)

        print(f"  NEW: total={new_pte} c={new_c:.1f} f={new_f:.1f} p={new_p:.1f}")

        new_metrics = result.get("fluency_metrics", {}) or {}
        old_pauses = (r["result_json"] or {}).get("fluency_metrics", {}).get("gap_pauses")
        new_pauses = new_metrics.get("gap_pauses")
        print(f"  pauses: {old_pauses} → {new_pauses}")

        new_rj = {
            **(r["result_json"] or {}),
            "content": new_c,
            "fluency": new_f,
            "pronunciation": new_p,
            "total": new_pte,
            "pte_score": new_pte,
            "transcript": result.get("transcript", ""),
            "word_scores": result.get("word_scores", []),
            "fluency_metrics": new_metrics,
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

    # Print summary before applying
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
            SET content_score = %s,
                fluency_score = %s,
                pronunciation_score = %s,
                score = %s,
                result_json = %s
            WHERE id = %s
            """,
            (nc, nf, np_, npte, Json(nrj), aid),
        )

    # Recompute attempt total_score (sum of all answer scores, not just RA)
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
