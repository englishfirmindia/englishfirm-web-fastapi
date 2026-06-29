"""
One-shot retrofill: re-score Sheetal's mock attempt 7196 speaking answers.

Background: all 32 speaking attempts in this mock session were marked
scoring_status=complete but their transcripts came back empty and
content/fluency scored 0.0 due to an OpenAI + Azure outage on 2026-06-25.
This script re-runs _score_speaking_v2 (the production scorer) per attempt
using the audio that's still in S3, recomputes the rubric-weighted PTE total
via _compute_question_score, and writes back to attempt_answers and the
parent practice_attempts.total_score.

Usage:
    python3 scripts/retrofill_sheetal_mock_7196_speaking.py            # dry run
    python3 scripts/retrofill_sheetal_mock_7196_speaking.py --apply    # commit
    python3 scripts/retrofill_sheetal_mock_7196_speaking.py read_aloud # filter
    python3 scripts/retrofill_sheetal_mock_7196_speaking.py read_aloud --apply
"""
import os
import sys
import urllib.parse
import json
import boto3
import psycopg2
from psycopg2.extras import Json, RealDictCursor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from services.speaking_scorer import _score_speaking_v2  # noqa: E402
from services.scoring.azure_scorer import _compute_question_score  # noqa: E402

ATTEMPT_ID = 7196
USER_ID = 101
AWS_PROFILE = "englishfirm"
AWS_REGION = "ap-southeast-2"
PTE_FLOOR, PTE_CEILING = 10, 90
SPEAKING_TYPES = (
    "read_aloud", "repeat_sentence", "answer_short_question",
    "describe_image", "retell_lecture",
    "summarize_group_discussion", "respond_to_situation",
)

filter_type = None
apply_mode = False
for a in sys.argv[1:]:
    if a == "--apply":
        apply_mode = True
    elif a in SPEAKING_TYPES:
        filter_type = a
    else:
        print(f"Unknown arg: {a}"); sys.exit(1)


def _dsn():
    return os.environ["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://")


def _s3_bytes(audio_url: str) -> bytes:
    parsed = urllib.parse.urlparse(audio_url)
    bucket = parsed.netloc.split(".")[0]
    key = parsed.path.lstrip("/")
    s3 = boto3.session.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION).client("s3")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _build_scorer_args(qt, content_json, evaluation_json):
    """Mirror the per-router prep in routers/speaking/*.py."""
    cj = content_json or {}
    ej = evaluation_json or {}
    ca = (ej.get("correctAnswers") or {}) if isinstance(ej.get("correctAnswers"), dict) else {}
    args = dict(reference_text="", key_points=[], expected_answers=[], stimulus_audio_url="")
    if qt == "read_aloud":
        args["reference_text"] = (cj.get("passage") or "").strip()
    elif qt == "repeat_sentence":
        args["reference_text"] = (cj.get("transcript") or ca.get("transcript") or "").strip()
    elif qt == "answer_short_question":
        ea = ca.get("acceptedVariants") or ([ca["answer"]] if ca.get("answer") else [])
        args["expected_answers"] = ea or []
    elif qt == "describe_image":
        args["key_points"] = ca.get("keyPoints") or []
    elif qt in ("retell_lecture", "summarize_group_discussion", "respond_to_situation"):
        # Routers all read "keyPoints"; RTS evaluation_json may also have "taskPoints" — fall back.
        args["key_points"] = ca.get("keyPoints") or ca.get("taskPoints") or []
        args["stimulus_audio_url"] = (cj.get("audio_url") or "").strip()
    return args


def _pte_from_pct(pct: float) -> int:
    return max(PTE_FLOOR, min(PTE_CEILING, int(round(10 + (pct or 0.0) * 80))))


def main():
    conn = psycopg2.connect(_dsn())
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)

    where_qt = "" if filter_type is None else " AND aa.question_type = %s"
    params = (ATTEMPT_ID,) + ((filter_type,) if filter_type else ())
    cur.execute(
        f"""
        SELECT aa.id, aa.question_id, aa.question_type, aa.audio_url,
               aa.score, aa.content_score, aa.fluency_score, aa.pronunciation_score,
               aa.result_json,
               q.content_json, qe.evaluation_json
        FROM attempt_answers aa
        JOIN questions_from_apeuni q ON q.question_id = aa.question_id
        LEFT JOIN question_evaluation_apeuni qe ON qe.question_id = aa.question_id
        WHERE aa.attempt_id = %s
          AND aa.question_type = ANY(%s)
          {where_qt}
        ORDER BY aa.id
        """,
        (ATTEMPT_ID, list(SPEAKING_TYPES)) + ((filter_type,) if filter_type else ()),
    )
    rows = cur.fetchall()
    print(f"Found {len(rows)} speaking answers in attempt {ATTEMPT_ID}"
          f"{' (filter='+filter_type+')' if filter_type else ''}")

    updates = []
    failures = []
    for r in rows:
        aid = r["id"]; qid = r["question_id"]; qt = r["question_type"]
        print(f"\n=== aid={aid} qid={qid} type={qt} ===")
        print(f"  OLD: pte={r['score']} c={r['content_score']} f={r['fluency_score']} p={r['pronunciation_score']}")
        try:
            audio_bytes = _s3_bytes(r["audio_url"])
            args = _build_scorer_args(qt, r["content_json"], r["evaluation_json"])
            result = _score_speaking_v2(
                user_id=USER_ID,
                question_id=qid,
                audio_bytes=audio_bytes,
                task_type=qt,
                **args,
            )
            new_c = float(result.get("content", 0))
            new_f = float(result.get("fluency", 0))
            new_p = float(result.get("pronunciation", 0))
            content_llm_scored = bool(result.get("content_llm_scored", False))
            raw_for_pte = {
                "content": new_c, "fluency": new_f, "pronunciation": new_p,
                "scoring": "complete",
                "content_llm_scored": content_llm_scored,
            }
            is_correct = result.get("is_correct")
            if is_correct is not None:
                raw_for_pte["is_correct"] = bool(is_correct)
            q = _compute_question_score(qt, raw_for_pte)
            new_pte = _pte_from_pct(q["pct"])
            transcript = (result.get("transcript") or "").strip()
            print(f"  NEW: pte={new_pte} c={new_c:.1f} f={new_f:.1f} p={new_p:.1f}  transcript={transcript[:80]!r}")

            new_rj = {
                **(r["result_json"] or {}),
                "content": new_c, "fluency": new_f, "pronunciation": new_p,
                "total": new_pte, "pte_score": new_pte,
                "transcript": result.get("transcript", ""),
                "word_scores": result.get("word_scores", []),
                "fluency_metrics": result.get("fluency_metrics", {}),
                "content_llm_scored": content_llm_scored,
                "scoring_warnings": result.get("scoring_warnings", []),
                "retrofill_2026_06_26": {
                    "reason": "Re-score after 2026-06-25 OpenAI+Azure outage left transcripts empty and content/fluency at 0",
                    "old": {
                        "score": int(r["score"] or 0),
                        "content": float(r["content_score"] or 0),
                        "fluency": float(r["fluency_score"] or 0),
                        "pronunciation": float(r["pronunciation_score"] or 0),
                    },
                },
            }
            updates.append((aid, new_c, new_f, new_p, new_pte, new_rj))
        except Exception as ex:
            print(f"  ✗ FAILED: {ex}")
            failures.append((aid, qid, qt, str(ex)))

    print("\n=== SUMMARY ===")
    print(f"{'aid':>5} {'qid':>6} {'type':<28} | {'old_pte':>7} {'old_c':>6} {'old_f':>6} {'old_p':>6} → {'new_pte':>7} {'new_c':>6} {'new_f':>6} {'new_p':>6}")
    for aid, nc, nf, np_, npte, _ in updates:
        old = next(r for r in rows if r["id"] == aid)
        print(
            f"{aid:>5} {old['question_id']:>6} {old['question_type']:<28} | "
            f"{old['score']:>7} {float(old['content_score']):>6.1f} {float(old['fluency_score']):>6.1f} {float(old['pronunciation_score']):>6.1f} → "
            f"{npte:>7} {nc:>6.1f} {nf:>6.1f} {np_:>6.1f}"
        )
    if failures:
        print(f"\n✗ {len(failures)} failures:")
        for aid, qid, qt, ex in failures:
            print(f"  aid={aid} qid={qid} qt={qt}: {ex}")

    if not apply_mode:
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
