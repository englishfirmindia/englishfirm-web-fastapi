"""
One-shot re-score: recover Hasini's (user 121) sectional attempt 15698 answers
that were flipped to `failed` with scoring_warnings=["scoring_infrastructure_timeout"]
because the AI/transcription-scored speaking questions didn't finish within the
300s pending-reaper window when all 32 questions scored at once (2026-08-22).

The audio is intact in S3, so we re-run the *production* scorer one question at a
time (no concurrent burst → no timeout), feeding the SAME inputs the live submit
routers feed:
  - answer_short_question           → expected_answers (evaluation.correctAnswers.acceptedVariants | [answer])
  - retell_lecture / summarize_group_discussion / ptea_respond_situation
                                    → key_points (evaluation.correctAnswers.keyPoints) + stimulus_audio_url (content_json.audio_url)

PTE per question via the production helper `_compute_question_score` so totals stay
consistent with live scoring.

Usage:
    arch -arm64 ./venv/bin/python scripts/rescore_hasini_15698_timeout.py            # dry run
    arch -arm64 ./venv/bin/python scripts/rescore_hasini_15698_timeout.py --apply    # commit
"""
import os
import sys
import urllib.parse
import boto3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from sqlalchemy.orm.attributes import flag_modified  # noqa: E402
from db.database import SessionLocal  # noqa: E402
from db.models import AttemptAnswer, PracticeAttempt, QuestionFromApeuni  # noqa: E402
from services.speaking_scorer import _score_speaking_v2  # noqa: E402
from services.scoring.azure_scorer import _compute_question_score  # noqa: E402

ATTEMPT_ID = 15698
USER_ID = 121
AWS_PROFILE = "englishfirm"
AWS_REGION = "ap-southeast-2"
PTE_FLOOR, PTE_CEILING = 10, 90
WARNING = "scoring_infrastructure_timeout"

APPLY = "--apply" in sys.argv


def _s3_bytes(audio_url: str) -> bytes:
    parsed = urllib.parse.urlparse(audio_url)
    bucket = parsed.netloc.split(".")[0]
    key = parsed.path.lstrip("/")
    s3 = boto3.session.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION).client("s3")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _pte_from_pct(pct: float) -> int:
    return max(PTE_FLOOR, min(PTE_CEILING, int(round(10 + (pct or 0.0) * 80))))


def _inputs_for(qtype: str, q: QuestionFromApeuni):
    """Mirror the live submit routers' input derivation."""
    content = q.content_json or {} if q else {}
    ev = (q.evaluation.evaluation_json or {}) if (q and q.evaluation) else {}
    ca = ev.get("correctAnswers", {}) or {}
    if qtype == "answer_short_question":
        expected = ca.get("acceptedVariants") or ([ca["answer"]] if ca.get("answer") else [])
        return {"expected_answers": expected}
    # retell_lecture / summarize_group_discussion / ptea_respond_situation
    return {
        "key_points": ca.get("keyPoints") or [],
        "stimulus_audio_url": (content.get("audio_url") or "").strip(),
    }


def main():
    db = SessionLocal()
    rows = (
        db.query(AttemptAnswer)
        .filter(
            AttemptAnswer.attempt_id == ATTEMPT_ID,
            AttemptAnswer.result_json["scoring_warnings"].astext.contains(WARNING),
        )
        .order_by(AttemptAnswer.id)
        .all()
    )
    print(f"Found {len(rows)} timed-out answers in attempt {ATTEMPT_ID}")

    updates = []
    for r in rows:
        qtype = r.question_type
        q = (
            db.query(QuestionFromApeuni)
            .filter(QuestionFromApeuni.question_id == r.question_id)
            .first()
        )
        kwargs = _inputs_for(qtype, q)
        print(f"\n=== q={r.question_id} aid={r.id} type={qtype} ===")
        print(f"  inputs: { {k: (len(v) if isinstance(v, list) else v) for k, v in kwargs.items()} }")
        if not r.audio_url:
            print("  !! no audio_url on this row — SKIP")
            continue
        try:
            audio_bytes = _s3_bytes(r.audio_url)
        except Exception as e:
            print(f"  !! S3 fetch failed: {e} — SKIP")
            continue

        result = _score_speaking_v2(
            user_id=USER_ID,
            question_id=r.question_id,
            audio_bytes=audio_bytes,
            reference_text="",
            task_type=qtype,
            **kwargs,
        )
        new_c = float(result.get("content", 0))
        new_f = float(result.get("fluency", 0))
        new_p = float(result.get("pronunciation", 0))
        raw = {
            "content": new_c, "fluency": new_f, "pronunciation": new_p,
            "scoring": "complete",
            "content_llm_scored": (result.get("fluency_metrics", {}) or {}).get("content_method") == "llm_keypoints",
        }
        qscore = _compute_question_score(qtype, raw)
        new_pte = _pte_from_pct(qscore["pct"])
        print(f"  OLD: pte={r.score} (timed out) → NEW: pte={new_pte} c={new_c:.1f} f={new_f:.1f} p={new_p:.1f}")
        print(f"       transcript: {(result.get('transcript','') or '')[:80]!r}")
        updates.append((r, new_c, new_f, new_p, new_pte, result))

    print("\n=== SUMMARY ===")
    print(f"{'aid':>6} {'qid':>6} {'type':28} | {'new_pte':>7} {'c':>5} {'f':>5} {'p':>5}")
    for r, nc, nf, np_, npte, _ in updates:
        print(f"{r.id:>6} {r.question_id:>6} {r.question_type:28} | {npte:>7} {nc:>5.1f} {nf:>5.1f} {np_:>5.1f}")

    if not APPLY:
        print("\n[DRY RUN] No DB writes. Re-run with --apply to commit.")
        db.close()
        return

    print("\n[APPLY] Writing to RDS...")
    for r, nc, nf, np_, npte, result in updates:
        rj = dict(r.result_json or {})
        # clear the timeout warning; record the recovery
        rj["scoring_warnings"] = [w for w in (rj.get("scoring_warnings") or []) if w != WARNING]
        rj.pop("scoring_failed_at", None)
        rj.update({
            "content": nc, "fluency": nf, "pronunciation": np_,
            "total": npte, "pte_score": npte,
            "transcript": result.get("transcript", ""),
            "word_scores": result.get("word_scores", []),
            "fluency_metrics": result.get("fluency_metrics", {}),
            "rescored_2026_08_22": {"reason": "recover scoring_infrastructure_timeout (attempt 15698)"},
        })
        r.content_score = nc
        r.fluency_score = nf
        r.pronunciation_score = np_
        r.score = npte
        r.result_json = rj
        flag_modified(r, "result_json")

    attempt = db.query(PracticeAttempt).filter(PracticeAttempt.id == ATTEMPT_ID).first()
    total = (
        db.query(AttemptAnswer)
        .with_entities(AttemptAnswer.score)
        .filter(AttemptAnswer.attempt_id == ATTEMPT_ID)
        .all()
    )
    attempt.total_score = sum(s[0] or 0 for s in total)
    attempt.scoring_status = "complete"
    db.commit()
    print(f"  attempt {ATTEMPT_ID} total_score → {attempt.total_score}, scoring_status → complete")
    print("\nDONE.")
    db.close()


if __name__ == "__main__":
    main()
