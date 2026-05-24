"""
SGD-specific fluency-only retrofill for Nimisha's mock 3782.

The generic retrofill script (re-running _score_speaking_v2) is risky for
SGD because (a) Claude LLM content scoring is non-deterministic and would
shift content_score on replay, and (b) Azure freeform can fail intermittently
and zero out pronunciation. Both happened in the dry-run.

This script avoids both issues: it re-detects pauses on the normalized audio
locally, then re-runs ONLY the fluency formula + cross-penalty using cached
wpm / sentence_count / content / raw-pronunciation values. Content is left
untouched. Pronunciation is recovered to its raw form (cached_p / cached_mP)
and re-dampened with the new cross-multiplier.

Usage:
    python3 scripts/retrofill_nimisha_mock_3782_sgd_fluency_only.py            # dry run
    python3 scripts/retrofill_nimisha_mock_3782_sgd_fluency_only.py --apply
"""
import os
import sys
import math
import urllib.parse
import boto3
import psycopg2
from psycopg2.extras import Json, RealDictCursor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from pydub import AudioSegment           # noqa: E402
from pydub.silence import detect_silence  # noqa: E402
from services.speaking_scorer import _normalize_for_silence_detection, _cross_multiplier  # noqa: E402
from services.scoring.speaking_config_service import get_speaking_config  # noqa: E402
from services.scoring.azure_scorer import _compute_question_score  # noqa: E402

ATTEMPT_ID = 3782
TASK_TYPE = "summarize_group_discussion"
APPLY = "--apply" in sys.argv
AWS_PROFILE = "englishfirm"
AWS_REGION = "ap-southeast-2"
PTE_FLOOR, PTE_CEILING = 10, 90


def _dsn():
    return os.environ["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://")


def _s3_bytes(audio_url: str) -> bytes:
    parsed = urllib.parse.urlparse(audio_url)
    bucket = parsed.netloc.split(".")[0]
    key = parsed.path.lstrip("/")
    s3 = boto3.session.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION).client("s3")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _redetect_gap_pauses(audio_bytes: bytes, cfg) -> int:
    """Run normalized pause detection and return within-speech pause count."""
    seg = AudioSegment.from_file(__import__("io").BytesIO(audio_bytes))
    seg_norm, _ = _normalize_for_silence_detection(seg)
    sils = detect_silence(
        seg_norm,
        min_silence_len=cfg.pause_min_ms,
        silence_thresh=cfg.silence_thresh_dbfs,
    )
    total_ms = len(seg)
    gap = 0
    for s, e in sils:
        if s <= cfg.pause_leading_tol_ms:
            continue
        if (total_ms - e) <= cfg.pause_trailing_tol_ms:
            continue
        gap += 1
    return gap


def _pause_penalty(total_pauses: int, sentence_count: int, cfg) -> float:
    s_clamped = min(
        max(sentence_count, cfg.pause_penalty_sentence_clamp_min),
        cfg.pause_penalty_sentence_clamp_max,
    )
    if cfg.pause_penalty_max_pauses_mult is not None:
        max_pauses = max(1, math.ceil(sentence_count * cfg.pause_penalty_max_pauses_mult))
        formula_const = max_pauses + 1
    else:
        max_pauses = cfg.pause_penalty_max_pauses
        formula_const = cfg.pause_penalty_formula_constant
    if total_pauses < s_clamped:
        return 100.0
    if total_pauses <= max_pauses:
        return max(0.0, min(100.0,
            100.0 * (max_pauses - total_pauses) / (formula_const - s_clamped)
        ))
    return 0.0


def _wpm_band(wpm: float, cfg) -> float:
    if wpm < cfg.wpm_floor or wpm > cfg.wpm_ceiling or wpm <= 0:
        return 0.0
    if wpm < cfg.wpm_plateau_low:
        return cfg.wpm_peak_score - cfg.wpm_slope_per_wpm * (cfg.wpm_plateau_low - wpm)
    if wpm <= cfg.wpm_plateau_high:
        return cfg.wpm_peak_score
    return max(0.0, cfg.wpm_peak_score - cfg.wpm_slope_per_wpm * (wpm - cfg.wpm_plateau_high))


def _pte_from_pct(pct: float) -> int:
    return max(PTE_FLOOR, min(PTE_CEILING, int(round(10 + (pct or 0.0) * 80))))


def main():
    cfg = get_speaking_config(TASK_TYPE)
    if cfg is None:
        print(f"No speaking config row for task_type={TASK_TYPE}")
        sys.exit(1)
    print(f"cfg: wpm_floor={cfg.wpm_floor} ceiling={cfg.wpm_ceiling} "
          f"pause_min={cfg.pause_min_ms}ms thresh={cfg.silence_thresh_dbfs}dBFS "
          f"max_mult={cfg.pause_penalty_max_pauses_mult}")
    print(f"cross-penalty: uses={cfg.uses_cross_penalty} "
          f"thresh={cfg.pronunciation_content_threshold} "
          f"floor={cfg.pronunciation_content_floor} "
          f"slope={cfg.pronunciation_content_slope}")

    conn = psycopg2.connect(_dsn())
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute(
        """
        SELECT aa.id, aa.question_id, aa.audio_url,
               aa.score, aa.content_score, aa.fluency_score, aa.pronunciation_score,
               aa.result_json
        FROM attempt_answers aa
        WHERE aa.attempt_id = %s AND aa.question_type = %s
        ORDER BY aa.id
        """,
        (ATTEMPT_ID, TASK_TYPE),
    )
    rows = cur.fetchall()
    print(f"\nFound {len(rows)} {TASK_TYPE} answers")

    updates = []
    for r in rows:
        aid = r["id"]; qid = r["question_id"]
        rj = r["result_json"] or {}
        fm = rj.get("fluency_metrics", {})
        cached_wpm = float(fm.get("wpm", 0))
        cached_sentences = int(fm.get("sentence_count", 1))
        cached_hesitations = int(fm.get("hesitation_count", 0))
        cached_gap_pauses = int(fm.get("gap_pauses", 0))
        cached_xm = fm.get("cross_multipliers", {}) or {}
        cached_mP = float(cached_xm.get("mP", 1.0))

        content = float(r["content_score"])
        old_fluency = float(r["fluency_score"])
        old_pronunciation = float(r["pronunciation_score"])
        raw_pronunciation = old_pronunciation / cached_mP if cached_mP > 0 else old_pronunciation

        print(f"\n=== q={qid} aid={aid} ===")
        print(f"  OLD: pte={r['score']} c={content} f={old_fluency} p={old_pronunciation}  "
              f"(wpm={cached_wpm:.1f} sentences={cached_sentences} "
              f"hesitations={cached_hesitations} gap_pauses={cached_gap_pauses} mP={cached_mP})")
        print(f"  recovered raw_p (= p / mP): {raw_pronunciation:.2f}")

        # New pause detection on normalized audio
        audio_bytes = _s3_bytes(r["audio_url"])
        new_gap_pauses = _redetect_gap_pauses(audio_bytes, cfg)
        new_total_pauses = new_gap_pauses + cached_hesitations

        # New scores
        wpm_band = _wpm_band(cached_wpm, cfg)
        pause_pen = _pause_penalty(new_total_pauses, cached_sentences, cfg)
        new_fluency = 0.0 if cached_wpm < cfg.wpm_floor or cached_wpm > cfg.wpm_ceiling else min(wpm_band, pause_pen)

        # New cross-multiplier (only mP for SGD)
        new_mP = 1.0
        if cfg.uses_cross_penalty:
            new_mP = _cross_multiplier(
                min(content, new_fluency),
                cfg.pronunciation_content_threshold or 100.0,
                cfg.pronunciation_content_floor or 0.5,
                cfg.pronunciation_content_slope or 0.005,
            )
        new_pronunciation = max(0.0, raw_pronunciation * new_mP)

        # PTE total via production rubric
        raw_for_total = {
            "content": content, "fluency": new_fluency, "pronunciation": new_pronunciation,
            "scoring": "complete",
            "content_llm_scored": fm.get("content_method") == "llm_keypoints",
        }
        q = _compute_question_score(TASK_TYPE, raw_for_total)
        new_pte = _pte_from_pct(q["pct"])

        print(f"  pauses: gap {cached_gap_pauses} → {new_gap_pauses}  "
              f"total {fm.get('total_pauses')} → {new_total_pauses}")
        print(f"  wpm_band={wpm_band:.1f} pause_penalty={pause_pen:.1f} mP={new_mP:.3f}")
        print(f"  NEW: pte={new_pte} c={content:.1f} f={new_fluency:.1f} p={new_pronunciation:.1f}")

        new_fm = {
            **fm,
            "gap_pauses": new_gap_pauses,
            "total_pauses": new_total_pauses,
            "pause_penalty_score": round(pause_pen, 1),
            "wpm_band_score": round(wpm_band, 1),
            "cross_multipliers": {"mC": 1.0, "mF": 1.0, "mP": round(new_mP, 3)},
        }
        new_rj = {
            **rj,
            "fluency": new_fluency,
            "pronunciation": new_pronunciation,
            "total": new_pte,
            "pte_score": new_pte,
            "fluency_metrics": new_fm,
            "retrofill_2026_05_24_sgd": {
                "reason": "Fluency-only retrofill after loudness-normalized pause detection",
                "old": {
                    "score": int(r["score"]),
                    "fluency": old_fluency,
                    "pronunciation": old_pronunciation,
                    "gap_pauses": cached_gap_pauses,
                    "mP": cached_mP,
                },
            },
        }
        updates.append((aid, content, new_fluency, new_pronunciation, new_pte, new_rj))

    print("\n=== SUMMARY ===")
    print(f"{'aid':>5} {'qid':>5} | {'old_pte':>7} {'old_f':>6} {'old_p':>6} → {'new_pte':>7} {'new_f':>6} {'new_p':>6}  (content unchanged)")
    for aid, c, nf, np_, npte, _ in updates:
        old = next(r for r in rows if r["id"] == aid)
        print(
            f"{aid:>5} {old['question_id']:>5} | "
            f"{old['score']:>7} {float(old['fluency_score']):>6.1f} {float(old['pronunciation_score']):>6.1f} → "
            f"{npte:>7} {nf:>6.1f} {np_:>6.1f}"
        )

    if not APPLY:
        print("\n[DRY RUN] No DB updates. Re-run with --apply to commit.")
        return

    print("\n[APPLY] Writing to RDS...")
    for aid, c, nf, np_, npte, nrj in updates:
        cur.execute(
            """
            UPDATE attempt_answers
            SET fluency_score = %s, pronunciation_score = %s,
                score = %s, result_json = %s
            WHERE id = %s
            """,
            (nf, np_, npte, Json(nrj), aid),
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
