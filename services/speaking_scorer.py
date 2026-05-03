"""
Shared background scoring for all speaking practice types.
Downloads audio from S3, runs Azure scoring, applies rubric weights + PTE formula.
Ported from englishfirm-app-fastapi/services/question_service.py _score_free_form_bg.
"""
import logging
import re
import threading
import requests as _requests

import core.config as config

logger = logging.getLogger(__name__)
from services.s3_service import generate_presigned_url
from services.session_service import store_score, update_speaking_score_in_db
from services.scoring.azure_scorer import _compute_question_score

from core.logging_config import get_logger

log = get_logger(__name__)


def _pte_score(pct: float) -> int:
    return max(config.PTE_FLOOR, min(config.PTE_CEILING, round(config.PTE_BASE + pct * config.PTE_SCALE)))


_WPM_RULE_TYPES = {
    'read_aloud',
    'repeat_sentence',
    'describe_image',
    'retell_lecture',
    'respond_to_situation',
    'ptea_respond_situation',
    'summarize_group_discussion',
}


def _maybe_apply_wpm_rule_speaking(
    user_id: int,
    question_id: int,
    question_type: str,
    audio_bytes: bytes,
    word_scores: list,
    content: float,
    fluency: float,
    pronunciation: float,
):
    """
    Apply wpm_scoring_rules to speaking subscores for the question types in
    _WPM_RULE_TYPES. Covers practice + sectional + mock uniformly — there is
    no parent-attempt gate; the rule applies any time one of these question
    types is scored.

    Words counted: word_scores entries where error_type is neither
    'Omission' nor 'Insertion'. Fail-open on any error.

    answer_short_question is excluded — its scoring is binary (correct/
    incorrect content only) and has no fluency/pronunciation dimension to
    penalise.
    """
    if question_type not in _WPM_RULE_TYPES:
        return content, fluency, pronunciation
    try:
        from db.database import SessionLocal
        from sqlalchemy import text as sql_text
        from pydub import AudioSegment
        import io

        words_spoken = sum(
            1 for w in (word_scores or [])
            if isinstance(w, dict) and w.get('error_type') not in ('Omission', 'Insertion')
        )
        if words_spoken <= 0:
            return content, fluency, pronunciation

        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        duration_sec = seg.duration_seconds
        if duration_sec <= 0:
            return content, fluency, pronunciation

        wpm = words_spoken * 60.0 / duration_sec

        db = SessionLocal()
        try:
            row = db.execute(sql_text("""
                SELECT id, mode, penalty, label
                FROM wpm_scoring_rules
                WHERE (min_wpm IS NULL OR :wpm >= min_wpm)
                  AND (max_wpm IS NULL OR :wpm <  max_wpm)
                ORDER BY id
                LIMIT 1
            """), {"wpm": wpm}).first()
        finally:
            db.close()

        if row is None:
            return content, fluency, pronunciation

        log.info(
            "[WPM] q=%s type=%s user=%s words=%s dur=%.2fs wpm=%.1f rule=%s(%s) mode=%s penalty=%s",
            question_id, question_type, user_id, words_spoken, duration_sec, wpm,
            row.id, row.label, row.mode, row.penalty,
        )

        if row.mode == 'zero_out':
            return 0.0, 0.0, 0.0
        if row.mode == 'subtract':
            p = float(row.penalty)
            return (
                max(0.0, float(content) - p),
                max(0.0, float(fluency) - p),
                max(0.0, float(pronunciation) - p),
            )
        return content, fluency, pronunciation
    except Exception as e:
        log.error("[WPM] rule application failed (fail-open): %s", e)
        return content, fluency, pronunciation


def _get_stimulus_key_points(question_type: str, audio_url: str) -> list:
    """Transcribe stimulus audio + GPT-extract key points for RL/RTS/SGD."""
    try:
        from services.azure_speech_service import transcribe_audio_full
        from services.llm_content_scoring_service import extract_key_points
        presigned = generate_presigned_url(audio_url)
        resp = _requests.get(presigned, timeout=30)
        resp.raise_for_status()
        transcript = transcribe_audio_full(resp.content)
        if transcript:
            return extract_key_points(transcript, question_type)
    except Exception as e:
        log.error(f"[SCORER] Stimulus key-point extraction failed ({question_type}): {e}")
    return []


def _run_scoring(
    user_id: int,
    question_id: int,
    question_type: str,
    audio_url: str,
    reference_text: str = "",
    key_points: list = None,
    expected_answers: list = None,
    stimulus_audio_url: str = "",
):
    from services.azure_speech_service import score_read_aloud, transcribe_and_score_free

    if key_points is None:
        key_points = []
    if expected_answers is None:
        expected_answers = []

    try:
        presigned = generate_presigned_url(audio_url)
        audio_resp = _requests.get(presigned, timeout=30)
        audio_resp.raise_for_status()
        audio_bytes = audio_resp.content

        content = 0.0
        fluency = 0.0
        pronunciation = 0.0
        transcript = ""
        word_scores = []
        content_llm_scored = False
        extra = {}

        if question_type in ("read_aloud", "repeat_sentence") and reference_text:
            raw = score_read_aloud(audio_bytes, reference_text)
            content       = raw["content"]
            fluency       = raw["fluency"]
            pronunciation = raw["pronunciation"]
            transcript    = raw.get("transcript", "")
            word_scores   = raw.get("word_scores", [])

        elif question_type == "answer_short_question":
            raw = transcribe_and_score_free(audio_bytes)
            transcript    = raw.get("transcript", "")
            fluency       = raw.get("fluency", 0)
            pronunciation = raw.get("pronunciation", 0)
            word_scores   = raw.get("word_scores", [])
            is_correct = False
            if expected_answers:
                t_lower = transcript.lower().strip()
                # Match each accepted variant as a whole phrase with word
                # boundaries. Any variant match → correct. The DB-provided
                # acceptedVariants list already covers article/phrasing
                # alternatives ("diaper", "a diaper", "the diaper"), so we
                # don't need to split the variant into per-word tokens —
                # which would re-introduce false positives like "a sheep"
                # matching "a banana" on the article alone.
                for ans in expected_answers:
                    ans_lower = ans.lower().strip()
                    if not ans_lower:
                        continue
                    if re.search(rf'\b{re.escape(ans_lower)}\b', t_lower):
                        is_correct = True
                        break
            content = 100.0 if is_correct else 0.0
            extra = {"is_correct": is_correct}

        elif question_type == "describe_image":
            raw = transcribe_and_score_free(audio_bytes)
            transcript    = raw.get("transcript", "")
            fluency       = raw.get("fluency", 0)
            pronunciation = raw.get("pronunciation", 0)
            word_scores   = raw.get("word_scores", [])
            if key_points and transcript:
                from services.llm_content_scoring_service import score_content_with_llm
                content = float(score_content_with_llm(transcript, key_points, question_type))
                content_llm_scored = True
            else:
                content = min(len(transcript.split()) / 40, 1.0) * 50.0 if transcript else 0.0

        else:
            # RL, RTS, SGD
            raw = transcribe_and_score_free(audio_bytes)
            transcript    = raw.get("transcript", "")
            fluency       = raw.get("fluency", 0)
            pronunciation = raw.get("pronunciation", 0)
            word_scores   = raw.get("word_scores", [])
            if not key_points and stimulus_audio_url:
                key_points = _get_stimulus_key_points(question_type, stimulus_audio_url)
            if key_points and transcript:
                from services.llm_content_scoring_service import score_content_with_llm
                content = float(score_content_with_llm(transcript, key_points, question_type))
                content_llm_scored = True
            else:
                _targets = {
                    "retell_lecture": 60,
                    "respond_to_situation": 30,
                    "summarize_group_discussion": 50,
                }
                target = _targets.get(question_type, 40)
                content = min(len(transcript.split()) / target, 1.0) * 50.0 if transcript else 0.0

        # WPM rule: applies to RA / RS / DI / RL / RTS / SGD across practice +
        # sectional + mock. No-op for ASQ. See _WPM_RULE_TYPES + wpm_scoring_rules.
        content, fluency, pronunciation = _maybe_apply_wpm_rule_speaking(
            user_id=user_id,
            question_id=question_id,
            question_type=question_type,
            audio_bytes=audio_bytes,
            word_scores=word_scores,
            content=content,
            fluency=fluency,
            pronunciation=pronunciation,
        )

        # Rubric-weighted PTE score (uniform for all types)
        computed = _compute_question_score(question_type, {
            "content": content,
            "fluency": fluency,
            "pronunciation": pronunciation,
            "content_llm_scored": content_llm_scored,
            **extra,
        })
        pte = _pte_score(computed["pct"])

        store_score(user_id, question_id, {
            "scoring":       "complete",
            "content":       round(content, 1),
            "fluency":       fluency,
            "pronunciation": pronunciation,
            "total":         max(config.PTE_FLOOR, pte),
            "transcript":    transcript,
            "word_scores":   word_scores,
            **extra,
        })
        update_speaking_score_in_db(
            user_id=user_id,
            question_id=question_id,
            content=round(content, 1),
            pronunciation=pronunciation,
            fluency=fluency,
            total=pte,
            transcript=transcript,
            word_scores=word_scores,
        )
        log.info(f"[SCORER] q={question_id} type={question_type} content={content:.1f} " f"fluency={fluency} pronunciation={pronunciation} pte={pte}")
        if question_type == "answer_short_question":
            log.info(f"[ASQ] q={question_id} " f"user_said={transcript!r} " f"expected={expected_answers!r} " f"is_correct={extra.get('is_correct')} " f"pte={pte}")

    except Exception as e:
        store_score(user_id, question_id, {"scoring": "error", "error": str(e)})
        if "no speech recognised" in str(e).lower():
            logger.warning(
                "[SCORER] user=%s question=%s type=%s no speech recognised",
                user_id, question_id, question_type,
            )
        else:
            logger.error(
                "[SCORER ERROR] user=%s question=%s type=%s exception=%s: %s",
                user_id, question_id, question_type, type(e).__name__, e,
            )
        # Always mark AttemptAnswer complete so background aggregation is never blocked
        update_speaking_score_in_db(
            user_id=user_id,
            question_id=question_id,
            content=0.0,
            pronunciation=0.0,
            fluency=0.0,
            total=0,
            transcript="",
            word_scores=[],
        )


def kick_off_scoring(
    user_id: int,
    question_id: int,
    question_type: str,
    audio_url: str,
    reference_text: str = "",
    key_points: list = None,
    expected_answers: list = None,
    stimulus_audio_url: str = "",
):
    t = threading.Thread(
        target=_run_scoring,
        args=(user_id, question_id, question_type, audio_url,
              reference_text, key_points, expected_answers, stimulus_audio_url),
        daemon=True,
    )
    t.start()
