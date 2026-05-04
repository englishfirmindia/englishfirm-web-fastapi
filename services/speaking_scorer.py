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


_FLUENCY_FORMULA_TYPES = {
    'read_aloud',
    'repeat_sentence',
    'describe_image',
    'retell_lecture',
    'respond_to_situation',
    'ptea_respond_situation',
    'summarize_group_discussion',
}


def _silence_ratio_pct(seg) -> float:
    """
    Return the within-speech silence ratio as a percentage [0, 100].
    Excludes leading and trailing silence (those aren't pauses, just dead air
    before/after the user spoke). Uses pydub silence detection at -30 dBFS,
    minimum 300 ms — same parameters used in our diagnostic ffmpeg path.
    """
    try:
        from pydub.silence import detect_silence
    except Exception:
        return 0.0
    total_ms = len(seg)
    if total_ms <= 0:
        return 0.0
    # detect_silence returns [start_ms, end_ms] for each silent stretch
    silences = detect_silence(seg, min_silence_len=300, silence_thresh=-30)
    if not silences:
        return 0.0
    total_silence_ms = sum(end - start for start, end in silences)
    # leading: silence at the very start (within 100 ms)
    leading_ms = (silences[0][1] - silences[0][0]) if silences[0][0] <= 100 else 0
    # if a near-continuous second silence follows the leading one (gap < 50 ms), include it too
    if (
        len(silences) >= 2
        and silences[0][0] <= 100
        and silences[1][0] - silences[0][1] < 50
    ):
        leading_ms += silences[1][1] - silences[1][0]
    # trailing: silence ending within 100 ms of audio end
    trailing_ms = (silences[-1][1] - silences[-1][0]) if (total_ms - silences[-1][1]) <= 100 else 0
    within_ms = total_silence_ms - leading_ms - trailing_ms
    speech_window_ms = total_ms - leading_ms - trailing_ms
    if speech_window_ms <= 0:
        return 0.0
    return max(0.0, min(100.0, within_ms / speech_window_ms * 100.0))


def _apply_speaking_fluency_formula(
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
    Replace Azure's FluencyScore with a deterministic formula based on WPM
    and within-speech silence ratio. Content (CompletenessScore) and
    pronunciation (AccuracyScore) pass through Azure-as-is.

    Formula:
        if silence_pct > 20  OR  wpm < 100  OR  wpm > 200:
            fluency = 0
        else:
            wpm_score      = 100 - 2.5 * (140 - wpm)   if 100 <= wpm < 140
                             100                        if 140 <= wpm <= 180
                             100 - 5 * (wpm - 180)      if 180 < wpm <= 200
            silence_score  = 5 * (20 - silence_pct)
            fluency        = min(wpm_score, silence_score)

    Words counted for WPM: word_scores entries where error_type is neither
    'Omission' nor 'Insertion'. Same rule as before.

    Silence ratio: within-speech (excludes leading/trailing dead air) via
    pydub.silence.detect_silence at -30 dBFS, min 300 ms. Same as before.

    Coverage: types in _FLUENCY_FORMULA_TYPES, across practice + sectional + mock.
    answer_short_question is excluded.

    NOTE: wpm_scoring_rules and silence_ratio_rules tables are no longer
    queried by this code path. They are kept in DB for easy revert.

    Fail-open: any error keeps Azure's original c/f/p (logged).
    """
    if question_type not in _FLUENCY_FORMULA_TYPES:
        return content, fluency, pronunciation
    try:
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
        silence_pct = _silence_ratio_pct(seg)

        # Hard-fail conditions
        if silence_pct > 20.0 or wpm < 100.0 or wpm > 200.0:
            new_fluency = 0.0
            wpm_score_str = "-"
            sil_score_str = "-"
            reason_parts = []
            if silence_pct > 20.0:
                reason_parts.append(f"sil>{silence_pct:.1f}%")
            if wpm < 100.0:
                reason_parts.append(f"wpm<100({wpm:.1f})")
            if wpm > 200.0:
                reason_parts.append(f"wpm>200({wpm:.1f})")
            reason = ",".join(reason_parts)
        else:
            if wpm < 140.0:
                wpm_score = 100.0 - 2.5 * (140.0 - wpm)
            elif wpm <= 180.0:
                wpm_score = 100.0
            else:  # 180 < wpm <= 200
                wpm_score = 100.0 - 5.0 * (wpm - 180.0)

            silence_score = 5.0 * (20.0 - silence_pct)
            new_fluency = min(wpm_score, silence_score)
            new_fluency = max(0.0, min(100.0, new_fluency))
            wpm_score_str = f"{wpm_score:.1f}"
            sil_score_str = f"{silence_score:.1f}"
            reason = "ok"

        log.info(
            "[FLUENCY_FORMULA] q=%s type=%s user=%s words=%s dur=%.2fs "
            "wpm=%.1f sil=%.1f%% wpm_score=%s sil_score=%s "
            "azure_f=%.1f → new_f=%.1f reason=%s",
            question_id, question_type, user_id, words_spoken, duration_sec,
            wpm, silence_pct, wpm_score_str, sil_score_str,
            float(fluency), new_fluency, reason,
        )

        return float(content), new_fluency, float(pronunciation)
    except Exception as e:
        log.error("[FLUENCY_FORMULA] application failed (fail-open, keeping azure fluency): %s", e)
        return content, fluency, pronunciation


def _transcribe_azure_with_whisper_parallel(audio_bytes: bytes) -> dict:
    """
    Run Azure transcribe_and_score_free + Whisper transcription concurrently.
    Returns Azure's full result dict, with 'transcript' replaced by Whisper's
    output when available. Adds audit fields:
      - 'azure_transcript':   raw Azure transcript
      - 'whisper_transcript': raw Whisper transcript ('' if failed)
      - 'transcript_source':  'whisper' | 'azure_fallback' | 'none'

    Whisper runs in a daemon thread that joins after Azure's longer call
    returns. Net latency is max(Azure, Whisper) instead of sum.

    Used by the LLM-content-scored branches (DI, RL, RTS, ptea_RTS, SGD)
    where Azure's en-US ASR mishears domain vocabulary. Whisper's broader
    language model handles technical PTE terms (Radiata, chipper, fallout)
    and accented English better.
    """
    from services.azure_speech_service import transcribe_and_score_free

    holder = {}

    def _whisper_worker():
        try:
            from services.whisper_service import transcribe_with_whisper
            holder['whisper'] = transcribe_with_whisper(audio_bytes)
        except Exception as e:
            log.error("[WHISPER] worker thread failed: %s", e)
            holder['whisper'] = ""

    t = threading.Thread(target=_whisper_worker, daemon=True)
    t.start()

    azure_result = transcribe_and_score_free(audio_bytes)
    azure_transcript = azure_result.get('transcript', '') if isinstance(azure_result, dict) else ''

    # Cap Whisper wait at 15s in case the API hangs.
    t.join(timeout=15.0)
    whisper_transcript = holder.get('whisper', '')

    if whisper_transcript:
        chosen = whisper_transcript
        source = 'whisper'
    elif azure_transcript:
        chosen = azure_transcript
        source = 'azure_fallback'
    else:
        chosen = ''
        source = 'none'

    log.info(
        "[WHISPER] azure_len=%d whisper_len=%d source=%s",
        len(azure_transcript), len(whisper_transcript), source,
    )

    return {
        **(azure_result if isinstance(azure_result, dict) else {}),
        'transcript': chosen,
        'azure_transcript': azure_transcript,
        'whisper_transcript': whisper_transcript,
        'transcript_source': source,
    }


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
            # Whisper transcribes in parallel with Azure for better content
            # scoring on PTE-specific vocab + accented English. Azure still
            # provides word_scores, fluency, pronunciation as before.
            raw = _transcribe_azure_with_whisper_parallel(audio_bytes)
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
            # RL, RTS, ptea_RTS, SGD — Whisper-parallel transcription too.
            raw = _transcribe_azure_with_whisper_parallel(audio_bytes)
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

        # Speaking fluency formula replaces Azure's FluencyScore for the
        # 7 types in _FLUENCY_FORMULA_TYPES across practice + sectional + mock.
        # Content (CompletenessScore) and pronunciation (AccuracyScore) pass
        # through Azure-as-is. ASQ is excluded.
        content, fluency, pronunciation = _apply_speaking_fluency_formula(
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
