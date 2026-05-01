"""
Azure Cognitive Services — Pronunciation Assessment + STT
Used for Read Aloud, Repeat Sentence, and other speaking tasks.

Score scale: all values returned by this module are 0-100 (raw Azure scale).
speaking_scoring_service divides by 100.0 to normalise.
"""

import os
import json
import time
import hmac
import base64
import hashlib
import logging
import threading
import concurrent.futures
import requests
import urllib.parse
from typing import Optional

from core.logging_config import get_logger

log = get_logger(__name__)


logger = logging.getLogger(__name__)

AZURE_SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "australiaeast")


def _get_stt_token() -> str:
    """Exchange subscription key for a short-lived bearer token."""
    url = f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    last_exc: Exception = RuntimeError("_get_stt_token: no attempts made")
    for attempt in range(1, 4):
        try:
            r = requests.post(url, headers={"Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY}, timeout=10)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_exc = exc
            logger.warning("[AZURE] _get_stt_token attempt=%d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2)
    raise last_exc


def _aac_to_wav_pcm(aac_bytes: bytes) -> bytes:
    """Convert AAC bytes to 16kHz mono WAV/PCM using pydub + ffmpeg."""
    import io
    from pydub import AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(aac_bytes), format="aac")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def _any_audio_to_wav_pcm(audio_bytes: bytes) -> bytes:
    """Convert any audio format (MP3, AAC, M4A, WAV…) to 16kHz mono WAV/PCM.
    Used for stimulus transcription where files may be MP3 (RL/SGD) or AAC (RS).
    """
    import io
    from pydub import AudioSegment
    # Let pydub/ffmpeg auto-detect the format
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def assess_pronunciation(
    audio_bytes: bytes,
    reference_text: str,
    language: str = "en-US",
    granularity: str = "Word",
    enable_miscue: bool = True,
) -> dict:
    """
    Use Azure Speech SDK for pronunciation assessment.
    Returns a normalised dict with AccuracyScore, FluencyScore, and word-level results.
    Retries up to 3 times on any exception; re-raises after all attempts fail.
    """
    if not AZURE_SPEECH_KEY:
        raise RuntimeError("AZURE_SPEECH_KEY not configured")

    import azure.cognitiveservices.speech as speechsdk

    last_exc: Exception = RuntimeError("assess_pronunciation: no attempts made")
    for attempt in range(1, 4):
        try:
            wav_bytes = _any_audio_to_wav_pcm(audio_bytes)

            speech_config = speechsdk.SpeechConfig(
                subscription=AZURE_SPEECH_KEY,
                region=AZURE_SPEECH_REGION,
            )
            speech_config.speech_recognition_language = language

            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Word,
                enable_miscue=enable_miscue,
            )

            audio_stream = speechsdk.audio.PushAudioInputStream()
            audio_config  = speechsdk.audio.AudioConfig(stream=audio_stream)
            recognizer    = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            pronunciation_config.apply_to(recognizer)

            audio_stream.write(wav_bytes)
            audio_stream.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(recognizer.recognize_once)
                try:
                    result = future.result(timeout=180)
                except concurrent.futures.TimeoutError:
                    raise RuntimeError("Azure: recognize_once timed out after 180s")

            if result.reason == speechsdk.ResultReason.NoMatch:
                raise RuntimeError("Azure: no speech recognised")
            if result.reason == speechsdk.ResultReason.Canceled:
                raise RuntimeError(f"Azure cancelled: {result.cancellation_details.reason}")

            pa_result = speechsdk.PronunciationAssessmentResult(result)

            words = []
            for w in pa_result.words:
                words.append({
                    "Word":          w.word,
                    "AccuracyScore": w.accuracy_score,
                    "ErrorType":     w.error_type,
                })

            raw_json = json.loads(result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult, "{}"
            ))
            pa_json = raw_json.get("NBest", [{}])[0].get("PronunciationAssessment", {})
            completeness = float(pa_json.get("CompletenessScore", 0) or 0)

            return {
                "AccuracyScore":     pa_result.accuracy_score,
                "FluencyScore":      pa_result.fluency_score,
                "CompletenessScore": completeness,
                "recognized_text":   result.text,
                "Words":             words,
            }

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[AZURE] assess_pronunciation attempt=%d/3 failed — %s: %s",
                attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)

    logger.error(
        "[AZURE] assess_pronunciation failed after 3 attempts — "
        "scoring_status remains pending. exception=%s: %s",
        type(last_exc).__name__, last_exc,
    )
    raise last_exc


def transcribe_and_score_free(audio_bytes: bytes) -> dict:
    """
    Free-form STT + fluency/pronunciation scoring (no reference text).
    Used for Describe Image, Retell Lecture, Respond to Situation.
    Returns:
        {
            "transcript":    str,
            "fluency":       float (0-100),
            "pronunciation": float (0-100),
            "word_scores":   list,
        }
    Raises on failure — retries are handled inside assess_pronunciation().
    """
    azure_result = assess_pronunciation(
        audio_bytes=audio_bytes,
        reference_text="",
        enable_miscue=False,
    )
    accuracy_score = azure_result.get("AccuracyScore", 0)
    fluency_score  = azure_result.get("FluencyScore", 0)
    transcript     = azure_result.get("recognized_text", "")

    word_scores = []
    for w in azure_result.get("Words", []):
        word_scores.append({
            "word":       w.get("Word", ""),
            "error_type": w.get("ErrorType", "None"),
            "accuracy":   w.get("AccuracyScore", 0),
        })

    return {
        "transcript":    transcript,
        "fluency":       round(float(fluency_score), 1),
        "pronunciation": round(float(accuracy_score), 1),
        "word_scores":   word_scores,
    }


def score_read_aloud(audio_bytes: bytes, reference_text: str) -> dict:
    """
    Full Read Aloud / Repeat Sentence scoring pipeline.
    Returns all scores on 0-100 scale (raw Azure values, no scaling).
        {
            "content":       0-100,
            "pronunciation": 0-100,
            "fluency":       0-100,
            "total":         0-100,   # average of three
            "recognized_text": str,
            "word_scores":   [...],
        }
    """
    azure_result = assess_pronunciation(
        audio_bytes=audio_bytes,
        reference_text=reference_text,
        granularity="Word",
        enable_miscue=True,
    )

    accuracy_score = azure_result.get("AccuracyScore", 0)
    fluency_score  = azure_result.get("FluencyScore",  0)
    recognized     = azure_result.get("recognized_text", "")

    log.info(f"[AZURE] AccuracyScore={accuracy_score} FluencyScore={fluency_score} text='{recognized[:60]}'")

    pronunciation = round(float(accuracy_score), 1)
    fluency       = round(float(fluency_score), 1)
    content       = round(float(azure_result.get("CompletenessScore", 0) or 0), 1)
    total         = round((content + pronunciation + fluency) / 3, 1)

    word_scores = []
    for w in azure_result.get("Words", []):
        word_scores.append({
            "word":       w.get("Word", ""),
            "error_type": w.get("ErrorType", "None"),
            "accuracy":   w.get("AccuracyScore", 0),
        })

    return {
        "content":         content,
        "pronunciation":   pronunciation,
        "fluency":         fluency,
        "total":           total,
        "recognized_text": recognized,
        "word_scores":     word_scores,
    }


def transcribe_audio_short(audio_bytes: bytes) -> str:
    """
    Pure STT for short audio (< 30s, e.g. Repeat Sentence stimulus).
    Uses recognize_once() — stops on first silence.
    Returns transcript string. Retries up to 3 times with 180s timeout.
    """
    if not AZURE_SPEECH_KEY:
        return ""
    import azure.cognitiveservices.speech as speechsdk
    for attempt in range(1, 4):
        try:
            wav_bytes = _any_audio_to_wav_pcm(audio_bytes)
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            audio_stream  = speechsdk.audio.PushAudioInputStream()
            audio_config  = speechsdk.audio.AudioConfig(stream=audio_stream)
            recognizer    = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            audio_stream.write(wav_bytes)
            audio_stream.close()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(recognizer.recognize_once)
                try:
                    result = future.result(timeout=180)
                except concurrent.futures.TimeoutError:
                    raise RuntimeError("Azure: recognize_once timed out after 180s")
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            return ""
        except Exception as exc:
            logger.warning("[AZURE] transcribe_audio_short attempt=%d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2)
    logger.error("[AZURE] transcribe_audio_short failed after 3 attempts")
    return ""


def transcribe_audio_full(audio_bytes: bytes) -> str:
    """
    Pure STT for long audio (e.g. Retell Lecture ~90s, SGD ~180s).
    Uses continuous recognition so the full audio is transcribed.
    Returns full transcript string. Retries up to 3 times with 180s timeout.
    """
    if not AZURE_SPEECH_KEY:
        return ""
    import azure.cognitiveservices.speech as speechsdk
    for attempt in range(1, 4):
        try:
            wav_bytes = _any_audio_to_wav_pcm(audio_bytes)
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            audio_stream  = speechsdk.audio.PushAudioInputStream()
            audio_config  = speechsdk.audio.AudioConfig(stream=audio_stream)
            recognizer    = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

            all_text = []
            done_evt = threading.Event()

            def _recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    all_text.append(evt.result.text)

            def _stop(evt):
                done_evt.set()

            recognizer.recognized.connect(_recognized)
            recognizer.session_stopped.connect(_stop)
            recognizer.canceled.connect(_stop)

            audio_stream.write(wav_bytes)
            audio_stream.close()

            recognizer.start_continuous_recognition()
            done_evt.wait(timeout=180)
            recognizer.stop_continuous_recognition()

            return " ".join(all_text)
        except Exception as exc:
            logger.warning("[AZURE] transcribe_audio_full attempt=%d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2)
    logger.error("[AZURE] transcribe_audio_full failed after 3 attempts")
    return ""
