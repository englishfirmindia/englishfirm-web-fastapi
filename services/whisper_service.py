"""
Thin wrapper around OpenAI Whisper-1 for English speech transcription.

Used by services.speaking_scorer for content-scored question types
(DI, RL, RTS, ptea_RTS, SGD) where Azure's en-US ASR struggles with
domain-specific PTE vocabulary (e.g. "Radiata", "chipper", "fallout"),
particularly under accented English.

Whisper output replaces Azure's transcript ONLY for the LLM key-point
scoring step. Azure is still called for pronunciation, word_scores,
and fluency/word-level data — Whisper does not provide those.

Fail-open: any exception returns empty string, which the caller treats
as a fallback signal to use the Azure transcript.
"""
import io
import os

from openai import OpenAI

from core.logging_config import get_logger

log = get_logger(__name__)

_client = None


def _get_client():
    """Lazy-init the OpenAI client. Reads OPENAI_API_KEY from env."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client


def transcribe_with_whisper(audio_bytes: bytes, language: str = "en") -> str:
    """
    Send audio bytes to Whisper-1 and return the recognized text.

    Returns "" on any failure (fail-open). Caller is responsible for
    falling back to an alternate transcript source.

    Whisper API accepts: flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav,
    webm. Raw AAC (which is what the Flutter clients upload) is NOT in
    that list — the API returns 400 for ".aac" filename. We convert AAC
    to 16 kHz mono WAV via pydub/ffmpeg first. The conversion is fast
    (~300-500 ms) and runs in the parallel Whisper thread anyway, so it
    doesn't add to wall-clock latency.
    """
    if not audio_bytes:
        return ""
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # 16 kHz mono is the format Whisper internally uses; reduces upload size.
        seg = seg.set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        seg.export(wav_io, format="wav")
        wav_io.seek(0)
        wav_io.name = "audio.wav"  # filename hint for the API

        result = _get_client().audio.transcriptions.create(
            model="whisper-1",
            file=wav_io,
            language=language,
            response_format="text",
        )
        text = str(result).strip()
        log.info("[WHISPER] ok len=%d preview=%r", len(text), text[:80])
        return text
    except Exception as e:
        log.error("[WHISPER] transcription failed (fail-open): %s", e)
        return ""
