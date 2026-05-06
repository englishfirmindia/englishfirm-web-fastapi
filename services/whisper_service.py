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

# Whisper-1 silently strips disfluencies (um/uh/ah/er/erm) by default. PTE
# fluency scoring needs them surfaced. A short filler-rich `prompt=` biases
# the decoder so disfluencies stay in the transcript with word timestamps.
# Keep neutral so it doesn't bias toward any non-filler vocabulary.
_FILLER_PROMPT = "Umm, let me think. Uh, hmm, ah, er, erm. Aah."

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
            prompt=_FILLER_PROMPT,
        )
        text = str(result).strip()
        log.info("[WHISPER] ok len=%d preview=%r", len(text), text[:80])
        return text
    except Exception as e:
        log.error("[WHISPER] transcription failed (fail-open): %s", e)
        return ""


def transcribe_with_whisper_words(audio_bytes: bytes, language: str = "en") -> dict:
    """
    Transcribe with word-level timestamps. Returns:

      {
        "transcript": str,
        "words":      [{"text": str, "start": float, "end": float}, ...],
        "duration":   float,           # whisper-reported audio duration (seconds)
      }

    On any failure returns the same shape with empty values:

      {"transcript": "", "words": [], "duration": 0.0}

    Used by read_aloud v2 scoring path which needs per-word timing for
    pause detection (inter-word gap >= 500 ms) + speech-window WPM.
    """
    empty = {"transcript": "", "words": [], "duration": 0.0}
    if not audio_bytes:
        return empty
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        seg.export(wav_io, format="wav")
        wav_io.seek(0)
        wav_io.name = "audio.wav"

        result = _get_client().audio.transcriptions.create(
            model="whisper-1",
            file=wav_io,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            prompt=_FILLER_PROMPT,
        )
        # `result` is a Pydantic-like object; .words may be missing if Whisper
        # didn't return any (extremely short / silent clip).
        words_raw = getattr(result, "words", None) or []
        words = [
            {"text": w.word, "start": float(w.start), "end": float(w.end)}
            for w in words_raw
        ]
        transcript = (getattr(result, "text", None) or "").strip()
        duration = float(getattr(result, "duration", 0.0) or 0.0)
        log.info(
            "[WHISPER] words ok words=%d dur=%.2fs preview=%r",
            len(words), duration, transcript[:80],
        )
        return {"transcript": transcript, "words": words, "duration": duration}
    except Exception as e:
        log.error("[WHISPER] words transcription failed (fail-open): %s", e)
        return empty
