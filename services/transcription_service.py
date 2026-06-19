"""
On-demand SST audio transcription via OpenAI Whisper.

SST questions in `questions_from_apeuni` carry only `audio_url` — no
transcript. The AI scorer's LLM content gate requires a reference text,
so without a transcript the gate is silently skipped and SST scoring
falls back to heuristic-only.

This service lazily transcribes the audio on first scoring attempt and
persists the result to `content_json.transcript` via JSONB merge so
subsequent submissions hit the cache.

Cost: ~$0.006/min × ~90s avg = ~$0.009 per question, lifetime.
Failure mode: returns "" so callers fall back to heuristic-only scoring.
"""
import io
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Optional

import openai as _openai_module
from openai import OpenAI
from sqlalchemy import text

from db.database import SessionLocal

log = logging.getLogger(__name__)

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_AUDIO_DOWNLOAD_TIMEOUT_S = 30
_WHISPER_TIMEOUT_S = 60
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2


def _filename_from_url(url: str) -> str:
    """Whisper SDK requires a file with a name attribute so it can sniff
    the audio container. Pull the trailing path component, default to .mp3."""
    path = urllib.parse.urlparse(url).path
    name = os.path.basename(path) or "audio.mp3"
    if "." not in name:
        name = name + ".mp3"
    return name


def _sniff_audio_extension(audio_bytes: bytes) -> Optional[str]:
    """Inspect the first ~12 bytes of `audio_bytes` and return the audio
    container extension Whisper should be told about — e.g. `'m4a'`,
    `'mp3'`, `'ogg'`, `'wav'`, `'flac'`, `'webm'`. Returns None when the
    bytes don't match a known signature (caller falls back to whatever
    extension the URL implies).

    Whisper's SDK routes by filename extension, not content type, so a
    file uploaded to S3 with the wrong extension (e.g. M4A bytes saved
    as `*.mp3` — found in apeuni's SST library, q=20679 "Female
    Reindeer") hard-fails with 400 "Invalid file format". Sniffing the
    magic bytes lets us route around source-data mislabels without a
    rename across the entire bucket.
    """
    if not audio_bytes or len(audio_bytes) < 12:
        return None
    head = audio_bytes[:12]
    # M4A / MP4 audio — `ftyp` box at offset 4-7, brand at 8-11
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B ", b"isom", b"mp42", b"mp41", b"dash"):
            return "m4a"
        return "mp4"
    # MP3 with ID3v2 tag
    if head[:3] == b"ID3":
        return "mp3"
    # MP3 raw — frame sync 11-bit `1111 1111 111x` (0xFFE…)
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"
    # WAV — `RIFF....WAVE`
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    # FLAC
    if head[:4] == b"fLaC":
        return "flac"
    # OGG (Vorbis/Opus container)
    if head[:4] == b"OggS":
        return "ogg"
    # WebM (EBML)
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    return None


def _coerce_filename(audio_bytes: bytes, filename: str) -> str:
    """Return a filename whose extension matches the sniffed container
    format. Leaves the basename intact; only swaps the extension when
    the bytes disagree with what `filename` claims. Falls through to
    the input filename when sniff returns None — preserves legacy
    behaviour for any container we don't recognise.
    """
    sniffed = _sniff_audio_extension(audio_bytes)
    if not sniffed:
        return filename
    base, _, current_ext = filename.rpartition(".")
    if not base:
        base = filename  # no extension in source
    if current_ext.lower() == sniffed:
        return filename
    corrected = f"{base}.{sniffed}"
    log.warning(
        "[Whisper] filename mismatch — '%s' bytes look like %s; routing as '%s'",
        filename, sniffed, corrected,
    )
    return corrected


def _download_audio(url: str) -> Optional[bytes]:
    """Download the SST audio file via a presigned URL.

    The `apeuni-questions-audio` bucket recently moved to private access —
    raw anonymous HTTPS GETs now return 403, which was silently flipping
    every SST scoring run into the word-count heuristic fallback. Mint a
    short-lived presigned URL with IAM auth and use that for the actual
    GET. If presigning fails (config error, bad URL), fall back to the
    raw URL so behaviour matches the legacy path for any bucket that's
    still public.
    """
    fetch_url = url
    try:
        from services.s3_service import generate_presigned_url
        fetch_url = generate_presigned_url(url, expires_in=300)
    except Exception as exc:
        # Bad URL format / S3 config — fall through to the raw URL. If the
        # bucket is private the raw GET will 403, which the warning below
        # already captures.
        log.warning(
            "[Whisper] presign failed url=%s: %s — falling back to raw URL",
            url, exc,
        )

    try:
        with urllib.request.urlopen(fetch_url, timeout=_AUDIO_DOWNLOAD_TIMEOUT_S) as resp:
            return resp.read()
    except Exception as exc:
        log.warning("[Whisper] audio download failed url=%s: %s", url, exc)
        return None


def _transcribe(audio_bytes: bytes, filename: str) -> str:
    """Call Whisper, returning text. Empty string on failure.

    Filename is content-sniffed before submission: when the magic bytes
    disagree with the input extension (e.g. apeuni's `sst_0666.mp3`
    which is actually M4A) we override `buf.name` to the right
    extension. Without this, Whisper hard-fails 400 "Invalid file
    format" and the SST scorer falls through to heuristic-only.
    """
    routed_filename = _coerce_filename(audio_bytes, filename)
    last_exc: Exception = RuntimeError("_transcribe: no attempts made")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            buf = io.BytesIO(audio_bytes)
            buf.name = routed_filename
            resp = _openai.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
                language="en",
                timeout=_WHISPER_TIMEOUT_S,
            )
            return (resp.text or "").strip()
        except _openai_module.AuthenticationError as exc:
            log.error("[Whisper] AuthenticationError — not retrying: %s", exc)
            return ""
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[Whisper] _transcribe attempt=%d/%d failed — %s: %s",
                attempt, _MAX_ATTEMPTS, type(exc).__name__, exc,
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S)

    log.error(
        "[Whisper] _transcribe failed after %d attempts — %s: %s",
        _MAX_ATTEMPTS, type(last_exc).__name__, last_exc,
    )
    return ""


_ALLOWED_TABLES = {"questions", "questions_from_apeuni"}


def _persist_transcript(table_name: str, question_id: int, transcript: str) -> None:
    """Idempotent JSONB merge on a short-lived session so the caller's
    transaction state is unaffected. Concurrent writers produce identical
    content — last-writer-wins is safe."""
    if table_name not in _ALLOWED_TABLES:
        log.error("[Whisper] refusing to persist to unknown table=%s qid=%d",
                  table_name, question_id)
        return

    s = SessionLocal()
    try:
        s.execute(
            text(
                f"UPDATE {table_name} "
                "SET content_json = content_json || CAST(:patch AS jsonb) "
                "WHERE question_id = :qid"
            ),
            {"patch": json.dumps({"transcript": transcript}), "qid": question_id},
        )
        s.commit()
    except Exception as exc:
        log.warning("[Whisper] persist failed qid=%d: %s", question_id, exc)
        s.rollback()
    finally:
        s.close()


def get_or_create_sst_transcript(question) -> str:
    """
    Returns the reference transcript for an SST question. If not yet cached
    in `content_json.transcript`, downloads `audio_url`, runs Whisper,
    persists the result, and returns it.

    `question` is an ORM row from `questions` or `questions_from_apeuni`.
    Returns "" on any failure — caller falls back to heuristic-only scoring.
    """
    content_json = question.content_json or {}
    cached = content_json.get("transcript")
    if cached:
        return cached

    audio_url = content_json.get("audio_url", "")
    if not audio_url:
        return ""

    audio_bytes = _download_audio(audio_url)
    if not audio_bytes:
        return ""

    transcript = _transcribe(audio_bytes, _filename_from_url(audio_url))
    if not transcript:
        return ""

    table_name = getattr(question, "__tablename__", "")
    _persist_transcript(table_name, question.question_id, transcript)
    log.info(
        "[Whisper] cached transcript qid=%d table=%s chars=%d",
        question.question_id, table_name, len(transcript),
    )
    return transcript
