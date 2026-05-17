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
    """Call Whisper, returning text. Empty string on failure."""
    last_exc: Exception = RuntimeError("_transcribe: no attempts made")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            buf = io.BytesIO(audio_bytes)
            buf.name = filename
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
