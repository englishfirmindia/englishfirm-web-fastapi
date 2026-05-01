"""
Mic check endpoint — called before sectional exam starts.
Records a short test phrase, checks volume and Azure word recognition.
"""
import re
import subprocess
import tempfile
import os

from fastapi import APIRouter, Depends, File, UploadFile

from core.dependencies import get_current_user
from db.models import User
from services.azure_speech_service import assess_pronunciation

from core.logging_config import get_logger

log = get_logger(__name__)


router = APIRouter(tags=["Mic Check"])

_TEST_PHRASE   = "The quick brown fox"
_DB_SILENT     = -60.0   # below this → silent
_DB_TOO_QUIET  = -42.0   # below this → too quiet


_LAST_MIC_CHECK_PATH = "/tmp/mic_check_last.aac"


def _mean_volume_db(audio_bytes: bytes) -> float:
    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", path, "-af", "volumedetect", "-f", "null", "/dev/null"],
            capture_output=True, text=True,
        )
        m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", r.stderr)
        import shutil
        shutil.copy(path, _LAST_MIC_CHECK_PATH)
        log.info(f"[mic-check] saved → {_LAST_MIC_CHECK_PATH} mean_db={float(m.group(1)) if m else -91.0:.1f}")
        return float(m.group(1)) if m else -91.0
    finally:
        os.unlink(path)


@router.post("/mic-check")
async def mic_check(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    audio_bytes = await audio.read()

    mean_db = _mean_volume_db(audio_bytes)

    words_recognised = 0
    try:
        az = assess_pronunciation(audio_bytes=audio_bytes, reference_text=_TEST_PHRASE)
        words_recognised = sum(
            1 for w in az.get("Words", []) if w.get("ErrorType") != "Omission"
        )
    except Exception:
        pass

    total_words = len(_TEST_PHRASE.split())

    if mean_db < _DB_SILENT:
        verdict = "silent"
    elif mean_db < _DB_TOO_QUIET:
        verdict = "too_quiet"
    else:
        verdict = "good"

    # Good volume but Azure heard nothing → likely noise, not speech
    if verdict == "good" and words_recognised == 0:
        verdict = "silent"

    return {
        "mean_volume_db":  round(mean_db, 1),
        "words_recognised": words_recognised,
        "total_words":      total_words,
        "verdict":          verdict,
    }
