"""
Shared background scoring for all speaking practice types.
Downloads audio from S3, runs Azure scoring, stores result.
"""
import threading
import requests as _requests

from services.s3_service import generate_presigned_url
from services.session_service import store_score

_WORD_COUNT_TARGET = {
    "describe_image": 40,
    "retell_lecture": 60,
    "respond_to_situation": 30,
    "summarize_group_discussion": 50,
}


def _run_scoring(user_id: int, question_id: int, question_type: str, audio_url: str, reference_text: str = ""):
    from services.azure_speech_service import score_read_aloud, transcribe_and_score_free
    try:
        presigned = generate_presigned_url(audio_url)
        audio_resp = _requests.get(presigned, timeout=30)
        audio_resp.raise_for_status()
        audio_bytes = audio_resp.content

        if question_type in ("read_aloud", "repeat_sentence") and reference_text:
            raw = score_read_aloud(audio_bytes, reference_text)
            store_score(user_id, question_id, {
                "scoring":       "complete",
                "content":       raw["content"],
                "pronunciation": raw["pronunciation"],
                "fluency":       raw["fluency"],
                "total":         raw["total"],
                "word_scores":   raw.get("word_scores", []),
            })
        elif question_type == "answer_short_question":
            raw = transcribe_and_score_free(audio_bytes)
            store_score(user_id, question_id, {
                "scoring":       "complete",
                "transcript":    raw.get("transcript", ""),
                "content":       0.0,
                "fluency":       raw.get("fluency", 0),
                "pronunciation": raw.get("pronunciation", 0),
            })
        else:
            # DI, RL, RTS, SGD — fluency + pronunciation from Azure, word-count proxy for content
            raw = transcribe_and_score_free(audio_bytes)
            transcript = raw.get("transcript", "")
            target = _WORD_COUNT_TARGET.get(question_type, 40)
            content = min(len(transcript.split()) / target, 1.0) * 50.0 if transcript else 0.0
            store_score(user_id, question_id, {
                "scoring":       "complete",
                "transcript":    transcript,
                "content":       round(content, 1),
                "fluency":       raw.get("fluency", 0),
                "pronunciation": raw.get("pronunciation", 0),
                "word_scores":   raw.get("word_scores", []),
            })

    except Exception as e:
        store_score(user_id, question_id, {"scoring": "error", "error": str(e)})


def kick_off_scoring(user_id: int, question_id: int, question_type: str, audio_url: str, reference_text: str = ""):
    t = threading.Thread(
        target=_run_scoring,
        args=(user_id, question_id, question_type, audio_url, reference_text),
        daemon=True,
    )
    t.start()
