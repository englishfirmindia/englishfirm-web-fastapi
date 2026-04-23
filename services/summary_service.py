"""
Summary service — generates a structured trainer note at end of session.
Uses a cheap GPT-3.5-turbo call so it doesn't add latency to the main response.
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

log = logging.getLogger(__name__)

_SUMMARY_SYSTEM = """You are a PTE trainer writing brief notes after a coaching session.

Given a conversation, write a structured note (max 120 words) covering:
- What the student revealed about themselves (motivation, schedule, anxiety, etc.)
- Any progress made or scores discussed
- Concerns or blockers mentioned
- What to follow up on next session

Format as short bullet points under these 4 headings:
Revealed: ...
Progress: ...
Concerns: ...
Next: ...

Be specific. Use the student's actual words where possible."""


async def generate_session_summary(
    messages: list[dict],
    student_name: str,
) -> Optional[str]:
    """
    Takes the last N conversation messages and writes a structured trainer note.
    Returns the summary string, or None on failure.
    """
    if not messages:
        return None

    # Build a readable transcript from messages
    transcript_lines = []
    for m in messages[-12:]:   # last 12 messages max
        role = "Student" if m["role"] == "user" else "Trainer"
        transcript_lines.append(f"{role}: {m['content'][:300]}")
    transcript = "\n".join(transcript_lines)

    try:
        response = _client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user",   "content": f"Student name: {student_name}\n\nTranscript:\n{transcript}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        summary = response.choices[0].message.content.strip()
        log.info("[SUMMARY] generated for %s (%d chars)", student_name, len(summary))
        return summary

    except Exception as e:
        log.warning("[SUMMARY] failed: %s", e)
        return None


def save_session_summary(user_id: int, summary: str, db) -> None:
    """Writes summary to student_trainer_profiles.last_session_summary."""
    from db.models import StudentTrainerProfile
    from datetime import datetime, timezone

    profile = db.query(StudentTrainerProfile).filter(
        StudentTrainerProfile.user_id == user_id
    ).first()

    if profile:
        profile.last_session_summary = summary
        profile.updated_at = datetime.now(timezone.utc)
        db.commit()
        log.info("[SUMMARY] saved for user_id=%s", user_id)
