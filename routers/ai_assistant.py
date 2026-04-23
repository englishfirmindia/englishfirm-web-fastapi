"""
AI chat router for EnglishFirm web.

Endpoints:
  POST /chat        — synchronous reply
  POST /chat/stream — SSE streaming reply
"""

import os
import json
import time
from typing import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import anthropic

from db.database import get_db
from db.models import User, Conversation, Message
from core.dependencies import get_current_user

router = APIRouter(prefix="/chat", tags=["AI Chat"])

SYSTEM_PROMPT = (
    "You are an expert PTE Academic English tutor at EnglishFirm. "
    "Help students improve their English and PTE scores. "
    "Be encouraging, specific, and practical in your feedback. "
    "Focus on PTE-specific skills: speaking fluency, pronunciation, "
    "writing coherence, reading comprehension, and listening accuracy."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_conversation(db: Session, user_id: int) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter_by(user_id=user_id, status="active")
        .order_by(Conversation.last_message_at.desc())
        .first()
    )
    if not conv:
        conv = Conversation(user_id=user_id, status="active", message_count=0)
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv


def _get_history(db: Session, conversation_id: int, limit: int = 20) -> list:
    msgs = (
        db.query(Message)
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in reversed(msgs)]


# ─────────────────────────────────────────────────────────────────────────────
# Request model
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat
# ─────────────────────────────────────────────────────────────────────────────

@router.post("")
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"reply": "AI chat is not configured yet."}

    conv = _get_or_create_conversation(db, current_user.id)
    history = _get_history(db, conv.id)

    # Save user message
    db.add(Message(conversation_id=conv.id, role="user", content=req.message))
    conv.message_count = (conv.message_count or 0) + 1
    conv.last_message_at = datetime.now(timezone.utc)
    db.commit()

    client = anthropic.Anthropic(api_key=api_key)
    messages = history + [{"role": "user", "content": req.message}]

    reply = "Sorry, I'm having trouble right now. Please try again."
    for attempt in range(1, 4):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
                timeout=30,
            )
            reply = response.content[0].text
            break
        except anthropic.AuthenticationError:
            reply = "AI chat is not configured correctly."
            break
        except Exception as e:
            if attempt < 3:
                time.sleep(2)
            else:
                reply = "Sorry, I'm having trouble right now. Please try again."

    # Save assistant message
    db.add(Message(conversation_id=conv.id, role="assistant", content=reply))
    conv.message_count = (conv.message_count or 0) + 1
    conv.last_message_at = datetime.now(timezone.utc)
    db.commit()

    return {"reply": reply}


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat/stream  (SSE)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        async def _no_key():
            yield "data: AI chat is not configured yet.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_key(), media_type="text/event-stream")

    conv = _get_or_create_conversation(db, current_user.id)
    history = _get_history(db, conv.id)

    db.add(Message(conversation_id=conv.id, role="user", content=req.message))
    conv.last_message_at = datetime.now(timezone.utc)
    db.commit()

    messages = history + [{"role": "user", "content": req.message}]

    async def _stream() -> AsyncGenerator[str, None]:
        client = anthropic.Anthropic(api_key=api_key)
        full_reply: list[str] = []
        try:
            with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
                timeout=30,
            ) as stream:
                for text in stream.text_stream:
                    full_reply.append(text)
                    yield f"data: {json.dumps({'chunk': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            reply_text = "".join(full_reply)
            if reply_text:
                db.add(Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=reply_text,
                ))
                conv.message_count = (conv.message_count or 0) + 2
                conv.last_message_at = datetime.now(timezone.utc)
                db.commit()
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
