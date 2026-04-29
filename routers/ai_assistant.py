"""
AI chat router for EnglishFirm web — full EF Coach port from mobile.

Endpoints:
  POST /chat                              — synchronous reply
  POST /chat/stream                       — SSE streaming reply
  GET  /chat/history                      — messages in active conversation
  POST /chat/conversations/new            — archive active conversation, start fresh
  GET  /chat/conversations                — list all closed conversations
  GET  /chat/conversations/{id}/messages  — messages in a past conversation
"""

import json
import time
import asyncio
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from pydantic import BaseModel

from db.database import get_db
from db.models import User, Conversation, Message
from core.dependencies import get_current_user

from mcp_server.tools import get_trainer_profile, get_new_practice_since
from services.coach_session_service import (
    detect_and_handle_session_boundary,
    compute_phase,
    compute_completeness_flags,
)
from services.summary_service import generate_session_summary, save_session_summary
from services.ai_service import get_ai_reply, stream_ai_reply

import logging
log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["AI Chat"])

SUMMARY_TRIGGER_COUNT = 6


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ─────────────────────────────────────────────────────────────────────────────
# Background task: generate + save session summary
# ─────────────────────────────────────────────────────────────────────────────

async def _run_summary_if_needed(
    conversation_id: int,
    user_id: int,
    student_name: str,
    message_count: int,
    db: Session,
):
    if message_count < SUMMARY_TRIGGER_COUNT:
        return

    if message_count % SUMMARY_TRIGGER_COUNT != 0:
        return

    try:
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(12)
            .all()
        )
        msg_dicts = [{"role": m.role, "content": m.content} for m in reversed(messages)]
        summary = await generate_session_summary(msg_dicts, student_name)
        if summary:
            save_session_summary(user_id, summary, db)
    except Exception as e:
        log.warning("[SUMMARY_BG] failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: get or create active conversation
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_active_conversation(user_id: int, db: Session) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id, Conversation.status == "active")
        .first()
    )
    if conv is None:
        conv = Conversation(user_id=user_id, status="active", message_count=0)
        db.add(conv)
        db.flush()
        db.refresh(conv)
    return conv


# ─────────────────────────────────────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conversation = _get_or_create_active_conversation(current_user.id, db)

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    db.flush()

    trainer_profile = get_trainer_profile(current_user.id, db)

    last_session_at_dt = None
    if trainer_profile.get("last_session_at"):
        try:
            last_session_at_dt = datetime.fromisoformat(trainer_profile["last_session_at"])
            if last_session_at_dt.tzinfo is None:
                last_session_at_dt = last_session_at_dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    detect_and_handle_session_boundary(current_user.id, db)
    trainer_profile = get_trainer_profile(current_user.id, db)

    new_practice = get_new_practice_since(current_user.id, db, last_session_at_dt)
    if new_practice:
        log.info("[PROACTIVE] %d new practice attempts since last session", len(new_practice))

    phase = compute_phase(trainer_profile, new_practice)
    completeness_flags = compute_completeness_flags(trainer_profile)

    log.info("[SESSION] user=%s session=#%s phase=%s missing=%d",
             current_user.username,
             trainer_profile.get("session_count"),
             phase,
             len(completeness_flags))

    recent_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.id.desc())
        .limit(10)
        .all()
    )
    conversation_messages = [
        {"role": m.role, "content": m.content}
        for m in reversed(recent_messages)
        if m.id != user_message.id
    ]

    from datetime import date
    days_to_exam = None
    if current_user.exam_date:
        try:
            days_to_exam = (current_user.exam_date - date.today()).days
            if days_to_exam < 0:
                days_to_exam = 0
        except Exception:
            pass

    user_context = {
        "username":          current_user.username,
        "score_requirement": current_user.score_requirement,
        "exam_date":         current_user.exam_date.isoformat() if current_user.exam_date else None,
        "days_to_exam":      days_to_exam,
    }

    reply = await get_ai_reply(
        db=db,
        conversation=conversation,
        user_message=request.message,
        user_context=user_context,
        user=current_user,
        user_id=current_user.id,
        trainer_profile=trainer_profile,
        phase=phase,
        completeness_flags=completeness_flags,
        new_practice=new_practice,
        conversation_messages=conversation_messages,
    )

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=reply,
    )
    db.add(assistant_message)

    conversation.message_count += 2
    conversation.last_message_at = func.now()

    db.commit()

    background_tasks.add_task(
        _run_summary_if_needed,
        conversation_id=conversation.id,
        user_id=current_user.id,
        student_name=current_user.username,
        message_count=conversation.message_count,
        db=db,
    )

    return ChatResponse(response=reply)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming chat endpoint (SSE)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    t0 = time.perf_counter()
    def ms(label: str, since: float) -> float:
        elapsed = (time.perf_counter() - since) * 1000
        log.info("[TTFT] %-30s %6.0f ms", label, elapsed)
        return time.perf_counter()

    conversation = _get_or_create_active_conversation(current_user.id, db)

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    # Commit the user's message *before* invoking the LLM. This way the
    # row survives even if the client disconnects mid-stream (Cmd-Tab
    # away, Chrome freezes the tab, network blip), so the next history
    # fetch already includes the question and the UI can rebuild
    # correctly. The assistant message is committed separately at the
    # end of the stream.
    db.commit()
    db.refresh(user_message)
    t = ms("db:conversation+user_msg", t0)

    trainer_profile = get_trainer_profile(current_user.id, db)
    t = ms("db:trainer_profile_1", t)

    last_session_at_dt = None
    if trainer_profile.get("last_session_at"):
        try:
            last_session_at_dt = datetime.fromisoformat(trainer_profile["last_session_at"])
            if last_session_at_dt.tzinfo is None:
                last_session_at_dt = last_session_at_dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    detect_and_handle_session_boundary(current_user.id, db)
    t = ms("db:session_boundary", t)

    trainer_profile = get_trainer_profile(current_user.id, db)
    t = ms("db:trainer_profile_2", t)

    new_practice       = get_new_practice_since(current_user.id, db, last_session_at_dt)
    phase              = compute_phase(trainer_profile, new_practice)
    completeness_flags = compute_completeness_flags(trainer_profile)
    t = ms("db:new_practice+phase", t)

    recent_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.id.desc())
        .limit(10)
        .all()
    )
    conversation_messages = [
        {"role": m.role, "content": m.content}
        for m in reversed(recent_messages)
        if m.id != user_message.id
    ]
    t = ms("db:recent_messages", t)

    log.info("[TTFT] %-30s %6.0f ms  ← setup done", "TOTAL_SETUP", (time.perf_counter() - t0) * 1000)

    from datetime import date
    days_to_exam = None
    if current_user.exam_date:
        try:
            days_to_exam = (current_user.exam_date - date.today()).days
            if days_to_exam < 0:
                days_to_exam = 0
        except Exception:
            pass

    user_context = {
        "username":          current_user.username,
        "score_requirement": current_user.score_requirement,
        "exam_date":         current_user.exam_date.isoformat() if current_user.exam_date else None,
        "days_to_exam":      days_to_exam,
    }

    conv_id   = conversation.id
    user_id   = current_user.id
    username  = current_user.username
    msg_count = conversation.message_count

    async def event_generator():
        full_text   = ""
        first_chunk = True
        try:
            async for chunk in stream_ai_reply(
                db=db,
                user_message=request.message,
                user_context=user_context,
                user=current_user,
                user_id=user_id,
                trainer_profile=trainer_profile,
                phase=phase,
                completeness_flags=completeness_flags,
                new_practice=new_practice,
                conversation_messages=conversation_messages,
                request_t0=t0,
            ):
                if first_chunk:
                    first_chunk = False
                    log.info("[STREAM] first chunk %.0f ms", (time.perf_counter() - t0) * 1000)
                full_text += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

        except Exception as e:
            log.error("[STREAM] LLM error: %s", e)
            err = "Sorry, I'm having trouble right now. Please try again shortly."
            full_text = err
            yield f"data: {json.dumps({'chunk': err})}\n\n"

        try:
            assistant_message = Message(
                conversation_id=conv_id,
                role="assistant",
                content=full_text or "",
            )
            db.add(assistant_message)
            conversation.message_count = msg_count + 2
            conversation.last_message_at = func.now()
            db.commit()

            asyncio.create_task(_run_summary_if_needed(
                conversation_id=conv_id,
                user_id=user_id,
                student_name=username,
                message_count=conversation.message_count,
                db=db,
            ))
        except Exception as e:
            log.error("[STREAM] DB save failed: %s", e)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Current conversation history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/history")
async def chat_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id,
            Conversation.status == "active",
        )
        .first()
    )
    if conversation is None:
        return {"messages": []}

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.id.asc())
        .all()
    )
    return {
        "messages": [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Start a new conversation (archive the active one)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/conversations/new")
async def new_conversation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id,
            Conversation.status == "active",
        )
        .first()
    )
    if conv is not None:
        first_msg = (
            db.query(Message)
            .filter(Message.conversation_id == conv.id, Message.role == "user")
            .order_by(Message.id.asc())
            .first()
        )
        if first_msg:
            raw = first_msg.content.strip()
            conv.title = raw[:60] + "…" if len(raw) > 60 else raw
        else:
            conv.title = "Chat"

        conv.status = "closed"
        conv.closed_at = func.now()
        db.commit()

    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# List all closed conversations (sidebar history)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/conversations")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    convs = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id,
            Conversation.status == "closed",
        )
        .order_by(Conversation.id.desc())
        .limit(50)
        .all()
    )
    return {
        "conversations": [
            {
                "id": c.id,
                "title": c.title or "Chat",
                "message_count": c.message_count,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in convs
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Messages in a specific past conversation
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/conversations/{conv_id}/messages")
async def get_conversation_messages(
    conv_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conv_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv_id)
        .order_by(Message.id.asc())
        .all()
    )
    return {
        "messages": [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
    }
