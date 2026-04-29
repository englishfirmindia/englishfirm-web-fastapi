"""
AI chat router for EnglishFirm web — full EF Coach port from mobile.

Endpoints:
  POST /chat        — synchronous reply
  POST /chat/stream — SSE streaming reply
"""

import json
import time
import asyncio
from datetime import datetime, timezone

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

# Trigger summary after this many messages in a conversation
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

    # Only summarise at every SUMMARY_TRIGGER_COUNT boundary (6, 12, 18...)
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

    # ── 1. Get or create active conversation ──────────────────────────────────
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id,
            Conversation.status == "active",
        )
        .first()
    )
    if conversation is None:
        conversation = Conversation(
            user_id=current_user.id,
            status="active",
            message_count=0,
        )
        db.add(conversation)
        db.flush()
        db.refresh(conversation)

    # ── 2. Persist user message ───────────────────────────────────────────────
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    db.flush()

    # ── 3. Load trainer profile ───────────────────────────────────────────────
    trainer_profile = get_trainer_profile(current_user.id, db)

    # ── 4. Session boundary detection ─────────────────────────────────────────
    last_session_at_dt = None
    if trainer_profile.get("last_session_at"):
        try:
            last_session_at_dt = datetime.fromisoformat(trainer_profile["last_session_at"])
            if last_session_at_dt.tzinfo is None:
                last_session_at_dt = last_session_at_dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    detect_and_handle_session_boundary(current_user.id, db)
    # Refresh after session update
    trainer_profile = get_trainer_profile(current_user.id, db)

    # ── 5. Check for new practice since last session ──────────────────────────
    new_practice = get_new_practice_since(current_user.id, db, last_session_at_dt)
    if new_practice:
        log.info("[PROACTIVE] %d new practice attempts since last session", len(new_practice))

    # ── 6. Compute phase + completeness flags ─────────────────────────────────
    phase = compute_phase(trainer_profile, new_practice)
    completeness_flags = compute_completeness_flags(trainer_profile)

    log.info("[SESSION] user=%s session=#%s phase=%s missing=%d",
             current_user.username,
             trainer_profile.get("session_count"),
             phase,
             len(completeness_flags))

    # ── 7. Load recent conversation history ───────────────────────────────────
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
        if m.id != user_message.id  # exclude the one we just added
    ]

    # ── 8. Build thin user context ────────────────────────────────────────────
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

    # ── 9. Get AI reply ───────────────────────────────────────────────────────
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

    # ── 10. Persist assistant message ─────────────────────────────────────────
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=reply,
    )
    db.add(assistant_message)

    conversation.message_count += 2
    conversation.last_message_at = func.now()

    db.commit()

    # ── 11. Background: generate session summary if threshold reached ─────────
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

    # ── 1. Get or create active conversation ──────────────────────────────────
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id,
            Conversation.status == "active",
        )
        .first()
    )
    if conversation is None:
        conversation = Conversation(
            user_id=current_user.id,
            status="active",
            message_count=0,
        )
        db.add(conversation)
        db.flush()
        db.refresh(conversation)

    # ── 2. Persist user message ───────────────────────────────────────────────
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    db.flush()
    t = ms("db:conversation+user_msg", t0)

    # ── 3. Trainer profile (first load) ───────────────────────────────────────
    trainer_profile = get_trainer_profile(current_user.id, db)
    t = ms("db:trainer_profile_1", t)

    # ── 4. Session boundary detection ─────────────────────────────────────────
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

    # ── 5. Trainer profile (refresh after session update) ─────────────────────
    trainer_profile = get_trainer_profile(current_user.id, db)
    t = ms("db:trainer_profile_2", t)

    # ── 6. New practice + phase ───────────────────────────────────────────────
    new_practice       = get_new_practice_since(current_user.id, db, last_session_at_dt)
    phase              = compute_phase(trainer_profile, new_practice)
    completeness_flags = compute_completeness_flags(trainer_profile)
    t = ms("db:new_practice+phase", t)

    # ── 7. Recent conversation history ────────────────────────────────────────
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

    log.info("[STREAM] >>> REQUEST  user=%s  msg=%r", username, request.message[:80])

    async def event_generator():
        full_text    = ""
        first_chunk  = True
        chunk_count  = 0
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
                    log.info("[STREAM] --- FIRST CHUNK sent  %.0f ms", (time.perf_counter() - t0) * 1000)
                full_text  += chunk
                chunk_count += 1
                if chunk_count % 10 == 0:
                    log.info("[STREAM] --- chunk #%d  total_chars=%d", chunk_count, len(full_text))
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            log.info("[STREAM] --- LLM done  chunks=%d  total_chars=%d  %.0f ms",
                     chunk_count, len(full_text), (time.perf_counter() - t0) * 1000)

        except Exception as e:
            log.error("[STREAM] !!! LLM ERROR: %s", e)
            err = "Sorry, I'm having trouble right now. Please try again shortly."
            full_text = err
            yield f"data: {json.dumps({'chunk': err})}\n\n"

        try:
            log.info("[STREAM] --- saving assistant message to DB (%d chars)", len(full_text))
            assistant_message = Message(
                conversation_id=conv_id,
                role="assistant",
                content=full_text or "",
            )
            db.add(assistant_message)
            conversation.message_count = msg_count + 2
            conversation.last_message_at = func.now()
            db.commit()
            log.info("[STREAM] --- DB saved OK")

            asyncio.create_task(_run_summary_if_needed(
                conversation_id=conv_id,
                user_id=user_id,
                student_name=username,
                message_count=conversation.message_count,
                db=db,
            ))
            log.info("[STREAM] <<< DONE  total=%.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            log.error("[STREAM] !!! DB save failed: %s", e)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# History endpoint
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
        .order_by(Message.id.desc())
        .limit(50)
        .all()
    )
    return {
        "messages": [
            {"role": m.role, "content": m.content}
            for m in reversed(messages)
        ]
    }
