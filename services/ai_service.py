"""
AI service — orchestration layer.

Architecture:
  Flutter → FastAPI → MCP tools → rich context + trainer profile
                    ├── Claude Sonnet  (TRAINING_PLAN, ESSAY_FEEDBACK)
                    └── Claude Sonnet + tools   (GENERAL — with tool calling to save student info)

Flow per message:
  1. Build rich context (parallel DB queries)
  2. Load trainer profile + session data
  3. Compute phase + completeness flags
  4. Build system prompt (phase-aware, personalised)
  5. Call Claude with tool calling enabled
  6. If tool call → execute → re-call Claude
  7. Return final text response
"""

import os
import json
import logging
import asyncio
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timezone

from dotenv import load_dotenv
import anthropic
from sqlalchemy.orm import Session

from mcp_server.tools import build_rich_context

load_dotenv()

ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL    = "claude-sonnet-4-6"

anthropic_client       = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
async_anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Small-talk router
# ─────────────────────────────────────────────────────────────────────────────

class ChatIntent(Enum):
    GREETING        = "greeting"
    FAREWELL        = "farewell"
    THANKS          = "thanks"
    ACKNOWLEDGEMENT = "acknowledgement"
    OTHER           = "other"
    EMPTY           = "empty"


class SmallTalkRouter:
    _greetings      = {"hi", "hello", "hey", "good morning", "good afternoon", "good night", "hlo", "yo", "sup", "how are you"}
    _farewells      = {"bye", "goodbye", "see you", "cya", "later"}
    _thanks         = {"thanks", "thank you", "thx"}
    _acknowledgements = {"ok", "okay", "k", "cheers", "hmm"}
    _negation_words = {"is", "does", "how", "what", "when", "where", "which", "why", "are", "did"}

    @classmethod
    def classify(cls, raw: str) -> ChatIntent:
        text = raw.strip().lower()
        if not text:              return ChatIntent.EMPTY
        if cls._matches(cls._greetings,       text): return ChatIntent.GREETING
        if cls._matches(cls._farewells,        text): return ChatIntent.FAREWELL
        if cls._matches(cls._thanks,           text): return ChatIntent.THANKS
        if cls._matches(cls._acknowledgements, text): return ChatIntent.ACKNOWLEDGEMENT
        return ChatIntent.OTHER

    @classmethod
    def reply_for(cls, intent: ChatIntent) -> str:
        return {
            ChatIntent.GREETING:        "👋 Hi! Ready to work on your PTE prep?",
            ChatIntent.FAREWELL:        "👋 Goodbye! Come back anytime to practice.",
            ChatIntent.THANKS:          "👍 You're welcome! Keep practicing.",
            ChatIntent.ACKNOWLEDGEMENT: "✅ Got it! Let's continue when you're ready.",
        }.get(intent, "")

    @classmethod
    def _matches(cls, patterns: set, text: str) -> bool:
        for word in cls._negation_words:
            if word in text:
                return False
        return any(text.startswith(p) for p in patterns)


# ─────────────────────────────────────────────────────────────────────────────
# Task router — decides which Claude mode to use
# ─────────────────────────────────────────────────────────────────────────────

class TaskType(Enum):
    TRAINING_PLAN  = "training_plan"
    ESSAY_FEEDBACK = "essay_feedback"
    GENERAL        = "general"


_TRAINING_KEYWORDS = {
    "study plan", "training plan", "practice plan", "schedule",
    "what should i practice", "what to practice", "how to improve",
    "improve my score", "weak area", "focus on", "prioritize",
    "next step", "what to study", "roadmap", "preparation plan",
}

_ESSAY_KEYWORDS = {
    "check my essay", "review my essay", "essay feedback",
    "my essay", "swt feedback", "check my summary",
    "review my summary", "my writing", "is my essay good",
    "grade my essay", "mark my essay", "evaluate my writing",
}


def _detect_task(message: str) -> TaskType:
    lower = message.lower()
    if any(kw in lower for kw in _TRAINING_KEYWORDS):
        return TaskType.TRAINING_PLAN
    if any(kw in lower for kw in _ESSAY_KEYWORDS):
        return TaskType.ESSAY_FEEDBACK
    return TaskType.GENERAL


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_weak_areas(weak: list) -> str:
    if not weak:
        return "  No significant weak areas identified yet."
    lines = []
    for w in weak[:5]:
        pct = round(w["pct"] * 100)
        lines.append(f"  • {w['task_type'].replace('_', ' ').title()}: {pct}% avg ({w['attempt_count']} attempts)")
    return "\n".join(lines)


def _fmt_history(history: list) -> str:
    if not history:
        return "  No completed exams yet."
    lines = []
    for h in history:
        lines.append(f"  • {h['module'].title()}: last={h['last_score']}, best={h['best_score']}, attempts={h['attempts']}")
    return "\n".join(lines)


def _fmt_recent_scores(recent: list) -> str:
    if not recent:
        return "  No recent scores yet."
    lines = []
    for r in recent:
        lines.append(f"  • {r['module'].title()}: {r['score']} pts ({r.get('completed_at', '')[:10]})")
    return "\n".join(lines)


def _fmt_new_practice(new_practice: list) -> str:
    if not new_practice:
        return ""
    lines = []
    for p in new_practice:
        delta_str = ""
        if p.get("delta") is not None:
            sign = "+" if p["delta"] >= 0 else ""
            delta_str = f" ({sign}{p['delta']} vs last)"
        lines.append(f"  • {p['module'].title()}: score {p['score']}{delta_str}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Claude handlers (non-streaming)
# ─────────────────────────────────────────────────────────────────────────────

async def _claude_training_plan(user_message: str, context: dict) -> str:
    weak_text    = _fmt_weak_areas(context.get("weak_areas", []))
    history_text = _fmt_history(context.get("exam_history", []))
    days         = context.get("days_to_exam")
    target       = context.get("score_requirement")
    name         = context.get("username", "Student")

    logger.info("[CLAUDE/TRAINING_PLAN] user=%s days_to_exam=%s target=%s", name, days, target)

    system = f"""You are an expert PTE Academic coach from Englishfirm.
You are creating a personalised study plan for {name}.

Student data:
- Target score: {target}
- Days to exam: {days if days is not None else 'unknown'}
- Exam history:
{history_text}
- Weak areas (worst first):
{weak_text}

Rules:
- Be specific and actionable. Reference the actual weak areas above.
- Structure: Priority tasks this week → Daily routine → Quick wins
- Keep under 300 words. Use markdown with short bullet points.
- Address {name} by name. Trainer tone — encouraging but direct.
- Do NOT invent scores or tasks not listed above."""

    response = await asyncio.to_thread(
        anthropic_client.messages.create,
        model=CLAUDE_MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": f"{system}\n\nStudent asked: {user_message}"}],
    )
    reply = response.content[0].text
    logger.info("[CLAUDE/TRAINING_PLAN] tokens=%s preview=%.200r", response.usage.output_tokens, reply)
    return reply


async def _claude_essay_feedback(user_message: str, context: dict) -> str:
    name   = context.get("username", "Student")
    target = context.get("score_requirement")

    logger.info("[CLAUDE/ESSAY_FEEDBACK] user=%s target=%s", name, target)

    system = f"""You are an expert PTE Academic writing coach from Englishfirm.
{name} has shared their writing for feedback. Their target score is {target}.

Evaluate against PTE Academic criteria:
- Content (relevance, main points covered)
- Form (word count, format compliance)
- Grammar (sentence structure, accuracy)
- Vocabulary (range, appropriateness)
- Spelling & Mechanics

Format:
# ✍️ Writing Feedback

**Overall:** [1-2 sentence summary]

**Strengths:**
- [specific strength]

**Improvements:**
- [specific, actionable improvement]

**Score Estimate:** [range e.g. 55-65]

*Tip:* [one concrete action to apply immediately]

Keep under 220 words. Be specific — quote from their text where helpful."""

    response = await asyncio.to_thread(
        anthropic_client.messages.create,
        model=CLAUDE_MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": f"{system}\n\n{user_message}"}],
    )
    reply = response.content[0].text
    logger.info("[CLAUDE/ESSAY_FEEDBACK] tokens=%s preview=%.200r", response.usage.output_tokens, reply)
    return reply


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point (non-streaming)
# ─────────────────────────────────────────────────────────────────────────────

async def get_ai_reply(
    db: Session,
    conversation,
    user_message: str,
    user_context: Dict,
    user=None,
    user_id: Optional[int] = None,
    trainer_profile: Optional[dict] = None,
    phase: str = "coaching",
    completeness_flags: Optional[list] = None,
    new_practice: Optional[list] = None,
    conversation_messages: Optional[list] = None,
) -> str:
    try:
        if not user_message.strip():
            raise ValueError("Empty message")

        # ── Small talk filter ─────────────────────────────────────────────────
        intent = SmallTalkRouter.classify(user_message)
        if intent not in (ChatIntent.OTHER, ChatIntent.EMPTY):
            return SmallTalkRouter.reply_for(intent)

        # ── Build rich context via MCP tools ──────────────────────────────────
        if user is not None and user_id is not None:
            rich_context = await asyncio.to_thread(build_rich_context, user, user_id, db)
        else:
            rich_context = user_context

        # ── Defaults ─────────────────────────────────────────────────────────
        trainer_profile      = trainer_profile or {}
        completeness_flags   = completeness_flags or []
        new_practice         = new_practice or []
        conversation_messages = conversation_messages or []

        # ── Task routing ──────────────────────────────────────────────────────
        task = _detect_task(user_message)
        logger.info("[ROUTE] %-20s phase=%s", task.value, phase)
        logger.info("[ROUTE] msg=%.120r", user_message)

        if task == TaskType.TRAINING_PLAN:
            return await _claude_training_plan(user_message, rich_context)

        if task == TaskType.ESSAY_FEEDBACK:
            return await _claude_essay_feedback(user_message, rich_context)

        # ── GENERAL: Claude + on-demand tools ────────────────────────────────
        from services.claude_router import get_reply as _claude_get_reply
        return await _claude_get_reply(
            user_message=user_message,
            context=rich_context,
            trainer_profile=trainer_profile,
            phase=phase,
            completeness_flags=completeness_flags,
            new_practice=new_practice,
            conversation_messages=conversation_messages,
            user_id=user_id,
            db=db,
            username=rich_context.get("username", "Student"),
        )

    except ValueError as ve:
        logger.error("[AIService] Validation: %s", ve)
        return "Invalid request. Please provide a valid message."
    except Exception as e:
        logger.error("[AIService] Error: %s: %s", type(e).__name__, e)
        return "Sorry, I'm having trouble right now. Please try again shortly."


# ─────────────────────────────────────────────────────────────────────────────
# Streaming: Claude handlers
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_claude_training_plan(user_message: str, context: dict):
    weak_text    = _fmt_weak_areas(context.get("weak_areas", []))
    history_text = _fmt_history(context.get("exam_history", []))
    days         = context.get("days_to_exam")
    target       = context.get("score_requirement")
    name         = context.get("username", "Student")

    logger.info("[STREAM/CLAUDE/TRAINING_PLAN] user=%s days_to_exam=%s target=%s", name, days, target)

    system = f"""You are an expert PTE Academic coach from Englishfirm.
You are creating a personalised study plan for {name}.

Student data:
- Target score: {target}
- Days to exam: {days if days is not None else 'unknown'}
- Exam history:
{history_text}
- Weak areas (worst first):
{weak_text}

Rules:
- Be specific and actionable. Reference the actual weak areas above.
- Structure: Priority tasks this week → Daily routine → Quick wins
- Keep under 300 words. Use markdown with short bullet points.
- Address {name} by name. Trainer tone — encouraging but direct.
- Do NOT invent scores or tasks not listed above."""

    async with async_anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": f"{system}\n\nStudent asked: {user_message}"}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _stream_claude_essay_feedback(user_message: str, context: dict):
    name   = context.get("username", "Student")
    target = context.get("score_requirement")

    logger.info("[STREAM/CLAUDE/ESSAY_FEEDBACK] user=%s target=%s", name, target)

    system = f"""You are an expert PTE Academic writing coach from Englishfirm.
{name} has shared their writing for feedback. Their target score is {target}.

Evaluate against PTE Academic criteria:
- Content (relevance, main points covered)
- Form (word count, format compliance)
- Grammar (sentence structure, accuracy)
- Vocabulary (range, appropriateness)
- Spelling & Mechanics

Format:
# ✍️ Writing Feedback

**Overall:** [1-2 sentence summary]

**Strengths:**
- [specific strength]

**Improvements:**
- [specific, actionable improvement]

**Score Estimate:** [range e.g. 55-65]

*Tip:* [one concrete action to apply immediately]

Keep under 220 words. Be specific — quote from their text where helpful."""

    async with async_anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": f"{system}\n\n{user_message}"}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ─────────────────────────────────────────────────────────────────────────────
# Streaming: public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def stream_ai_reply(
    db: Session,
    user_message: str,
    user_context: Dict,
    user=None,
    user_id: Optional[int] = None,
    trainer_profile: Optional[dict] = None,
    phase: str = "coaching",
    completeness_flags: Optional[list] = None,
    new_practice: Optional[list] = None,
    conversation_messages: Optional[list] = None,
    request_t0: Optional[float] = None,
):
    """Async generator yielding text chunks for SSE streaming."""
    import time as _time
    t0 = request_t0 or _time.perf_counter()

    def _ms(label: str, since: float) -> float:
        elapsed = (_time.perf_counter() - since) * 1000
        logger.info("[TTFT] %-30s %6.0f ms", label, elapsed)
        return _time.perf_counter()

    try:
        if not user_message.strip():
            raise ValueError("Empty message")

        # ── Small talk filter ─────────────────────────────────────────────────
        intent = SmallTalkRouter.classify(user_message)
        if intent not in (ChatIntent.OTHER, ChatIntent.EMPTY):
            yield SmallTalkRouter.reply_for(intent)
            return

        # ── Build rich context (5 parallel DB queries) ────────────────────────
        t = _time.perf_counter()
        if user is not None and user_id is not None:
            rich_context = await asyncio.to_thread(build_rich_context, user, user_id, db)
        else:
            rich_context = user_context
        _ms("svc:build_rich_context", t)

        trainer_profile       = trainer_profile or {}
        completeness_flags    = completeness_flags or []
        new_practice          = new_practice or []
        conversation_messages = conversation_messages or []

        # ── Task routing ──────────────────────────────────────────────────────
        task = _detect_task(user_message)
        logger.info("[TTFT] %-30s route=%s phase=%s",
                    "svc:task_detected", task.value, phase)

        if task == TaskType.TRAINING_PLAN:
            async for chunk in _stream_claude_training_plan(user_message, rich_context):
                yield chunk
            return

        if task == TaskType.ESSAY_FEEDBACK:
            async for chunk in _stream_claude_essay_feedback(user_message, rich_context):
                yield chunk
            return

        # ── GENERAL: Claude + on-demand tools (streaming) ────────────────────
        from services.claude_router import stream_reply as _claude_stream_reply
        async for chunk in _claude_stream_reply(
            user_message=user_message,
            context=rich_context,
            trainer_profile=trainer_profile,
            phase=phase,
            completeness_flags=completeness_flags,
            new_practice=new_practice,
            conversation_messages=conversation_messages,
            user_id=user_id,
            db=db,
            username=rich_context.get("username", "Student"),
        ):
            yield chunk

    except Exception as e:
        logger.error("[STREAM] Error: %s: %s", type(e).__name__, e)
        yield "Sorry, I'm having trouble right now. Please try again shortly."
