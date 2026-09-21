"""
Coach-bubble specialised stream handlers.

Two lightweight phases used by the frontend coach bubble state machine:

  • coach_bubble_deviation — user free-typed a question on Q2 ("Do you find
    any section difficult in particular?") instead of clicking Yes/No.
    Claude replies in 1-2 sentences and steers them back to the buttons.
    No tools, no user context, no session summary.

  • coach_bubble_booking — user answered Yes to Q3 ("Would you like to book
    a free demo?"). Claude's ONLY job is to collect (preferred_day,
    preferred_time), compute the concrete Sydney date+time, and call
    schedule_demo. Phone is auto-injected from the user's profile so
    Claude doesn't ask. Tool set restricted to schedule_demo only.

Kept out of ai_service.py + claude_router.py so the full-coach flow (with
its 700-line system prompt, 5-query rich context, and tool suite) is not
affected by these narrow flows.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Optional

from sqlalchemy.orm import Session

from services.claude_router import CLAUDE_MODEL, _client, _async_client, _extract_text, _extract_tool_calls, _fake_stream
from services.tool_registry import (
    ToolContext,
    TOOL_REGISTRY,
    execute_tools_parallel,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — deviation
# ─────────────────────────────────────────────────────────────────────────────

_DEVIATION_SYSTEM = (
    "You are the EnglishFirm EF Coach mini-chat, embedded in a floating "
    "help bubble on the app.englishfirm.com home screen. The user just "
    "asked a question instead of clicking Yes/No on the guided prompt "
    "\"Do you find any section difficult in particular?\".\n\n"
    "Rules — follow strictly:\n"
    "  1. Answer their question in ONE to TWO short sentences. No bullets, "
    "     no lists, no markdown.\n"
    "  2. Do NOT pitch a demo, do NOT reveal what the next question is, "
    "     do NOT ask them a question back.\n"
    "  3. End with the EXACT string: (Tap Yes or No above to continue.)\n"
    "  4. Never mention that you are an AI, or apologise for limitations.\n"
    "  5. If the question is unrelated to PTE, still answer briefly, then "
    "     hint that the buttons above will get them help faster."
)


async def stream_deviation(user_message: str) -> AsyncGenerator[str, None]:
    """Short-form Claude reply for the Q2 free-text deviation branch."""
    log.info("[COACH_BUBBLE/DEVIATION] streaming reply, len=%d", len(user_message))
    async with _async_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=_DEVIATION_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text_chunk in stream.text_stream:
            yield text_chunk


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — booking
# ─────────────────────────────────────────────────────────────────────────────

def _sydney_today_str() -> str:
    """Return today's date in Australia/Sydney as YYYY-MM-DD.

    Rough DST switch (first Sunday of Oct / first Sunday of Apr) is fine
    for a system-prompt hint — Claude uses this to compute scheduled_date
    from phrases like "next Tuesday", and any small drift is corrected
    by the tool's own YYYY-MM-DD validation.
    """
    now_utc = datetime.now(timezone.utc)
    y = now_utc.year
    oct_1 = datetime(y, 10, 1, tzinfo=timezone.utc)
    oct_dst_start = oct_1 + timedelta(days=(6 - oct_1.weekday()) % 7)
    apr_1 = datetime(y, 4, 1, tzinfo=timezone.utc)
    apr_dst_end = apr_1 + timedelta(days=(6 - apr_1.weekday()) % 7)
    offset = timedelta(hours=10) if apr_dst_end <= now_utc < oct_dst_start else timedelta(hours=11)
    return (now_utc + offset).strftime("%Y-%m-%d")


def _build_booking_system(user_phone: str, user_target_score: Optional[int],
                          today_syd: str, first_name: str) -> str:
    target = f"{user_target_score}+" if user_target_score else "unspecified"
    return (
        f"You are the EnglishFirm EF Coach mini-chat. The user (first name "
        f"{first_name}, target PTE {target}) has clicked \"Yes, book my free "
        f"demo\" on the coach bubble.\n\n"
        f"Your ONLY job is to collect:\n"
        f"  1. preferred_day  (e.g. 'Tuesday', 'this Friday')\n"
        f"  2. preferred_time (e.g. '2pm', 'morning')\n\n"
        f"Phone is already on file: {user_phone}. Do NOT ask for phone. "
        f"Do NOT ask for name or email — you already have them.\n\n"
        f"Today's date in Australia/Sydney is {today_syd}. Once you have "
        f"both preferred_day and preferred_time, compute:\n"
        f"  - scheduled_date (YYYY-MM-DD) — the actual next matching date "
        f"in Sydney from today\n"
        f"  - scheduled_time (HH:MM 24h in Sydney) — e.g. '14:00' for 2pm, "
        f"'09:00' for morning\n"
        f"THEN call the schedule_demo tool with all five fields plus "
        f"phone='{user_phone}'.\n\n"
        f"Style rules:\n"
        f"  - Keep every reply to ONE short sentence.\n"
        f"  - No markdown, no bullets, no emoji.\n"
        f"  - If the user asks anything unrelated, redirect politely back "
        f"    to picking a time.\n"
        f"  - If the user is vague (e.g. 'anytime'), suggest 'How about "
        f"    tomorrow at 3pm Sydney time?' — never guess a full booking "
        f"    without their agreement.\n"
        f"  - Do NOT list available times — just ask what works for them."
    )


async def stream_booking(
    *,
    user_message: str,
    user,                                    # SQLAlchemy User row (or None)
    user_id: Optional[int],
    db: Session,
    conversation_messages: list,
) -> AsyncGenerator[str, None]:
    """Streaming Claude reply for the Q3=Yes booking flow.

    Behaves like [claude_router.stream_reply] but with a minimal system
    prompt, no rich-context blocks, and the tool set restricted to
    schedule_demo only. Yields the tool's success marker `<<BOOKING_CONFIRMED>>`
    (inline in Claude's final reply) so the frontend can transition the
    bubble to its `booked` terminal state.
    """
    username = getattr(user, "username", None) or "Student"
    first_name = username.split()[0] if username else "Student"
    phone = (getattr(user, "phone", None) or "").strip()
    target_score = getattr(user, "score_requirement", None)

    if not phone:
        # Shouldn't happen — bubble is only shown to logged-in Google-Ads
        # users, and signup requires a phone. Degrade gracefully.
        log.warning("[COACH_BUBBLE/BOOKING] no phone on file user_id=%s — aborting", user_id)
        for chunk in _fake_stream(
            "Sorry, we don't have your phone on file — please email "
            "support@englishfirm.com to book. "
        ):
            yield chunk
        return

    system_prompt = _build_booking_system(
        user_phone=phone,
        user_target_score=target_score,
        today_syd=_sydney_today_str(),
        first_name=first_name,
    )

    # ── Restricted tool set: schedule_demo only ──────────────────────────────
    schedule_demo_schema = TOOL_REGISTRY["schedule_demo"]["schema"]
    tools_arg = [schedule_demo_schema]

    # ── Build message history (last N messages of THIS conversation only) ──
    messages = []
    for m in (conversation_messages or [])[-10:]:
        messages.append({"role": m["role"], "content": m["content"][:1000]})
    messages.append({"role": "user", "content": user_message})

    ctx = ToolContext(user_id=user_id or 0, db=db, username=username)

    MAX_ROUNDTRIPS = 3  # ask day, ask time, call tool — 3 is plenty

    for roundtrip in range(MAX_ROUNDTRIPS + 1):
        log.info("[COACH_BUBBLE/BOOKING] roundtrip=%d", roundtrip)

        if roundtrip < MAX_ROUNDTRIPS:
            response = await asyncio.to_thread(
                _client.messages.create,
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=system_prompt,
                tools=tools_arg,
                messages=messages,
            )
            if response.stop_reason == "end_turn":
                text = _extract_text(response.content)
                for chunk in _fake_stream(text):
                    yield chunk
                return

            raw_calls = _extract_tool_calls(response.content)
            if not raw_calls:
                text = _extract_text(response.content)
                for chunk in _fake_stream(text):
                    yield chunk
                return

            # Only allow schedule_demo — everything else silently blocked.
            allowed = [c for c in raw_calls if c["name"] == "schedule_demo"]
            if not allowed:
                log.warning("[COACH_BUBBLE/BOOKING] non-allowlist tool call blocked")
                text = _extract_text(response.content)
                for chunk in _fake_stream(text):
                    yield chunk
                return

            # Execute (may fail Zapier — tool returns apology text; may
            # succeed — tool returns "<<BOOKING_CONFIRMED>> ...").
            tool_results = await execute_tools_parallel(allowed, ctx)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})

        else:
            # Final roundtrip: stream Claude's summary of the tool result.
            log.warning("[COACH_BUBBLE/BOOKING] max roundtrips — streaming final")
            async with _async_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=system_prompt,
                tools=tools_arg,
                messages=messages,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    yield text_chunk
            return
