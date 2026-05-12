"""
Claude Router — handles the GENERAL route using Claude + on-demand tools.

Architecture:
  1. Build phase-aware system prompt (same logic as before)
  2. Call Claude with tool schemas
  3. If tool_use → execute tools in parallel → call Claude again
  4. Hard cap: MAX_ROUNDTRIPS=2, MAX_TOOL_CALLS=5
  5. Return text (or stream it)

Adding a new tool: add it to tool_registry.py only. No changes needed here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

import anthropic
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from services.tool_registry import (
    ToolContext,
    TOOL_CALL_LIMITS,
    execute_tool,
    execute_tools_parallel,
    get_tool_schemas,
)

load_dotenv()

log = logging.getLogger(__name__)

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL   = "claude-sonnet-4-6"

_client        = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
_async_client  = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)

# ── Hard guardrail caps (enforced in code, not just prompt) ───────────────────
MAX_ROUNDTRIPS    = 2
MAX_TOOL_CALLS    = 5


# ─────────────────────────────────────────────────────────────────────────────
# Phase instructions (same as before — preserved exactly)
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_INSTRUCTIONS = {
    "intake": """CURRENT PHASE: intake
You are meeting this student and building your understanding of their situation.
Do NOT jump straight to PTE tips. First understand the person.
Ask 1 question from the PROFILE GAPS list naturally within your response.
Do not interrogate — weave the question in after giving some useful information.
Once you have enough context, your next responses can shift to coaching.""",

    "planning": """CURRENT PHASE: planning
The student's profile is complete. Your primary goal this session is to generate
a structured, personalised training plan based on their actual data.
If the student asks anything, answer it AND include a plan or direct them toward one.
When generating a plan, call save_student_info with plan_text.""",

    "coaching": """CURRENT PHASE: coaching
You are in active coaching mode. Answer the student's question using your tools.
Reference the student's actual weak areas and history where relevant.
Check in on plan adherence if it's been a few days since the plan was made.""",

    "review": """CURRENT PHASE: review
The student has completed practice since your last session.
Open your response by acknowledging their recent practice results specifically.
Comment on what improved, what still needs work, and adjust advice accordingly.
Then answer their question.""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_scoring_contributions(data: Optional[dict]) -> str:
    """Format the canonical task→section contribution table for the system prompt."""
    if not data:
        return ""
    section_order = ["speaking", "writing", "reading", "listening"]
    lines = [
        "\nPTE SCORING CONTRIBUTIONS — CANONICAL (derived from RDS pte_question_weightage):",
        "Max points each task can contribute to each section's pte_score (10–90 scale).",
        "A task only contributes to the sections listed below for it. Sections not listed = zero contribution.",
    ]
    any_section = False
    for s in section_order:
        items = data.get(s) or []
        if not items:
            continue
        any_section = True
        lines.append(f"  {s.title()}:")
        for it in items:
            lines.append(f"    • {it['label']}: up to ~{it['max_pts']:.1f} pts")
    if not any_section:
        return ""
    return "\n".join(lines) + "\n"


def _fmt_speaking_task_breakdowns(data: Optional[dict]) -> str:
    if not data or not data.get("tasks"):
        return ""

    source = (data.get("source") or "").upper()
    attempt_id = data.get("attempt_id")
    completed = (data.get("completed_at") or "")[:10]
    header = f"\nLATEST {source} — PER-QUESTION SPEAKING SCORES (attempt #{attempt_id}, {completed}):"
    lines = [header]

    for task_type, answers in data["tasks"].items():
        title = task_type.replace("_", " ").title()
        lines.append(f"  {title} ({len(answers)} questions):")
        for ans in answers[:5]:
            parts = [f"q{ans['q']}: score={ans.get('score')}"]
            if ans.get("max_score") is not None:
                parts.append(f"max={ans['max_score']}")
            if ans.get("content") is not None:
                parts.append(f"content={ans['content']}")
            if ans.get("fluency") is not None:
                parts.append(f"fluency={ans['fluency']}")
            if ans.get("pronunciation") is not None:
                parts.append(f"pron={ans['pronunciation']}")
            if ans.get("pte_score") is not None:
                parts.append(f"pte={ans['pte_score']}")
            lines.append("    • " + ", ".join(parts))
        if len(answers) > 5:
            lines.append(f"    • ...({len(answers) - 5} more)")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(
    context:            dict,
    trainer_profile:    dict,
    phase:              str,
    completeness_flags: list,
    new_practice:       list,
    pre_retrieved:      Optional[str] = None,
) -> str:
    name    = context.get("username", "Student")
    target  = context.get("score_requirement", "unknown")
    days    = context.get("days_to_exam", "unknown")

    # ── Weak areas ────────────────────────────────────────────────────────────
    weak_areas = context.get("weak_areas", [])
    if weak_areas:
        weak_lines = "\n".join(
            f"  • {w['task_type']}: {int(w['pct']*100)}% avg ({w['attempt_count']} attempts)"
            for w in weak_areas[:6]
        )
    else:
        weak_lines = "  No weak area data yet."

    # ── Exam history ──────────────────────────────────────────────────────────
    history = context.get("exam_history", [])
    if history:
        hist_lines = "\n".join(
            f"  • {h['module'].title()}: best {h['best_score']}, last {h['last_score']}, avg {h['avg_score']} ({h['attempts']} attempts)"
            for h in history[:5]
        )
    else:
        hist_lines = "  No exam history yet."

    # ── Recent scores ─────────────────────────────────────────────────────────
    recent = context.get("recent_scores", [])
    if recent:
        recent_lines = "\n".join(
            f"  • {r['module'].title()}: {r['score']} pts ({(r.get('completed_at') or '')[:10]})"
            for r in recent[:3]
        )
    else:
        recent_lines = "  No recent scores."

    # ── Last session summary ──────────────────────────────────────────────────
    summary = trainer_profile.get("last_session_summary")
    summary_block = f"\nLAST SESSION SUMMARY:\n{summary}\n" if summary else "\nLAST SESSION SUMMARY:\nThis is the first session.\n"

    # ── New practice (proactive opener) ──────────────────────────────────────
    proactive_block = ""
    if new_practice:
        lines = []
        for p in new_practice:
            delta_str = ""
            if p.get("delta") is not None:
                sign = "+" if p["delta"] >= 0 else ""
                delta_str = f" ({sign}{p['delta']} vs last)"
            lines.append(f"  • {p['module'].title()}: score {p['score']}{delta_str}")
        proactive_block = (
            "\nPROACTIVE OPENER — start your response by acknowledging this before answering:\n"
            + "\n".join(lines)
            + "\nBe encouraging if improved, constructive if dropped.\n"
        )

    # ── Speaking per-question breakdown (latest mock/sectional) ───────────────
    speaking_tasks_block = _fmt_speaking_task_breakdowns(
        context.get("speaking_task_breakdowns")
    )

    # ── Canonical task→section contribution table ────────────────────────────
    contributions_block = _fmt_scoring_contributions(
        context.get("scoring_contributions")
    )

    # ── Pre-retrieved knowledge (forced search on factual keywords) ──────────
    pre_retrieved_block = ""
    if pre_retrieved:
        pre_retrieved_block = (
            "\nPRE-RETRIEVED KNOWLEDGE — already searched on behalf of this question:\n"
            "Use this as your authoritative source. Do NOT call search_pte_knowledge\n"
            "again unless this content is clearly insufficient for the student's question.\n"
            "---\n"
            f"{pre_retrieved}\n"
            "---\n"
        )

    # ── Phase instruction ─────────────────────────────────────────────────────
    phase_block = _PHASE_INSTRUCTIONS.get(phase, _PHASE_INSTRUCTIONS["coaching"])

    # ── Profile gaps ──────────────────────────────────────────────────────────
    gaps_block = ""
    if completeness_flags:
        gap_lines = "\n".join(
            f"  - {f['description']} ({f['field']})"
            for f in completeness_flags[:3]
        )
        gaps_block = f"\nPROFILE GAPS — ask 1 naturally per response, never interrogate:\n{gap_lines}\n"

    return f"""You are an expert PTE Academic trainer from Englishfirm coaching {name}.
You have full knowledge of PTE scoring, strategies, and this student's practice history.

STUDENT PROFILE:
  Name:             {name}
  Target score:     {target}
  Days to exam:     {days}
  Motivation:       {trainer_profile.get('motivation') or 'not yet known'}
  Study time:       {f"{trainer_profile.get('study_hours_per_day')}h/day ({trainer_profile.get('study_schedule')})" if trainer_profile.get('study_hours_per_day') else 'not yet known'}
  Prior attempts:   {trainer_profile.get('prior_pte_attempts') if trainer_profile.get('prior_pte_attempts') is not None else 'not yet known'}
  Anxiety level:    {trainer_profile.get('anxiety_level') or 'not yet known'}
  Self-weak area:   {trainer_profile.get('biggest_weakness_self') or 'not yet known'}

Recent scores:
{recent_lines}

Exam history:
{hist_lines}

Weak areas (from practice data, worst first):
{weak_lines}
{speaking_tasks_block}{contributions_block}{pre_retrieved_block}{summary_block}{proactive_block}
{phase_block}
{gaps_block}
KNOWLEDGE BASE RULES (mandatory — no exceptions):
- The PTE SCORING CONTRIBUTIONS block above is canonical and authoritative. For ANY question about which tasks contribute to which section, or how many points a task is worth, answer directly from that block. NEVER call search_pte_knowledge for contribution/weighting questions.
- If a PRE-RETRIEVED KNOWLEDGE block is present above, treat it as the authoritative source for this turn. Answer from it. Only call search_pte_knowledge if it is clearly insufficient.
- For OTHER factual questions about PTE (strategies, tips, exam format, prep advice) that are not covered by the blocks above, you MUST call search_pte_knowledge. Never answer such questions from general training knowledge.
- If search_pte_knowledge returns "NO_RELEVANT_CONTENT" or "SEARCH_UNAVAILABLE", respond with exactly: "I don't have information on that in my knowledge base." Do not supplement with general knowledge under any circumstances.
- After 2 searches with no relevant results, respond: "I don't have information on that in my knowledge base." Do not fall back to general knowledge.

TOOL GUARDRAILS (mandatory):
- Call search_pte_knowledge at most 2× per turn.
- Call get_last_attempt_breakdown at most 1× per turn.
- Call get_milestones at most 1× per turn.
- Call get_attempt_detail at most 1× per turn, only when the student asks about specific question performance.
- Call get_recent_task_answers at most 1× per turn. Use this if get_attempt_detail returns "No completed attempts found" but the student insists they have practiced — this tool also surfaces in-progress practice sessions and individual practice answers.
- Call save_student_info at most 1× per turn, only when the student reveals new personal info.
- Never call a tool to retrieve information already present in this system prompt.
- Never guess or ask for the student's user_id — it is injected by the system.
- Only discuss PTE Academic. For off-topic questions: "I'm here to help with PTE prep."
- Never invent scores or attempt data not returned by a tool.

FORMATTING:
- Minimal markdown only: **bold** labels and bullet points (-). No ## or ### headers.
- Max 3 sections, max 5 bullets per section.
- Under 220 words for factual answers. Plans can be longer.
- Short follow-up replies: skip heading and tip, answer naturally.
- Trainer tone: encouraging, direct, specific. Address {name} by name occasionally."""


# ─────────────────────────────────────────────────────────────────────────────
# Forced retrieval — keyword-triggered eager search_pte_knowledge
# ─────────────────────────────────────────────────────────────────────────────
#
# Why: Claude has been observed to ignore the "MUST search" guardrail and answer
# factual scoring/strategy questions from training data (e.g. claiming Read Aloud
# contributes to Reading). We run the search eagerly on the server before Claude
# ever sees the message, then inject the result into the system prompt as
# PRE-RETRIEVED KNOWLEDGE so it can't be skipped.

_FACTUAL_KEYWORDS = (
    # scoring / contribution terms
    "contribute", "contribution", "contributes", "counts toward", "counts towards",
    "feed into", "feeds into", "affect reading", "affect speaking", "affect writing",
    "affect listening", "weight", "weighted", "weighting", "weightage",
    "max points", "maximum points", "max marks", "points to", "how many points",
    "scoring", "how is it scored", "how scored", "score breakdown", "scoring rule",
    "reading score", "speaking score", "writing score", "listening score",
    "pte formula", "score formula", "scaled score", "raw score",
    # task-format / strategy
    "strategy", "tips", "rules", "format", "how to score",
    "time limit", "word limit", "recording time", "prep time",
    "scoring criteria", "judging", "judged on", "judge",
    # explicit fact-check phrasing
    "is it true", "does it contribute", "does it affect",
)

_TASK_NAME_PATTERNS = (
    "read aloud", "repeat sentence", "describe image", "retell lecture",
    "re-tell lecture", "answer short question", "summarize group discussion",
    "respond to a situation", "respond to situation",
    "summarize written text", "swt", "write essay",
    "fill in the blanks", "fib", "reorder paragraph", "re-order paragraph",
    "highlight incorrect words", "hiw", "highlight correct summary", "hcs",
    "select missing word", "smw", "write from dictation", "wfd",
    "summarize spoken text", "sst",
    "multiple choice", "mcq", "drag and drop",
)

_SECTION_WORDS = ("reading", "speaking", "writing", "listening", "score", "scores")


def _should_force_retrieval(message: str) -> bool:
    m = (message or "").lower().strip()
    if not m or len(m) < 4:
        return False
    if any(kw in m for kw in _FACTUAL_KEYWORDS):
        return True
    if any(t in m for t in _TASK_NAME_PATTERNS) and any(w in m for w in _SECTION_WORDS):
        return True
    return False


def _pre_retrieve_knowledge(user_message: str, ctx: ToolContext) -> Optional[str]:
    """
    Eagerly call search_pte_knowledge if the user message looks like a factual
    PTE question. Returns the retrieved text on success, None otherwise.
    """
    if not _should_force_retrieval(user_message):
        return None
    try:
        result = execute_tool(
            "search_pte_knowledge",
            {"query": user_message[:200]},
            ctx,
        )
    except Exception as e:
        log.warning("[CLAUDE_ROUTER] pre-retrieve failed: %s", e)
        return None
    if not result:
        return None
    if result.startswith("NO_RELEVANT_CONTENT") or result.startswith("SEARCH_UNAVAILABLE"):
        log.info("[CLAUDE_ROUTER] pre-retrieve: %s", result[:60])
        return None
    log.info("[CLAUDE_ROUTER] pre-retrieve hit (%d chars)", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool call extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tool_calls(content: list) -> list[dict]:
    """Extract tool_use blocks from Claude response content."""
    calls = []
    for block in content:
        if getattr(block, "type", None) == "tool_use":
            calls.append({
                "id":   block.id,
                "name": block.name,
                "args": block.input or {},
            })
    return calls


def _extract_text(content: list) -> str:
    """Extract text from Claude response content blocks."""
    parts = [b.text for b in content if getattr(b, "type", None) == "text" and b.text]
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Core async handler (non-streaming)
# ─────────────────────────────────────────────────────────────────────────────

async def get_reply(
    user_message:       str,
    context:            dict,
    trainer_profile:    dict,
    phase:              str,
    completeness_flags: list,
    new_practice:       list,
    conversation_messages: list,
    user_id:            int,
    db:                 Session,
    username:           str = "Student",
) -> str:
    ctx = ToolContext(user_id=user_id, db=db, username=username)

    # ── Forced retrieval for factual questions (Lever 1) ──────────────────────
    pre_retrieved = await asyncio.to_thread(_pre_retrieve_knowledge, user_message, ctx)

    system_prompt = _build_system_prompt(
        context, trainer_profile, phase, completeness_flags, new_practice,
        pre_retrieved=pre_retrieved,
    )

    # Build message history
    messages = []
    for m in conversation_messages[-10:]:
        messages.append({"role": m["role"], "content": m["content"][:1000]})
    messages.append({"role": "user", "content": user_message})

    tool_call_counts: dict[str, int] = {k: 0 for k in TOOL_CALL_LIMITS}
    total_tool_calls = 0
    # Pre-retrieval consumed one search_pte_knowledge call — Claude may do at
    # most 1 more (per TOOL_CALL_LIMITS["search_pte_knowledge"] = 2).
    if pre_retrieved is not None:
        tool_call_counts["search_pte_knowledge"] = 1
        total_tool_calls = 1

    for roundtrip in range(MAX_ROUNDTRIPS + 1):
        log.info("[CLAUDE_ROUTER] roundtrip=%d", roundtrip)

        response = await asyncio.to_thread(
            _client.messages.create,
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt,
            tools=get_tool_schemas(),
            messages=messages,
        )

        # ── End turn → return text ────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            return _extract_text(response.content) or "Sorry, I couldn't generate a response."

        # ── Hard cap on roundtrips ────────────────────────────────────────────
        if roundtrip >= MAX_ROUNDTRIPS:
            log.warning("[GUARDRAIL] max roundtrips reached")
            text = _extract_text(response.content)
            return text or "I've reached my processing limit. Please try again."

        # ── Tool use ──────────────────────────────────────────────────────────
        raw_calls = _extract_tool_calls(response.content)
        if not raw_calls:
            return _extract_text(response.content) or "No response generated."

        # Apply per-tool limits and total cap
        allowed_calls = []
        for call in raw_calls:
            name = call["name"]
            limit = TOOL_CALL_LIMITS.get(name, 1)
            if tool_call_counts.get(name, 0) >= limit:
                log.warning("[GUARDRAIL] tool=%s exceeded limit=%d, skipping", name, limit)
                allowed_calls.append({**call, "_blocked": True})
                continue
            if total_tool_calls >= MAX_TOOL_CALLS:
                log.warning("[GUARDRAIL] total tool calls cap (%d) reached", MAX_TOOL_CALLS)
                allowed_calls.append({**call, "_blocked": True})
                continue
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
            total_tool_calls += 1
            allowed_calls.append(call)

        # Execute allowed calls in parallel, return blocked ones as limit messages
        to_execute = [c for c in allowed_calls if not c.get("_blocked")]
        blocked    = [c for c in allowed_calls if c.get("_blocked")]

        tool_results = []
        if to_execute:
            tool_results = await execute_tools_parallel(to_execute, ctx)

        for c in blocked:
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": c["id"],
                "content":     "Tool call limit reached. Use information already available.",
            })

        # Append Claude's response + tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})

    return "Sorry, I ran into an issue. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# Streaming async generator
# ─────────────────────────────────────────────────────────────────────────────

async def stream_reply(
    user_message:       str,
    context:            dict,
    trainer_profile:    dict,
    phase:              str,
    completeness_flags: list,
    new_practice:       list,
    conversation_messages: list,
    user_id:            int,
    db:                 Session,
    username:           str = "Student",
) -> AsyncGenerator[str, None]:
    ctx = ToolContext(user_id=user_id, db=db, username=username)

    # ── Forced retrieval for factual questions (Lever 1) ──────────────────────
    pre_retrieved = await asyncio.to_thread(_pre_retrieve_knowledge, user_message, ctx)

    system_prompt = _build_system_prompt(
        context, trainer_profile, phase, completeness_flags, new_practice,
        pre_retrieved=pre_retrieved,
    )

    messages = []
    for m in conversation_messages[-10:]:
        messages.append({"role": m["role"], "content": m["content"][:1000]})
    messages.append({"role": "user", "content": user_message})

    tool_call_counts: dict[str, int] = {k: 0 for k in TOOL_CALL_LIMITS}
    total_tool_calls = 0
    if pre_retrieved is not None:
        tool_call_counts["search_pte_knowledge"] = 1
        total_tool_calls = 1

    for roundtrip in range(MAX_ROUNDTRIPS + 1):
        log.info("[CLAUDE_ROUTER/STREAM] roundtrip=%d", roundtrip)

        # ── Non-final roundtrips: use non-streaming to collect tool calls ─────
        if roundtrip < MAX_ROUNDTRIPS:
            response = await asyncio.to_thread(
                _client.messages.create,
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=system_prompt,
                tools=get_tool_schemas(),
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                # Claude answered without tools — stream the text char by char
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

            # Apply limits
            allowed_calls, blocked = [], []
            for call in raw_calls:
                name  = call["name"]
                limit = TOOL_CALL_LIMITS.get(name, 1)
                if tool_call_counts.get(name, 0) >= limit or total_tool_calls >= MAX_TOOL_CALLS:
                    log.warning("[GUARDRAIL/STREAM] tool=%s blocked", name)
                    blocked.append(call)
                    continue
                tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
                total_tool_calls += 1
                allowed_calls.append(call)

            tool_results = []
            if allowed_calls:
                tool_results = await execute_tools_parallel(allowed_calls, ctx)
            for c in blocked:
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": c["id"],
                    "content":     "Tool call limit reached.",
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})

        else:
            # ── Final roundtrip: stream the response ──────────────────────────
            log.warning("[GUARDRAIL/STREAM] max roundtrips reached, streaming final")
            async with _async_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=system_prompt,
                tools=get_tool_schemas(),
                messages=messages,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    yield text_chunk
            return

    # ── Last roundtrip: stream ────────────────────────────────────────────────
    async with _async_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        tools=get_tool_schemas(),
        messages=messages,
    ) as stream:
        async for text_chunk in stream.text_stream:
            yield text_chunk


def _fake_stream(text: str, chunk_size: int = 20):
    """Yield text in small chunks to simulate streaming for non-streamed Claude calls."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
