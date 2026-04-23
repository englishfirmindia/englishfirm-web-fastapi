"""
Tool Registry — central definition of all Claude tools.

Adding a new tool:
  1. Add its schema to TOOL_REGISTRY
  2. Add its handler to _HANDLERS
  Done. claude_router picks it up automatically.

ToolContext is passed to every handler so tools have auth'd user context
without relying on globals.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
VECTOR_STORE_ID = "vs_68a7e0e054dc8191aa4704c083d228ec"

_openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── Max chars returned per tool call (prevents context blowout) ───────────────
_MAX_RESULT_CHARS = 3000

# ── Per-tool call limits (enforced in claude_router, declared here) ───────────
TOOL_CALL_LIMITS: dict[str, int] = {
    "search_pte_knowledge":      2,
    "get_last_attempt_breakdown": 1,
    "get_milestones":            1,
    "save_student_info":         1,
    "get_attempt_detail":        1,
}


# ─────────────────────────────────────────────────────────────────────────────
# ToolContext — injected by the router, never trusted from Claude
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolContext:
    user_id:  int
    db:       Session
    username: str = "Student"


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (Anthropic format)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict] = {

    "search_pte_knowledge": {
        "schema": {
            "name": "search_pte_knowledge",
            "description": (
                "Search the PTE Academic knowledge base for exam rules, scoring, "
                "task formats, strategies, and tips. Use this when the student asks "
                "a factual question about PTE that isn't in the student profile."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query — be precise for best results.",
                    }
                },
                "required": ["query"],
            },
        },
    },

    "get_last_attempt_breakdown": {
        "schema": {
            "name": "get_last_attempt_breakdown",
            "description": (
                "Get this student's most recent attempt score for each PTE task type "
                "from the database. Use when the student asks about their scores, "
                "performance, or how they did on a specific task."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },

    "get_milestones": {
        "schema": {
            "name": "get_milestones",
            "description": (
                "Get this student's achieved milestones and their next milestone to unlock. "
                "Use when celebrating progress or motivating the student."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },

    "get_attempt_detail": {
        "schema": {
            "name": "get_attempt_detail",
            "description": (
                "Get a question-by-question breakdown of the student's most recent attempt "
                "for a specific question type. Use when the student asks how they did on "
                "individual questions, wants to review a specific task in detail, or asks "
                "which questions they got wrong."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question_type": {
                        "type": "string",
                        "description": (
                            "Use exactly one of these values:\n"
                            "  read_aloud — Read Aloud\n"
                            "  repeat_sentence — Repeat Sentence\n"
                            "  answer_short_question — Answer Short Question\n"
                            "  describe_image — Describe Image\n"
                            "  retell_lecture — Re-tell Lecture\n"
                            "  summarize_group_discussion — Summarize Group Discussion\n"
                            "  ptea_respond_situation — Respond to a Situation\n"
                            "  summarize_written_text — Summarize Written Text\n"
                            "  write_essay — Write Essay\n"
                            "  mcq_single — Reading Multiple Choice (Single)\n"
                            "  mcq_multiple — Reading Multiple Choice (Multiple)\n"
                            "  reading_fib_drop_down — Reading Fill in the Blanks\n"
                            "  reading_drag_and_drop — Reading Fill in the Blanks (Drag & Drop)\n"
                            "  reorder_paragraphs — Re-order Paragraphs\n"
                            "  listening_wfd — Write from Dictation\n"
                            "  listening_fib — Listening Fill in the Blanks\n"
                            "  listening_hcs — Highlight Correct Summary\n"
                            "  listening_smw — Select Missing Word\n"
                            "  listening_mcq_single — Listening MCQ (Single)\n"
                            "  listening_mcq_multiple — Listening MCQ (Multiple)\n"
                            "  summarize_spoken_text — Summarize Spoken Text\n"
                            "  highlight_incorrect_words — Highlight Incorrect Words"
                        ),
                    }
                },
                "required": ["question_type"],
            },
        },
    },

    "save_student_info": {
        "schema": {
            "name": "save_student_info",
            "description": (
                "Save information the student has revealed during conversation. "
                "Call this when the student mentions motivation, study schedule, "
                "anxiety, prior PTE attempts, or biggest weakness. All fields optional."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "motivation": {
                        "type": "string",
                        "description": "Why they need PTE (e.g. 'Australian PR visa')",
                    },
                    "study_hours_per_day": {
                        "type": "number",
                        "description": "Hours per day they can study",
                    },
                    "study_schedule": {
                        "type": "string",
                        "description": "When they study (e.g. '1hr morning, 30min evening')",
                    },
                    "prior_pte_attempts": {
                        "type": "integer",
                        "description": "Number of previous PTE attempts (0 if first time)",
                    },
                    "anxiety_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Student's anxiety or confidence level",
                    },
                    "biggest_weakness_self": {
                        "type": "string",
                        "description": "What they personally feel is their biggest weakness",
                    },
                    "plan_text": {
                        "type": "string",
                        "description": "The training plan generated for the student",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Handlers — one per tool, always take (args, ctx) and return str
# ─────────────────────────────────────────────────────────────────────────────

# Whitelist for save_student_info to prevent prompt-injection DB writes
_SAVE_FIELD_WHITELIST = {
    "motivation", "study_hours_per_day", "study_schedule",
    "prior_pte_attempts", "anxiety_level", "biggest_weakness_self", "plan_text",
}

_ANXIETY_VALUES = {"low", "medium", "high"}


def _handle_search_pte_knowledge(args: dict, ctx: ToolContext) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "No query provided."
    try:
        results = _openai_client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=query,
            max_num_results=5,
        )
        chunks = [r.content[0].text for r in results.data if r.content]
        text = "\n\n".join(chunks[:5]) if chunks else "No relevant content found."
        return text[:_MAX_RESULT_CHARS]
    except Exception as e:
        log.warning("[TOOL] search_pte_knowledge error: %s", e)
        return "Search unavailable. Use your PTE expertise to answer."


def _handle_get_last_attempt_breakdown(args: dict, ctx: ToolContext) -> str:
    try:
        from mcp_server.tools import get_last_attempt_breakdown
        data = get_last_attempt_breakdown(ctx.user_id, ctx.db)
        if not data:
            return "No completed practice attempts found for this student."

        lines = []
        # First surface sub-task breakdown from sectional attempts
        for qt, info in data.items():
            # If this is a sectional attempt with sub-task breakdown, expand it
            sub_tasks = {k: v for k, v in info.items()
                         if isinstance(v, dict) and "pct" in v and "count" in v}
            if sub_tasks:
                for sub, sub_info in sub_tasks.items():
                    pct   = int(sub_info.get("pct", 0) * 100)
                    avg   = round(sub_info.get("avg_total", 0), 1)
                    maxp  = sub_info.get("max_points", 1)
                    count = sub_info.get("count", 0)
                    lines.append(f"{sub}: avg {avg}/{maxp} ({pct}%) over {count} questions")
            else:
                # Simple pass/fail question type (e.g. individual mcq)
                score = info.get("total_score", 0)
                total = info.get("total_questions", 1)
                pct   = int((score / total) * 100) if total else 0
                # Skip zero-data entries (incomplete attempts)
                if total > 0 and (score > 0 or total <= 5):
                    lines.append(f"{qt}: {score}/{total} ({pct}%)")

        if not lines:
            return "No meaningful attempt data found yet."
        return "\n".join(lines)[:_MAX_RESULT_CHARS]
    except Exception as e:
        log.warning("[TOOL] get_last_attempt_breakdown error: %s", e)
        return "Score data unavailable."


def _handle_get_milestones(args: dict, ctx: ToolContext) -> str:
    try:
        from mcp_server.tools import get_milestones
        data = get_milestones(ctx.user_id, ctx.db)
        achieved = data.get("achieved", [])
        next_m   = data.get("next_milestone")
        total    = data.get("total_practices", 0)

        lines = [f"Total practice sessions: {total}"]
        if achieved:
            lines.append(f"Milestones achieved ({len(achieved)}):")
            for m in achieved[-5:]:  # last 5 to keep short
                lines.append(f"  ✅ {m['label']}")
        else:
            lines.append("No milestones achieved yet.")
        if next_m:
            lines.append(
                f"Next milestone: {next_m['label']} "
                f"({next_m['progress']}/{next_m['target']} sessions)"
            )
        return "\n".join(lines)[:_MAX_RESULT_CHARS]
    except Exception as e:
        log.warning("[TOOL] get_milestones error: %s", e)
        return "Milestone data unavailable."


def _handle_get_attempt_detail(args: dict, ctx: ToolContext) -> str:
    qt = str(args.get("question_type", "")).strip()
    if not qt:
        return "No question_type provided."
    try:
        from mcp_server.tools import get_attempt_detail
        data = get_attempt_detail(ctx.user_id, qt, ctx.db)
        if not data:
            return f"No completed attempts found for question type: {qt}"

        lines = [
            f"Question type: {data['question_type']}",
            f"Completed: {(data.get('completed_at') or '')[:10]}",
            f"Total score: {data['total_score']} / {data['total_questions']} questions",
            "",
        ]
        for a in data["answers"]:
            parts = [f"Q{a['q']} (id={a['question_id']}): score={a['score']}"]
            if a.get("pte_score") is not None:
                parts.append(f"pte={a['pte_score']}")
            if a.get("content") is not None:
                parts.append(f"content={a['content']}")
            if a.get("fluency") is not None:
                parts.append(f"fluency={a['fluency']}")
            if a.get("pronunciation") is not None:
                parts.append(f"pron={a['pronunciation']}")
            if a.get("hits") is not None:
                parts.append(f"hits={a['hits']}/{a.get('total_words', '?')}")
            if a.get("max_score") is not None:
                parts.append(f"max={a['max_score']}")
            if a.get("scoring_status") not in (None, "complete", "scored"):
                parts.append(f"[{a['scoring_status']}]")
            lines.append("  " + "  ".join(parts))

        return "\n".join(lines)[:_MAX_RESULT_CHARS]
    except Exception as e:
        log.warning("[TOOL] get_attempt_detail error: %s", e)
        return "Attempt detail unavailable."


def _handle_save_student_info(args: dict, ctx: ToolContext) -> str:
    # Validate and sanitise before writing
    clean: dict = {}
    for field, value in args.items():
        if field not in _SAVE_FIELD_WHITELIST:
            log.warning("[GUARDRAIL] save_student_info rejected unknown field: %s", field)
            continue
        if field == "anxiety_level" and value not in _ANXIETY_VALUES:
            log.warning("[GUARDRAIL] save_student_info invalid anxiety_level: %s", value)
            continue
        if field == "study_hours_per_day":
            try:
                value = float(value)
                if value < 0 or value > 24:
                    continue
            except (TypeError, ValueError):
                continue
        if field == "prior_pte_attempts":
            try:
                value = int(value)
                if value < 0:
                    continue
            except (TypeError, ValueError):
                continue
        clean[field] = value

    if not clean:
        return "Nothing to save."

    try:
        from mcp_server.tools import save_trainer_info
        result = save_trainer_info(ctx.user_id, ctx.db, **clean)
        saved = result.get("saved", [])
        log.info("[TOOL] save_student_info saved=%s for user_id=%s", saved, ctx.user_id)
        return f"Saved: {', '.join(saved)}"
    except Exception as e:
        log.warning("[TOOL] save_student_info error: %s", e)
        return "Could not save student info."


# ── Handler dispatch map ──────────────────────────────────────────────────────

_HANDLERS: dict[str, Any] = {
    "search_pte_knowledge":       _handle_search_pte_knowledge,
    "get_last_attempt_breakdown": _handle_get_last_attempt_breakdown,
    "get_milestones":             _handle_get_milestones,
    "get_attempt_detail":         _handle_get_attempt_detail,
    "save_student_info":          _handle_save_student_info,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_tool_schemas(names: Optional[list[str]] = None) -> list[dict]:
    """Return Anthropic-format tool schemas. If names=None, return all."""
    keys = names if names else list(TOOL_REGISTRY.keys())
    return [TOOL_REGISTRY[k]["schema"] for k in keys if k in TOOL_REGISTRY]


def execute_tool(name: str, args: dict, ctx: ToolContext) -> str:
    """Execute a single tool synchronously. Returns string result."""
    handler = _HANDLERS.get(name)
    if not handler:
        log.warning("[TOOL] unknown tool requested: %s", name)
        return f"Tool '{name}' is not available."
    try:
        return handler(args, ctx)
    except Exception as e:
        log.error("[TOOL] %s execution error: %s", name, e)
        return "Tool error. Please continue without this data."


async def execute_tools_parallel(
    calls: list[dict],   # list of {"id": str, "name": str, "args": dict}
    ctx: ToolContext,
    timeout_secs: float = 8.0,
) -> list[dict]:
    """
    Execute multiple tool calls in parallel via asyncio.gather.
    Wall time = max(individual tools), not sum.
    Each result: {"tool_use_id": str, "content": str}
    """
    async def _run_one(call: dict) -> dict:
        name = call["name"]
        args = call["args"]
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(execute_tool, name, args, ctx),
                timeout=timeout_secs,
            )
        except asyncio.TimeoutError:
            log.warning("[GUARDRAIL] tool timeout: %s after %.0fs", name, timeout_secs)
            result = "Tool timed out. Continue without this data."
        return {
            "type":        "tool_result",
            "tool_use_id": call["id"],
            "content":     result,
        }

    return list(await asyncio.gather(*[_run_one(c) for c in calls]))
