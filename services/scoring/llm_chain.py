"""
Cross-provider LLM judge with automatic primary → backup fallback.

Each provider gets 3 attempts with 2-s linear back-off and a 30-s
per-attempt timeout (mirrors the existing `_call_llm` retry envelope in
`llm_content_scoring_service.py`). On full exhaustion of the primary,
the same prompt is retried via the backup. Both providers failed →
the caller receives `data=None` + `warning="llm_unavailable"` and is
expected to surface the warning into `result_json.scoring_warnings`.

Caller signature:
    judge_json(prompt, primary="gpt", backup="claude") -> {
        "data":    dict | None,    # parsed JSON; None if both failed
        "source":  str | None,     # "gpt-4o-mini" | "claude-haiku-4-5" | None
        "warning": str | None,     # "primary_<x>_failed" | "llm_unavailable"
    }

Used by `score_content_with_llm` and `extract_key_points` to close the
DI / RL / RTS / SGD content-scoring gap where a single GPT outage today
floors the user's score to 0 with no recovery path.
"""
import json
import os
import time
from typing import Literal, Optional

from core.logging_config import get_logger

log = get_logger(__name__)

_GPT_MODEL = "gpt-4o-mini"
_CLAUDE_MODEL = "claude-haiku-4-5"

# Lazy clients — avoid import cost when the chain isn't exercised.
_openai_client = None
_anthropic_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            raw = parts[1].lstrip("json").strip()
    return raw


def _call_gpt(prompt: str, max_tokens: int) -> dict:
    """Three-attempt retry against gpt-4o-mini. Raises after full exhaustion."""
    import openai as _openai_mod
    last_exc: Exception = RuntimeError("_call_gpt: no attempts made")
    for attempt in range(1, 4):
        try:
            resp = _get_openai().chat.completions.create(
                model=_GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=30,
            )
            raw = _strip_fences(resp.choices[0].message.content or "")
            return json.loads(raw)
        except _openai_mod.AuthenticationError as exc:
            log.error("[LLM_CHAIN] GPT auth error — not retrying: %s", exc)
            raise
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[LLM_CHAIN] GPT attempt=%d/3 failed — %s: %s",
                attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)
    raise last_exc


def _call_claude(prompt: str, max_tokens: int) -> dict:
    """Three-attempt retry against claude-haiku-4-5. Raises after full exhaustion."""
    last_exc: Exception = RuntimeError("_call_claude: no attempts made")
    # Claude wraps JSON in prose occasionally; nudge for clean JSON.
    final_prompt = prompt + "\n\nReturn JSON only, no preamble, no markdown fences."
    for attempt in range(1, 4):
        try:
            resp = _get_anthropic().messages.create(
                model=_CLAUDE_MODEL,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": final_prompt}],
                timeout=30,
            )
            text_parts = [
                getattr(block, "text", "")
                for block in (resp.content or [])
                if getattr(block, "type", "") == "text"
            ]
            raw = _strip_fences("".join(text_parts) or "{}")
            return json.loads(raw)
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[LLM_CHAIN] Claude attempt=%d/3 failed — %s: %s",
                attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)
    raise last_exc


def _call(provider: str, prompt: str, max_tokens: int) -> dict:
    return _call_gpt(prompt, max_tokens) if provider == "gpt" else _call_claude(prompt, max_tokens)


def _model_label(provider: str) -> str:
    return _GPT_MODEL if provider == "gpt" else _CLAUDE_MODEL


def judge_json(
    prompt: str,
    *,
    max_tokens: int = 200,
    primary: Literal["gpt", "claude"] = "gpt",
    backup: Literal["gpt", "claude"] = "claude",
) -> dict:
    """Run `prompt` through `primary`; on exhaustion, retry via `backup`.

    Returns:
        {"data": <dict|None>, "source": <model_id|None>, "warning": <code|None>}

    - `data` is the parsed JSON object from whichever provider scored.
    - `source` is the model id of the provider that produced the score.
    - `warning` is `None` on happy path, `primary_<x>_failed` when backup
       scored, `llm_unavailable` when both failed.

    Setting `primary == backup` short-circuits the fallback and behaves
    like a single-provider call.
    """
    # Single-provider mode: no fallback wanted.
    if primary == backup:
        try:
            data = _call(primary, prompt, max_tokens)
            return {"data": data, "source": _model_label(primary), "warning": None}
        except Exception:
            return {"data": None, "source": None, "warning": "llm_unavailable"}

    # Primary attempt
    try:
        data = _call(primary, prompt, max_tokens)
        return {"data": data, "source": _model_label(primary), "warning": None}
    except Exception as primary_exc:
        log.warning(
            "[LLM_CHAIN] primary=%s exhausted, falling back to %s: %s",
            primary, backup, primary_exc,
        )

    # Backup attempt
    try:
        data = _call(backup, prompt, max_tokens)
        log.info(
            "[FALLBACK] axis=llm primary=%s backup=%s reason=primary_exhausted_retries",
            primary, backup,
        )
        return {
            "data": data,
            "source": _model_label(backup),
            "warning": f"primary_{primary}_failed",
        }
    except Exception as backup_exc:
        log.error(
            "[LLM_CHAIN] both providers failed: primary=%s backup=%s err=%s",
            primary, backup, backup_exc,
        )
        return {"data": None, "source": None, "warning": "llm_unavailable"}
