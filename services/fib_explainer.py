"""AI explanations for reading FIB-DD and FIB-dropdown answers.

For each blank, produces one teacher-friendly sentence explaining why the
correct word fits. Single LLM call per submission. Claude Haiku 4.5 primary;
GPT-4o fallback if Haiku fails. Returns empty list on total failure so the
UI degrades gracefully (no explanations shown, score card unchanged).

Public API:
  build_passage(content_json) -> str
  generate_fib_explanations(passage, blanks) -> list[dict]
"""
import json
import os
import time
from typing import Optional

import anthropic
from anthropic import Anthropic
from openai import OpenAI, AuthenticationError

from core.logging_config import get_logger

log = get_logger(__name__)

_HAIKU_MODEL = "claude-haiku-4-5"
_GPT4O_MODEL = "gpt-4o"

_anthropic_client: Optional[Anthropic] = None
_openai_client: Optional[OpenAI] = None


def _get_anthropic() -> Optional[Anthropic]:
    global _anthropic_client
    if _anthropic_client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        _anthropic_client = Anthropic(api_key=key)
    return _anthropic_client


def _get_openai() -> Optional[OpenAI]:
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        _openai_client = OpenAI(api_key=key)
    return _openai_client

_SYSTEM_PROMPT = """You are a friendly PTE Academic reading tutor.

For each blank in a fill-in-the-blanks question, write ONE short sentence
(max 25 words) explaining WHY the correct word fits — grammar, collocation,
or meaning from the passage. Use a warm, teacher-friendly tone. Do not
mention "PTE", "blank N", or repeat the word back.

If the student's answer was wrong, focus your sentence on why the CORRECT
word is right — not on shaming the student's choice. If their answer was
right, briefly confirm why it works.

Return ONLY this JSON shape, no commentary:
{
  "explanations": [
    {"blank_id": "1", "explanation": "..."},
    {"blank_id": "2", "explanation": "..."}
  ]
}
"""


def build_passage(content_json: dict) -> str:
    """Reconstruct a flat passage with `___1___` blank markers from the
    question's content_json. Supports both the contentBlocks format and the
    legacy passage+options format. Returns empty string on unknown shape."""
    if not content_json or not isinstance(content_json, dict):
        return ""

    # New contentBlocks format. Real DB shape uses `blockType` and `text`;
    # `type` and `content` are kept as fallbacks in case other surfaces emit
    # the older naming.
    blocks = content_json.get("contentBlocks")
    if isinstance(blocks, list) and blocks:
        parts: list = []
        blank_idx = 0
        for b in blocks:
            if not isinstance(b, dict):
                continue
            btype = b.get("blockType") or b.get("type")
            if btype == "text":
                parts.append(str(b.get("text") or b.get("content") or ""))
            elif btype in ("blank", "dropdown", "dragdrop"):
                blank_idx += 1
                parts.append(f" ___{blank_idx}___ ")
        joined = "".join(parts).strip()
        if joined:
            return joined

    # Legacy passage format with [BLANK] / [blank] / {{blank}} markers
    raw = content_json.get("passage") or content_json.get("text") or ""
    if isinstance(raw, str) and raw:
        import re
        idx = [0]
        def _sub(_m):
            idx[0] += 1
            return f" ___{idx[0]}___ "
        return re.sub(r"\[BLANK\]|\[blank\]|\{\{blank\}\}|_{2,}", _sub, raw).strip()

    return ""


def generate_fib_explanations(
    passage: str,
    blanks: list,
    timeout_seconds: float = 15.0,
) -> list:
    """Generate one-sentence explanations for each blank.

    `blanks` is a list of dicts: [{blank_id, correct, user_answer, is_correct}].
    Returns a list of dicts mirroring input plus an `explanation` string.
    On total LLM failure returns []."""
    if not blanks:
        return []
    if not passage or not passage.strip():
        return []

    user_block = _build_user_block(passage, blanks)
    parsed = _call_haiku(user_block, timeout_seconds)
    scorer = "haiku"
    if parsed is None:
        parsed = _call_gpt4o(user_block, timeout_seconds)
        scorer = "gpt4o"
    if parsed is None:
        log.warning("[FIB-EXPLAIN] both Haiku and GPT-4o failed")
        return []

    # The LLM is told to respond with numeric blank_ids ("1", "2", …) while
    # routers send "blank_1", "blank_2", …. Index by every plausible alias so
    # the lookup matches regardless of which form Haiku/GPT-4o echoes back.
    by_id: dict = {}
    for item in parsed.get("explanations", []) or []:
        if not isinstance(item, dict):
            continue
        bid = str(item.get("blank_id") or item.get("blankId") or "").strip()
        exp = str(item.get("explanation") or "").strip()
        if not bid or not exp:
            continue
        for alias in _id_aliases(bid):
            by_id.setdefault(alias, exp)

    out: list = []
    for b in blanks:
        bid = str(b.get("blank_id"))
        explanation = ""
        for alias in _id_aliases(bid):
            if alias in by_id:
                explanation = by_id[alias]
                break
        out.append({
            "blank_id": bid,
            "correct": b.get("correct"),
            "user_answer": b.get("user_answer"),
            "is_correct": bool(b.get("is_correct")),
            "explanation": explanation,
            "scorer": scorer,
        })
    return out


def _id_aliases(bid: str) -> list:
    """Return all forms the LLM might use for a blank id. e.g. 'blank_1'
    matches '1', 'blank_1', 'Blank 1', '1.', and vice versa."""
    raw = str(bid).strip()
    bare = raw
    for prefix in ("blank_", "Blank ", "blank ", "Blank_"):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break
    bare = bare.rstrip(".").strip()
    seen: list = []
    for x in (raw, bare, f"blank_{bare}", f"Blank {bare}"):
        if x and x not in seen:
            seen.append(x)
    return seen


def _build_user_block(passage: str, blanks: list) -> str:
    rows = []
    for b in blanks:
        # Strip the "blank_" prefix so the LLM sees the bare numeric form
        # used in the JSON example in _SYSTEM_PROMPT and echoes it back.
        bare_id = str(b.get("blank_id") or "").replace("blank_", "").replace("Blank ", "").strip()
        rows.append(
            f"  Blank {bare_id}: correct=\"{b.get('correct')}\", "
            f"student=\"{b.get('user_answer')}\", "
            f"correct?={'yes' if b.get('is_correct') else 'no'}"
        )
    return (
        f"PASSAGE (blanks marked ___1___, ___2___, …):\n{passage}\n\n"
        f"BLANKS:\n" + "\n".join(rows) + "\n\n"
        f"For each blank, write ONE short teacher-friendly sentence "
        f"explaining why the correct word fits. Return JSON only."
    )


def _call_haiku(user_block: str, timeout: float) -> Optional[dict]:
    client = _get_anthropic()
    if client is None:
        return None
    last_exc: Exception = RuntimeError("haiku: no attempts made")
    for attempt in range(1, 3):
        try:
            resp = client.messages.create(
                model=_HAIKU_MODEL,
                max_tokens=1024,
                temperature=0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_block}],
                timeout=timeout,
            )
            text = "".join(
                getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
            return json.loads(text)
        except anthropic.AuthenticationError as exc:
            log.error("[FIB-EXPLAIN haiku] auth error — not retrying: %s", exc)
            return None
        except Exception as exc:
            last_exc = exc
            log.warning("[FIB-EXPLAIN haiku] attempt=%d/2 failed — %s: %s",
                        attempt, type(exc).__name__, exc)
            if attempt < 2:
                time.sleep(1)
    log.warning("[FIB-EXPLAIN haiku] failed after 2 attempts: %s", last_exc)
    return None


def _call_gpt4o(user_block: str, timeout: float) -> Optional[dict]:
    client = _get_openai()
    if client is None:
        return None
    last_exc: Exception = RuntimeError("gpt4o: no attempts made")
    for attempt in range(1, 3):
        try:
            resp = client.chat.completions.create(
                model=_GPT4O_MODEL,
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_block},
                ],
                timeout=timeout,
            )
            text = (resp.choices[0].message.content or "").strip()
            return json.loads(text)
        except AuthenticationError as exc:
            log.error("[FIB-EXPLAIN gpt4o] auth error — not retrying: %s", exc)
            return None
        except Exception as exc:
            last_exc = exc
            log.warning("[FIB-EXPLAIN gpt4o] attempt=%d/2 failed — %s: %s",
                        attempt, type(exc).__name__, exc)
            if attempt < 2:
                time.sleep(1)
    log.warning("[FIB-EXPLAIN gpt4o] failed after 2 attempts: %s", last_exc)
    return None
