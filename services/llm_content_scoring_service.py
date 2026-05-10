"""
LLM content scoring for speaking and writing tasks.
Ported from englishfirm-app-fastapi. Uses gpt-4o-mini.

Return types for the speaking-side helpers (extract_key_points and
score_content_with_llm) are dicts so callers can distinguish a *real*
zero ("the student was off-topic") from a *synthetic* zero ("the LLM
was unreachable"). See W7/W8 in the speaking-pipeline reliability work.
"""
import json
import logging
import os
import time
from typing import Optional

import openai as _openai_module
from openai import OpenAI

log = logging.getLogger(__name__)

_openai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_LLM_MODEL = "gpt-4o-mini"


def _call_llm(prompt: str, max_tokens: int = 200) -> dict:
    last_exc: Exception = RuntimeError("_call_llm: no attempts made")
    for attempt in range(1, 4):
        try:
            resp = _openai.chat.completions.create(
                model=_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            return json.loads(raw)
        except _openai_module.AuthenticationError as exc:
            log.error("[LLM] AuthenticationError — not retrying: %s", exc)
            raise
        except Exception as exc:
            last_exc = exc
            log.warning(
                "[LLM] _call_llm attempt=%d/3 failed — %s: %s",
                attempt, type(exc).__name__, exc,
            )
            if attempt < 3:
                time.sleep(2)

    log.error(
        "[LLM] _call_llm failed after 3 attempts — exception=%s: %s",
        type(last_exc).__name__, last_exc,
    )
    raise last_exc


# ── Stimulus key-point extraction (RL / RTS / SGD) ───────────────────────────

def extract_key_points(transcript: str, question_type: str) -> dict:
    """Extract 4-5 key points from a stimulus transcript for RL/RTS/SGD.

    Returns a dict so the caller can distinguish "no key points because the
    transcript is empty / LLM said none" from "no key points because the LLM
    call itself failed":
        {
            "key_points": list[str],  # may be []
            "scored":     bool,        # True if LLM returned cleanly
            "warning_code": Optional[str],
        }
    """
    if not transcript.strip():
        return {"key_points": [], "scored": True, "warning_code": None}

    _prompts = {
        "retell_lecture": (
            "Extract exactly 4-5 key points from this lecture transcript. "
            "Each key point must be a complete factual statement that a good student retelling should cover. "
            "Return a JSON array of strings only, no other text.\n\n"
            f"Transcript:\n{transcript}"
        ),
        "summarize_group_discussion": (
            "Extract exactly 4-5 key points from this group discussion transcript. "
            "Each should capture a distinct main viewpoint or argument expressed by participants. "
            "Return a JSON array of strings only, no other text.\n\n"
            f"Transcript:\n{transcript}"
        ),
        "respond_to_situation": (
            "Based on this situation prompt, extract 3-4 key elements that a good response must address. "
            "Return a JSON array of strings only, no other text.\n\n"
            f"Situation:\n{transcript}"
        ),
    }
    prompt = _prompts.get(question_type, (
        "Extract 4-5 key points that a good student response should cover. "
        "Return a JSON array of strings only.\n\n"
        f"Text:\n{transcript}"
    ))
    last_exc: Exception = RuntimeError("extract_key_points: no attempts made")
    for attempt in range(1, 4):
        try:
            resp = _openai.chat.completions.create(
                model=_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            result = json.loads(raw)
            kp = result if isinstance(result, list) else []
            return {"key_points": kp, "scored": True, "warning_code": None}
        except _openai_module.AuthenticationError as exc:
            log.error("[LLM] extract_key_points AuthenticationError: %s", exc)
            return {
                "key_points": [],
                "scored": False,
                "warning_code": "content_keypoints_unavailable",
            }
        except Exception as exc:
            last_exc = exc
            log.warning("[LLM] extract_key_points attempt=%d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2)
    log.error("[LLM] extract_key_points failed after 3 attempts: %s", last_exc)
    return {
        "key_points": [],
        "scored": False,
        "warning_code": "content_keypoints_unavailable",
    }


# ── Speaking content scoring (DI / RL / RTS / SGD) ───────────────────────────

def score_content_with_llm(student_transcript: str, key_points: list, question_type: str) -> dict:
    """Score student response against key points.

    Returns a dict so the caller (and ultimately the user) can tell a real
    zero apart from an LLM-unavailable zero, and so the trainer review can
    surface the LLM's reasoning instead of dropping it on the log floor:
        {
            "score":        float,             # 0-100
            "scored":       bool,              # True if LLM returned cleanly
            "reasoning":    Optional[str],     # only present when scored
            "warning_code": Optional[str],     # set when scored=False
        }

    A return with scored=False and score=0.0 means "no LLM signal — do not
    treat this 0 as the student's content score". The caller is expected to
    surface the warning_code via result_json.scoring_warnings.
    """
    if not student_transcript.strip():
        return {"score": 0.0, "scored": True, "reasoning": None, "warning_code": None}
    if not key_points:
        # No key points to score against — caller must decide whether to
        # treat that as legitimate or as a degraded path.
        return {"score": 0.0, "scored": True, "reasoning": None, "warning_code": None}

    kp_text = "\n".join(f"- {kp}" for kp in key_points)

    _prompts = {
        "retell_lecture": (
            "You are scoring a PTE Retell Lecture response.\n\n"
            f"Key points the student should cover:\n{kp_text}\n\n"
            f"Student's response:\n{student_transcript}\n\n"
            "Score how well the student covered the key points (0-100). "
            "Accurate paraphrasing counts — exact wording is not required. "
            'Return JSON only: {"score": <int 0-100>, "reasoning": "<one sentence>"}'
        ),
        "summarize_group_discussion": (
            "You are scoring a PTE Summarize Group Discussion response.\n\n"
            f"Key viewpoints from the discussion:\n{kp_text}\n\n"
            f"Student's response:\n{student_transcript}\n\n"
            "Score how well the student summarised the discussion viewpoints (0-100). "
            'Return JSON only: {"score": <int 0-100>, "reasoning": "<one sentence>"}'
        ),
        "respond_to_situation": (
            "You are scoring a PTE Respond to Situation response.\n\n"
            f"Key elements the response must address:\n{kp_text}\n\n"
            f"Student's response:\n{student_transcript}\n\n"
            "Score how well the student addressed all required elements (0-100). "
            'Return JSON only: {"score": <int 0-100>, "reasoning": "<one sentence>"}'
        ),
        "describe_image": (
            "You are scoring a PTE Describe Image response.\n\n"
            f"Key elements the student should describe:\n{kp_text}\n\n"
            f"Student's response:\n{student_transcript}\n\n"
            "Score how accurately and completely the student described the image (0-100). "
            'Return JSON only: {"score": <int 0-100>, "reasoning": "<one sentence>"}'
        ),
    }
    prompt = _prompts.get(question_type, _prompts["describe_image"])

    try:
        result = _call_llm(prompt)
        score = max(0.0, min(100.0, float(result.get("score", 0))))
        reasoning = (result.get("reasoning") or "").strip() or None
        log.info("[LLM-CONTENT] %s score=%.1f reason=%s", question_type, score, reasoning or "")
        return {
            "score": score,
            "scored": True,
            "reasoning": reasoning,
            "warning_code": None,
        }
    except Exception as e:
        log.warning("[LLM-CONTENT] Fallback for %s: %s", question_type, e)
        return {
            "score": 0.0,
            "scored": False,
            "reasoning": None,
            "warning_code": "content_llm_unavailable",
        }


# ── SWT ───────────────────────────────────────────────────────────────────────

def score_swt_content(passage: str, user_text: str) -> int:
    """Score SWT content 0-4. Raises on LLM failure — caller stays pending."""
    if not user_text.strip():
        return 0
    prompt = (
        "You are a PTE Academic examiner scoring a Summarize Written Text response.\n\n"
        f"PASSAGE:\n{passage[:1500]}\n\n"
        f"STUDENT SUMMARY:\n{user_text}\n\n"
        "Score CONTENT only (0-4):\n"
        "4 = accurately captures the main idea and key supporting details\n"
        "3 = captures the main idea, minor gaps or inaccuracies\n"
        "2 = captures some ideas but significant gaps or inaccuracies\n"
        "1 = minimal relevant content, mostly off-topic\n"
        "0 = completely off-topic, empty, or verbatim copy of passage\n\n"
        'Return JSON only: {"content": <0-4>, "reasoning": "<one sentence>"}'
    )
    result = _call_llm(prompt)
    score = max(0, min(4, int(result.get("content", 0))))
    log.info("[LLM-SWT] content=%d reason=%s", score, result.get("reasoning", ""))
    return score


# ── WE ────────────────────────────────────────────────────────────────────────

def score_we_content(essay_prompt: str, user_text: str) -> int:
    """Score WE content 0-3. Raises on LLM failure — caller stays pending."""
    if not user_text.strip():
        return 0
    prompt = (
        "You are a PTE Academic examiner scoring a Write Essay response.\n\n"
        f"ESSAY PROMPT:\n{essay_prompt[:500]}\n\n"
        f"STUDENT ESSAY:\n{user_text[:2000]}\n\n"
        "Score CONTENT only (0-3):\n"
        "3 = fully addresses the topic with well-developed arguments and relevant examples\n"
        "2 = adequately addresses the topic but arguments lack development or depth\n"
        "1 = partially addresses the topic, significant gaps\n"
        "0 = does not address the topic, completely irrelevant or plagiarised prompt\n\n"
        'Return JSON only: {"content": <0-3>, "reasoning": "<one sentence>"}'
    )
    result = _call_llm(prompt)
    score = max(0, min(3, int(result.get("content", 0))))
    log.info("[LLM-WE] content=%d reason=%s", score, result.get("reasoning", ""))
    return score


# ── SST ───────────────────────────────────────────────────────────────────────

def score_sst_content(reference_answer: str, user_text: str) -> int:
    """Score SST content 0-4. Raises on LLM failure — caller stays pending."""
    if not user_text.strip():
        return 0
    prompt = (
        "You are a PTE Academic examiner scoring a Summarize Spoken Text response.\n\n"
        f"REFERENCE ANSWER (what the audio was about):\n{reference_answer[:1500]}\n\n"
        f"STUDENT SUMMARY:\n{user_text}\n\n"
        "The student listened to a lecture and must summarise it in 50-70 words.\n"
        "Score CONTENT only (0-4):\n"
        "4 = accurately captures main ideas and key details\n"
        "3 = captures main ideas, minor gaps\n"
        "2 = partial capture, significant gaps\n"
        "1 = minimal relevant content\n"
        "0 = off-topic or empty\n\n"
        'Return JSON only: {"content": <0-4>, "reasoning": "<one sentence>"}'
    )
    result = _call_llm(prompt)
    score = max(0, min(4, int(result.get("content", 0))))
    log.info("[LLM-SST] content=%d reason=%s", score, result.get("reasoning", ""))
    return score
