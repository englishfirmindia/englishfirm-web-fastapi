"""
LLM content scoring for speaking and writing tasks.
Ported from englishfirm-app-fastapi. Uses gpt-4o-mini.
"""
import json
import logging
import os

from openai import OpenAI

log = logging.getLogger(__name__)

_openai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_LLM_MODEL = "gpt-4o-mini"


def _call_llm(prompt: str, max_tokens: int = 200) -> dict:
    resp = _openai.chat.completions.create(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return json.loads(raw)


# ── Stimulus key-point extraction (RL / RTS / SGD) ───────────────────────────

def extract_key_points(transcript: str, question_type: str) -> list:
    """Extract 4-5 key points from a stimulus transcript for RL/RTS/SGD."""
    if not transcript.strip():
        return []

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
    try:
        resp = _openai.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception as e:
        log.error("[LLM] Key point extraction failed (%s): %s", question_type, e)
        return []


# ── Speaking content scoring (DI / RL / RTS / SGD) ───────────────────────────

def score_content_with_llm(student_transcript: str, key_points: list, question_type: str) -> float:
    """Score student response against key points. Returns 0-100."""
    if not key_points or not student_transcript.strip():
        return 0.0

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
        log.info("[LLM-CONTENT] %s score=%.1f reason=%s", question_type, score, result.get("reasoning", ""))
        return score
    except Exception as e:
        log.warning("[LLM-CONTENT] Fallback for %s: %s", question_type, e)
        return 0.0


# ── SWT ───────────────────────────────────────────────────────────────────────

def score_swt_content(passage: str, user_text: str) -> int:
    """Score SWT content 0-4. Falls back to word-count heuristic on failure."""
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
    try:
        result = _call_llm(prompt)
        score = max(0, min(4, int(result.get("content", 0))))
        log.info("[LLM-SWT] content=%d reason=%s", score, result.get("reasoning", ""))
        return score
    except Exception as e:
        log.warning("[LLM-SWT] fallback: %s", e)
        return min(4, max(1, len(user_text.split()) // 10))


# ── WE ────────────────────────────────────────────────────────────────────────

def score_we_content(essay_prompt: str, user_text: str) -> int:
    """Score WE content 0-3. Falls back to word-count heuristic on failure."""
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
    try:
        result = _call_llm(prompt)
        score = max(0, min(3, int(result.get("content", 0))))
        log.info("[LLM-WE] content=%d reason=%s", score, result.get("reasoning", ""))
        return score
    except Exception as e:
        log.warning("[LLM-WE] fallback: %s", e)
        wc = len(user_text.split())
        return 3 if wc >= 200 else 2 if wc >= 120 else 1 if wc >= 60 else 0


# ── SST ───────────────────────────────────────────────────────────────────────

def score_sst_content(reference_answer: str, user_text: str) -> int:
    """Score SST content 0-4. Falls back to word-count heuristic on failure."""
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
    try:
        result = _call_llm(prompt)
        score = max(0, min(4, int(result.get("content", 0))))
        log.info("[LLM-SST] content=%d reason=%s", score, result.get("reasoning", ""))
        return score
    except Exception as e:
        log.warning("[LLM-SST] fallback: %s", e)
        wc = len(user_text.split())
        return min(4, max(0, wc // 15))
