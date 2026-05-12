"""Typed request bodies for non-speaking submit endpoints.

Replacing `payload: dict = Body(...)` with these models converts a
missing-field 500 into a structured 422 response and pins the answer
shape at the boundary so downstream scorers can trust the inputs.

One model per answer shape — most endpoints share a shape, so models
are reused across writing/reading/listening routers.

Speaking submits are not covered here; they are validated via
core.security_helpers.safe_question_id + assert_audio_url_owned which
already enforce shape and ownership before hitting the scorer.
"""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ── Shared base ───────────────────────────────────────────────────────────────
class _BaseSubmit(BaseModel):
    session_id: str = Field(..., min_length=1)
    question_id: int
    # Reading practice stopwatch sends elapsed seconds spent on this question.
    # Persisted into result_json so trainer review and analytics can see speed
    # per question. None when the client doesn't send it (e.g. mock / sectional
    # submits, mobile clients that haven't been updated).
    time_on_question_seconds: Optional[int] = None


# ── Text answer (SWT, WE, SST, WFD) ───────────────────────────────────────────
class TextSubmitRequest(_BaseSubmit):
    """Free-text answer. `text` accepted as legacy alias."""
    user_answer: Optional[str] = None
    user_text: Optional[str] = None
    text: Optional[str] = None  # legacy alias

    def resolved_text(self) -> str:
        return self.user_answer or self.user_text or self.text or ""


# ── Dict-of-blanks answer (FIB-R, FIB-DD, FIB-L) ─────────────────────────────
# Accepts either dict (e.g. {"blank_1": "ans"}) or positional list — list form
# is normalised to a dict by callers, kept here to preserve compat with older
# clients.
_AnswersInput = Union[Dict[str, str], List[str]]


class AnswersDictSubmitRequest(_BaseSubmit):
    user_answers: Optional[_AnswersInput] = None
    answers: Optional[_AnswersInput] = None  # legacy alias

    def resolved_raw(self) -> _AnswersInput:
        if self.user_answers is not None:
            return self.user_answers
        if self.answers is not None:
            return self.answers
        return {}


# ── Single-option answer (MCS-R, HCS, MCS-L, SMW) ────────────────────────────
class SingleOptionSubmitRequest(_BaseSubmit):
    """One option chosen from a list."""
    selected_option: Optional[str] = None
    selected_option_ids: Optional[List[str]] = None

    def resolved_option(self) -> str:
        if self.selected_option:
            return self.selected_option
        if self.selected_option_ids:
            return self.selected_option_ids[0]
        return ""


# ── Multi-option answer (MCM-R, MCM-L) ───────────────────────────────────────
class MultiOptionSubmitRequest(_BaseSubmit):
    """Several options chosen from a list."""
    selected_options: Optional[List[str]] = None
    selected_option_ids: Optional[List[str]] = None

    def resolved_options(self) -> List[str]:
        return self.selected_options or self.selected_option_ids or []


# ── Sequence answer (RP) ─────────────────────────────────────────────────────
class SequenceSubmitRequest(_BaseSubmit):
    """Ordered list of paragraph IDs."""
    user_sequence: Optional[List[str]] = None
    paragraphs: Optional[List[str]] = None  # legacy alias

    def resolved_sequence(self) -> List[str]:
        return self.user_sequence or self.paragraphs or []


# ── Highlight-incorrect-words (HIW) ──────────────────────────────────────────
class HIWSubmitRequest(_BaseSubmit):
    """Either explicit clicked words, or 0-based indices into the passage."""
    highlighted_words: Optional[List[str]] = None
    highlighted_indices: Optional[List[int]] = None
