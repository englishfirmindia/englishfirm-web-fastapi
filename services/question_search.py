"""Shared keyword-search filter for question list endpoints.

Replaces the legacy title-only filter (`title ILIKE '%search%'`) which missed
hits in the passage / prompt / question / transcript body. The new helper
ORs across:

  - title (existing behaviour)
  - content_json.passage / .prompt / .question / .text / .situation_text
  - question_id (exact match if the search term parses as int)

Postgres JSONB `->>` operator extracts each field as text; SQLAlchemy emits
the equivalent `content_json -> 'passage' ->> 0` cast via `.astext`. No index
required — for the current ~10k-row pool a sequential scan is sub-100ms.
Add a pg_trgm GIN index later if usage volume grows.

Pure / stateless / no I/O — just composes SQLAlchemy clauses.
"""
from typing import Optional

from sqlalchemy import or_

from db.models import QuestionFromApeuni


# content_json keys that hold the question body across task types. All are
# searched on every list query — most rows only populate one or two keys, so
# the OR fans out cheaply.
_CONTENT_TEXT_KEYS = (
    "passage",
    "prompt",
    "question",
    "text",
    "situation_text",
)


def apply_search_filter(query, search: Optional[str]):
    """OR-search across title + content_json text fields + question_id.

    No-op if `search` is None / empty / whitespace.
    """
    if not search or not search.strip():
        return query
    needle = f"%{search.strip()}%"

    cj = QuestionFromApeuni.content_json
    clauses = [QuestionFromApeuni.title.ilike(needle)]
    for key in _CONTENT_TEXT_KEYS:
        clauses.append(cj[key].astext.ilike(needle))

    # Exact question_id match for trainer-style "go find #11150" lookups.
    qid_int = _maybe_int(search.strip())
    if qid_int is not None:
        clauses.append(QuestionFromApeuni.question_id == qid_int)

    return query.filter(or_(*clauses))


def _maybe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None
