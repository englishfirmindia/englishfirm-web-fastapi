"""Regression tests for `practiced_questions_subq` + `practiced_question_ids_in`
in services/question_list_helper.py.

These helpers replace the prior UQA-based "Done" filter across all 22
practice list endpoints (shipped 2026-06-02). The previous logic queried
`user_question_attempts` for the practiced subquery, which counted ALL
modes (practice + sectional + mock) — diverging from:

  * The recency-map source (`fetch_practice_recency_map`) — practice-only
  * The last-answer endpoint (`/api/v1/user/last-answer`) — practice-only

That divergence produced:
  * Mock/sectional-attempted questions showing in "Done" with a "Never
    practiced" bucket header (the recency map didn't have them)
  * Clicking such items showed empty answer (last-answer correctly
    returned null because no practice-mode answer existed)
  * Orphan UQA rows (no matching attempt_answers) — 3,451 rows across
    3 users from an April 2026 batch — also showed in "Done" with empty
    answers when clicked

The new helpers source from `attempt_answers JOIN practice_attempts WHERE
filter_type='practice'`, aligning all three sources of truth. Orphan UQAs
are implicitly excluded (no underlying answer = not in Done).

These tests pin:
  1. Practice-mode answers ARE counted as practised
  2. Mock-mode answers are NOT counted
  3. Sectional-mode answers are NOT counted
  4. Cleared answers are NOT counted (scoring_status='cleared')
  5. The orphan-UQA pattern produces an empty result (no answer row exists)
  6. Question-type aliases work (reading_fib / reading_drag_and_drop)
  7. Both helpers (subq + ids_in) agree on the same set of practiced qids
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import JSON


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return compiler.visit_JSON(JSON())


@compiles(ARRAY, "sqlite")
def _compile_array_sqlite(type_, compiler, **kw):
    return "TEXT"


from db.models import (
    AttemptAnswer,
    PracticeAttempt,
    QuestionFromApeuni,
    User,
    UserQuestionAttempt,
)
from services.question_list_helper import (
    practiced_question_ids_in,
    practiced_questions_subq,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    for model in (User, QuestionFromApeuni, PracticeAttempt, AttemptAnswer,
                  UserQuestionAttempt):
        model.__table__.create(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    sess = TestingSessionLocal()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


# ── Builders ─────────────────────────────────────────────────────────────────

def _user(db, uid=1):
    u = User(id=uid, username=f"u{uid}", email=f"u{uid}@ef.com", hashed_password="x")
    db.add(u); db.commit(); return u


def _question(db, qid, qtype="read_aloud", module="speaking"):
    q = QuestionFromApeuni(
        question_id=qid, module=module, question_type=qtype,
        title=f"Q{qid}", difficulty_level=1, content_json={},
    )
    db.add(q); db.commit(); return q


def _attempt(db, user_id, *, filter_type="practice",
             module="speaking", question_type="read_aloud", sid_suffix=""):
    a = PracticeAttempt(
        user_id=user_id, session_id=f"sess-{user_id}-{filter_type}-{sid_suffix}",
        module=module, question_type=question_type, filter_type=filter_type,
        total_questions=1, total_score=0, questions_answered=0,
        status="active", scoring_status="pending",
    )
    db.add(a); db.commit(); db.refresh(a); return a


def _answer(db, attempt, qid, *, qtype="read_aloud", scoring_status="complete", score=70):
    aa = AttemptAnswer(
        attempt_id=attempt.id, question_id=qid, question_type=qtype,
        user_answer_json={}, result_json={"pte_score": score},
        score=score, scoring_status=scoring_status,
    )
    db.add(aa); db.commit(); return aa


def _uqa(db, user_id, qid, qtype="read_aloud", module="speaking"):
    """Direct UQA row — simulates orphan / legacy data."""
    r = UserQuestionAttempt(
        user_id=user_id, question_id=qid, question_type=qtype, module=module,
    )
    db.add(r); db.commit(); return r


def _ids_from_subq(db, user_id, question_type):
    """Resolve the subquery into a concrete set of qids."""
    subq = practiced_questions_subq(db, user_id, question_type)
    rows = db.query(QuestionFromApeuni.question_id).filter(
        QuestionFromApeuni.question_id.in_(subq)
    ).all()
    return {r[0] for r in rows}


# ── Practice-only contract ──────────────────────────────────────────────────


def test_practice_mode_answer_is_included(db_session):
    u = _user(db_session)
    _question(db_session, 101)
    a = _attempt(db_session, u.id, filter_type="practice")
    _answer(db_session, a, 101)

    qids = _ids_from_subq(db_session, u.id, "read_aloud")
    assert qids == {101}

    ids = practiced_question_ids_in(db_session, u.id, "read_aloud", [101, 999])
    assert ids == {101}


def test_mock_mode_answer_is_EXCLUDED(db_session):
    """The whole point of Option A — mock-mode answers don't count as
    'practised'. They should NOT appear in the Done filter."""
    u = _user(db_session)
    _question(db_session, 201)
    a = _attempt(db_session, u.id, filter_type="mock", module="mock", sid_suffix="m1")
    _answer(db_session, a, 201)

    assert _ids_from_subq(db_session, u.id, "read_aloud") == set()
    assert practiced_question_ids_in(db_session, u.id, "read_aloud", [201]) == set()


def test_sectional_mode_answer_is_EXCLUDED(db_session):
    """Same as above for sectional — graded-context exposure doesn't bubble
    a question into the Done list."""
    u = _user(db_session)
    _question(db_session, 301)
    a = _attempt(db_session, u.id, filter_type="sectional",
                 question_type="sectional", sid_suffix="s1")
    _answer(db_session, a, 301)

    assert _ids_from_subq(db_session, u.id, "read_aloud") == set()
    assert practiced_question_ids_in(db_session, u.id, "read_aloud", [301]) == set()


def test_cleared_answer_is_EXCLUDED(db_session):
    """Cleared answers (Clear button) flip the question back to 'New' and
    must not show in Done."""
    u = _user(db_session)
    _question(db_session, 401)
    a = _attempt(db_session, u.id, filter_type="practice")
    _answer(db_session, a, 401, scoring_status="cleared")

    assert _ids_from_subq(db_session, u.id, "read_aloud") == set()
    assert practiced_question_ids_in(db_session, u.id, "read_aloud", [401]) == set()


def test_orphan_uqa_is_IGNORED(db_session):
    """The bug we just fixed: orphan UQA rows (no matching attempt_answers)
    used to show in Done with empty answer screens. Now they're invisible."""
    u = _user(db_session)
    _question(db_session, 501)
    _uqa(db_session, u.id, 501)   # UQA exists, attempt_answers does NOT

    assert _ids_from_subq(db_session, u.id, "read_aloud") == set()
    assert practiced_question_ids_in(db_session, u.id, "read_aloud", [501]) == set()


def test_mix_practice_mock_sectional_only_practice_counts(db_session):
    """One user, 3 different qids — one practice, one mock, one sectional.
    Only the practice qid appears."""
    u = _user(db_session)
    for qid in (601, 602, 603): _question(db_session, qid)
    pa = _attempt(db_session, u.id, filter_type="practice", sid_suffix="p")
    ma = _attempt(db_session, u.id, filter_type="mock", module="mock", sid_suffix="m")
    sa = _attempt(db_session, u.id, filter_type="sectional",
                  question_type="sectional", sid_suffix="s")
    _answer(db_session, pa, 601)
    _answer(db_session, ma, 602)
    _answer(db_session, sa, 603)

    qids = _ids_from_subq(db_session, u.id, "read_aloud")
    assert qids == {601}

    page_qids = [601, 602, 603]
    assert practiced_question_ids_in(db_session, u.id, "read_aloud", page_qids) == {601}


def test_reading_fib_alias_drag_and_drop_is_counted(db_session):
    """Existing alias: reading_fib_drop_down screen lists questions stored
    as reading_drag_and_drop in attempt_answers. The helper must look up both
    aliases via _QUESTION_TYPE_ALIASES."""
    u = _user(db_session)
    _question(db_session, 701, qtype="reading_fib", module="reading")
    pa = _attempt(db_session, u.id, filter_type="practice",
                  question_type="reading_fib", module="reading", sid_suffix="r")
    # Answer stored under the alias `reading_drag_and_drop`
    _answer(db_session, pa, 701, qtype="reading_drag_and_drop")

    # Querying for either form should find it
    assert _ids_from_subq(db_session, u.id, "reading_fib") == {701}
    assert _ids_from_subq(db_session, u.id, "reading_drag_and_drop") == {701}


def test_per_page_helper_restricts_correctly(db_session):
    """practiced_question_ids_in only returns qids that are BOTH practised
    AND in the restrict_to list — so it can drive per-page enrichment without
    overfetching."""
    u = _user(db_session)
    for qid in (801, 802, 803, 804): _question(db_session, qid)
    pa = _attempt(db_session, u.id, filter_type="practice")
    _answer(db_session, pa, 801)
    # Create a second practice attempt for the second qid (need unique session_id)
    pa2 = _attempt(db_session, u.id, filter_type="practice", sid_suffix="2")
    _answer(db_session, pa2, 802)

    # Page contains 803 + 802 + 999. Only 802 is practised among those.
    page = [803, 802, 999]
    assert practiced_question_ids_in(db_session, u.id, "read_aloud", page) == {802}


def test_subq_and_in_helper_agree(db_session):
    """Sanity check: the two helpers must return the same set when given
    the same input."""
    u = _user(db_session)
    for qid in (901, 902, 903): _question(db_session, qid)
    pa = _attempt(db_session, u.id, filter_type="practice")
    _answer(db_session, pa, 901)
    pa2 = _attempt(db_session, u.id, filter_type="practice", sid_suffix="2")
    _answer(db_session, pa2, 902)
    pa3 = _attempt(db_session, u.id, filter_type="mock", module="mock", sid_suffix="m")
    _answer(db_session, pa3, 903)

    via_subq = _ids_from_subq(db_session, u.id, "read_aloud")
    via_ids_in = practiced_question_ids_in(
        db_session, u.id, "read_aloud", [901, 902, 903]
    )
    assert via_subq == via_ids_in == {901, 902}


def test_empty_restrict_to_returns_empty_set(db_session):
    """No-op fast-path: when the page contains no qids, return empty without
    a DB query."""
    u = _user(db_session)
    assert practiced_question_ids_in(db_session, u.id, "read_aloud", []) == set()


def test_different_question_types_are_isolated(db_session):
    """A practice attempt for read_aloud must NOT show up when querying for
    listening_wfd. Avoids cross-type leaks."""
    u = _user(db_session)
    _question(db_session, 1001, qtype="read_aloud")
    pa = _attempt(db_session, u.id, filter_type="practice",
                  question_type="read_aloud")
    _answer(db_session, pa, 1001, qtype="read_aloud")

    assert _ids_from_subq(db_session, u.id, "read_aloud") == {1001}
    assert _ids_from_subq(db_session, u.id, "listening_wfd") == set()
