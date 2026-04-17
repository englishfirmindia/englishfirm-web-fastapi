from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "reading"
    q.question_type = "reading_fib_drop_down"
    q.difficulty_level = 2
    q.time_limit_seconds = 120
    q.content_json = {"passage": "The ___ runs ___.", "wordbank": ["dog", "fast", "cat", "slowly"]}
    q.evaluation = MagicMock()
    q.evaluation.evaluation_json = {
        "correctAnswers": {"1": "dog", "2": "fast"},
        "scoringRules": {"marksPerBlank": 1, "isCaseSensitive": False, "trimWhitespace": True},
    }
    return q


@patch("routers.reading.fib_drag_drop.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "fdd1",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "reading", "question_type": "reading_fib_drop_down",
                        "difficulty_level": 2, "time_limit_seconds": 120, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/reading/fib-drag-drop/start", json={})
    assert response.status_code == 200
    assert response.json()["session_id"] == "fdd1"


@patch("routers.reading.fib_drag_drop.get_scorer")
@patch("routers.reading.fib_drag_drop.mark_submitted")
@patch("routers.reading.fib_drag_drop.get_session")
def test_submit_returns_score(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(1)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {1: q},
        "score": 0,
        "submitted_questions": set(),
    }
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=90, raw_score=1.0, is_async=False,
        breakdown={"hits": 2, "total": 2, "blank_results": {"1": True, "2": True}}
    )

    response = client.post("/api/v1/questions/reading/fib-drag-drop/submit", json={
        "session_id": "fdd1", "question_id": 1, "user_answers": {"1": "dog", "2": "fast"}
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pte_score"] == 90
    assert data["breakdown"]["hits"] == 2


@patch("routers.reading.fib_drag_drop.get_session")
def test_submit_question_not_found(mock_get_session, client):
    mock_get_session.return_value = {"user_id": 1, "questions": {}, "score": 0, "submitted_questions": set()}
    response = client.post("/api/v1/questions/reading/fib-drag-drop/submit", json={
        "session_id": "fdd1", "question_id": 999, "user_answers": {"1": "dog"}
    })
    assert response.status_code == 404
