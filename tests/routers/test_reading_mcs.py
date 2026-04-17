from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "reading"
    q.question_type = "reading_mcs"
    q.difficulty_level = 3
    q.time_limit_seconds = 60
    q.content_json = {"passage": "Test passage", "question": "What is correct?", "options": ["A", "B", "C", "D"]}
    q.evaluation = MagicMock()
    q.evaluation.evaluation_json = {
        "correctAnswers": {"correctOption": "A"},
        "scoringRules": {"marksPerCorrect": 1},
    }
    return q


@patch("routers.reading.mcs.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "s1",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "reading", "question_type": "reading_mcs",
                        "difficulty_level": 3, "time_limit_seconds": 60, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/reading/mcs/start", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "s1"
    assert data["total_questions"] == 1


@patch("routers.reading.mcs.get_scorer")
@patch("routers.reading.mcs.mark_submitted")
@patch("routers.reading.mcs.get_session")
def test_submit_correct_answer(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(1)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {1: q},
        "score": 0,
        "submitted_questions": set(),
    }
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=72, raw_score=1.0, is_async=False, breakdown={"is_correct": True}
    )

    response = client.post("/api/v1/questions/reading/mcs/submit", json={
        "session_id": "s1", "question_id": 1, "selected_option": "A"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pte_score"] == 72
    assert data["is_async"] is False
    assert data["breakdown"]["is_correct"] is True


@patch("routers.reading.mcs.get_scorer")
@patch("routers.reading.mcs.mark_submitted")
@patch("routers.reading.mcs.get_session")
def test_submit_wrong_answer(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(2)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {2: q},
        "score": 0,
        "submitted_questions": set(),
    }
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=10, raw_score=0.0, is_async=False, breakdown={"is_correct": False}
    )

    response = client.post("/api/v1/questions/reading/mcs/submit", json={
        "session_id": "s1", "question_id": 2, "selected_option": "B"
    })
    assert response.status_code == 200
    assert response.json()["pte_score"] == 10


@patch("routers.reading.mcs.get_session")
def test_submit_question_not_found(mock_get_session, client):
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {},
        "score": 0,
        "submitted_questions": set(),
    }
    response = client.post("/api/v1/questions/reading/mcs/submit", json={
        "session_id": "s1", "question_id": 999, "selected_option": "A"
    })
    assert response.status_code == 404
