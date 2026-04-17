from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "reading"
    q.question_type = "reading_mcm"
    q.difficulty_level = 3
    q.time_limit_seconds = 90
    q.content_json = {"passage": "Test passage", "question": "Select all correct?", "options": ["A", "B", "C", "D"]}
    q.evaluation = MagicMock()
    q.evaluation.evaluation_json = {
        "correctAnswers": {"correctOptions": ["A", "C"]},
        "scoringRules": {"marksPerCorrect": 1, "deductPerWrong": 0},
    }
    return q


@patch("routers.reading.mcm.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "s2",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "reading", "question_type": "reading_mcm",
                        "difficulty_level": 3, "time_limit_seconds": 90, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/reading/mcm/start", json={})
    assert response.status_code == 200
    assert response.json()["session_id"] == "s2"


@patch("routers.reading.mcm.get_scorer")
@patch("routers.reading.mcm.mark_submitted")
@patch("routers.reading.mcm.get_session")
def test_submit_partial_correct(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(1)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {1: q},
        "score": 0,
        "submitted_questions": set(),
    }
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=46, raw_score=0.5, is_async=False,
        breakdown={"correct_selected": ["A"], "wrong_selected": [], "score": 1, "max_possible": 2}
    )

    response = client.post("/api/v1/questions/reading/mcm/submit", json={
        "session_id": "s2", "question_id": 1, "selected_options": ["A"]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pte_score"] == 46
    assert data["is_async"] is False


@patch("routers.reading.mcm.get_session")
def test_submit_question_not_found(mock_get_session, client):
    mock_get_session.return_value = {"user_id": 1, "questions": {}, "score": 0, "submitted_questions": set()}
    response = client.post("/api/v1/questions/reading/mcm/submit", json={
        "session_id": "s2", "question_id": 999, "selected_options": ["A"]
    })
    assert response.status_code == 404
