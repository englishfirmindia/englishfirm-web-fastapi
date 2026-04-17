from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "writing"
    q.question_type = "write_essay"
    q.difficulty_level = 3
    q.time_limit_seconds = 1200
    q.content_json = {"topic": "Should technology replace teachers?"}
    q.evaluation = None
    return q


@patch("routers.writing.write_essay.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "we1",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "writing", "question_type": "write_essay",
                        "difficulty_level": 3, "time_limit_seconds": 1200, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/writing/write-essay/start", json={})
    assert response.status_code == 200
    assert response.json()["session_id"] == "we1"


@patch("routers.writing.write_essay.get_scorer")
@patch("routers.writing.write_essay.mark_submitted")
@patch("routers.writing.write_essay.get_session")
def test_submit_returns_score(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(1)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {1: q},
        "score": 0,
        "submitted_questions": set(),
    }
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=75, raw_score=0.8125, is_async=False,
        breakdown={"ai_raw": 81.25, "task_type": "we"}
    )

    response = client.post("/api/v1/questions/writing/write-essay/submit", json={
        "session_id": "we1", "question_id": 1,
        "user_answer": "Technology can support teachers but cannot replace the human element..."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pte_score"] == 75
    assert data["is_async"] is False
    assert data["breakdown"]["task_type"] == "we"


@patch("routers.writing.write_essay.get_session")
def test_submit_question_not_found(mock_get_session, client):
    mock_get_session.return_value = {"user_id": 1, "questions": {}, "score": 0, "submitted_questions": set()}
    response = client.post("/api/v1/questions/writing/write-essay/submit", json={
        "session_id": "we1", "question_id": 999, "user_answer": "Some essay."
    })
    assert response.status_code == 404
