from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "writing"
    q.question_type = "summarize_written_text"
    q.difficulty_level = 3
    q.time_limit_seconds = 600
    q.content_json = {"passage": "Climate change affects global temperatures..."}
    q.evaluation = None
    return q


@patch("routers.writing.summarize_written_text.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "swt1",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "writing", "question_type": "summarize_written_text",
                        "difficulty_level": 3, "time_limit_seconds": 600, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/writing/summarize-written-text/start", json={})
    assert response.status_code == 200
    assert response.json()["session_id"] == "swt1"


@patch("routers.writing.summarize_written_text.get_scorer")
@patch("routers.writing.summarize_written_text.mark_submitted")
@patch("routers.writing.summarize_written_text.get_session")
def test_submit_returns_score(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(1)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {1: q},
        "score": 0,
        "submitted_questions": set(),
    }
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=68, raw_score=0.725, is_async=False,
        breakdown={"ai_raw": 72.5, "task_type": "swt"}
    )

    response = client.post("/api/v1/questions/writing/summarize-written-text/submit", json={
        "session_id": "swt1", "question_id": 1,
        "user_answer": "Climate change impacts global temperatures significantly."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pte_score"] == 68
    assert data["is_async"] is False


@patch("routers.writing.summarize_written_text.get_session")
def test_submit_question_not_found(mock_get_session, client):
    mock_get_session.return_value = {"user_id": 1, "questions": {}, "score": 0, "submitted_questions": set()}
    response = client.post("/api/v1/questions/writing/summarize-written-text/submit", json={
        "session_id": "swt1", "question_id": 999, "user_answer": "Some answer."
    })
    assert response.status_code == 404
