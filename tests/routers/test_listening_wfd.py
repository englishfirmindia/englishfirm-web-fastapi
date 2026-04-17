from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "listening"
    q.question_type = "listening_wfd"
    q.difficulty_level = 2
    q.time_limit_seconds = 60
    q.content_json = {"audio_url": "s3://bucket/audio.mp3"}
    q.evaluation = MagicMock()
    q.evaluation.evaluation_json = {
        "correctAnswers": {"transcript": "The quick brown fox jumps"},
        "scoringRules": {},
    }
    return q


@patch("routers.listening.wfd.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "wfd1",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "listening", "question_type": "listening_wfd",
                        "difficulty_level": 2, "time_limit_seconds": 60, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/listening/wfd/start", json={})
    assert response.status_code == 200
    assert response.json()["session_id"] == "wfd1"


@patch("routers.listening.wfd.get_scorer")
@patch("routers.listening.wfd.mark_submitted")
@patch("routers.listening.wfd.get_session")
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
        breakdown={"hits": 5, "total": 5}
    )

    response = client.post("/api/v1/questions/listening/wfd/submit", json={
        "session_id": "wfd1", "question_id": 1,
        "user_text": "The quick brown fox jumps"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pte_score"] == 90
    assert data["is_async"] is False
    assert data["breakdown"]["hits"] == 5


@patch("routers.listening.wfd.get_session")
def test_submit_question_not_found(mock_get_session, client):
    mock_get_session.return_value = {"user_id": 1, "questions": {}, "score": 0, "submitted_questions": set()}
    response = client.post("/api/v1/questions/listening/wfd/submit", json={
        "session_id": "wfd1", "question_id": 999, "user_text": "Some text"
    })
    assert response.status_code == 404
