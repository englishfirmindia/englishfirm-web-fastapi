from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "listening"
    q.question_type = "listening_mcs"
    q.difficulty_level = 2
    q.time_limit_seconds = 90
    q.content_json = {"audio_url": "s3://bucket/audio.mp3", "question": "What is the main idea?",
                       "options": ["A", "B", "C", "D"]}
    q.evaluation = MagicMock()
    q.evaluation.evaluation_json = {
        "correctAnswers": {"correctOption": "B"},
        "scoringRules": {"marksPerCorrect": 1},
    }
    return q


@patch("routers.listening.mcs.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "lmcs1",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "listening", "question_type": "listening_mcs",
                        "difficulty_level": 2, "time_limit_seconds": 90, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/listening/mcs/start", json={})
    assert response.status_code == 200
    assert response.json()["session_id"] == "lmcs1"


@patch("routers.listening.mcs.get_scorer")
@patch("routers.listening.mcs.mark_submitted")
@patch("routers.listening.mcs.get_session")
def test_submit_correct(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(1)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {1: q},
        "score": 0,
        "submitted_questions": set(),
    }
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=90, raw_score=1.0, is_async=False,
        breakdown={"is_correct": True, "correct_option": "B", "selected_option": "B"}
    )

    response = client.post("/api/v1/questions/listening/mcs/submit", json={
        "session_id": "lmcs1", "question_id": 1, "selected_option": "B"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pte_score"] == 90
    assert data["breakdown"]["is_correct"] is True


@patch("routers.listening.mcs.get_session")
def test_submit_question_not_found(mock_get_session, client):
    mock_get_session.return_value = {"user_id": 1, "questions": {}, "score": 0, "submitted_questions": set()}
    response = client.post("/api/v1/questions/listening/mcs/submit", json={
        "session_id": "lmcs1", "question_id": 999, "selected_option": "A"
    })
    assert response.status_code == 404
