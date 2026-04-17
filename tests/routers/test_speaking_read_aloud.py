from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "speaking"
    q.question_type = "read_aloud"
    q.difficulty_level = 2
    q.time_limit_seconds = 75
    q.content_json = {"passage": "The sun rises in the east and sets in the west."}
    q.evaluation = None
    return q


@patch("routers.speaking.read_aloud.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "ra1",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "speaking", "question_type": "read_aloud",
                        "difficulty_level": 2, "time_limit_seconds": 75, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/speaking/read-aloud/start", json={})
    assert response.status_code == 200
    assert response.json()["session_id"] == "ra1"


@patch("routers.speaking.read_aloud.get_scorer")
@patch("routers.speaking.read_aloud.mark_submitted")
@patch("routers.speaking.read_aloud.get_session")
def test_submit_returns_pending(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(1)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {1: q},
        "score": 0,
        "submitted_questions": set(),
    }
    # Azure scorer returns is_async=True
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=0, raw_score=0.0, is_async=True,
        breakdown={"status": "pending", "task_type": "read_aloud"}
    )

    response = client.post("/api/v1/questions/speaking/read-aloud/submit", json={
        "session_id": "ra1", "question_id": 1,
        "audio_url": "https://bucket.s3.amazonaws.com/recordings/1/ra/1/abc.aac"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["scoring_status"] == "pending"
    assert data["message"] == "submitted"


@patch("routers.speaking.read_aloud.get_score_from_store")
def test_poll_score_pending(mock_store, client):
    mock_store.return_value = None
    response = client.get("/api/v1/questions/speaking/read-aloud/score/1")
    assert response.status_code == 200
    assert response.json()["scoring"] == "pending"


@patch("routers.speaking.read_aloud.get_score_from_store")
def test_poll_score_ready(mock_store, client):
    mock_store.return_value = {
        "scoring": "complete",
        "pte_score": 58,
        "content": 65.0,
        "fluency": 72.0,
        "pronunciation": 68.0,
    }
    response = client.get("/api/v1/questions/speaking/read-aloud/score/1")
    assert response.status_code == 200
    data = response.json()
    assert data["scoring"] == "complete"
    assert data["pte_score"] == 58


def test_audio_url_endpoint_exists(client):
    """Verify the /audio-url endpoint is registered (will fail S3 call but route should exist)."""
    # Route is registered — a missing s3_url param gives 422, not 404
    response = client.get("/api/v1/questions/speaking/read-aloud/audio-url")
    assert response.status_code == 422  # Unprocessable — missing required query param s3_url
