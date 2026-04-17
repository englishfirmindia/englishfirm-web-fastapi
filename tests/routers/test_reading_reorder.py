from unittest.mock import patch, MagicMock
from services.scoring.base import ScoringResult


def _make_question(question_id: int):
    q = MagicMock()
    q.question_id = question_id
    q.module = "reading"
    q.question_type = "reorder_paragraphs"
    q.difficulty_level = 3
    q.time_limit_seconds = 120
    q.content_json = {"paragraphs": [{"id": "A", "text": "First."}, {"id": "B", "text": "Second."}]}
    q.evaluation = MagicMock()
    q.evaluation.evaluation_json = {
        "correctAnswers": {"correctSequence": ["A", "B"]},
        "scoringRules": {"marksPerAdjacentPair": 1},
    }
    return q


@patch("routers.reading.reorder_paragraphs.start_session")
def test_start_returns_session(mock_start, client):
    mock_start.return_value = {
        "session_id": "ro1",
        "total_questions": 1,
        "questions": [{"question_id": 1, "module": "reading", "question_type": "reorder_paragraphs",
                        "difficulty_level": 3, "time_limit_seconds": 120, "content_json": {}}],
    }
    response = client.post("/api/v1/questions/reading/reorder-paragraphs/start", json={})
    assert response.status_code == 200
    assert response.json()["session_id"] == "ro1"


@patch("routers.reading.reorder_paragraphs.get_scorer")
@patch("routers.reading.reorder_paragraphs.mark_submitted")
@patch("routers.reading.reorder_paragraphs.get_session")
def test_submit_correct_sequence(mock_get_session, mock_mark, mock_get_scorer, client):
    q = _make_question(1)
    mock_get_session.return_value = {
        "user_id": 1,
        "questions": {1: q},
        "score": 0,
        "submitted_questions": set(),
    }
    mock_get_scorer.return_value.score.return_value = ScoringResult(
        pte_score=90, raw_score=1.0, is_async=False,
        breakdown={"hits": 1, "total_pairs": 1, "pair_results": [True]}
    )

    response = client.post("/api/v1/questions/reading/reorder-paragraphs/submit", json={
        "session_id": "ro1", "question_id": 1, "user_sequence": ["A", "B"]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pte_score"] == 90
    assert data["breakdown"]["hits"] == 1


@patch("routers.reading.reorder_paragraphs.get_session")
def test_submit_question_not_found(mock_get_session, client):
    mock_get_session.return_value = {"user_id": 1, "questions": {}, "score": 0, "submitted_questions": set()}
    response = client.post("/api/v1/questions/reading/reorder-paragraphs/submit", json={
        "session_id": "ro1", "question_id": 999, "user_sequence": ["A", "B"]
    })
    assert response.status_code == 404
