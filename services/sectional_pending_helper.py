"""Cross-device resume support for the 4 sectional modules.

`SectionalSavedSession` in the Flutter web client stores a "resume me"
breadcrumb in browser localStorage — that breaks when the student logs
in from a second device or browser, even though the backend's
`practice_attempts` table still holds the pending row.

This helper backs the new `GET /sectional/{module}/in-progress` endpoint
each of the 4 sectional routers exposes. The endpoint JOINs
`practice_attempts` (status='pending') with `practice_session_states`
(must not be expired) so the client gets the authoritative list from
the server, regardless of which device started the test.

Mirrors the shape used by `GET /mock/pending` (see routers/mock.py
:mock_pending) — keeping the two patterns aligned makes the Flutter
caller code symmetric.
"""
from __future__ import annotations

from typing import List, Dict, Any

from sqlalchemy import text as _sql_text
from sqlalchemy.orm import Session


def fetch_pending_sectionals(
    db: Session, user_id: int, module: str
) -> List[Dict[str, Any]]:
    """Return the user's in-progress sectional attempts for `module`.

    Only surfaces rows whose paired `practice_session_states.expires_at`
    is still in the future — anything older is functionally lost (the
    resume endpoint would 404), so don't dangle a dead "Resume" CTA in
    front of the user.

    Each result includes `test_number` parsed from `task_breakdown` so
    the client can route directly to `/sectional/exam?module=X&
    test_number=N`. Ordered newest-saved first so the most recently
    paused test sits at the top of the resume list.
    """
    rows = db.execute(_sql_text(
        """
        SELECT pa.id                                AS attempt_id,
               pa.session_id                        AS session_id,
               pa.started_at                        AS started_at,
               (pa.task_breakdown ->> 'test_number')::int AS test_number,
               COUNT(aa.id)                         AS submitted_count,
               pss.updated_at                       AS state_updated_at,
               pss.expires_at                       AS state_expires_at
        FROM   practice_attempts pa
        JOIN   practice_session_states pss ON pss.session_id = pa.session_id
        LEFT   JOIN attempt_answers aa     ON aa.attempt_id  = pa.id
        WHERE  pa.user_id      = :uid
          AND  pa.module       = :module
          AND  pa.question_type = 'sectional'
          AND  pa.status       = 'pending'
          AND  pss.expires_at  > NOW()
        GROUP  BY pa.id, pa.session_id, pa.started_at, pa.task_breakdown,
                  pss.updated_at, pss.expires_at
        ORDER  BY pss.updated_at DESC
        """
    ), {"uid": user_id, "module": module}).all()

    return [
        {
            "attempt_id":      r.attempt_id,
            "session_id":      r.session_id,
            "test_number":     r.test_number,
            "started_at":      r.started_at.isoformat() if r.started_at else None,
            "submitted_count": int(r.submitted_count or 0),
            "last_saved_at":   r.state_updated_at.isoformat() if r.state_updated_at else None,
            "expires_at":      r.state_expires_at.isoformat() if r.state_expires_at else None,
        }
        for r in rows
    ]
