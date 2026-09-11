-- 2026-09-11 — Add nudge_events funnel log
-- Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS nudge_events (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    nudge_id    VARCHAR(64) NOT NULL,
    event_type  VARCHAR(32) NOT NULL,
    meta        JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_nudge_events_user_id      ON nudge_events(user_id);
CREATE INDEX IF NOT EXISTS ix_nudge_events_nudge_id     ON nudge_events(nudge_id);
CREATE INDEX IF NOT EXISTS ix_nudge_events_event_type   ON nudge_events(event_type);
CREATE INDEX IF NOT EXISTS ix_nudge_events_created_at   ON nudge_events(created_at);

-- Verify
SELECT COUNT(*) AS rows_before FROM nudge_events;
