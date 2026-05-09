-- Migration: auth_refresh_tokens table for stored refresh-token rotation
-- Date: 2026-05-09
-- Applied to: postgres database on database-1 (ap-southeast-2)
--
-- Replaces the single 30-day stateless JWT model with:
--   - short-lived (30 min) JWT access token (stateless, unchanged validation)
--   - long-lived (30 day) refresh token, hashed + stored here, rotated on use
--
-- token_family: each refresh token chain shares an ID. If a revoked token in
-- a family is presented again (replay attack), the entire family is revoked
-- — single source of compromise containment.
--
-- Idempotent — safe to re-run.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()

CREATE TABLE IF NOT EXISTS auth_refresh_tokens (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash    VARCHAR(64) NOT NULL UNIQUE,    -- SHA-256 hex of the raw token
    token_family  UUID NOT NULL,                  -- chain ID for replay/theft detection
    issued_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ NOT NULL,
    revoked_at    TIMESTAMPTZ,
    replaced_by   UUID REFERENCES auth_refresh_tokens(id) ON DELETE SET NULL,
    user_agent    TEXT,
    ip_address    INET,
    last_used_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_refresh_user_active
    ON auth_refresh_tokens(user_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_refresh_family
    ON auth_refresh_tokens(token_family);
CREATE INDEX IF NOT EXISTS ix_refresh_expires
    ON auth_refresh_tokens(expires_at);

COMMIT;

-- Verify:
--   \d auth_refresh_tokens
--   SELECT count(*) FROM auth_refresh_tokens;
