"""
Auth-token issuance and rotation.

Replaces the legacy "single 30-day JWT" model with:
  - access JWT  : 30 min, stateless, validated by core/dependencies.py
  - refresh tok : 30 days, hashed in auth_refresh_tokens, single-use rotating

Each successful /auth/refresh issues a brand-new 30-day refresh clock, so
an actively-used session never expires.

Replay detection: if a revoked refresh token is presented again, every
token sharing its `token_family` is revoked at once — the entire chain is
considered compromised.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from jose import jwt
from fastapi import Request
from sqlalchemy.orm import Session

import core.config as config
from core.logging_config import get_logger
from db.models import AuthRefreshToken

log = get_logger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _new_raw_refresh() -> str:
    """256-bit URL-safe random token. Hash is what's stored."""
    return secrets.token_urlsafe(48)


def _client_ip(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    fwd = request.headers.get("x-forwarded-for") or ""
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


def _user_agent(request: Optional[Request]) -> Optional[str]:
    return request.headers.get("user-agent") if request else None


# ── access token ──────────────────────────────────────────────────────────────
def issue_access_token(user_id: int) -> str:
    """30-minute JWT carrying `sub` = user_id. Stateless."""
    exp = _now() + timedelta(minutes=config.JWT_ACCESS_EXPIRY_MINUTES)
    payload = {"sub": str(user_id), "exp": int(exp.timestamp())}
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


# ── refresh token ─────────────────────────────────────────────────────────────
def issue_token_pair(
    db: Session,
    user_id: int,
    request: Optional[Request] = None,
    token_family: Optional[uuid.UUID] = None,
) -> Tuple[str, str, AuthRefreshToken]:
    """
    Issue a fresh (access_token, refresh_token_raw, refresh_row) triple.

    `token_family` is reused on rotation; left None for fresh logins (a new
    family is created).
    """
    access_token = issue_access_token(user_id)
    raw_refresh = _new_raw_refresh()
    family = token_family or uuid.uuid4()
    row = AuthRefreshToken(
        id=uuid.uuid4(),
        user_id=user_id,
        token_hash=_hash_token(raw_refresh),
        token_family=family,
        issued_at=_now(),
        expires_at=_now() + timedelta(days=config.JWT_REFRESH_EXPIRY_DAYS),
        user_agent=_user_agent(request),
        ip_address=_client_ip(request),
    )
    db.add(row)
    db.flush()  # populate row.id without committing — caller commits
    return access_token, raw_refresh, row


def rotate_refresh_token(
    db: Session,
    raw_refresh: str,
    request: Optional[Request] = None,
) -> Optional[Tuple[str, str]]:
    """
    Validate `raw_refresh` and issue a new pair.

    Returns (new_access, new_refresh_raw) on success.
    Returns None on any failure (expired / unknown / replayed).

    Replay handling: if the presented token row is *already revoked* and
    we are outside the rotation grace window, every token in its family is
    revoked — the chain is presumed compromised.
    """
    h = _hash_token(raw_refresh)
    # Lock the row for the duration of this transaction so two concurrent
    # tabs presenting the same refresh token can't both run the rotation
    # logic and produce inconsistent `revoked_at` + `replaced_by` chains.
    # Postgres SELECT … FOR UPDATE serialises the rotators: the second
    # request waits, then sees the row already revoked + replaced_by set,
    # and goes through the grace-window branch instead of stomping the
    # first rotator's chain.
    row = (
        db.query(AuthRefreshToken)
        .filter(AuthRefreshToken.token_hash == h)
        .with_for_update()
        .first()
    )
    if row is None:
        log.warning("[AUTH_REFRESH] unknown token presented")
        return None

    now = _now()

    # Expired absolutely — refuse
    if row.expires_at <= now:
        log.info("[AUTH_REFRESH] expired token presented (user=%s)", row.user_id)
        return None

    # Revoked — check rotation grace window before assuming compromise
    if row.revoked_at is not None:
        grace_end = row.revoked_at + timedelta(seconds=config.REFRESH_ROTATION_GRACE_SECONDS)
        if now <= grace_end and row.replaced_by is not None:
            # Concurrent-tab race: the previous request rotated this token a
            # few seconds ago. Re-issue against the replacement chain rather
            # than penalising the user. We do NOT chain a second rotation —
            # just emit a fresh access JWT scoped to the same family. This
            # avoids token-table bloat from racing tabs.
            log.info("[AUTH_REFRESH] grace-window replay accepted (user=%s family=%s)",
                     row.user_id, row.token_family)
            access_token = issue_access_token(row.user_id)
            # Issue a new refresh that piggybacks on the existing replacement
            # — keeps families coherent without adding another revoked row.
            # Same FOR UPDATE pattern on the replacement row — guards
            # against two concurrent grace-path requests both reading the
            # replacement, both generating new raw bytes, and both
            # overwriting `replacement.token_hash` (last commit wins, the
            # other user's token becomes orphaned).
            replacement = (
                db.query(AuthRefreshToken)
                .filter(AuthRefreshToken.id == row.replaced_by)
                .with_for_update()
                .first()
            )
            if replacement and replacement.revoked_at is None:
                # Hand the caller the most recent live token in this family.
                # Rare path; we generate fresh raw bytes and replace the
                # replacement row's hash, keeping it single-use.
                new_raw = _new_raw_refresh()
                replacement.token_hash = _hash_token(new_raw)
                replacement.last_used_at = now
                db.add(replacement)
                return access_token, new_raw
            # Replacement chain broken — fall through to compromise path
        # Outside grace OR no replacement chain → treat as theft.
        _revoke_family(db, row.token_family)
        log.warning("[AUTH_REFRESH] REPLAY DETECTED — family %s revoked (user=%s)",
                    row.token_family, row.user_id)
        return None

    # Happy path: revoke this row, issue a new one in the same family
    row.revoked_at = now
    row.last_used_at = now
    new_access, new_raw, new_row = issue_token_pair(
        db, user_id=row.user_id, request=request, token_family=row.token_family,
    )
    row.replaced_by = new_row.id
    db.add(row)
    return new_access, new_raw


def revoke_refresh_token(db: Session, raw_refresh: str) -> bool:
    """Mark this refresh token revoked. Used by /auth/logout. Idempotent."""
    if not raw_refresh:
        return False
    h = _hash_token(raw_refresh)
    row = db.query(AuthRefreshToken).filter(AuthRefreshToken.token_hash == h).first()
    if row is None or row.revoked_at is not None:
        return False
    row.revoked_at = _now()
    db.add(row)
    return True


def revoke_all_for_user(db: Session, user_id: int) -> int:
    """Logout-all-devices: revoke every active refresh token for the user.
    Returns number of rows revoked."""
    now = _now()
    n = (
        db.query(AuthRefreshToken)
        .filter(AuthRefreshToken.user_id == user_id, AuthRefreshToken.revoked_at.is_(None))
        .update({AuthRefreshToken.revoked_at: now}, synchronize_session=False)
    )
    return int(n or 0)


def _revoke_family(db: Session, family_id: uuid.UUID) -> None:
    now = _now()
    db.query(AuthRefreshToken).filter(
        AuthRefreshToken.token_family == family_id,
        AuthRefreshToken.revoked_at.is_(None),
    ).update({AuthRefreshToken.revoked_at: now}, synchronize_session=False)
