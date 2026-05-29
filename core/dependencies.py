from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
import core.config as config

# auto_error=False so a missing Authorization header doesn't 401 immediately —
# we want to fall through to the cookie before rejecting.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def _extract_token(request: Request, header_token: Optional[str]) -> Optional[str]:
    if header_token:
        return header_token
    return request.cookies.get(config.SESSION_COOKIE_NAME)


def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    raw = _extract_token(request, token)
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        payload = jwt.decode(raw, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def try_get_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Soft-auth dependency: returns the authenticated User if the
    Authorization header (or session cookie) carries a valid JWT,
    otherwise returns None instead of raising 401.

    Use ONLY for endpoints that must accept anonymous requests by design.
    Built for the frontend telemetry endpoint, which must work even
    before login (boot errors, login-page JS errors) and from
    `navigator.sendBeacon` calls which cannot carry the Authorization
    header. Do NOT use on any endpoint that touches user-scoped data —
    keep `get_current_user` for those.
    """
    raw = _extract_token(request, token)
    if not raw:
        return None
    try:
        payload = jwt.decode(raw, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return None
    except JWTError:
        return None
    return db.query(User).filter(User.id == int(user_id)).first()


def get_subscription_context(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """FastAPI dependency: resolve the active SubscriptionContext for the
    authenticated user. Returned by GET /subscription/me and consumed by
    every gated endpoint via EnforceLimit (Week 3).

    No caching yet — single indexed query, sub-millisecond on RDS. Add a
    TTL cache once Stripe webhooks (Week 4) need invalidation hooks."""
    from services.billing.subscription_context import resolve_subscription_context
    return resolve_subscription_context(db, user.id)
