"""
Trainer authentication: stateless JWTs with a separate `aud` claim.

Tokens are signed with the same `JWT_SECRET_KEY` as user tokens but carry
`aud='trainer'`, so the student `get_current_user` dependency rejects them
and the trainer `get_current_trainer` dependency rejects student tokens.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

import core.config as config
from db.database import get_db
from db.models import Trainer


# Standalone bearer scheme — different tokenUrl than user auth so OpenAPI
# docs render two distinct "Authorize" buttons.
trainer_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/trainer/auth/verify-otp",
    scheme_name="TrainerBearer",
    auto_error=True,
)


def make_trainer_token(trainer: Trainer) -> str:
    """Mint a JWT carrying `aud='trainer'`. Lifetime = TRAINER_JWT_EXPIRY_HOURS."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": f"trainer:{trainer.id}",
        "trainer_id": trainer.id,
        "email": trainer.email,
        "aud": config.TRAINER_JWT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=config.TRAINER_JWT_EXPIRY_HOURS)).timestamp()),
    }
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def decode_trainer_token(token: str) -> dict:
    """Decode + verify audience. Raises JWTError on any failure."""
    return jwt.decode(
        token,
        config.JWT_SECRET_KEY,
        algorithms=[config.JWT_ALGORITHM],
        audience=config.TRAINER_JWT_AUDIENCE,
    )


def get_current_trainer(
    token: str = Depends(trainer_oauth2_scheme),
    db: Session = Depends(get_db),
) -> Trainer:
    """FastAPI dependency: extracts the trainer from the bearer token."""
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired trainer token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_trainer_token(token)
        trainer_id: Optional[int] = payload.get("trainer_id")
        if trainer_id is None:
            raise creds_exc
    except JWTError:
        raise creds_exc

    trainer = db.query(Trainer).filter(Trainer.id == int(trainer_id)).first()
    if trainer is None or not trainer.is_active:
        # Active-flag check on every request: admin can revoke a trainer
        # by flipping is_active=false even before the JWT expires.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Trainer no longer authorized",
        )
    return trainer
