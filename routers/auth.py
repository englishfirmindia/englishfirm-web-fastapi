from datetime import datetime, timedelta, date
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import jwt, jwk as jose_jwk
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests as _requests

import core.config as config
from db.database import get_db
from db.models import User
from services.zapier import send_signup_webhook

router = APIRouter(prefix="/auth", tags=["Auth"])
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    phone: Optional[str] = None
    score_requirement: Optional[int] = None
    exam_date: Optional[str] = None


class GoogleAuthRequest(BaseModel):
    id_token: str


class AppleAuthRequest(BaseModel):
    identity_token: str


def _make_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(days=config.JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def _parse_exam_date(exam_date_str: Optional[str]) -> Optional[date]:
    if not exam_date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(exam_date_str, fmt).date()
        except ValueError:
            continue
    return None


@router.post("/signup", status_code=201)
def signup(req: SignupRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    parsed_exam_date = _parse_exam_date(req.exam_date)
    user = User(
        username=req.username,
        email=req.email,
        hashed_password=_pwd.hash(req.password),
        phone=req.phone,
        score_requirement=req.score_requirement,
        exam_date=parsed_exam_date,
    )
    db.add(user)
    db.commit()
    background_tasks.add_task(
        send_signup_webhook,
        student_name=req.username,
        phone_number=req.phone,
        exam_date=parsed_exam_date,
    )
    return {"message": "Account created"}


@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not _pwd.verify(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    return {"access_token": _make_token(user.id), "token_type": "bearer"}


@router.post("/google")
def google_auth(req: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Verify a Google id_token and return a JWT. Creates the user if they don't exist."""
    try:
        resp = _requests.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={req.id_token}",
            timeout=10,
        )
        resp.raise_for_status()
        token_info = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Google token verification failed: {exc}")

    if "error" in token_info:
        raise HTTPException(status_code=401, detail=token_info.get("error_description", "Invalid Google token"))

    email = token_info.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Google token missing email")

    if not token_info.get("email_verified", False):
        raise HTTPException(status_code=401, detail="Google email not verified")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        name = token_info.get("name") or token_info.get("given_name") or email.split("@")[0]
        user = User(
            username=name,
            email=email,
            hashed_password=_pwd.hash(token_info.get("sub", "")),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return {"access_token": _make_token(user.id), "token_type": "bearer"}


@router.post("/apple")
def apple_auth(req: AppleAuthRequest, db: Session = Depends(get_db)):
    """
    Verify an Apple identity_token (RS256, signed by Apple) and return a JWT.
    Creates the user if they don't exist. Audience is whitelisted against
    APPLE_ALLOWED_AUDIENCES so only our own iOS + web apps are accepted.
    """
    try:
        keys_resp = _requests.get(
            "https://appleid.apple.com/auth/keys", timeout=10
        )
        keys_resp.raise_for_status()
        jwks = keys_resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Could not fetch Apple public keys: {exc}"
        )

    try:
        unverified_header = jwt.get_unverified_header(req.identity_token)
    except Exception as exc:
        raise HTTPException(
            status_code=401, detail=f"Malformed Apple identity token: {exc}"
        )

    kid = unverified_header.get("kid")
    matching_key = next(
        (k for k in jwks.get("keys", []) if k.get("kid") == kid), None
    )
    if not matching_key:
        raise HTTPException(status_code=401, detail="Apple signing key not found")

    try:
        public_key = jose_jwk.construct(matching_key)
        # Decode without aud check; we'll whitelist explicitly below.
        claims = jwt.decode(
            req.identity_token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=401, detail=f"Invalid Apple identity token: {exc}"
        )

    aud = claims.get("aud")
    if aud not in config.APPLE_ALLOWED_AUDIENCES:
        raise HTTPException(
            status_code=401, detail="Apple token audience not allowed"
        )

    email = claims.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Apple token missing email")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        name = email.split("@")[0]
        # `sub` is Apple's stable user ID — fine as a placeholder password seed.
        user = User(
            username=name,
            email=email,
            hashed_password=_pwd.hash(claims.get("sub", "")),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return {"access_token": _make_token(user.id), "token_type": "bearer"}
