import os
import time
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base, Session

import core.config as config

log = logging.getLogger(__name__)


class RetrySession(Session):
    _MAX_RETRIES = 3
    _RETRY_DELAY = 0.5

    def execute(self, statement, params=None, **kw):
        last_exc: Exception = RuntimeError("execute: no attempts made")
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return super().execute(statement, params, **kw)
            except OperationalError as exc:
                last_exc = exc
                log.warning("[DB] execute attempt=%d/3 failed: %s", attempt, exc)
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY * attempt)
        raise last_exc

    def commit(self):
        last_exc: Exception = RuntimeError("commit: no attempts made")
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return super().commit()
            except OperationalError as exc:
                last_exc = exc
                log.warning("[DB] commit attempt=%d/3 failed: %s", attempt, exc)
                if attempt < self._MAX_RETRIES:
                    super().rollback()
                    time.sleep(self._RETRY_DELAY * attempt)
        raise last_exc

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Single, shared engine for the entire app
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=config.DB_POOL_SIZE,
    max_overflow=config.DB_MAX_OVERFLOW,
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=RetrySession,
)

# Base class for models
Base = declarative_base()


def init_db():
    """
    Create tables if they do not exist.
    Safe to call multiple times.
    """
    from db import models  # IMPORTANT: registers models
    Base.metadata.create_all(bind=engine)


# Dependency for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
