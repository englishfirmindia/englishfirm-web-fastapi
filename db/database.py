import os
import time
import logging
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base, Session

import core.config as config

log = logging.getLogger(__name__)

# Log any DB statement that runs longer than this. Surfaces real slow queries
# in CloudWatch with the SQL template + duration so we can attribute slowness
# to specific routes without enabling pg_stat_statements (which needs a DB
# restart). Threshold is intentionally well below the statement_timeout so we
# see the long tail before it gets cancelled.
_SLOW_QUERY_MS = 500


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
    # statement_timeout caps any single query at 10s. Anything truly stuck is
    # killed by Postgres and the connection released, so workers don't pile up
    # behind a slow query while ALB waits its 60s idle timeout.
    connect_args={"options": "-c statement_timeout=10000"},
)


@event.listens_for(engine, "before_cursor_execute")
def _record_query_start(conn, cursor, statement, parameters, context, executemany):
    context._query_start = time.monotonic()


@event.listens_for(engine, "after_cursor_execute")
def _log_slow_query(conn, cursor, statement, parameters, context, executemany):
    started = getattr(context, "_query_start", None)
    if started is None:
        return
    duration_ms = (time.monotonic() - started) * 1000
    if duration_ms < _SLOW_QUERY_MS:
        return
    # Collapse whitespace + truncate so a 5KB JOIN doesn't blow up the log.
    sql = " ".join(statement.split())
    if len(sql) > 400:
        sql = sql[:400] + "…"
    log.warning("[SLOW_QUERY] ms=%.0f sql=%s", duration_ms, sql)

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
