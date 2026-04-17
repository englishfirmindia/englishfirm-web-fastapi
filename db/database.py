import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import core.config as config

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
