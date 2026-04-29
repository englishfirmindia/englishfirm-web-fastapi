"""
One-shot setup for the human-trainer sharing system.

Idempotent. Run from the repo root with the venv active:
    python scripts/init_trainer_tables.py

Steps:
  1. Loads .env so DATABASE_URL is available
  2. Calls Base.metadata.create_all() — creates the 4 new tables if missing
  3. Upserts the seed trainer rows in TRAINER_SEEDS

Re-running this script is safe: existing tables and trainer rows are
left untouched (no overwrites).
"""

import os
import sys

# Make project root importable when run from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"))

from db.database import Base, engine, SessionLocal  # noqa: E402
from db import models  # noqa: F401, E402  — registers all models on Base
from db.models import Trainer  # noqa: E402


# Edit this list to add / remove trainer whitelisting rows.
TRAINER_SEEDS = [
    {"email": "nimishaelizabethjames99@gmail.com", "display_name": "Nimisha"},
]


def ensure_tables() -> None:
    """Create any tables that don't yet exist (no-op for ones that do)."""
    print("[init] creating tables (idempotent) ...")
    Base.metadata.create_all(bind=engine)
    print("[init] done.")


def upsert_trainers() -> None:
    """Insert any seed trainers that aren't already in the table."""
    db = SessionLocal()
    try:
        for seed in TRAINER_SEEDS:
            email = seed["email"].strip().lower()
            display_name = seed["display_name"].strip()

            existing = db.query(Trainer).filter(Trainer.email == email).first()
            if existing:
                print(
                    f"[seed] trainer already present: id={existing.id} "
                    f"email={existing.email} active={existing.is_active}"
                )
                continue

            row = Trainer(email=email, display_name=display_name, is_active=True)
            db.add(row)
            db.commit()
            db.refresh(row)
            print(
                f"[seed] inserted trainer id={row.id} "
                f"email={row.email} name={row.display_name}"
            )
    finally:
        db.close()


def main() -> None:
    if not os.getenv("DATABASE_URL"):
        raise SystemExit(
            "DATABASE_URL not set. Source .env or export it before running."
        )
    ensure_tables()
    upsert_trainers()
    print("[init] all good.")


if __name__ == "__main__":
    main()
