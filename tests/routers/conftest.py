import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Patch DATABASE_URL before any app import so db/database.py doesn't raise
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from main import app
from db.database import get_db
from core.dependencies import get_current_user
from db.models import User


@pytest.fixture
def mock_user():
    u = MagicMock(spec=User)
    u.id = 1
    u.email = "test@test.com"
    return u


@pytest.fixture
def client(mock_user):
    def override_db():
        yield MagicMock()

    def override_user():
        return mock_user

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    yield TestClient(app)
    app.dependency_overrides.clear()
