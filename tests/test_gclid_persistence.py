"""Regression guard for gclid persistence — 2026-07-17.

Before this feature, the frontend captured `?gclid=…` on landing, used
its presence to set `from_google_ads=True`, and DISCARDED the raw string.
That prevented us from ever matching a specific user back to their
click (Google Ads UI look-up, Offline Conversions API upload, etc.).

These tests pin the three touchpoints that must all stay in sync:
  1. `SignupRequest` accepts `gclid: Optional[str]`
  2. `GoogleAuthRequest` accepts `gclid: Optional[str]`
  3. The User() constructor call in both signup paths passes it through
     with defensive truncation to the schema's VARCHAR(200) limit.
"""
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def test_signup_request_declares_gclid_field():
    src = (REPO / "routers/auth.py").read_text()
    # Look for the field declaration inside SignupRequest — the class runs
    # through end-of-file, so a plain 'gclid:' match at module scope is fine.
    assert "class SignupRequest(BaseModel):" in src
    assert "gclid:        Optional[str] = None" in src, (
        "SignupRequest must declare `gclid: Optional[str] = None` so the "
        "frontend can forward it. Without this pydantic silently drops it "
        "and every gads signup loses attribution."
    )


def test_google_auth_request_declares_gclid_field():
    src = (REPO / "routers/auth.py").read_text()
    assert "class GoogleAuthRequest(BaseModel):" in src
    # Both classes declare `gclid:        Optional[str] = None` — enforce
    # the field appears twice (once per BaseModel).
    assert src.count("gclid:        Optional[str] = None") >= 2, (
        "GoogleAuthRequest must also declare `gclid` for the OAuth signup "
        "path; otherwise gads users who use 'Continue with Google' lose "
        "attribution silently."
    )


def test_signup_persists_gclid_to_user_row():
    src = (REPO / "routers/auth.py").read_text()
    # The User(...) constructor call in the signup() handler must pass
    # gclid through. Search for the truncation pattern which is the
    # canonical shape we wrote.
    assert "gclid=(req.gclid[:200] if req.gclid else None)" in src, (
        "signup() and google() handlers must persist req.gclid onto the "
        "User row. The `[:200]` truncation matches the VARCHAR(200) "
        "schema limit."
    )
    # Both signup and google handlers use the same line — must appear >= 2
    assert src.count("gclid=(req.gclid[:200] if req.gclid else None)") >= 2


def test_user_model_declares_gclid_column():
    src = (REPO / "db/models.py").read_text()
    assert "gclid           = Column(String(200), nullable=True)" in src, (
        "User model must declare the gclid column with matching type "
        "(VARCHAR(200)). Without this SQLAlchemy will raise on assignment."
    )


def test_startup_migration_adds_gclid_column():
    src = (REPO / "main.py").read_text()
    assert '"gclid           VARCHAR(200)"' in src, (
        "Startup migration must include `gclid VARCHAR(200)` so first "
        "deploy against prod RDS adds the column via "
        "ALTER TABLE ... ADD COLUMN IF NOT EXISTS."
    )
    # And the partial index that keeps lookups cheap
    assert "ix_users_gclid ON users (gclid) WHERE gclid IS NOT NULL" in src, (
        "Partial index on gclid (WHERE NOT NULL) is required so "
        "conversion-upload lookups don't full-scan the users table."
    )
