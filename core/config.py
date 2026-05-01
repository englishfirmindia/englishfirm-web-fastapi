import os

# ── AWS / S3 ──────────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_S3_REGION", "ap-southeast-2")
S3_RECORDINGS_BUCKET = os.getenv("S3_RECORDINGS_BUCKET", "apeuni-user-recordings")
S3_QUESTIONS_BUCKET = os.getenv("S3_QUESTIONS_BUCKET", "apeuni-questions-audio")
PRESIGNED_READ_EXPIRY_SECONDS = 3600   # 1 hour
PRESIGNED_UPLOAD_EXPIRY_SECONDS = 300  # 5 minutes

# ── Azure Speech ──────────────────────────────────────────────────────────────
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "australiaeast")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 40

# ── Auth (JWT) ────────────────────────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_DAYS = 30

# ── Auth (web session cookie) ─────────────────────────────────────────────────
# Web clients use an httpOnly cookie holding the same JWT, so the token is not
# reachable from JavaScript. The iOS client keeps using `Authorization: Bearer`
# and ignores Set-Cookie — both transports are accepted by get_current_user.
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "ef_session")
# Secure=True is required for cross-site cookie use over HTTPS. Localhost is
# treated as secure by modern browsers, so this default is safe for dev too.
# Override with COOKIE_SECURE=false only if you know what you're doing.
SESSION_COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() == "true"
# Lax (not Strict): Strict breaks the post-Google/Apple OAuth top-level redirect
# back to our origin. Lax still blocks the common cross-site CSRF cases.
SESSION_COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")
# Optional cookie domain — set for cross-subdomain auth (e.g. ".englishfirm.com"
# so api.englishfirm.com sets a cookie readable from app.englishfirm.com).
# Leave empty in dev (cookie defaults to host-only).
SESSION_COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", "") or None
SESSION_COOKIE_MAX_AGE_SECONDS = JWT_EXPIRY_DAYS * 24 * 60 * 60

# ── Session defaults ──────────────────────────────────────────────────────────
SESSION_QUESTION_LIMIT = 20

# ── PTE Score formula — max(PTE_FLOOR, min(PTE_CEILING, round(PTE_BASE + pct * PTE_SCALE))) ──
PTE_FLOOR = 10
PTE_CEILING = 90
PTE_BASE = 10
PTE_SCALE = 80

# ── Trainer (OTP login + sharing) ─────────────────────────────────────────────
TRAINER_OTP_LENGTH = 6
TRAINER_OTP_EXPIRY_MINUTES = 10
TRAINER_OTP_MAX_ATTEMPTS = 5
TRAINER_OTP_RATE_LIMIT_PER_HOUR = 5
TRAINER_JWT_EXPIRY_HOURS = 24
TRAINER_AUDIO_PRESIGN_TTL_SECONDS = 259_200  # 3 days

# Trainer-side audience claim — keeps the same JWT_SECRET_KEY but isolates
# trainer tokens from student tokens via the `aud` field.
TRAINER_JWT_AUDIENCE = "trainer"
USER_JWT_AUDIENCE = "user"

# ── Email delivery ────────────────────────────────────────────────────────────
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@englishfirm.local")
EMAIL_WEBHOOK_URL = os.getenv("EMAIL_WEBHOOK_URL", "")  # optional Zapier/etc.

# ── Frontend (used to build links inside outbound emails) ─────────────────────
# Override in production with the deployed web app URL, e.g. https://app.englishfirm.com
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8080")

# ── CORS allowlist ────────────────────────────────────────────────────────────
# Comma-separated origins allowed to call the API. Defaults to FRONTEND_URL plus
# common local-dev hosts. Override in production.
CORS_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        f"{FRONTEND_URL},"
        "http://localhost:8080,http://127.0.0.1:8080,"
        "http://localhost:5000,http://127.0.0.1:5000",
    ).split(",")
    if o.strip()
]

# ── Password reset (stateless, JWT-based) ─────────────────────────────────────
PASSWORD_RESET_TOKEN_EXPIRY_MINUTES = 15
PASSWORD_RESET_TOKEN_PURPOSE = "password_reset"

# ── Apple Sign-In ─────────────────────────────────────────────────────────────
# Comma-separated list of allowed `aud` values inside Apple identity_tokens.
# Should include the iOS bundle ID and the web Apple Services ID.
# Example: "com.englishfirm.assistant,com.englishfirm.web"
APPLE_ALLOWED_AUDIENCES = [
    a.strip()
    for a in os.getenv(
        "APPLE_ALLOWED_AUDIENCES",
        "com.englishfirm.assistant,com.englishfirm.web",
    ).split(",")
    if a.strip()
]
