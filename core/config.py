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
DB_POOL_SIZE = 5
DB_MAX_OVERFLOW = 10

# ── Auth (JWT) ────────────────────────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# ── Session defaults ──────────────────────────────────────────────────────────
SESSION_QUESTION_LIMIT = 20

# ── PTE Score formula — max(PTE_FLOOR, min(PTE_CEILING, round(PTE_BASE + pct * PTE_SCALE))) ──
PTE_FLOOR = 10
PTE_CEILING = 90
PTE_BASE = 10
PTE_SCALE = 80
