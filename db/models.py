from sqlalchemy import Column, Integer, String, DateTime, Date, ForeignKey, Text, Boolean, JSON, CheckConstraint, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy import ForeignKey



class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    phone = Column(String, nullable=True)
    score_requirement = Column(Integer, nullable=True)
    exam_date = Column(Date, nullable=True)

    current_plan = Column(String, nullable=True)
    plan_started_at = Column(Date, nullable=True)
    plan_end_at = Column(Date, nullable=True)

    status = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True)

    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    status = Column(String(20), nullable=False, default="active")
    pinned_summary = Column(Text, nullable=True)

    message_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    role = Column(String(20), nullable=False)  # user | assistant
    content = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")



class Question(Base):
    __tablename__ = "questions"

    question_id = Column(Integer, primary_key=True, index=True)

    module = Column(String(50), nullable=False)
    question_type = Column(String(50), nullable=False)

    title = Column(Text, nullable=True)

    difficulty_level = Column(
        Integer,
        CheckConstraint("difficulty_level BETWEEN 1 AND 5"),
        nullable=True,
    )

    time_limit_seconds = Column(Integer, nullable=True)

    content_json = Column(JSONB, nullable=False)

    tags = Column(ARRAY(Text), nullable=True)

    is_active = Column(Boolean, default=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    evaluation = relationship(
        "QuestionEvaluation",
        back_populates="question",
        uselist=False,
        cascade="all, delete",
    )


class QuestionEvaluation(Base):
    __tablename__ = "question_evaluation"

    question_id = Column(
        Integer,
        ForeignKey("questions.question_id", ondelete="CASCADE"),
        primary_key=True,
    )

    evaluation_json = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    updated_at = Column(DateTime(timezone=True), nullable=True)

    question = relationship(
        "Question",
        back_populates="evaluation",
    )

class QuestionFromApeuni(Base):
    __tablename__ = "questions_from_apeuni"

    question_id = Column(Integer, primary_key=True, index=True)

    question_number_from_apeuni = Column(Integer, nullable=True)

    module = Column(String(50), nullable=False)
    question_type = Column(String(50), nullable=False)

    title = Column(Text, nullable=True)

    difficulty_level = Column(
        Integer,
        CheckConstraint("difficulty_level BETWEEN 1 AND 5"),
        nullable=True,
    )

    time_limit_seconds = Column(Integer, nullable=True)

    content_json = Column(JSONB, nullable=False)

    tags = Column(ARRAY(Text), nullable=True)

    is_prediction = Column(Boolean, default=False)
    audio_duration_seconds = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    evaluation = relationship(
        "QuestionEvaluationApeuni",
        back_populates="question",
        uselist=False,
        cascade="all, delete",
    )


class QuestionEvaluationApeuni(Base):
    __tablename__ = "question_evaluation_apeuni"

    question_id = Column(
        Integer,
        ForeignKey("questions_from_apeuni.question_id", ondelete="CASCADE"),
        primary_key=True,
    )

    evaluation_json = Column(JSONB, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    question = relationship(
        "QuestionFromApeuni",
        back_populates="evaluation",
    )


class UserQuestionAttempt(Base):
    """Tracks which questions a user has already practiced.
    Used to exclude practiced questions from sectional exam selection.
    """
    __tablename__ = "user_question_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id = Column(Integer, nullable=False, index=True)
    question_type = Column(String(50), nullable=False)
    module = Column(String(50), nullable=False)
    attempted_at = Column(DateTime(timezone=True), server_default=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# Milestones
# ─────────────────────────────────────────────────────────────────────────────

class Milestone(Base):
    __tablename__ = "milestones"

    id           = Column(Integer, primary_key=True, index=True)
    number       = Column(Integer, unique=True, nullable=False)
    name         = Column(String(50), nullable=False)
    emoji        = Column(String(10), nullable=False)
    category     = Column(String(50), nullable=False)   # Onboarding | Foundation | Sectional Tests | Mock Test
    total_points = Column(Integer, nullable=False, default=25)
    description  = Column(String(255), nullable=True)

    tasks = relationship("MilestoneTask", back_populates="milestone", order_by="MilestoneTask.sort_order")


class MilestoneTask(Base):
    __tablename__ = "milestone_tasks"

    id                  = Column(Integer, primary_key=True, index=True)
    milestone_id        = Column(Integer, ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False, index=True)
    task_code           = Column(String(50), nullable=False)
    task_label          = Column(String(100), nullable=False)
    task_type           = Column(String(20), nullable=False)  # action | learn | quiz | practice | bonus | mock_scored
    target_count        = Column(Integer, nullable=False, default=1)
    points              = Column(Integer, nullable=True)
    pass_threshold      = Column(String(20), nullable=True)   # NULL | "80" | "user_target"
    complete_on_target  = Column(Boolean, default=False)
    sort_order          = Column(Integer, nullable=False, default=1)
    app_route           = Column(String(100), nullable=True)

    milestone = relationship("Milestone", back_populates="tasks")
    tiers     = relationship("MilestoneTaskTier", back_populates="task")


class MilestoneTaskTier(Base):
    __tablename__ = "milestone_task_tiers"

    id          = Column(Integer, primary_key=True, index=True)
    task_id     = Column(Integer, ForeignKey("milestone_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    tier_label  = Column(String(50), nullable=False)
    condition   = Column(String(50), nullable=False)  # below_target | near_target | hit_target
    points      = Column(Integer, nullable=False)

    task = relationship("MilestoneTask", back_populates="tiers")


class UserMilestoneTaskProgress(Base):
    __tablename__ = "user_milestone_task_progress"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    task_id       = Column(Integer, ForeignKey("milestone_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    current_count = Column(Integer, nullable=False, default=0)
    is_complete   = Column(Boolean, default=False)
    points_earned = Column(Integer, nullable=True)
    completed_at  = Column(DateTime(timezone=True), nullable=True)
    updated_at    = Column(DateTime(timezone=True), server_default=func.now())

# ─────────────────────────────────────────────────────────────────────────────
# Practice Attempts + Attempt Answers  (sectional / mock scoring)
# ─────────────────────────────────────────────────────────────────────────────

class PracticeAttempt(Base):
    __tablename__ = "practice_attempts"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id          = Column(String, unique=True, nullable=False, index=True)
    module              = Column(String(50), nullable=False)
    question_type       = Column(String(50), nullable=False)
    filter_type         = Column(String(20), nullable=False)
    total_questions     = Column(Integer, nullable=False)
    total_score         = Column(Integer, nullable=False, default=0)
    questions_answered  = Column(Integer, nullable=False, default=0)
    status              = Column(String(20), nullable=False, default="pending")
    scoring_status      = Column(String(20), nullable=True, default="pending")
    task_breakdown           = Column(JSONB, nullable=True)
    selected_question_ids    = Column(JSONB, nullable=True)
    started_at          = Column(DateTime(timezone=True), server_default=func.now())
    completed_at        = Column(DateTime(timezone=True), nullable=True)

    answers = relationship("AttemptAnswer", back_populates="attempt", cascade="all, delete")


# ─────────────────────────────────────────────────────────────────────────────
# Student Trainer Profile  (long-term AI trainer memory)
# ─────────────────────────────────────────────────────────────────────────────

class StudentTrainerProfile(Base):
    __tablename__ = "student_trainer_profiles"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )

    # Conversation phase: intake → planning → coaching → review
    phase          = Column(String(20), nullable=False, default="intake")
    session_count  = Column(Integer, nullable=False, default=0)
    last_session_at = Column(DateTime(timezone=True), nullable=True)

    # Gathered through conversation (tool calling fills these)
    motivation              = Column(Text, nullable=True)   # why they need PTE
    study_hours_per_day     = Column(Float, nullable=True)
    study_schedule          = Column(Text, nullable=True)   # "1hr morning, 30min evening"
    prior_pte_attempts      = Column(Integer, nullable=True)
    anxiety_level           = Column(String(10), nullable=True)  # low / medium / high
    learning_style          = Column(Text, nullable=True)
    biggest_weakness_self   = Column(Text, nullable=True)   # what student thinks is weak

    # Training plan
    plan_text          = Column(Text, nullable=True)
    plan_generated_at  = Column(DateTime(timezone=True), nullable=True)

    # Session continuity
    last_session_summary = Column(Text, nullable=True)

    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), nullable=True)


class AttemptAnswer(Base):
    __tablename__ = "attempt_answers"

    id                  = Column(Integer, primary_key=True, index=True)
    attempt_id          = Column(Integer, ForeignKey("practice_attempts.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id         = Column(Integer, nullable=False)
    question_type       = Column(String(50), nullable=False)
    user_answer_json    = Column(JSONB, nullable=False, default=dict)
    correct_answer_json = Column(JSONB, nullable=False, default=dict)
    result_json         = Column(JSONB, nullable=False, default=dict)
    score               = Column(Integer, nullable=False, default=0)
    content_score       = Column(Float, nullable=True)
    fluency_score       = Column(Float, nullable=True)
    pronunciation_score = Column(Float, nullable=True)
    audio_url           = Column(String(500), nullable=True)
    trainer_feedback    = Column(String(2000), nullable=True)
    scoring_status      = Column(String(20), nullable=True, default="pending")
    submitted_at        = Column(DateTime(timezone=True), server_default=func.now())

    attempt = relationship("PracticeAttempt", back_populates="answers")


# ─────────────────────────────────────────────────────────────────────────────
# Student Milestones
# ─────────────────────────────────────────────────────────────────────────────

class StudentMilestone(Base):
    __tablename__ = "student_milestones"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    milestone_key = Column(String(50), nullable=False)   # e.g. "practice_10", "speaking_above_65"
    achieved_at   = Column(DateTime(timezone=True), server_default=func.now())
    metadata_     = Column("metadata", JSONB, nullable=True)  # extra context (score, module, etc.)

    __table_args__ = (
        # Each milestone awarded at most once per user
        __import__('sqlalchemy').UniqueConstraint("user_id", "milestone_key", name="uq_user_milestone"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Human Trainer Sharing  (admin-managed whitelist + per-attempt shares + notes)
# ─────────────────────────────────────────────────────────────────────────────

class Trainer(Base):
    """
    Admin-managed whitelist of human trainers.
    Trainers do NOT appear in the `users` table — they authenticate by
    email + OTP and receive a separate JWT (audience='trainer').
    """
    __tablename__ = "trainers"

    id           = Column(Integer, primary_key=True, index=True)
    email        = Column(String(255), unique=True, index=True, nullable=False)
    display_name = Column(String(120), nullable=False)
    is_active    = Column(Boolean, nullable=False, default=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), nullable=True)


class TrainerOtp(Base):
    """
    One-time codes for trainer login. New rows on every request-otp call;
    older unconsumed rows for the same email are invalidated when a new
    code is issued.
    """
    __tablename__ = "trainer_otps"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), nullable=False, index=True)
    code          = Column(String(6), nullable=False)
    expires_at    = Column(DateTime(timezone=True), nullable=False)
    consumed_at   = Column(DateTime(timezone=True), nullable=True)
    attempts_left = Column(Integer, nullable=False, default=5)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    ip            = Column(String(45), nullable=True)
    user_agent    = Column(Text, nullable=True)


class TrainerShare(Base):
    """
    A student-initiated share of one specific PracticeAttempt with one
    specific trainer. Re-sharing after revoke creates a new row; the old
    revoked row stays for audit.
    """
    __tablename__ = "trainer_shares"

    id              = Column(Integer, primary_key=True, index=True)
    attempt_id      = Column(Integer, ForeignKey("practice_attempts.id", ondelete="CASCADE"), nullable=False, index=True)
    student_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    trainer_id      = Column(Integer, ForeignKey("trainers.id", ondelete="CASCADE"), nullable=False, index=True)
    shared_at       = Column(DateTime(timezone=True), server_default=func.now())
    revoked_at      = Column(DateTime(timezone=True), nullable=True)
    first_viewed_at = Column(DateTime(timezone=True), nullable=True)
    last_viewed_at  = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Only one ACTIVE share per (attempt, trainer); revoked rows excluded.
        __import__('sqlalchemy').Index(
            "ix_trainer_shares_active",
            "attempt_id",
            "trainer_id",
            unique=True,
            postgresql_where=__import__('sqlalchemy').text("revoked_at IS NULL"),
        ),
    )


class TrainerNote(Base):
    """
    Notes left by a trainer on a shared attempt (or specific question
    inside that attempt). Persists even if the share is revoked — student
    keeps full visibility forever.
    """
    __tablename__ = "trainer_notes"

    id          = Column(Integer, primary_key=True, index=True)
    share_id    = Column(Integer, ForeignKey("trainer_shares.id", ondelete="CASCADE"), nullable=False, index=True)
    attempt_id  = Column(Integer, ForeignKey("practice_attempts.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id = Column(Integer, nullable=True, index=True)  # null = attempt-level note
    trainer_id  = Column(Integer, ForeignKey("trainers.id", ondelete="CASCADE"), nullable=False, index=True)
    body        = Column(Text, nullable=False)
    rating      = Column(Integer, nullable=True)  # optional 1-5 rubric
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), nullable=True)
    deleted_at  = Column(DateTime(timezone=True), nullable=True)
