"""Database tables.

exams         one uploaded answer schema (questions + marking scheme stored as JSON)
submissions   one student's answer script for an exam, with its status and totals
answer_evals  one row per question per script: AI suggestion, evidence, flags, final mark
"""
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _now() -> datetime:
    return datetime.now()


class Exam(Base):
    __tablename__ = "exams"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    schema_file: Mapped[str] = mapped_column(String(300))
    total_marks: Mapped[float] = mapped_column(Float)
    schema_json: Mapped[dict] = mapped_column(JSON)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan", order_by="Submission.id")


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"))
    file_name: Mapped[str] = mapped_column(String(300))
    file_path: Mapped[str] = mapped_column(String(500))
    student_name: Mapped[str] = mapped_column(String(200), default="")
    roll_no: Mapped[str] = mapped_column(String(50), default="")
    # pending -> processing -> evaluated -> finalized   (or error)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error: Mapped[str] = mapped_column(Text, default="")
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    total_suggested: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    manual_marks: Mapped[float] = mapped_column(Float, default=0.0)
    n_flagged: Mapped[int] = mapped_column(Integer, default=0)
    total_final: Mapped[float | None] = mapped_column(Float, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    exam: Mapped[Exam] = relationship(back_populates="submissions")
    answers: Mapped[list["AnswerEval"]] = relationship(
        back_populates="submission", cascade="all, delete-orphan", order_by="AnswerEval.position")


class AnswerEval(Base):
    __tablename__ = "answer_evals"

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"))
    position: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(20))
    qtype: Mapped[str] = mapped_column(String(20))
    max_marks: Mapped[float] = mapped_column(Float)
    answer_text: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    method: Mapped[str] = mapped_column(String(20))
    suggested: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline: Mapped[float | None] = mapped_column(Float, nullable=True)
    dnn_marks: Mapped[float | None] = mapped_column(Float, nullable=True)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    counted: Mapped[bool] = mapped_column(Boolean, default=True)
    points: Mapped[list] = mapped_column(JSON, default=list)   # per-point status + evidence
    flags: Mapped[list] = mapped_column(JSON, default=list)
    final_marks: Mapped[float | None] = mapped_column(Float, nullable=True)  # teacher's decision
    teacher_comment: Mapped[str] = mapped_column(Text, default="")

    submission: Mapped[Submission] = relationship(back_populates="answers")