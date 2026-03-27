from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class ProgressRecord(Base):
    __tablename__ = "progress_records"
    __table_args__ = (UniqueConstraint("user_id", "lesson_id", name="uq_progress_user_lesson"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    lesson_id: Mapped[str] = mapped_column(String(120), index=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SnippetRecord(Base):
    __tablename__ = "snippet_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    language: Mapped[str] = mapped_column(String(40), index=True)
    title: Mapped[str] = mapped_column(String(140))
    code: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CourseRecord(Base):
    __tablename__ = "course_records"
    __table_args__ = (UniqueConstraint("language", "version", name="uq_course_language_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    language: Mapped[str] = mapped_column(String(40), index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    level: Mapped[str] = mapped_column(String(40), default="beginner", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="published", nullable=False)
    model: Mapped[str] = mapped_column(String(120), default="openrouter/auto", nullable=False)
    generated_by_prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    modules: Mapped[list["CourseModuleRecord"]] = relationship(
        "CourseModuleRecord",
        back_populates="course",
        cascade="all, delete-orphan",
    )


class CourseModuleRecord(Base):
    __tablename__ = "course_module_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("course_records.id", ondelete="CASCADE"), index=True)
    module_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(180))
    objective: Mapped[str] = mapped_column(Text)

    course: Mapped[CourseRecord] = relationship("CourseRecord", back_populates="modules")
    lessons: Mapped[list["CourseLessonRecord"]] = relationship(
        "CourseLessonRecord",
        back_populates="module",
        cascade="all, delete-orphan",
    )


class CourseLessonRecord(Base):
    __tablename__ = "course_lesson_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    module_id: Mapped[int] = mapped_column(ForeignKey("course_module_records.id", ondelete="CASCADE"), index=True)
    lesson_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(200))
    explanation: Mapped[str] = mapped_column(Text)
    example_code: Mapped[str] = mapped_column(Text)
    exercise: Mapped[str] = mapped_column(Text)

    module: Mapped[CourseModuleRecord] = relationship("CourseModuleRecord", back_populates="lessons")


class CourseGenerationJobRecord(Base):
    __tablename__ = "course_generation_job_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    language: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    model: Mapped[str] = mapped_column(String(120), default="openrouter/auto", nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UserPreferenceRecord(Base):
    __tablename__ = "user_preference_records"
    __table_args__ = (UniqueConstraint("user_id", name="uq_user_preference_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    theme: Mapped[str] = mapped_column(String(30), default="light", nullable=False)
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
