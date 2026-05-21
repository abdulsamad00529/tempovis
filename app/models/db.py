"""SQLAlchemy ORM models for persisting analyses, alerts, and feedback."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, BigInteger
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRecord(Base):
    """Persisted record of every completed analysis."""

    __tablename__ = "analysis_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    task: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    domain: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    series_names: Mapped[list] = mapped_column(JSONB, nullable=False)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning_steps: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list
    )
    anomalies: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    trends: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    raw_vlm_response: Mapped[str] = mapped_column(Text, nullable=False)
    processing_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    plot_stored_key: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )
    # New agentic fields
    iterations_taken: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tools_used: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    use_agent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_analysis_created_task", "created_at", "task"),
        Index("ix_analysis_domain_conf", "domain", "confidence"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisRecord id={self.id} task={self.task}>"


class Alert(Base):
    """An anomaly alert derived from an analysis that exceeded severity threshold."""

    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    domain: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True
    )  # "low" | "medium" | "high"
    anomaly_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "point" | "contextual" | "collective"
    timestamp_index: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    acknowledged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    __table_args__ = (
        Index("ix_alert_domain_severity", "domain", "severity"),
        Index("ix_alert_created", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Alert id={self.id} domain={self.domain} severity={self.severity}>"
        )


class Feedback(Base):
    """User correction / endorsement of a past analysis."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    correction: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Stored when correct=True so it can be replayed into FewShotLibrary
    plot_b64: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    added_to_library: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    __table_args__ = (Index("ix_feedback_analysis", "analysis_id"),)

    def __repr__(self) -> str:
        return (
            f"<Feedback id={self.id} analysis_id={self.analysis_id} "
            f"correct={self.correct}>"
        )


class CostLog(Base):
    """Per-call OpenAI cost record for billing and observability."""

    __tablename__ = "cost_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    analysis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    endpoint: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("ix_cost_logs_created", "created_at"),
        Index("ix_cost_logs_analysis", "analysis_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<CostLog id={self.id} model={self.model} cost=${self.cost_usd:.6f}>"
        )
