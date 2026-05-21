"""initial schema

Revision ID: dc8deae679a4
Revises:
Create Date: 2026-05-21 19:01:29.317886

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dc8deae679a4"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── analysis_records ──────────────────────────────────────────────────────
    op.create_table(
        "analysis_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("task", sa.String(64), nullable=False),
        sa.Column("domain", sa.String(64), nullable=True),
        sa.Column("series_names", postgresql.JSONB(), nullable=False),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "reasoning_steps",
            postgresql.JSONB(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "anomalies", postgresql.JSONB(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "trends", postgresql.JSONB(), nullable=False, server_default="[]"
        ),
        sa.Column("raw_vlm_response", sa.Text(), nullable=False),
        sa.Column("processing_ms", sa.Integer(), nullable=False),
        sa.Column("plot_stored_key", sa.String(256), nullable=True),
        sa.Column(
            "iterations_taken", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "tools_used", postgresql.JSONB(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "escalated", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column(
            "use_agent", sa.Boolean(), nullable=False, server_default="false"
        ),
    )
    op.create_index("ix_analysis_records_task", "analysis_records", ["task"])
    op.create_index(
        "ix_analysis_records_domain", "analysis_records", ["domain"]
    )
    op.create_index(
        "ix_analysis_created_task", "analysis_records", ["created_at", "task"]
    )
    op.create_index(
        "ix_analysis_domain_conf", "analysis_records", ["domain", "confidence"]
    )

    # ── alerts ────────────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("domain", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("anomaly_type", sa.String(32), nullable=False),
        sa.Column("timestamp_index", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "acknowledged", sa.Boolean(), nullable=False, server_default="false"
        ),
    )
    op.create_index("ix_alerts_analysis_id", "alerts", ["analysis_id"])
    op.create_index("ix_alerts_domain", "alerts", ["domain"])
    op.create_index("ix_alerts_severity", "alerts", ["severity"])
    op.create_index(
        "ix_alert_domain_severity", "alerts", ["domain", "severity"]
    )
    op.create_index("ix_alert_created", "alerts", ["created_at"])

    # ── feedback ──────────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("correction", sa.Text(), nullable=True),
        sa.Column("plot_b64", sa.Text(), nullable=True),
        sa.Column("reasoning_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column(
            "added_to_library",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.create_index("ix_feedback_analysis", "feedback", ["analysis_id"])


def downgrade() -> None:
    op.drop_index("ix_feedback_analysis", table_name="feedback")
    op.drop_table("feedback")

    op.drop_index("ix_alert_created", table_name="alerts")
    op.drop_index("ix_alert_domain_severity", table_name="alerts")
    op.drop_index("ix_alerts_severity", table_name="alerts")
    op.drop_index("ix_alerts_domain", table_name="alerts")
    op.drop_index("ix_alerts_analysis_id", table_name="alerts")
    op.drop_table("alerts")

    op.drop_index("ix_analysis_domain_conf", table_name="analysis_records")
    op.drop_index("ix_analysis_created_task", table_name="analysis_records")
    op.drop_index("ix_analysis_records_domain", table_name="analysis_records")
    op.drop_index("ix_analysis_records_task", table_name="analysis_records")
    op.drop_table("analysis_records")
