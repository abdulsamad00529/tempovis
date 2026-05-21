"""Pydantic request/response schemas for the TempoVis API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class AnalysisTask(str, Enum):
    anomaly_detection = "anomaly_detection"
    trend_analysis = "trend_analysis"
    forecasting = "forecasting"
    pattern_recognition = "pattern_recognition"
    comparative = "comparative"
    general = "general"


class PlotStyle(str, Enum):
    line = "line"
    area = "area"
    candlestick = "candlestick"
    heatmap = "heatmap"
    multi_panel = "multi_panel"


class SeriesFrequency(str, Enum):
    secondly = "S"
    minutely = "T"
    hourly = "H"
    daily = "D"
    weekly = "W"
    monthly = "M"


# ── Inbound ────────────────────────────────────────────────────────────────────

class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class TimeSeriesInput(BaseModel):
    """A named time series with optional metadata."""

    name: str = Field(..., min_length=1, max_length=128)
    points: list[TimeSeriesPoint] = Field(..., min_length=2)
    unit: str | None = Field(None, max_length=32, examples=["°C", "USD", "req/s"])
    frequency: SeriesFrequency | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("points")
    @classmethod
    def _points_sorted(cls, pts: list[TimeSeriesPoint]) -> list[TimeSeriesPoint]:
        return sorted(pts, key=lambda p: p.timestamp)


class AnalysisRequest(BaseModel):
    """Top-level request body for /analyze."""

    series: list[TimeSeriesInput] = Field(..., min_length=1, max_length=8)
    task: AnalysisTask = AnalysisTask.general
    question: str | None = Field(
        None,
        max_length=1024,
        description="Optional free-form question to focus the analysis",
    )
    plot_style: PlotStyle = PlotStyle.line
    chain_of_thought: bool = Field(
        True, description="Include step-by-step reasoning in the response"
    )

    @model_validator(mode="after")
    def _validate_comparative(self) -> "AnalysisRequest":
        if self.task == AnalysisTask.comparative and len(self.series) < 2:
            raise ValueError("Comparative analysis requires at least 2 series")
        return self


# ── Outbound ───────────────────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    step: int
    observation: str
    inference: str


class AnalysisResult(BaseModel):
    series_names: list[str]
    task: AnalysisTask
    summary: str
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    trends: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_vlm_response: str


class AnalysisResponse(BaseModel):
    request_id: str
    created_at: datetime
    result: AnalysisResult
    plot_base64: str | None = None
    processing_ms: int


# ── Agent ──────────────────────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class AgentState(BaseModel):
    """Typed state bag threaded through the LangGraph agent."""

    request_id: str
    analysis_request: AnalysisRequest
    messages: list[AgentMessage] = Field(default_factory=list)
    plot_base64: str | None = None
    iteration: int = 0
    final_result: AnalysisResult | None = None
    error: str | None = None
