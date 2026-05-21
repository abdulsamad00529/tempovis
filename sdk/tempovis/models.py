"""Pydantic response models matching the TempoVis API schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Anomaly(BaseModel):
    """A single anomaly identified within a time-series window.

    Attributes
    ----------
    type:             Classification of the anomaly.
    severity:         Impact severity.
    timestamp_index:  0-based sample index within the analysed window.
    description:      Human-readable explanation of why this is anomalous.
    """

    type: Literal["point", "contextual", "collective"]
    severity: Literal["low", "medium", "high"]
    timestamp_index: int = Field(..., ge=0)
    description: str = ""


class AnalysisResult(BaseModel):
    """Structured result returned by :meth:`TempoVis.analyze`.

    Attributes
    ----------
    anomalies:          All anomalies detected in the series (may be empty).
    explanation:        Full chain-of-thought reasoning from the VLM.
    confidence:         Model self-assessed certainty in [0, 1].
    trend:              Dominant trend across the analysed window.
    forecast_direction: Expected direction for the next ~20 % of the window.
    plot_url:           URL or path of the rendered plot artifact (may be None).
    iterations_taken:   Number of agentic loop iterations before completion.
    """

    anomalies: list[Anomaly] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    trend: Literal["up", "down", "flat", "cyclical"] = "flat"
    forecast_direction: Literal["up", "down", "flat", "uncertain"] = "uncertain"
    plot_url: str | None = None
    iterations_taken: int = 1
