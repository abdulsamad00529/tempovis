"""Shared pytest fixtures for TempoVis tests."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

# Prevent Settings from requiring real OPENAI_API_KEY in unit tests
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("POSTGRES_PASSWORD", "testpassword")


from app.models.schemas import AnalysisRequest, AnalysisTask, PlotStyle, TimeSeriesInput, TimeSeriesPoint


def _make_points(n: int = 60, start: datetime | None = None) -> list[TimeSeriesPoint]:
    base = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    import math, random
    random.seed(42)
    return [
        TimeSeriesPoint(
            timestamp=base + timedelta(hours=i),
            value=math.sin(i / 6) * 10 + random.gauss(0, 0.5) + i * 0.02,
        )
        for i in range(n)
    ]


@pytest.fixture
def sample_series() -> TimeSeriesInput:
    return TimeSeriesInput(name="cpu_usage", points=_make_points(), unit="%")


@pytest.fixture
def sample_request(sample_series) -> AnalysisRequest:
    return AnalysisRequest(
        series=[sample_series],
        task=AnalysisTask.anomaly_detection,
        plot_style=PlotStyle.line,
        chain_of_thought=True,
    )


@pytest.fixture
def two_series_request() -> AnalysisRequest:
    s1 = TimeSeriesInput(name="revenue", points=_make_points(90), unit="USD")
    s2 = TimeSeriesInput(
        name="sessions",
        points=_make_points(90, start=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        unit="count",
    )
    return AnalysisRequest(
        series=[s1, s2],
        task=AnalysisTask.comparative,
        plot_style=PlotStyle.line,
    )
