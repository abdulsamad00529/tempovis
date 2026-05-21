"""Unit tests for Pydantic schemas — validation edge cases."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    AnalysisRequest,
    AnalysisTask,
    PlotStyle,
    TimeSeriesInput,
    TimeSeriesPoint,
)


def _pt(ts: str, val: float) -> TimeSeriesPoint:
    return TimeSeriesPoint(timestamp=datetime.fromisoformat(ts), value=val)


class TestTimeSeriesInput:
    def test_valid_input(self):
        ts = TimeSeriesInput(
            name="test",
            points=[_pt("2024-01-01T00:00:00+00:00", 1.0),
                    _pt("2024-01-01T01:00:00+00:00", 2.0)],
        )
        assert ts.name == "test"

    def test_points_sorted_automatically(self):
        ts = TimeSeriesInput(
            name="x",
            points=[
                _pt("2024-01-01T02:00:00+00:00", 3.0),
                _pt("2024-01-01T00:00:00+00:00", 1.0),
                _pt("2024-01-01T01:00:00+00:00", 2.0),
            ],
        )
        timestamps = [p.timestamp for p in ts.points]
        assert timestamps == sorted(timestamps)

    def test_too_few_points_raises(self):
        with pytest.raises(ValidationError):
            TimeSeriesInput(
                name="bad",
                points=[_pt("2024-01-01T00:00:00+00:00", 1.0)],
            )

    def test_name_too_long_raises(self):
        with pytest.raises(ValidationError):
            TimeSeriesInput(name="x" * 200, points=[
                _pt("2024-01-01T00:00:00+00:00", 1.0),
                _pt("2024-01-01T01:00:00+00:00", 2.0),
            ])


class TestAnalysisRequest:
    def _two_series(self) -> list[TimeSeriesInput]:
        return [
            TimeSeriesInput(name="a", points=[
                _pt("2024-01-01T00:00:00+00:00", 1.0),
                _pt("2024-01-01T01:00:00+00:00", 2.0),
            ]),
            TimeSeriesInput(name="b", points=[
                _pt("2024-01-01T00:00:00+00:00", 3.0),
                _pt("2024-01-01T01:00:00+00:00", 4.0),
            ]),
        ]

    def test_valid_request(self):
        req = AnalysisRequest(series=self._two_series(), task=AnalysisTask.general)
        assert req.task == AnalysisTask.general

    def test_comparative_requires_two_series(self):
        single = [TimeSeriesInput(name="x", points=[
            _pt("2024-01-01T00:00:00+00:00", 1.0),
            _pt("2024-01-01T01:00:00+00:00", 2.0),
        ])]
        with pytest.raises(ValidationError, match="Comparative"):
            AnalysisRequest(series=single, task=AnalysisTask.comparative)

    def test_comparative_with_two_series_ok(self):
        req = AnalysisRequest(series=self._two_series(), task=AnalysisTask.comparative)
        assert req.task == AnalysisTask.comparative

    def test_defaults(self):
        req = AnalysisRequest(series=self._two_series())
        assert req.task == AnalysisTask.general
        assert req.plot_style == PlotStyle.line
        assert req.chain_of_thought is True
