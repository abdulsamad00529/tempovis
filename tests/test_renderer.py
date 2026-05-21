"""Unit tests for the plot rendering engine."""

from __future__ import annotations

import base64

import pytest

from app.models.schemas import AnalysisTask, PlotStyle
from app.services.ingestion import DataIngestionService
from app.services.renderer import PlotRenderer, RenderError


@pytest.fixture
def renderer():
    return PlotRenderer()


@pytest.fixture
def normalized_single(sample_series):
    svc = DataIngestionService()
    return svc.process([sample_series])


@pytest.fixture
def normalized_two(two_series_request):
    svc = DataIngestionService()
    return svc.process(two_series_request.series)


class TestPlotRenderer:
    def test_render_returns_string(self, renderer, normalized_single):
        b64 = renderer.render(normalized_single)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_render_valid_base64(self, renderer, normalized_single):
        b64 = renderer.render(normalized_single)
        raw = base64.b64decode(b64)
        # PNG magic bytes
        assert raw[:4] == b"\x89PNG"

    @pytest.mark.parametrize("style", [PlotStyle.line, PlotStyle.area, PlotStyle.multi_panel])
    def test_all_styles(self, renderer, normalized_single, style):
        b64 = renderer.render(normalized_single, style=style)
        assert len(b64) > 100

    def test_heatmap_style(self, renderer, normalized_single):
        b64 = renderer.render(normalized_single, style=PlotStyle.heatmap)
        assert base64.b64decode(b64)[:4] == b"\x89PNG"

    def test_multi_series_line(self, renderer, normalized_two):
        b64 = renderer.render(normalized_two, style=PlotStyle.line)
        assert len(b64) > 100

    def test_anomaly_markers(self, renderer, normalized_single):
        b64 = renderer.render(
            normalized_single,
            style=PlotStyle.line,
            task=AnalysisTask.anomaly_detection,
        )
        assert len(b64) > 100

    def test_trend_line(self, renderer, normalized_single):
        b64 = renderer.render(
            normalized_single,
            style=PlotStyle.line,
            task=AnalysisTask.trend_analysis,
        )
        assert len(b64) > 100

    def test_custom_title(self, renderer, normalized_single):
        b64 = renderer.render(normalized_single, title="My Custom Title")
        assert len(b64) > 100

    def test_base64_to_bytes_roundtrip(self, renderer, normalized_single):
        b64 = renderer.render(normalized_single)
        raw = PlotRenderer.base64_to_bytes(b64)
        assert raw[:4] == b"\x89PNG"
