"""Unit tests for the TempoVis SDK client."""

from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
import pytest
from tempovis import TempoVis
from tempovis.exceptions import (
    APIKeyError,
    InvalidDataError,
    QuotaExceededError,
)
from tempovis.models import AnalysisResult

# ── Shared fixtures ────────────────────────────────────────────────────────────

_MOCK_RESPONSE = {
    "request_id": "test-uuid",
    "domain": "ops",
    "result": {
        "series_names": ["cpu"],
        "task": "anomaly_detection",
        "summary": "Stable series with one point anomaly at sample 42.",
        "reasoning_steps": [],
        "anomalies": [
            {
                "type": "point",
                "severity": "medium",
                "timestamp_index": 42,
                "description": "Isolated spike above 3σ.",
            }
        ],
        "trends": [{"direction": "flat", "strength": "moderate"}],
        "confidence": 0.82,
        "raw_vlm_response": "",
    },
    "plot_artifact_url": "/plots/test.png",
    "iterations_taken": 2,
}

_CSV_CONTENT = "timestamp,value,channel\n2024-01-01T00:00:00Z,0.5,cpu\n2024-01-01T00:01:00Z,0.6,cpu\n"


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    """Temporary CSV file with minimal valid data."""
    p = tmp_path / "metrics.csv"
    p.write_text(_CSV_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def mock_transport() -> httpx.MockTransport:
    """httpx mock transport that returns _MOCK_RESPONSE for any POST."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_MOCK_RESPONSE)

    return httpx.MockTransport(handler)


@pytest.fixture()
def client(mock_transport: httpx.MockTransport) -> TempoVis:
    """TempoVis client wired to the mock transport."""
    c = TempoVis(base_url="http://test")
    # Monkeypatch httpx.Client to use the mock transport
    import httpx as _httpx

    original_client = _httpx.Client

    class _MockClient(_httpx.Client):
        def __init__(self, **kwargs):
            kwargs.setdefault("transport", mock_transport)
            super().__init__(**kwargs)

    _httpx.Client = _MockClient  # type: ignore[misc]
    yield c
    _httpx.Client = original_client  # type: ignore[misc]


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_analyze_with_filepath(client: TempoVis, csv_file: Path) -> None:
    """analyze() with a CSV path returns a populated AnalysisResult."""
    result = client.analyze(str(csv_file), domain="ops")

    assert isinstance(result, AnalysisResult)
    assert result.confidence == pytest.approx(0.82)
    assert len(result.anomalies) == 1
    assert result.anomalies[0].type == "point"
    assert result.anomalies[0].severity == "medium"
    assert result.anomalies[0].timestamp_index == 42
    assert result.plot_url == "/plots/test.png"
    assert result.iterations_taken == 2


def test_analyze_with_dataframe(client: TempoVis) -> None:
    """analyze() accepts a pandas DataFrame as input."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"],
        "value": [0.5, 0.6],
        "channel": ["cpu", "cpu"],
    })
    result = client.analyze(df, domain="ops")

    assert isinstance(result, AnalysisResult)
    assert result.confidence == pytest.approx(0.82)
    assert result.trend == "flat"


def test_analyze_dataframe_missing_value_column(client: TempoVis) -> None:
    """analyze() raises InvalidDataError when DataFrame has no 'value' column."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"timestamp": ["2024-01-01"], "metric": [1.0]})

    with pytest.raises(InvalidDataError, match="'value' column"):
        client.analyze(df)


def test_analyze_filepath_not_found(client: TempoVis, tmp_path: Path) -> None:
    """analyze() raises InvalidDataError for a missing file."""
    with pytest.raises(InvalidDataError, match="File not found"):
        client.analyze(str(tmp_path / "nonexistent.csv"))


def test_api_key_error_on_401() -> None:
    """analyze() raises APIKeyError on HTTP 401."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Unauthorized"})

    transport = httpx.MockTransport(handler)
    import httpx as _httpx

    original = _httpx.Client

    class _MockClient(_httpx.Client):
        def __init__(self, **kwargs):
            kwargs.setdefault("transport", transport)
            super().__init__(**kwargs)

    _httpx.Client = _MockClient  # type: ignore[misc]
    try:
        client = TempoVis(base_url="http://test")
        with pytest.raises(APIKeyError, match="Invalid or missing API key"):
            with tempfile.NamedTemporaryFile(
                suffix=".csv", mode="w", delete=False, encoding="utf-8"
            ) as f:
                f.write(_CSV_CONTENT)
                fname = f.name
            client.analyze(fname)
    finally:
        _httpx.Client = original  # type: ignore[misc]


def test_quota_exceeded_on_429() -> None:
    """analyze() raises QuotaExceededError on HTTP 429."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "Too many requests"})

    transport = httpx.MockTransport(handler)
    import httpx as _httpx

    original = _httpx.Client

    class _MockClient(_httpx.Client):
        def __init__(self, **kwargs):
            kwargs.setdefault("transport", transport)
            super().__init__(**kwargs)

    _httpx.Client = _MockClient  # type: ignore[misc]
    try:
        client = TempoVis(base_url="http://test")
        with pytest.raises(QuotaExceededError, match="Daily API call limit"):
            with tempfile.NamedTemporaryFile(
                suffix=".csv", mode="w", delete=False, encoding="utf-8"
            ) as f:
                f.write(_CSV_CONTENT)
                fname = f.name
            client.analyze(fname)
    finally:
        _httpx.Client = original  # type: ignore[misc]
