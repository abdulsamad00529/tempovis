"""Tests for app/services/reasoner.py.

Uses unittest.mock to avoid real OpenAI calls. All async tests run automatically
via asyncio_mode = auto (set in pytest.ini).
"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.core.config import get_settings
from app.services.reasoner import (
    AnomalyDetail,
    ReasonerError,
    ReasoningOutput,
    VLMReasoner,
    VLMResponse,
)
from app.services.renderer import PlotArtifact

# ── Shared test data ───────────────────────────────────────────────────────────

_VALID_JSON = {
    "description": "Single channel with a clear upward trend and one spike at sample 64.",
    "anomalies": [{"type": "point", "severity": "medium", "timestamp_index": 64}],
    "trend": "up",
    "forecast_direction": "up",
    "confidence": 0.82,
    "raw_reasoning": "Step 1: upward trend visible. Step 2: spike at index 64.",
}

_VALID_VLM_RESPONSE = f"""\
I will now analyse this plot step-by-step.

STEP 1 — The channel rises steadily from left to right.
STEP 2 — One isolated spike visible near the midpoint.
STEP 3 — Overall trend is upward; forecast: continuation upward.
STEP 4 — Signal is clean; confidence 0.82.

```json
{json.dumps(_VALID_JSON, indent=2)}
```
"""

_MALFORMED_VLM_RESPONSE = """\
Analysis complete.

```json
{
  "description": "Noisy flat signal",
  "anomalies": [{"type": "point", "severity": "low", "timestamp_index": 32}
  "trend": "flat",
  "forecast_direction": "flat",
  "confidence": 0.55,
  "raw_reasoning": "..."
}
```
"""

_REPAIRED_JSON = {
    "description": "Noisy flat signal.",
    "anomalies": [{"type": "point", "severity": "low", "timestamp_index": 32}],
    "trend": "flat",
    "forecast_direction": "flat",
    "confidence": 0.55,
    "raw_reasoning": "Repaired reasoning.",
}

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure lru_cache does not bleed between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def mock_openai_client():
    """A fully-mocked AsyncOpenAI client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture()
def reasoner(mock_openai_client, monkeypatch):
    """VLMReasoner wired to the mock client, with a fake API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-0000")
    get_settings.cache_clear()
    with patch("app.services.reasoner.AsyncOpenAI", return_value=mock_openai_client):
        svc = VLMReasoner()
    return svc, mock_openai_client


@pytest.fixture()
def sample_artifact():
    """Minimal PlotArtifact with fake image bytes."""
    raw = b"fake-png-data"
    return PlotArtifact(
        image_bytes=raw,
        base64_string=base64.b64encode(raw).decode(),
        metadata={
            "domain": "default",
            "n_channels": 2,
            "channels": ["acc_x", "acc_y"],
            "window_size": 128,
            "normalization": "zscore",
        },
    )


def _make_completion(content: str) -> MagicMock:
    """Build a fake chat completion object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ── ReasoningOutput schema ─────────────────────────────────────────────────────


class TestReasoningOutput:
    def test_valid_construction(self):
        out = ReasoningOutput(**_VALID_JSON)
        assert out.trend == "up"
        assert out.confidence == pytest.approx(0.82)
        assert len(out.anomalies) == 1
        assert out.anomalies[0].type == "point"

    def test_empty_anomalies_allowed(self):
        data = {**_VALID_JSON, "anomalies": []}
        out = ReasoningOutput(**data)
        assert out.anomalies == []

    def test_invalid_trend_rejected(self):
        with pytest.raises(ValidationError):
            ReasoningOutput(**{**_VALID_JSON, "trend": "sideways"})

    def test_invalid_forecast_direction_rejected(self):
        with pytest.raises(ValidationError):
            ReasoningOutput(**{**_VALID_JSON, "forecast_direction": "maybe"})

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ReasoningOutput(**{**_VALID_JSON, "confidence": 1.5})

    def test_negative_confidence_rejected(self):
        with pytest.raises(ValidationError):
            ReasoningOutput(**{**_VALID_JSON, "confidence": -0.1})


class TestAnomalyDetail:
    def test_all_types_accepted(self):
        for atype in ("point", "contextual", "collective"):
            a = AnomalyDetail(type=atype, severity="low", timestamp_index=10)
            assert a.type == atype

    def test_all_severities_accepted(self):
        for sev in ("low", "medium", "high"):
            a = AnomalyDetail(type="point", severity=sev, timestamp_index=0)
            assert a.severity == sev

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            AnomalyDetail(type="global", severity="low", timestamp_index=0)

    def test_negative_index_rejected(self):
        with pytest.raises(ValidationError):
            AnomalyDetail(type="point", severity="low", timestamp_index=-1)


# ── analyze() — happy path ─────────────────────────────────────────────────────


class TestVLMReasonerAnalyze:
    async def test_happy_path_returns_reasoning_output(self, reasoner, sample_artifact):
        svc, client = reasoner
        client.chat.completions.create.return_value = _make_completion(_VALID_VLM_RESPONSE)

        result = await svc.analyze(sample_artifact)

        assert isinstance(result, ReasoningOutput)
        assert result.trend == "up"
        assert result.forecast_direction == "up"
        assert result.confidence == pytest.approx(0.82)
        assert result.anomalies[0].timestamp_index == 64

    async def test_domain_context_included_in_messages(self, reasoner, sample_artifact):
        svc, client = reasoner
        client.chat.completions.create.return_value = _make_completion(_VALID_VLM_RESPONSE)

        await svc.analyze(sample_artifact, domain_context="HR range 60-100 bpm")

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_text = messages[1]["content"][0]["text"]
        assert "HR range 60-100 bpm" in user_text

    async def test_image_url_contains_base64(self, reasoner, sample_artifact):
        svc, client = reasoner
        client.chat.completions.create.return_value = _make_completion(_VALID_VLM_RESPONSE)

        await svc.analyze(sample_artifact)

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        image_part = messages[1]["content"][1]
        assert image_part["type"] == "image_url"
        assert sample_artifact.base64_string in image_part["image_url"]["url"]

    async def test_metadata_fields_in_user_message(self, reasoner, sample_artifact):
        svc, client = reasoner
        client.chat.completions.create.return_value = _make_completion(_VALID_VLM_RESPONSE)

        await svc.analyze(sample_artifact)

        call_args = client.chat.completions.create.call_args
        user_text = call_args.kwargs["messages"][1]["content"][0]["text"]
        assert "acc_x" in user_text
        assert "128" in user_text


# ── JSON repair fallback ───────────────────────────────────────────────────────


class TestJsonRepairFallback:
    async def test_repair_called_on_malformed_json(self, reasoner, sample_artifact):
        svc, client = reasoner
        repaired_response = json.dumps(_REPAIRED_JSON)

        # First call → malformed; second call (repair) → valid
        client.chat.completions.create.side_effect = [
            _make_completion(_MALFORMED_VLM_RESPONSE),
            _make_completion(repaired_response),
        ]

        result = await svc.analyze(sample_artifact)

        assert isinstance(result, ReasoningOutput)
        assert result.trend == "flat"
        assert result.confidence == pytest.approx(0.55)
        assert client.chat.completions.create.call_count == 2

    async def test_raises_when_repair_also_fails(self, reasoner, sample_artifact):
        svc, client = reasoner
        still_bad = '{"trend": "oops", "confidence": 99}'

        client.chat.completions.create.side_effect = [
            _make_completion(_MALFORMED_VLM_RESPONSE),
            _make_completion(still_bad),
        ]

        with pytest.raises(ReasonerError, match="JSON repair also failed"):
            await svc.analyze(sample_artifact)

    async def test_api_error_propagated_as_reasoner_error(self, reasoner, sample_artifact):
        svc, client = reasoner
        from openai import APIConnectionError

        client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

        with pytest.raises(ReasonerError, match="VLM API call failed"):
            await svc.analyze(sample_artifact)


# ── _extract_json_block ────────────────────────────────────────────────────────


class TestExtractJsonBlock:
    def test_extracts_from_fenced_block(self):
        text = 'Some text.\n```json\n{"a": 1}\n```\nMore text.'
        result = VLMReasoner._extract_json_block(text)
        assert result == '{"a": 1}'

    def test_returns_last_fence_when_multiple(self):
        text = '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```'
        result = VLMReasoner._extract_json_block(text)
        assert result == '{"b": 2}'

    def test_falls_back_to_bare_braces(self):
        text = 'No fence here. {"x": 42} some trailing text.'
        result = VLMReasoner._extract_json_block(text)
        assert json.loads(result) == {"x": 42}

    def test_raises_when_no_json_found(self):
        with pytest.raises(ValueError, match="No JSON block found"):
            VLMReasoner._extract_json_block("No JSON anywhere in this string.")

    def test_case_insensitive_fence(self):
        text = "```JSON\n{\"k\": true}\n```"
        result = VLMReasoner._extract_json_block(text)
        assert result == '{"k": true}'


# ── Legacy reason() API ────────────────────────────────────────────────────────


class TestLegacyReasonApi:
    async def test_returns_vlm_response(self, reasoner):
        svc, client = reasoner
        legacy_json = {
            "summary": "Stable signal with no anomalies.",
            "confidence": 0.9,
            "reasoning_steps": [
                {"step": 1, "observation": "Flat line", "inference": "Stable"}
            ],
            "anomalies": [],
            "trends": [{"direction": "flat", "strength": "strong",
                        "period": None, "description": "No drift"}],
        }
        response_text = f"Reasoning...\n```json\n{json.dumps(legacy_json)}\n```"
        client.chat.completions.create.return_value = _make_completion(response_text)

        from app.models.schemas import AnalysisTask
        result = await svc.reason(
            plot_base64="abc123",
            task=AnalysisTask.general,
            series_names=["sensor_a"],
        )

        assert isinstance(result, VLMResponse)
        assert result.result.confidence == pytest.approx(0.9)
        assert result.result.summary == "Stable signal with no anomalies."
        assert len(result.result.reasoning_steps) == 1

    async def test_raises_reasoner_error_on_bad_json(self, reasoner):
        svc, client = reasoner
        client.chat.completions.create.return_value = _make_completion(
            "No JSON here, just text."
        )

        from app.models.schemas import AnalysisTask
        with pytest.raises(ReasonerError, match="Failed to parse"):
            await svc.reason(
                plot_base64="abc123",
                task=AnalysisTask.general,
                series_names=["sensor_a"],
            )
