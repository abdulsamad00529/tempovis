"""Tests for FewShotLibrary and cosine similarity helper in reasoner.py."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import get_settings
from app.services.reasoner import (
    FewShotExample,
    FewShotLibrary,
    ReasoningOutput,
    VLMReasoner,
    _artifact_to_query,
    _cosine_sim,
    _example_from_record,
    _example_to_record,
)
from app.services.renderer import PlotArtifact

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def mock_openai_client():
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock()
    return client


def _make_artifact(domain: str = "default", channels: list[str] | None = None) -> PlotArtifact:
    raw = b"fake-png"
    return PlotArtifact(
        image_bytes=raw,
        base64_string=base64.b64encode(raw).decode(),
        metadata={
            "domain": domain,
            "n_channels": len(channels or ["ch0"]),
            "channels": channels or ["ch0"],
            "window_size": 128,
            "normalization": "zscore",
        },
    )


def _make_output(
    description: str = "flat signal",
    trend: str = "flat",
    confidence: float = 0.7,
) -> ReasoningOutput:
    return ReasoningOutput(
        description=description,
        anomalies=[],
        trend=trend,
        forecast_direction="flat",
        confidence=confidence,
        raw_reasoning="step 1...",
    )


def _make_embedding_response(vector: list[float]) -> MagicMock:
    data_item = MagicMock()
    data_item.embedding = vector
    resp = MagicMock()
    resp.data = [data_item]
    return resp


def _make_library(path: Path, client: MagicMock) -> FewShotLibrary:
    with patch("app.services.reasoner.AsyncOpenAI", return_value=client):
        return FewShotLibrary(path=path, openai_client=client)


# ── _cosine_sim ────────────────────────────────────────────────────────────────


class TestCosineSim:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self):
        assert _cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_known_angle(self):
        # 45-degree angle → cos(45°) ≈ 0.7071
        result = _cosine_sim([1.0, 0.0], [1.0, 1.0])
        assert result == pytest.approx(math.cos(math.pi / 4), abs=1e-6)


# ── _artifact_to_query ─────────────────────────────────────────────────────────


class TestArtifactToQuery:
    def test_contains_domain(self):
        artifact = _make_artifact(domain="clinical")
        q = _artifact_to_query(artifact, domain_context=None)
        assert "domain=clinical" in q

    def test_contains_channels(self):
        artifact = _make_artifact(channels=["heart_rate", "acc_x"])
        q = _artifact_to_query(artifact, domain_context=None)
        assert "heart_rate" in q
        assert "acc_x" in q

    def test_contains_domain_context(self):
        artifact = _make_artifact()
        q = _artifact_to_query(artifact, domain_context="HR 60-100 normal")
        assert "HR 60-100 normal" in q

    def test_omits_context_when_none(self):
        artifact = _make_artifact()
        q = _artifact_to_query(artifact, domain_context=None)
        assert "context=" not in q


# ── FewShotLibrary: add / retrieve ────────────────────────────────────────────


class TestFewShotLibraryAddRetrieve:
    async def test_add_increments_length(self, mock_openai_client, tmp_path):
        mock_openai_client.embeddings.create.return_value = _make_embedding_response(
            [1.0, 0.0, 0.0]
        )
        lib = _make_library(tmp_path / "lib.json", mock_openai_client)
        assert len(lib) == 0

        await lib.add(_make_artifact(), _make_output())
        assert len(lib) == 1

        await lib.add(_make_artifact(), _make_output())
        assert len(lib) == 2

    async def test_retrieve_returns_most_similar(self, mock_openai_client, tmp_path):
        """Example with embedding [1,0,0] should score higher for query [0.9,0.1,0]."""
        embeddings = [
            [1.0, 0.0, 0.0],   # ex0 — most similar to query
            [0.0, 1.0, 0.0],   # ex1 — orthogonal
            [0.0, 0.0, 1.0],   # ex2 — orthogonal
            [0.9, 0.1, 0.0],   # query embedding (4th call)
        ]
        mock_openai_client.embeddings.create.side_effect = [
            _make_embedding_response(e) for e in embeddings
        ]

        lib = _make_library(tmp_path / "lib.json", mock_openai_client)
        await lib.add(_make_artifact(domain="clinical"), _make_output("upward trend", "up"))
        await lib.add(_make_artifact(domain="iot"), _make_output("flat sensor", "flat"))
        await lib.add(_make_artifact(domain="financial"), _make_output("cyclical", "cyclical"))

        results = await lib.retrieve("query text", k=1)
        assert len(results) == 1
        assert results[0].output.trend == "up"   # ex0 is most similar

    async def test_retrieve_k_zero_returns_empty(self, mock_openai_client, tmp_path):
        mock_openai_client.embeddings.create.return_value = _make_embedding_response(
            [1.0, 0.0]
        )
        lib = _make_library(tmp_path / "lib.json", mock_openai_client)
        await lib.add(_make_artifact(), _make_output())

        results = await lib.retrieve("query", k=0)
        assert results == []

    async def test_retrieve_empty_library_returns_empty(self, mock_openai_client, tmp_path):
        lib = _make_library(tmp_path / "lib.json", mock_openai_client)
        results = await lib.retrieve("query", k=3)
        assert results == []

    async def test_retrieve_caps_at_available_examples(self, mock_openai_client, tmp_path):
        emb_seq = [[float(i), 0.0] for i in range(3)] + [[0.5, 0.0]]
        mock_openai_client.embeddings.create.side_effect = [
            _make_embedding_response(e) for e in emb_seq
        ]
        lib = _make_library(tmp_path / "lib.json", mock_openai_client)
        for _ in range(3):
            await lib.add(_make_artifact(), _make_output())

        results = await lib.retrieve("query", k=10)
        assert len(results) == 3


# ── FewShotLibrary: persistence ───────────────────────────────────────────────


class TestFewShotLibraryPersistence:
    async def test_save_creates_json_file(self, mock_openai_client, tmp_path):
        path = tmp_path / "lib.json"
        mock_openai_client.embeddings.create.return_value = _make_embedding_response([1.0, 0.0])
        lib = _make_library(path, mock_openai_client)
        await lib.add(_make_artifact(), _make_output())

        assert path.exists()
        records = json.loads(path.read_text())
        assert len(records) == 1
        assert "image_b64" in records[0]
        assert "embedding" in records[0]

    async def test_load_restores_examples(self, mock_openai_client, tmp_path):
        path = tmp_path / "lib.json"
        emb = [0.1, 0.9]
        mock_openai_client.embeddings.create.return_value = _make_embedding_response(emb)

        lib1 = _make_library(path, mock_openai_client)
        await lib1.add(_make_artifact(domain="iot"), _make_output("IoT sensor flat", "flat"))
        assert len(lib1) == 1

        # New instance loads from the same path
        lib2 = _make_library(path, mock_openai_client)
        assert len(lib2) == 1
        assert lib2._examples[0].domain == "iot"
        assert lib2._examples[0].output.trend == "flat"
        assert lib2._examples[0].embedding == pytest.approx(emb)

    def test_load_nonexistent_path_starts_empty(self, mock_openai_client, tmp_path):
        lib = _make_library(tmp_path / "does_not_exist.json", mock_openai_client)
        assert len(lib) == 0

    def test_load_corrupt_file_starts_empty(self, mock_openai_client, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json!!")
        lib = _make_library(path, mock_openai_client)
        assert len(lib) == 0


# ── Serialisation round-trip ───────────────────────────────────────────────────


class TestSerialisation:
    def test_example_round_trips_through_record(self):
        artifact = _make_artifact(domain="financial", channels=["price", "volume"])
        output = _make_output("price rises steadily", "up", 0.9)
        example = FewShotExample(
            artifact=artifact,
            output=output,
            embedding=[0.1, 0.2, 0.3],
            domain="financial",
        )

        record = _example_to_record(example)
        restored = _example_from_record(record)

        assert restored.domain == "financial"
        assert restored.output.trend == "up"
        assert restored.output.confidence == pytest.approx(0.9)
        assert restored.embedding == pytest.approx([0.1, 0.2, 0.3])
        assert restored.artifact.base64_string == artifact.base64_string
        assert restored.artifact.metadata == artifact.metadata


# ── VLMReasoner integration ────────────────────────────────────────────────────


class TestVLMReasonerFewShot:
    @pytest.fixture()
    def reasoner_and_client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        get_settings.cache_clear()
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_client.embeddings = MagicMock()
        mock_client.embeddings.create = AsyncMock()
        with patch("app.services.reasoner.AsyncOpenAI", return_value=mock_client):
            svc = VLMReasoner()
        return svc, mock_client

    async def test_no_library_sends_single_image(self, reasoner_and_client):
        svc, client = reasoner_and_client
        valid_json = {
            "description": "flat", "anomalies": [], "trend": "flat",
            "forecast_direction": "flat", "confidence": 0.5, "raw_reasoning": "...",
        }
        msg = MagicMock()
        msg.content = f"```json\n{json.dumps(valid_json)}\n```"
        choice = MagicMock()
        choice.message = msg
        completion = MagicMock()
        completion.choices = [choice]
        client.chat.completions.create.return_value = completion

        artifact = _make_artifact()
        result = await svc.analyze(artifact, few_shot_library=None)

        assert result.trend == "flat"
        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        # Without few-shot: exactly one image_url part
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["detail"] == "high"

    async def test_with_library_injects_example_images(
        self, reasoner_and_client, tmp_path
    ):
        svc, client = reasoner_and_client
        valid_json = {
            "description": "up trend", "anomalies": [], "trend": "up",
            "forecast_direction": "up", "confidence": 0.8, "raw_reasoning": "...",
        }
        msg = MagicMock()
        msg.content = f"```json\n{json.dumps(valid_json)}\n```"
        choice = MagicMock()
        choice.message = msg
        completion = MagicMock()
        completion.choices = [choice]
        client.chat.completions.create.return_value = completion

        emb = [1.0, 0.0]
        client.embeddings.create.return_value = _make_embedding_response(emb)

        lib = FewShotLibrary(path=tmp_path / "lib.json", openai_client=client)
        await lib.add(_make_artifact(domain="clinical"), _make_output("clinical upward", "up"))

        artifact = _make_artifact(domain="clinical")
        result = await svc.analyze(artifact, few_shot_library=lib, k=1)
        assert result.trend == "up"

        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        # 1 example (detail=low) + 1 query (detail=high)
        assert len(image_parts) == 2
        assert image_parts[0]["image_url"]["detail"] == "low"
        assert image_parts[-1]["image_url"]["detail"] == "high"
