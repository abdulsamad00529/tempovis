"""Integration tests for FastAPI routes.

Uses httpx.AsyncClient (ASGI transport) — no real DB, OpenAI, or Redis calls.
All external I/O is mocked via dependency_overrides and unittest.mock.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.schemas import (
    AnalysisResult,
    AnalysisTask,
    ReasoningStep,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────


def _make_series(n: int = 10) -> list[dict]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        {"timestamp": (base + timedelta(hours=i)).isoformat(), "value": float(i)}
        for i in range(n)
    ]


def _analyze_payload(**overrides) -> dict:
    payload: dict = {
        "series": _make_series(),
        "domain": "default",
        "use_agent": True,
        "task": "general",
        "plot_style": "line",
        "chain_of_thought": True,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def mock_agent_result() -> AnalysisResult:
    return AnalysisResult(
        series_names=["default"],
        task=AnalysisTask.general,
        summary="Test summary: linear upward trend detected.",
        reasoning_steps=[
            ReasoningStep(
                step=1,
                observation="Values increase steadily.",
                inference="Linear growth.",
            )
        ],
        anomalies=[],
        trends=[
            {
                "direction": "up",
                "strength": "strong",
                "period": None,
                "description": "Linear growth",
            }
        ],
        confidence=0.85,
        raw_vlm_response='{"summary": "linear trend"}',
    )


@pytest.fixture
def mock_db() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))), fetchall=MagicMock(return_value=[])))
    return session


@pytest_asyncio.fixture
async def client(mock_db) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with DB and Redis fully mocked."""
    from app.core.database import get_db

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_db] = _override_db

    # Mock Redis so lifespan doesn't fail without a running Redis
    with patch("app.core.redis.init_redis", AsyncMock()), \
         patch("app.core.redis.close_redis", AsyncMock()), \
         patch("app.core.redis.ping_redis", AsyncMock(return_value=True)), \
         patch("app.core.database.create_tables", AsyncMock()):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()


# ── GET /health ────────────────────────────────────────────────────────────────


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_liveness(self, client: AsyncClient):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_root_health(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "dependencies" in data

    @pytest.mark.asyncio
    async def test_detailed_health(self, client: AsyncClient):
        with patch("app.api.routes.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(
                return_value=MagicMock(data=[MagicMock(id="gpt-4o")])
            )
            MockOAI.return_value = mock_client

            resp = await client.get("/api/v1/health/detailed")

        assert resp.status_code == 200
        data = resp.json()
        assert "openai_api" in data
        assert "database" in data
        assert "memory_mb" in data


# ── POST /analyze ──────────────────────────────────────────────────────────────


class TestAnalyzeEndpoint:
    @pytest.mark.asyncio
    async def test_analyze_success(
        self, client: AsyncClient, mock_agent_result: AnalysisResult
    ):
        with patch("app.api.routes.AnalysisAgent") as MockAgent, \
             patch("app.api.routes.log_cost", AsyncMock(return_value=0.0025)), \
             patch("app.api.routes.get_storage") as mock_storage:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = (mock_agent_result, "fakeb64png==")
            MockAgent.return_value = mock_agent

            storage_inst = AsyncMock()
            storage_inst.upload = AsyncMock(return_value="/plots/test.png")
            mock_storage.return_value = storage_inst

            resp = await client.post("/api/v1/analyze", json=_analyze_payload())

        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert data["result"]["task"] == "general"
        assert data["result"]["confidence"] == pytest.approx(0.85)
        assert data["cost_usd"] == pytest.approx(0.0025)
        assert data["plot_artifact_url"] == "/plots/test.png"

    @pytest.mark.asyncio
    async def test_analyze_no_plot(
        self, client: AsyncClient, mock_agent_result: AnalysisResult
    ):
        """When agent returns no plot, plot_artifact_url should be None."""
        with patch("app.api.routes.AnalysisAgent") as MockAgent, \
             patch("app.api.routes.log_cost", AsyncMock(return_value=0.0)):
            mock_agent = AsyncMock()
            mock_agent.run.return_value = (mock_agent_result, None)
            MockAgent.return_value = mock_agent

            resp = await client.post("/api/v1/analyze", json=_analyze_payload())

        assert resp.status_code == 200
        assert resp.json()["plot_artifact_url"] is None

    @pytest.mark.asyncio
    async def test_analyze_agent_runtime_error(self, client: AsyncClient):
        with patch("app.api.routes.AnalysisAgent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.side_effect = RuntimeError("VLM timeout")
            MockAgent.return_value = mock_agent

            resp = await client.post("/api/v1/analyze", json=_analyze_payload())

        assert resp.status_code == 500
        assert "VLM timeout" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_analyze_too_few_points(self, client: AsyncClient):
        payload = _analyze_payload(series=_make_series(1))
        resp = await client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_invalid_domain(self, client: AsyncClient):
        payload = _analyze_payload(domain="nuclear")
        resp = await client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_missing_body(self, client: AsyncClient):
        resp = await client.post("/api/v1/analyze", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_clinical_domain(
        self, client: AsyncClient, mock_agent_result: AnalysisResult
    ):
        with patch("app.api.routes.AnalysisAgent") as MockAgent, \
             patch("app.api.routes.log_cost", AsyncMock(return_value=0.0)):
            mock_agent = AsyncMock()
            mock_agent.run.return_value = (mock_agent_result, None)
            MockAgent.return_value = mock_agent

            resp = await client.post(
                "/api/v1/analyze", json=_analyze_payload(domain="clinical")
            )

        assert resp.status_code == 200
        assert resp.json()["domain"] == "clinical"


# ── POST /analyze/batch ────────────────────────────────────────────────────────


class TestBatchAnalyzeEndpoint:
    @pytest.mark.asyncio
    async def test_batch_enqueues_job(self, client: AsyncClient):
        fake_task = MagicMock()
        fake_task.id = str(uuid.uuid4())

        with patch("app.api.routes.batch_analyze") as mock_batch:
            mock_batch.apply_async.return_value = fake_task

            payload = {
                "series_list": [_make_series(5), _make_series(5)],
                "domain": "default",
                "task": "general",
            }
            resp = await client.post("/api/v1/analyze/batch", json=payload)

        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"] == fake_task.id
        assert data["count"] == 2
        assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_batch_too_many_series(self, client: AsyncClient):
        payload = {
            "series_list": [_make_series(3)] * 51,  # exceeds max_length=50
            "domain": "default",
            "task": "general",
        }
        resp = await client.post("/api/v1/analyze/batch", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_job_status_pending(self, client: AsyncClient):
        job_id = str(uuid.uuid4())
        with patch("app.api.routes.AsyncResult") as MockResult:
            mock_r = MagicMock()
            mock_r.state = "PENDING"
            MockResult.return_value = mock_r

            resp = await client.get(f"/api/v1/analyze/batch/{job_id}")

        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_batch_job_status_done(self, client: AsyncClient):
        job_id = str(uuid.uuid4())
        sub_id = str(uuid.uuid4())

        with patch("app.api.routes.AsyncResult") as MockResult:
            call_count = 0

            def _side_effect(tid, app=None):
                nonlocal call_count
                call_count += 1
                m = MagicMock()
                if call_count == 1:
                    # parent job
                    m.state = "SUCCESS"
                    m.result = {"sub_task_ids": {"0": sub_id}, "count": 1}
                else:
                    # sub task
                    m.ready.return_value = True
                    m.result = {"status": "done", "domain": "default"}
                return m

            MockResult.side_effect = _side_effect

            resp = await client.get(f"/api/v1/analyze/batch/{job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ready_count"] == 1
        assert data["status"] == "done"


# ── GET /alerts ────────────────────────────────────────────────────────────────


class TestAlertsEndpoint:
    @pytest.mark.asyncio
    async def test_alerts_empty(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_alerts_invalid_severity(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts?severity_min=critical")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_alerts_pagination_params(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts?limit=5&offset=0&severity_min=high")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 5
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_alerts_domain_filter(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts?domain=clinical&severity_min=medium")
        assert resp.status_code == 200


# ── POST /feedback ─────────────────────────────────────────────────────────────


class TestFeedbackEndpoint:
    @pytest.mark.asyncio
    async def test_feedback_analysis_not_found(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/feedback",
            json={
                "analysis_id": str(uuid.uuid4()),
                "correct": True,
            },
        )
        # mock_db.get returns None → 404
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_feedback_invalid_uuid(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/feedback",
            json={"analysis_id": "not-a-uuid", "correct": False},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_feedback_correct_endorsement(self, client: AsyncClient, mock_db):
        from app.models.db import AnalysisRecord

        aid = uuid.uuid4()
        mock_record = MagicMock(spec=AnalysisRecord)
        mock_record.id = aid
        mock_record.summary = "Test"
        mock_record.confidence = 0.9
        mock_record.anomalies = []
        mock_record.trends = []
        mock_record.raw_vlm_response = ""
        mock_record.plot_stored_key = None
        mock_record.series_names = ["ch1"]
        mock_record.domain = "default"
        mock_db.get = AsyncMock(return_value=mock_record)

        with patch("app.api.routes._add_to_few_shot_library", AsyncMock(return_value=True)):
            resp = await client.post(
                "/api/v1/feedback",
                json={"analysis_id": str(aid), "correct": True},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["added_to_library"] is True
        assert "feedback_id" in data

    @pytest.mark.asyncio
    async def test_feedback_incorrect_submission(self, client: AsyncClient, mock_db):
        from app.models.db import AnalysisRecord

        aid = uuid.uuid4()
        mock_record = MagicMock(spec=AnalysisRecord)
        mock_record.id = aid
        mock_record.summary = "Needs correction"
        mock_record.confidence = 0.4
        mock_record.anomalies = []
        mock_record.trends = []
        mock_record.raw_vlm_response = ""
        mock_record.plot_stored_key = None
        mock_record.series_names = []
        mock_record.domain = "iot"
        mock_db.get = AsyncMock(return_value=mock_record)

        resp = await client.post(
            "/api/v1/feedback",
            json={
                "analysis_id": str(aid),
                "correct": False,
                "correction": "The anomaly at t=42 is a sensor glitch, not a real event.",
            },
        )

        assert resp.status_code == 201
        data = resp.json()
        assert data["added_to_library"] is False


# ── GET /analyses/{id} ─────────────────────────────────────────────────────────


class TestGetAnalysis:
    @pytest.mark.asyncio
    async def test_get_analysis_not_found(self, client: AsyncClient):
        resp = await client.get(f"/api/v1/analyses/{uuid.uuid4()}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_analysis_invalid_uuid(self, client: AsyncClient):
        resp = await client.get("/api/v1/analyses/not-a-uuid")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_get_analysis_found(self, client: AsyncClient, mock_db):
        from app.models.db import AnalysisRecord

        aid = uuid.uuid4()
        mock_record = MagicMock(spec=AnalysisRecord)
        mock_record.id = aid
        mock_record.created_at = datetime.now(timezone.utc)
        mock_record.task = "general"
        mock_record.summary = "Trend upward"
        mock_record.confidence = 0.88
        mock_record.series_names = ["cpu"]
        mock_record.reasoning_steps = [
            {"step": 1, "observation": "rising", "inference": "growth"}
        ]
        mock_record.anomalies = []
        mock_record.trends = []
        mock_record.raw_vlm_response = "{}"
        mock_record.processing_ms = 1500
        mock_db.get = AsyncMock(return_value=mock_record)

        resp = await client.get(f"/api/v1/analyses/{aid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == str(aid)
        assert data["result"]["confidence"] == pytest.approx(0.88)


# ── Rate limiting ──────────────────────────────────────────────────────────────


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_header_present(
        self, client: AsyncClient, mock_agent_result: AnalysisResult
    ):
        """X-Process-Time-Ms header injected by logging middleware."""
        with patch("app.api.routes.AnalysisAgent") as MockAgent, \
             patch("app.api.routes.log_cost", AsyncMock(return_value=0.0)):
            mock_agent = AsyncMock()
            mock_agent.run.return_value = (mock_agent_result, None)
            MockAgent.return_value = mock_agent

            resp = await client.post("/api/v1/analyze", json=_analyze_payload())

        assert "x-process-time-ms" in resp.headers
