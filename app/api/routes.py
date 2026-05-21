"""FastAPI route definitions for the TempoVis API."""

from __future__ import annotations

import base64
import logging
import time
import uuid

logger = logging.getLogger(__name__)
from datetime import datetime, timezone
from typing import Any  # noqa: F401 — used in BatchJobStatus.results / BatchAnalyzeResponse

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models.db import Alert, AnalysisRecord, Feedback
from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisResult,
    AnalysisTask,
)
from app.services.agent import AnalysisAgent, FinalAnalysis
from app.services.cost_tracker import log_cost
from app.services.storage import get_storage

router = APIRouter(prefix="/api/v1", tags=["analysis"])


# ── Request / response schemas (route-local) ───────────────────────────────────


class SeriesPoint(BaseModel):
    timestamp: datetime
    value: float
    channel: str = "value"


class AgentAnalyzeRequest(BaseModel):
    """Body for POST /analyze."""

    series: list[SeriesPoint] = Field(..., min_length=2)
    domain: str = Field("default", pattern="^(clinical|financial|iot|default)$")
    use_agent: bool = True
    task: AnalysisTask = AnalysisTask.general
    question: str | None = Field(None, max_length=1024)
    plot_style: str = "line"
    chain_of_thought: bool = True


class AgentAnalyzeResponse(BaseModel):
    request_id: str
    created_at: datetime
    domain: str
    use_agent: bool
    final_analysis: FinalAnalysis | None = None
    result: AnalysisResult | None = None
    plot_artifact_url: str | None = None
    processing_ms: int
    cost_usd: float = 0.0


class BatchAnalyzeRequest(BaseModel):
    """Body for POST /analyze/batch — enqueues one Celery task per series."""

    series_list: list[list[SeriesPoint]] = Field(..., min_length=1, max_length=50)
    domain: str = Field("default", pattern="^(clinical|financial|iot|default)$")
    task: AnalysisTask = AnalysisTask.general
    question: str | None = Field(None, max_length=1024)


class BatchAnalyzeResponse(BaseModel):
    job_id: str
    sub_task_ids: dict[str, str]
    count: int
    status: str = "queued"


class BatchJobStatus(BaseModel):
    job_id: str
    sub_task_ids: dict[str, str]
    results: dict[str, Any]
    ready_count: int
    total_count: int
    status: str


class AlertOut(BaseModel):
    id: str
    created_at: datetime
    analysis_id: str
    domain: str
    severity: str
    anomaly_type: str
    timestamp_index: int
    description: str
    confidence: float
    acknowledged: bool


class AlertsPage(BaseModel):
    items: list[AlertOut]
    total: int
    limit: int
    offset: int


class DetailedHealth(BaseModel):
    status: str
    openai_api: str
    database: str
    memory_mb: float
    uptime_s: float | None = None


class FeedbackRequest(BaseModel):
    analysis_id: str
    correct: bool
    correction: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    added_to_library: bool
    message: str


# ── Server start time ──────────────────────────────────────────────────────────

_START_TIME = time.monotonic()


# ── POST /analyze ──────────────────────────────────────────────────────────────


@router.post(
    "/analyze",
    response_model=AgentAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run agentic multimodal time series analysis",
)
async def analyze(
    request: Request,
    body: AgentAnalyzeRequest,
    db: AsyncSession = Depends(get_db),
) -> AgentAnalyzeResponse:
    """Render the series as a plot image, run the LangGraph agentic loop,
    return structured FinalAnalysis with chain-of-thought reasoning,
    and upload the plot artifact to configured storage."""
    cfg = get_settings()
    request_id = str(uuid.uuid4())
    start_ms = time.monotonic_ns() // 1_000_000

    from app.models.schemas import (  # noqa: PLC0415
        PlotStyle,
        TimeSeriesInput,
        TimeSeriesPoint,
    )

    ts_input = TimeSeriesInput(
        name=body.domain,
        points=[
            TimeSeriesPoint(timestamp=pt.timestamp, value=pt.value)
            for pt in body.series
        ],
    )
    legacy_req = AnalysisRequest(
        series=[ts_input],
        task=body.task,
        question=body.question,
        plot_style=PlotStyle(body.plot_style)
        if body.plot_style in PlotStyle.__members__
        else PlotStyle.line,
        chain_of_thought=body.chain_of_thought,
    )

    agent = AnalysisAgent()
    try:
        result, plot_b64 = await agent.run(legacy_req)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

    # ── Upload plot artifact ───────────────────────────────────────────────────
    plot_url: str | None = None
    if plot_b64:
        try:
            storage = get_storage()
            plot_bytes = base64.b64decode(plot_b64)
            plot_key = f"{request_id}.png"
            plot_url = await storage.upload(plot_key, plot_bytes)
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning("Plot upload failed: %s", exc)
            plot_url = None

    # ── Estimate & log OpenAI cost ─────────────────────────────────────────────
    # Token counts come from result.raw_vlm_response if the agent stored them,
    # else we estimate from character length (~4 chars/token).
    raw = result.raw_vlm_response or ""
    est_in = max(1, len(raw) // 4)
    est_out = max(1, len(result.summary) // 4)
    cost_usd = await log_cost(
        db,
        model=cfg.openai_model,
        input_tokens=est_in,
        output_tokens=est_out,
        endpoint="/api/v1/analyze",
        analysis_id=uuid.UUID(request_id),
    )
    # Propagate cost to middleware for access log
    request.state.token_cost_usd = cost_usd
    request.state.input_tokens = est_in
    request.state.output_tokens = est_out

    # ── Persist analysis record ───────────────────────────────────────────────
    try:
        record = AnalysisRecord(
            id=uuid.UUID(request_id),
            task=body.task.value,
            domain=body.domain,
            series_names=[pt.channel for pt in body.series[:8]],
            question=body.question,
            summary=result.summary,
            confidence=result.confidence,
            reasoning_steps=[rs.model_dump() for rs in result.reasoning_steps],
            anomalies=result.anomalies,
            trends=result.trends,
            raw_vlm_response=result.raw_vlm_response,
            processing_ms=elapsed_ms,
            use_agent=body.use_agent,
            iterations_taken=0,
            tools_used=[],
            escalated=False,
        )
        db.add(record)
        await db.flush()

        for anomaly in result.anomalies:
            severity = anomaly.get("severity", "low")
            if severity in ("medium", "high"):
                db.add(Alert(
                    analysis_id=uuid.UUID(request_id),
                    domain=body.domain,
                    severity=severity,
                    anomaly_type=anomaly.get("type", "point"),
                    timestamp_index=int(anomaly.get("timestamp_index", 0)),
                    description=result.summary[:512],
                    confidence=result.confidence,
                ))
        await db.flush()
    except Exception:
        await db.rollback()

    return AgentAnalyzeResponse(
        request_id=request_id,
        created_at=datetime.now(timezone.utc),
        domain=body.domain,
        use_agent=body.use_agent,
        result=result,
        plot_artifact_url=plot_url,
        processing_ms=elapsed_ms,
        cost_usd=cost_usd,
    )


# ── POST /analyze/batch ────────────────────────────────────────────────────────


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue async batch analysis via Celery",
)
async def analyze_batch(
    request: Request,
    body: BatchAnalyzeRequest,
) -> BatchAnalyzeResponse:
    """Fan out each series to a Celery worker. Returns a job_id to poll."""
    from app.tasks.analyze_task import batch_analyze

    batch_payload = [
        [{"timestamp": pt.timestamp.isoformat(), "value": pt.value, "channel": pt.channel}
         for pt in series]
        for series in body.series_list
    ]

    job = batch_analyze.apply_async(
        kwargs={
            "batch": batch_payload,
            "domain": body.domain,
            "task_type": body.task.value,
            "question": body.question,
        }
    )

    # The parent batch_analyze task fans out and returns sub_task_ids
    # We use the parent task id as the job_id
    return BatchAnalyzeResponse(
        job_id=job.id,
        sub_task_ids={},  # populated when job completes
        count=len(body.series_list),
    )


@router.get(
    "/analyze/batch/{job_id}",
    response_model=BatchJobStatus,
    summary="Poll async batch job status",
)
async def get_batch_status(job_id: str) -> BatchJobStatus:
    """Return current status + available results for a batch job."""
    from celery.result import AsyncResult

    from app.tasks.analyze_task import celery_app

    parent = AsyncResult(job_id, app=celery_app)

    if parent.state in ("PENDING", "STARTED", "RETRY"):
        return BatchJobStatus(
            job_id=job_id,
            sub_task_ids={},
            results={},
            ready_count=0,
            total_count=0,
            status=parent.state.lower(),
        )

    if parent.state == "FAILURE":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch job failed: {parent.result}",
        )

    batch_result: dict = parent.result or {}
    sub_ids: dict[str, str] = batch_result.get("sub_task_ids", {})

    results: dict[str, Any] = {}
    ready = 0
    for idx, tid in sub_ids.items():
        sub = AsyncResult(tid, app=celery_app)
        if sub.ready():
            results[idx] = sub.result
            ready += 1
        else:
            results[idx] = {"status": sub.state.lower()}

    overall = "done" if ready == len(sub_ids) else "running"
    return BatchJobStatus(
        job_id=job_id,
        sub_task_ids=sub_ids,
        results=results,
        ready_count=ready,
        total_count=len(sub_ids),
        status=overall,
    )


# ── GET /alerts ────────────────────────────────────────────────────────────────


@router.get(
    "/alerts",
    response_model=AlertsPage,
    summary="Paginated list of anomaly alerts",
)
async def list_alerts(
    domain: str | None = Query(None, description="Filter by domain"),
    severity_min: str = Query(
        "low",
        description="Minimum severity level",
        pattern="^(low|medium|high)$",
    ),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> AlertsPage:
    """Return a paginated list of anomaly alerts, filtered by domain and minimum severity."""
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = severity_rank.get(severity_min, 0)
    allowed = [s for s, r in severity_rank.items() if r >= min_rank]

    try:
        stmt = select(Alert)
        if domain:
            stmt = stmt.where(Alert.domain == domain)
        stmt = stmt.where(Alert.severity.in_(allowed))
        stmt = stmt.order_by(Alert.created_at.desc())

        count_q = select(Alert.id).where(Alert.severity.in_(allowed))
        if domain:
            count_q = count_q.where(Alert.domain == domain)
        total = len((await db.execute(count_q)).fetchall())

        rows = (await db.execute(stmt.limit(limit).offset(offset))).scalars().all()
        items = [
            AlertOut(
                id=str(r.id),
                created_at=r.created_at,
                analysis_id=str(r.analysis_id),
                domain=r.domain,
                severity=r.severity,
                anomaly_type=r.anomaly_type,
                timestamp_index=r.timestamp_index,
                description=r.description,
                confidence=r.confidence,
                acknowledged=r.acknowledged,
            )
            for r in rows
        ]
        return AlertsPage(items=items, total=total, limit=limit, offset=offset)
    except Exception as exc:
        logger.warning("Alerts DB query failed, returning empty results: %s", exc)
        return AlertsPage(items=[], total=0, limit=limit, offset=offset)


# ── Health ─────────────────────────────────────────────────────────────────────


@router.get("/health", tags=["ops"], summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "tempovis"}


@router.get(
    "/health/detailed",
    response_model=DetailedHealth,
    tags=["ops"],
    summary="Detailed health — OpenAI, DB, memory",
)
async def health_detailed(db: AsyncSession = Depends(get_db)) -> DetailedHealth:
    import psutil

    openai_status = "unknown"
    db_status = "unknown"

    try:
        from openai import AsyncOpenAI

        cfg = get_settings()
        client = AsyncOpenAI(api_key=cfg.openai_api_key)
        models = await client.models.list()
        openai_status = "ok" if models.data else "no_models"
    except Exception as exc:
        openai_status = f"error: {type(exc).__name__}"

    try:
        await db.execute(select(1))  # type: ignore[arg-type]
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {type(exc).__name__}"

    proc = psutil.Process()
    mem_mb = proc.memory_info().rss / (1024 * 1024)

    overall = "ok" if openai_status == "ok" and db_status == "ok" else "degraded"
    return DetailedHealth(
        status=overall,
        openai_api=openai_status,
        database=db_status,
        memory_mb=round(mem_mb, 1),
        uptime_s=round(time.monotonic() - _START_TIME, 1),
    )


# ── POST /feedback ─────────────────────────────────────────────────────────────


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit correction or endorsement for a past analysis",
)
async def submit_feedback(
    body: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """Record a correction or endorsement for a previously completed analysis."""
    try:
        analysis_uuid = uuid.UUID(body.analysis_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid analysis_id UUID",
        )

    record: AnalysisRecord | None = await db.get(AnalysisRecord, analysis_uuid)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )

    feedback_id = str(uuid.uuid4())
    added_to_library = False

    try:
        fb = Feedback(
            id=uuid.UUID(feedback_id),
            analysis_id=analysis_uuid,
            correct=body.correct,
            correction=body.correction,
            reasoning_snapshot={
                "summary": record.summary,
                "confidence": record.confidence,
                "anomalies": record.anomalies,
                "trends": record.trends,
            },
        )
        db.add(fb)
        await db.flush()
    except Exception:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback",
        )

    if body.correct:
        try:
            added_to_library = await _add_to_few_shot_library(record)
            if added_to_library:
                fb.added_to_library = True
                await db.flush()
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning("FewShot library update failed: %s", exc)

    return FeedbackResponse(
        feedback_id=feedback_id,
        added_to_library=added_to_library,
        message=(
            "Feedback saved and added to few-shot library."
            if added_to_library
            else "Feedback saved."
        ),
    )


async def _add_to_few_shot_library(record: AnalysisRecord) -> bool:
    try:
        import base64
        from pathlib import Path

        from openai import AsyncOpenAI

        from app.services.reasoner import FewShotLibrary, ReasoningOutput
        from app.services.renderer import PlotArtifact

        cfg = get_settings()
        client = AsyncOpenAI(api_key=cfg.openai_api_key)
        lib = FewShotLibrary(
            path=Path("data/few_shot_library.json"),
            openai_client=client,
        )
        output = ReasoningOutput(
            description=record.summary,
            anomalies=[],
            trend=(record.trends[0].get("direction", "flat") if record.trends else "flat"),
            forecast_direction="uncertain",
            confidence=record.confidence,
            raw_reasoning=record.raw_vlm_response or "",
        )
        placeholder_bytes = (
            base64.b64decode(record.plot_stored_key) if record.plot_stored_key else b""
        )
        artifact = PlotArtifact(
            image_bytes=placeholder_bytes,
            base64_string=record.plot_stored_key or "",
            metadata={
                "domain": record.domain or "default",
                "n_channels": 1,
                "channels": record.series_names[:1],
                "window_size": 128,
                "normalization": "zscore",
            },
        )
        await lib.add(artifact, output)
        return True
    except Exception:
        return False


# ── GET /analyses/{id} ─────────────────────────────────────────────────────────


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Retrieve a past analysis by ID",
)
async def get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    """Retrieve a previously stored analysis record by its UUID."""
    try:
        uid = uuid.UUID(analysis_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID"
        )

    record: AnalysisRecord | None = await db.get(AnalysisRecord, uid)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found"
        )

    from app.models.schemas import ReasoningStep

    result = AnalysisResult(
        series_names=record.series_names,
        task=AnalysisTask(record.task),
        summary=record.summary,
        reasoning_steps=[ReasoningStep(**s) for s in record.reasoning_steps],
        anomalies=record.anomalies,
        trends=record.trends,
        confidence=record.confidence,
        raw_vlm_response=record.raw_vlm_response,
    )
    return AnalysisResponse(
        request_id=str(record.id),
        created_at=record.created_at,
        result=result,
        processing_ms=record.processing_ms,
    )
