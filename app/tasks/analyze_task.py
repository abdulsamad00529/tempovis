"""Celery tasks for asynchronous batch time-series analysis."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from celery import Celery, Task
from celery.utils.log import get_task_logger

from app.core.config import get_settings

logger = get_task_logger(__name__)

# ── Celery application ────────────────────────────────────────────────────────


def make_celery() -> Celery:
    cfg = get_settings()
    app = Celery(
        "tempovis",
        broker=cfg.celery_broker_url,
        backend=cfg.celery_result_backend,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        result_expires=86400,  # 24 h
        worker_prefetch_multiplier=1,
        task_acks_late=True,
    )
    return app


celery_app = make_celery()


# ── Helper: run async code inside a Celery (sync) worker ─────────────────────


def _run_async(coro):  # type: ignore[no-untyped-def]
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, coro)
                return fut.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Core analysis task ────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="tempovis.analyze_series",
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=300,
    time_limit=360,
)
def analyze_series(
    self: Task,
    series_payload: list[dict[str, Any]],
    domain: str = "default",
    task_type: str = "general",
    question: str | None = None,
    plot_style: str = "line",
    chain_of_thought: bool = True,
) -> dict[str, Any]:
    """Run AnalysisAgent for a single series payload. Returned as Celery result."""
    try:
        return _run_async(
            _async_analyze(
                series_payload=series_payload,
                domain=domain,
                task_type=task_type,
                question=question,
                plot_style=plot_style,
                chain_of_thought=chain_of_thought,
                task_id=self.request.id or "unknown",
            )
        )
    except Exception as exc:
        logger.exception("analyze_series failed: %s", exc)
        raise self.retry(exc=exc)


async def _async_analyze(
    *,
    series_payload: list[dict[str, Any]],
    domain: str,
    task_type: str,
    question: str | None,
    plot_style: str,
    chain_of_thought: bool,
    task_id: str,
) -> dict[str, Any]:
    from app.models.schemas import (
        AnalysisRequest,
        AnalysisTask,
        PlotStyle,
        TimeSeriesInput,
        TimeSeriesPoint,
    )
    from app.services.agent import AnalysisAgent

    ts_inputs = [
        TimeSeriesInput(
            name=s.get("name", domain),
            points=[
                TimeSeriesPoint(
                    timestamp=datetime.fromisoformat(p["timestamp"]),
                    value=float(p["value"]),
                )
                for p in s["points"]
            ],
            unit=s.get("unit"),
        )
        for s in series_payload
    ]

    req = AnalysisRequest(
        series=ts_inputs,
        task=AnalysisTask(task_type),
        question=question,
        plot_style=PlotStyle(plot_style) if plot_style in PlotStyle.__members__ else PlotStyle.line,
        chain_of_thought=chain_of_thought,
    )

    agent = AnalysisAgent()
    result, plot_b64 = await agent.run(req)

    return {
        "task_id": task_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "result": result.model_dump(mode="json"),
        "plot_base64": plot_b64,
    }


# ── Batch dispatch task ────────────────────────────────────────────────────────


@celery_app.task(name="tempovis.batch_analyze")
def batch_analyze(
    batch: list[dict[str, Any]],
    domain: str = "default",
    task_type: str = "general",
    question: str | None = None,
) -> dict[str, Any]:
    """Fan out a list of series dicts to individual analyze_series tasks.

    Returns a dict mapping index → sub-task-id so the caller can poll each.
    """
    sub_ids: dict[str, str] = {}
    for i, series_item in enumerate(batch):
        payload = series_item if isinstance(series_item, list) else [series_item]
        sub = analyze_series.apply_async(
            kwargs={
                "series_payload": payload,
                "domain": domain,
                "task_type": task_type,
                "question": question,
            }
        )
        sub_ids[str(i)] = sub.id

    return {"sub_task_ids": sub_ids, "count": len(batch)}
