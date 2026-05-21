"""Agentic time-series analysis loop built with LangGraph.

Graph topology
--------------
    perceive → reason → evaluate ──high-conf──→ critique → END
                                 └─low-conf──→ act ──────→ perceive  (max 4 iters)
                                 └─escalate──→ END

Nodes
-----
perceive  : render current TimeSeriesWindow → PlotArtifact
reason    : VLMReasoner.analyze() → ReasoningOutput
evaluate  : gate on confidence_threshold; choose next node
act       : execute one of {expand_window, rerender, add_context, escalate}
critique  : model self-reviews its own output for consistency
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.models.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisTask,
    ReasoningStep,
)
from app.services.ingestion import DataIngestionService, IngestionError, TimeSeriesWindow
from app.services.reasoner import ReasonerError, ReasoningOutput, VLMReasoner
from app.services.renderer import PlotArtifact, PlotRenderer, RenderError, WindowRenderer

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 4

# Domain-specific context hints injected by the add_context tool
_DOMAIN_CONTEXTS: dict[str, str] = {
    "clinical": (
        "Normal HR: 60-100 bpm. SpO2 >= 95% normal. Sudden drops in SpO2 or "
        "HR spikes > 120 bpm should be flagged as high-severity anomalies."
    ),
    "financial": (
        "Look for regime changes, volatility clustering, and mean-reversion breaks. "
        "A > 3-sigma move within a single window is likely an anomaly."
    ),
    "iot": (
        "Sensor drift, stuck readings (constant values), and step-change "
        "discontinuities are common anomaly types in IoT streams."
    ),
    "default": (
        "Apply general statistical reasoning. Deviations beyond ±2.5 standard "
        "deviations may indicate anomalies."
    ),
}

# ── Output models ──────────────────────────────────────────────────────────────


class FinalAnalysis(BaseModel):
    """Structured output returned after the agentic loop completes."""
    final_reasoning: ReasoningOutput
    iterations_taken: int
    tools_used: list[str]
    escalated: bool


# ── AgentState dataclass (user-facing constructor) ─────────────────────────────


@dataclass
class AgentState:
    """User-facing state container; passed to WindowAgent to start a run."""
    windows: list[TimeSeriesWindow]
    current_reasoning: ReasoningOutput | None = None
    tool_calls_made: list[str] = field(default_factory=list)
    iteration: int = 0
    final_output: FinalAnalysis | None = None
    confidence_threshold: float = 0.75


# ── Internal LangGraph state (TypedDict) ───────────────────────────────────────


class _GState(TypedDict):
    windows: list                   # list[TimeSeriesWindow]
    window_idx: int                 # index of currently-active window
    current_artifact: Any           # PlotArtifact | None
    current_reasoning: Any          # ReasoningOutput | None
    tool_calls_made: list           # list[str]
    iteration: int
    final_output: Any               # FinalAnalysis | None
    confidence_threshold: float
    domain_context: str | None
    escalated: bool
    error: str | None
    # backward-compat fields (populated by AnalysisAgent wrapper)
    legacy_request: Any             # AnalysisRequest | None
    legacy_plot_b64: str | None


# ── Helpers ────────────────────────────────────────────────────────────────────


def _current_window(state: _GState) -> TimeSeriesWindow:
    return state["windows"][state["window_idx"]]


def _derivative_window(window: TimeSeriesWindow) -> TimeSeriesWindow:
    """Return a new window whose values are the first-difference of the signal."""
    raw = np.diff(window.raw_data, axis=0)
    norm = np.diff(window.normalized_data, axis=0)
    meta = {**window.metadata, "window_size": raw.shape[0], "derived": "diff"}
    return TimeSeriesWindow(
        raw_data=raw,
        normalized_data=norm,
        metadata=meta,
        domain_hint=window.domain_hint,
    )


# ── Node: perceive ─────────────────────────────────────────────────────────────


def perceive_node(state: _GState) -> dict:
    """Render the active TimeSeriesWindow → PlotArtifact."""
    try:
        window = _current_window(state)
        renderer = WindowRenderer()
        artifact = renderer.render(window)
        logger.info(
            "perceive: rendered window_idx=%d domain=%s size=%dB",
            state["window_idx"],
            window.domain_hint or "default",
            len(artifact.image_bytes),
        )
        return {"current_artifact": artifact, "error": None}
    except RenderError as exc:
        logger.error("perceive: render failed — %s", exc)
        return {"error": str(exc)}


# ── Node: reason ───────────────────────────────────────────────────────────────


async def reason_node(state: _GState) -> dict:
    """Send PlotArtifact to VLM, parse ReasoningOutput."""
    if state.get("error"):
        return {}

    artifact: PlotArtifact = state["current_artifact"]
    domain_context: str | None = state.get("domain_context")
    iteration = state["iteration"]

    try:
        reasoner = VLMReasoner()
        output = await reasoner.analyze(artifact, domain_context=domain_context)
        logger.info(
            "reason: confidence=%.2f trend=%s anomalies=%d iter=%d",
            output.confidence, output.trend, len(output.anomalies), iteration,
        )
        return {"current_reasoning": output}
    except ReasonerError as exc:
        logger.error("reason: VLM call failed — %s", exc)
        return {"error": str(exc)}


# ── Node: evaluate ─────────────────────────────────────────────────────────────


def evaluate_node(state: _GState) -> dict:
    """Pure logic gate — no side effects, just increments iteration counter."""
    return {"iteration": state["iteration"] + 1}


# ── Node: act ─────────────────────────────────────────────────────────────────


def act_node(state: _GState) -> dict:
    """Execute one remediation tool; update windows / context / escalation."""
    reasoning: ReasoningOutput = state["current_reasoning"]
    iteration = state["iteration"]
    tools_used: list[str] = list(state["tool_calls_made"])
    windows: list[TimeSeriesWindow] = list(state["windows"])
    window_idx: int = state["window_idx"]

    # Hard escalation: confidence critically low after first pass
    if reasoning.confidence < 0.25 and iteration >= 2:
        logger.warning("act: escalating — confidence=%.2f after %d iters", reasoning.confidence, iteration)
        tools_used.append("escalate")
        return {
            "tool_calls_made": tools_used,
            "escalated": True,
        }

    # Pick tool based on iteration depth (rotate through strategies)
    already_used = set(tools_used)
    tool: str

    if "expand_window" not in already_used and len(windows) > window_idx + 1:
        tool = "expand_window"
        # Merge current window raw/norm data with the next available window
        curr = windows[window_idx]
        nxt = windows[window_idx + 1]
        merged_raw = np.concatenate([curr.raw_data, nxt.raw_data], axis=0)
        merged_norm = np.concatenate([curr.normalized_data, nxt.normalized_data], axis=0)
        merged_meta = {
            **curr.metadata,
            "window_size": merged_raw.shape[0],
            "merged_from": [window_idx, window_idx + 1],
        }
        expanded = TimeSeriesWindow(
            raw_data=merged_raw,
            normalized_data=merged_norm,
            metadata=merged_meta,
            domain_hint=curr.domain_hint,
        )
        windows.insert(window_idx, expanded)          # prepend merged view
        logger.info("act: expand_window — merged windows %d+%d", window_idx, window_idx + 1)
        updates: dict = {
            "windows": windows,
            "tool_calls_made": tools_used + [tool],
        }

    elif "rerender" not in already_used:
        tool = "rerender"
        # Produce a derivative (first-difference) window for pattern clarity
        curr = windows[window_idx]
        if curr.raw_data.shape[0] > 1:
            deriv = _derivative_window(curr)
            windows.insert(window_idx, deriv)
        logger.info("act: rerender — injecting derivative window")
        updates = {
            "windows": windows,
            "tool_calls_made": tools_used + [tool],
        }

    elif "add_context" not in already_used:
        tool = "add_context"
        domain = (windows[window_idx].domain_hint or "default").lower()
        context = _DOMAIN_CONTEXTS.get(domain, _DOMAIN_CONTEXTS["default"])
        logger.info("act: add_context — domain=%s", domain)
        updates = {
            "domain_context": context,
            "tool_calls_made": tools_used + [tool],
        }

    else:
        # All non-escalation tools exhausted
        tool = "escalate"
        logger.warning("act: escalating — all tools exhausted at iter=%d", iteration)
        updates = {
            "tool_calls_made": tools_used + [tool],
            "escalated": True,
        }

    return updates


# ── Node: critique ─────────────────────────────────────────────────────────────

_CRITIQUE_SYSTEM = """\
You are a self-consistency reviewer for time-series analysis.
Given the analysis JSON below, check for logical contradictions and reply ONLY
with a JSON object: {"consistent": true|false, "issues": ["...", ...], "adjusted_confidence": 0.0-1.0}
"""


async def critique_node(state: _GState) -> dict:
    """Self-critique: the VLM reviews its own ReasoningOutput for consistency."""
    if state.get("error"):
        reasoning: ReasoningOutput = state.get("current_reasoning") or _fallback_reasoning()
        return {"final_output": _make_final(state, reasoning)}

    reasoning = state["current_reasoning"]

    try:
        import json

        from openai import AsyncOpenAI

        from app.core.config import get_settings

        cfg = get_settings()
        client = AsyncOpenAI(api_key=cfg.openai_api_key)

        payload = reasoning.model_dump(exclude={"raw_reasoning"})
        resp = await client.chat.completions.create(
            model=cfg.openai_model,
            messages=[
                {"role": "system", "content": _CRITIQUE_SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or "{}"

        # Strip code fences if present
        import re
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        critique = json.loads(m.group(1) if m else raw)

        adj_conf = float(critique.get("adjusted_confidence", reasoning.confidence))
        issues = critique.get("issues", [])
        consistent = critique.get("consistent", True)

        if not consistent and issues:
            logger.info("critique: inconsistencies found — %s", issues)

        # Rebuild reasoning with adjusted confidence
        adjusted = ReasoningOutput(
            description=reasoning.description,
            anomalies=reasoning.anomalies,
            trend=reasoning.trend,
            forecast_direction=reasoning.forecast_direction,
            confidence=adj_conf,
            raw_reasoning=reasoning.raw_reasoning + f"\n\n[Critique] {issues}",
        )
        logger.info("critique: adjusted confidence %.2f → %.2f", reasoning.confidence, adj_conf)
        return {"final_output": _make_final(state, adjusted)}

    except Exception as exc:
        logger.warning("critique: failed (%s) — using original reasoning", exc)
        return {"final_output": _make_final(state, reasoning)}


def _fallback_reasoning() -> ReasoningOutput:
    return ReasoningOutput(
        description="Analysis could not be completed.",
        anomalies=[],
        trend="flat",
        forecast_direction="uncertain",
        confidence=0.0,
        raw_reasoning="",
    )


def _make_final(state: _GState, reasoning: ReasoningOutput) -> FinalAnalysis:
    return FinalAnalysis(
        final_reasoning=reasoning,
        iterations_taken=state["iteration"],
        tools_used=list(state["tool_calls_made"]),
        escalated=bool(state.get("escalated", False)),
    )


# ── Routing ────────────────────────────────────────────────────────────────────


def _route_evaluate(state: _GState) -> str:
    if state.get("error"):
        return "critique"  # critique will build a fallback FinalAnalysis

    if state.get("escalated"):
        return "critique"

    reasoning: ReasoningOutput | None = state.get("current_reasoning")
    confidence = reasoning.confidence if reasoning else 0.0
    iteration = state["iteration"]

    if confidence >= state["confidence_threshold"] or iteration >= _MAX_ITERATIONS:
        return "critique"

    return "act"


def _route_act(state: _GState) -> str:
    if state.get("escalated") or state.get("error"):
        return END
    return "perceive"


# ── Graph assembly ─────────────────────────────────────────────────────────────


def _build_graph() -> StateGraph:
    g = StateGraph(_GState)

    g.add_node("perceive", perceive_node)
    g.add_node("reason", reason_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("act", act_node)
    g.add_node("critique", critique_node)

    g.add_edge(START, "perceive")
    g.add_edge("perceive", "reason")
    g.add_edge("reason", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        _route_evaluate,
        {"critique": "critique", "act": "act"},
    )
    g.add_conditional_edges(
        "act",
        _route_act,
        {"perceive": "perceive", END: END},
    )
    g.add_edge("critique", END)

    return g


_COMPILED_GRAPH = _build_graph().compile()


# ── WindowAgent — new public API ───────────────────────────────────────────────


class WindowAgent:
    """Run the full perceive→reason→evaluate→act→critique loop.

    Parameters
    ----------
    agent_state:        Initialised AgentState (holds windows + config).
    """

    async def run(self, agent_state: AgentState) -> FinalAnalysis:
        """Execute the graph and return a FinalAnalysis.

        Raises RuntimeError if no windows are supplied or the graph errors out.
        """
        if not agent_state.windows:
            raise RuntimeError("AgentState.windows must contain at least one TimeSeriesWindow")

        initial: _GState = {
            "windows": list(agent_state.windows),
            "window_idx": 0,
            "current_artifact": None,
            "current_reasoning": agent_state.current_reasoning,
            "tool_calls_made": list(agent_state.tool_calls_made),
            "iteration": agent_state.iteration,
            "final_output": agent_state.final_output,
            "confidence_threshold": agent_state.confidence_threshold,
            "domain_context": None,
            "escalated": False,
            "error": None,
            "legacy_request": None,
            "legacy_plot_b64": None,
        }

        logger.info(
            "WindowAgent starting: windows=%d confidence_threshold=%.2f",
            len(agent_state.windows),
            agent_state.confidence_threshold,
        )
        final_state: _GState = await _COMPILED_GRAPH.ainvoke(initial)

        output: FinalAnalysis | None = final_state.get("final_output")
        if output is None:
            raise RuntimeError("Graph terminated without producing FinalAnalysis")
        return output


# ── AnalysisAgent — backward-compatible wrapper used by routes.py ──────────────


class AnalysisAgent:
    """Legacy public interface consumed by app/api/routes.py.

    Converts an AnalysisRequest → TimeSeriesWindows → runs WindowAgent →
    converts FinalAnalysis → (AnalysisResult, plot_base64).
    """

    async def run(self, request: AnalysisRequest) -> tuple[AnalysisResult, str | None]:
        """Execute the full pipeline; return (AnalysisResult, plot_base64 | None)."""
        # ── 1. Ingest request series into TimeSeriesWindows ────────────────────
        windows: list[TimeSeriesWindow] = []
        for ts_input in request.series:
            try:
                df = _series_input_to_df(ts_input)
                domain = _task_to_domain(request.task)
                n_pts = len(ts_input.points)
                win_size = min(128, max(2, n_pts))
                svc = DataIngestionService(window_size=win_size)
                w = svc.ingest(df, domain_hint=domain)
                windows.extend(w)
            except (IngestionError, Exception) as exc:
                logger.warning("Ingestion failed for series '%s': %s", ts_input.name, exc)

        if not windows:
            raise RuntimeError("No windows could be produced from the supplied series")

        # ── 2. Run the agentic loop ────────────────────────────────────────────
        agent_state = AgentState(
            windows=windows,
            confidence_threshold=0.65,  # slightly lower for legacy path
        )
        agent = WindowAgent()
        final: FinalAnalysis = await agent.run(agent_state)
        r = final.final_reasoning

        # ── 3. Also produce a base64 plot for the response (last rendered) ─────
        plot_b64: str | None = None
        try:
            renderer = PlotRenderer()
            from app.services.ingestion import DataIngestionService as DIS
            # Render the first window with legacy PlotRenderer for backward compat
            legacy_svc = DIS()
            norm_series = legacy_svc.process(request.series)
            plot_b64 = renderer.render(norm_series, style=request.plot_style, task=request.task)
        except Exception as exc:
            logger.warning("Legacy plot render failed: %s", exc)

        # ── 4. Map FinalAnalysis → AnalysisResult ─────────────────────────────
        result = AnalysisResult(
            series_names=[s.name for s in request.series],
            task=request.task,
            summary=r.description,
            reasoning_steps=[
                ReasoningStep(
                    step=1,
                    observation=r.raw_reasoning[:200] if r.raw_reasoning else "",
                    inference=r.description,
                )
            ],
            anomalies=[a.model_dump() for a in r.anomalies],
            trends=[{"direction": r.trend, "forecast": r.forecast_direction}],
            confidence=r.confidence,
            raw_vlm_response=r.raw_reasoning,
        )
        return result, plot_b64


# ── Utilities ──────────────────────────────────────────────────────────────────


def _series_input_to_df(ts_input: Any) -> pd.DataFrame:
    """Convert a TimeSeriesInput Pydantic model to a DataFrame for ingestion."""
    rows = [
        {"timestamp": pt.timestamp, "value": float(pt.value), "channel_name": ts_input.name}
        for pt in ts_input.points
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _task_to_domain(task: AnalysisTask) -> str:
    return {
        AnalysisTask.anomaly_detection: "clinical",
        AnalysisTask.trend_analysis: "financial",
        AnalysisTask.forecasting: "financial",
        AnalysisTask.pattern_recognition: "iot",
        AnalysisTask.comparative: "default",
        AnalysisTask.general: "default",
    }.get(task, "default")
