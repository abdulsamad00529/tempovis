"""VLM reasoning service — sends rendered plots to GPT-4o and parses structured output.

Two APIs:

  New API  — accepts PlotArtifact, returns ReasoningOutput (Pydantic):
    VLMReasoner.analyze(PlotArtifact, domain_context?, few_shot_library?, k?) -> ReasoningOutput

  Legacy API — accepts raw base64, returns AnalysisResult (used by agent.py):
    VLMReasoner.reason(plot_base64, task, ...) -> VLMResponse

Few-shot ICL:
    FewShotLibrary stores (PlotArtifact, ReasoningOutput) pairs embedded with
    text-embedding-3-small on the description field.  retrieve(query, k) returns
    the top-k pairs by cosine similarity; they are injected into the prompt as
    in-context examples before the main query image.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.models.schemas import AnalysisResult, AnalysisTask, ReasoningStep
from app.services.renderer import PlotArtifact

logger = logging.getLogger(__name__)

# ── Daily call-limit helpers ───────────────────────────────────────────────────

_COUNTER_PREFIX = "tempovis:api_calls"


def _today_key() -> str:
    return f"{_COUNTER_PREFIX}:{datetime.now(UTC).strftime('%Y-%m-%d')}"


def _seconds_until_midnight() -> int:
    """Seconds from now until the next UTC midnight (used as Redis TTL)."""
    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


async def _check_and_increment() -> None:
    """Increment the daily OpenAI call counter and raise if the limit is exceeded.

    Uses Redis with a TTL that expires at the next UTC midnight so the counter
    resets automatically.  When Redis is unavailable the check is skipped silently
    so a Redis outage never blocks real analysis.
    """
    cfg = get_settings()
    try:
        from app.core.redis import get_redis  # local import to avoid circular deps
        redis = await get_redis()
        key = _today_key()
        count = await redis.incr(key)
        if count == 1:
            # First call today — set TTL so key self-destructs at midnight
            await redis.expire(key, _seconds_until_midnight())
        if count > cfg.max_calls_per_day:
            # Decrement so the counter stays accurate (this call won't be charged)
            await redis.decr(key)
            raise DailyLimitExceeded(
                f"Daily test limit reached ({cfg.max_calls_per_day} calls/day) — "
                "set DRY_RUN=true to continue testing without API calls"
            )
        logger.debug("OpenAI call counter: %d / %d today", count, cfg.max_calls_per_day)
    except DailyLimitExceeded:
        raise
    except Exception as exc:
        logger.warning("Redis call-counter unavailable, skipping limit check: %s", exc)

# Transient OpenAI errors that are safe to retry.
_RETRIABLE = (RateLimitError, APITimeoutError, APIConnectionError)


# ── Output schema (new API) ────────────────────────────────────────────────────

class AnomalyDetail(BaseModel):
    """A single anomaly identified by the VLM in the rendered window."""

    type: Literal["point", "contextual", "collective"]
    severity: Literal["low", "medium", "high"]
    timestamp_index: int = Field(..., ge=0, description="0-based sample index in the window")


class ReasoningOutput(BaseModel):
    """Structured VLM output for the new analyze() API.

    Fields
    ------
    description:        2-4 sentence visual summary of the plot.
    anomalies:          All anomalies found (may be empty).
    trend:              Overall trend across the window.
    forecast_direction: Expected direction for the next ~20 % of the window.
    confidence:         Model's self-assessed certainty [0, 1].
    raw_reasoning:      Verbatim chain-of-thought text produced by the model.
    """

    description: str
    anomalies: list[AnomalyDetail] = Field(default_factory=list)
    trend: Literal["up", "down", "flat", "cyclical"]
    forecast_direction: Literal["up", "down", "flat", "uncertain"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_reasoning: str


# ── Legacy output type (kept for agent.py) ─────────────────────────────────────

@dataclass(frozen=True)
class VLMResponse:
    raw_text: str
    result: AnalysisResult


# ── Exceptions ─────────────────────────────────────────────────────────────────

class ReasonerError(RuntimeError):
    """Raised when VLM call or response parsing fails after all retries."""


class DailyLimitExceeded(ReasonerError):
    """Raised when the Redis daily call counter reaches max_calls_per_day."""


# ── DRY_RUN mock responses ─────────────────────────────────────────────────────

def _mock_reasoning_output() -> ReasoningOutput:
    """Realistic hardcoded ReasoningOutput returned when DRY_RUN=true."""
    return ReasoningOutput(
        description=(
            "The series shows a stable baseline with moderate variance across the window. "
            "A brief collective anomaly is visible around sample 40-45 where all channels "
            "deviate upward simultaneously, suggesting a transient event. "
            "The overall trajectory is flat with a slight downward drift toward the end."
        ),
        anomalies=[
            AnomalyDetail(type="collective", severity="medium", timestamp_index=42),
            AnomalyDetail(type="point", severity="low", timestamp_index=67),
        ],
        trend="flat",
        forecast_direction="down",
        confidence=0.78,
        raw_reasoning=(
            "STEP 1 — VISUAL DESCRIPTION\n"
            "All channels are z-score normalised and oscillate within ±2σ. "
            "Channel 0 exhibits the highest variance; channels 1+ follow a broadly "
            "correlated pattern with a slight lag.\n\n"
            "STEP 2 — ANOMALY IDENTIFICATION\n"
            "Samples 42-45: simultaneous upward excursion across all channels — classified "
            "as collective, medium severity. Sample 67: isolated spike in channel 0 only — "
            "classified as point, low severity.\n\n"
            "STEP 3 — TREND AND FORECAST\n"
            "Window-level trend: flat. The terminal 10 samples show a mild negative slope; "
            "forecast direction: down.\n\n"
            "STEP 4 — CONFIDENCE\n"
            "Series length is adequate; normalisation is clean. Confidence: 0.78. "
            "[DRY_RUN — no real VLM call was made]"
        ),
    )


def _mock_vlm_response(task: AnalysisTask, series_names: list[str]) -> VLMResponse:
    """Realistic hardcoded VLMResponse for the legacy reason() API when DRY_RUN=true."""
    raw = json.dumps({
        "summary": (
            "Dry-run mock: the series is stable with one medium-severity collective "
            "anomaly around sample 42 and a minor point anomaly at sample 67. "
            "Overall trend is flat with a slight downward drift at the end of the window."
        ),
        "confidence": 0.78,
        "reasoning_steps": [
            {
                "step": 1,
                "observation": "All channels normalised; collective deviation at samples 42-45.",
                "inference": "Transient event affecting all channels simultaneously.",
            },
            {
                "step": 2,
                "observation": "Isolated spike at sample 67 in channel 0 only.",
                "inference": "Point anomaly, low severity.",
            },
        ],
        "anomalies": [
            {
                "timestamp": "sample_42",
                "severity": "medium",
                "description": "Collective upward excursion across all channels.",
            },
            {
                "timestamp": "sample_67",
                "severity": "low",
                "description": "Point spike in channel 0.",
            },
        ],
        "trends": [
            {
                "direction": "flat",
                "strength": "moderate",
                "period": None,
                "description": "Stable baseline across the full window.",
            }
        ],
    })
    reasoning_steps = [
        ReasoningStep(step=1, observation="Collective deviation at 42-45.", inference="Transient event."),
        ReasoningStep(step=2, observation="Spike at 67 in channel 0.", inference="Point anomaly."),
    ]
    result = AnalysisResult(
        series_names=series_names,
        task=task,
        summary=(
            "Dry-run mock: stable series, one medium collective anomaly at sample 42, "
            "one low point anomaly at sample 67. Trend: flat."
        ),
        reasoning_steps=reasoning_steps,
        anomalies=[
            {"timestamp": "sample_42", "severity": "medium",
             "description": "Collective upward excursion."},
            {"timestamp": "sample_67", "severity": "low",
             "description": "Point spike in channel 0."},
        ],
        trends=[{"direction": "flat", "strength": "moderate",
                 "period": None, "description": "Stable baseline."}],
        confidence=0.78,
        raw_vlm_response=raw,
    )
    return VLMResponse(raw_text=raw, result=result)


# ── Few-shot ICL ───────────────────────────────────────────────────────────────

@dataclass
class FewShotExample:
    """A stored (plot, analysis) pair with its description embedding."""
    artifact: PlotArtifact
    output: ReasoningOutput
    embedding: list[float]                  # text-embedding-3-small of output.description
    domain: str
    added_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; returns 0.0 on zero-norm."""
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    return float(np.dot(arr_a, arr_b) / denom) if denom > 0 else 0.0


def _artifact_to_query(artifact: PlotArtifact, domain_context: str | None) -> str:
    """Build a query string from artifact metadata for embedding-based retrieval.

    We embed metadata rather than a description because we don't yet have a VLM
    description for the query artifact.  Domain + channel names are a good proxy
    for the kind of visual pattern the VLM will see.
    """
    meta = artifact.metadata
    parts = [
        f"domain={meta.get('domain', 'default')}",
        f"channels={','.join(meta.get('channels', []))}",
        f"n_channels={meta.get('n_channels', 1)}",
        f"window_size={meta.get('window_size', 128)}",
        f"normalization={meta.get('normalization', 'zscore')}",
    ]
    if domain_context:
        parts.append(f"context={domain_context}")
    return " ".join(parts)


class FewShotLibrary:
    """Persisted store of (PlotArtifact, ReasoningOutput) pairs with embedding-based retrieval.

    Embeddings use OpenAI ``text-embedding-3-small`` on the ``description`` field of each
    stored ``ReasoningOutput``.  Retrieval uses cosine similarity so that semantically
    similar visual descriptions rank higher.

    The library is persisted as a JSON file; call ``add()`` and the file is updated
    immediately.  Pass an instance to ``VLMReasoner.analyze(few_shot_library=..., k=N)``
    to inject the top-N examples into the prompt.

    Parameters
    ----------
    path:          Path to the JSON persistence file (created automatically).
    openai_client: Async OpenAI client used for embedding calls.
    """

    _EMBED_MODEL = "text-embedding-3-small"

    def __init__(self, path: str | Path, openai_client: AsyncOpenAI) -> None:
        self._path = Path(path)
        self._client = openai_client
        self._examples: list[FewShotExample] = []
        self._load()

    # ── Public interface ───────────────────────────────────────────────────────

    async def add(self, artifact: PlotArtifact, output: ReasoningOutput) -> None:
        """Embed output.description, store the example, and persist to disk."""
        embedding = await self._embed(output.description)
        self._examples.append(
            FewShotExample(
                artifact=artifact,
                output=output,
                embedding=embedding,
                domain=artifact.metadata.get("domain", "default"),
            )
        )
        self._save()
        logger.info("FewShotLibrary: added example (total=%d)", len(self._examples))

    async def retrieve(self, query: str, k: int = 3) -> list[FewShotExample]:
        """Return the top-k examples most similar to ``query`` by cosine similarity.

        ``query`` is embedded with text-embedding-3-small and compared against each
        stored example's description embedding.
        """
        if not self._examples or k <= 0:
            return []
        query_emb = await self._embed(query)
        scored = [(_cosine_sim(query_emb, ex.embedding), ex) for ex in self._examples]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ex for _, ex in scored[:k]]

    def __len__(self) -> int:
        return len(self._examples)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        records = [_example_to_record(ex) for ex in self._examples]
        self._path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            records = json.loads(self._path.read_text(encoding="utf-8"))
            self._examples = [_example_from_record(r) for r in records]
            logger.info(
                "FewShotLibrary: loaded %d example(s) from %s",
                len(self._examples), self._path,
            )
        except Exception as exc:
            logger.warning(
                "FewShotLibrary: failed to load %s (%s); starting empty", self._path, exc
            )

    # ── Embedding ──────────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._EMBED_MODEL,
            input=text,
        )
        return response.data[0].embedding


def _example_to_record(ex: FewShotExample) -> dict:
    return {
        "image_b64": ex.artifact.base64_string,
        "metadata": ex.artifact.metadata,
        "reasoning_output": ex.output.model_dump(),
        "embedding": ex.embedding,
        "domain": ex.domain,
        "added_at": ex.added_at,
    }


def _example_from_record(record: dict) -> FewShotExample:
    b64 = record["image_b64"]
    artifact = PlotArtifact(
        image_bytes=base64.b64decode(b64),
        base64_string=b64,
        metadata=record["metadata"],
    )
    return FewShotExample(
        artifact=artifact,
        output=ReasoningOutput.model_validate(record["reasoning_output"]),
        embedding=record["embedding"],
        domain=record["domain"],
        added_at=record["added_at"],
    )


# ── Prompt templates ───────────────────────────────────────────────────────────

_ANALYZE_SYSTEM_PROMPT = """\
You are TempoVis — a specialist in multivariate time-series analysis.
You receive a rendered plot image of one or more normalized sensor / financial / clinical channels.

Work through FOUR explicit reasoning steps, then emit a single JSON block.

STEP 1 — VISUAL DESCRIPTION
Describe each channel: amplitude range, shape, dominant features (peaks, troughs, cycles,
discontinuities, noise level). Note cross-channel correlations if visible.

STEP 2 — ANOMALY IDENTIFICATION
Classify every anomaly you can identify into exactly one of:
  point      — isolated spike or dip clearly outside the local range
  contextual — value normal globally but anomalous given local neighborhood
  collective — sustained window that is collectively anomalous (drift, shift, plateau)
Estimate each anomaly's 0-based sample index within the window.

STEP 3 — TREND AND FORECAST
State the overall trend across the full window: up / down / flat / cyclical.
Project the next 20 % of the window forward: forecast_direction must be one of
up / down / flat / uncertain.

STEP 4 — CONFIDENCE
Assign confidence in [0, 1]. Reduce for noisy, short, or ambiguous series.

OUTPUT
After your step-by-step reasoning, output EXACTLY ONE JSON block fenced with ```json ... ```:

```json
{
  "description": "<2-4 sentences summarising the visual pattern>",
  "anomalies": [
    {"type": "point|contextual|collective", "severity": "low|medium|high", "timestamp_index": 0}
  ],
  "trend": "up|down|flat|cyclical",
  "forecast_direction": "up|down|flat|uncertain",
  "confidence": 0.85,
  "raw_reasoning": "<your step-by-step reasoning verbatim>"
}
```

RULES
- No JSON outside the fence.
- Do not add extra keys.
- Escape all inner quotes inside string values.
- anomalies may be [].
"""

_REPAIR_SYSTEM_PROMPT = """\
You are a JSON repair assistant. The previous model response contained malformed JSON.
Extract the intended values and return ONLY a valid, corrected JSON object (no code fence,
no explanation, no trailing text). The schema you must match exactly:

{
  "description": "string",
  "anomalies": [{"type": "point|contextual|collective", "severity": "low|medium|high",
                 "timestamp_index": integer}],
  "trend": "up|down|flat|cyclical",
  "forecast_direction": "up|down|flat|uncertain",
  "confidence": float,
  "raw_reasoning": "string"
}
"""

_LEGACY_SYSTEM_PROMPT = """\
You are TempoVis, an expert time-series analyst with deep expertise in anomaly detection,
trend analysis, forecasting, and pattern recognition. You are given one or more time-series
plots rendered as images.

Your job:
1. Carefully examine the visual patterns in the plot(s).
2. Reason step-by-step (Chain of Thought) before concluding.
3. Output a single JSON block (fenced with ```json ... ```) at the end of your response
   matching the schema below — no other JSON must appear outside that fence.

JSON schema:
{
  "summary":          "<concise summary of key findings, 2-4 sentences>",
  "confidence":       <float 0.0-1.0>,
  "reasoning_steps":  [{"step": <int>, "observation": "<str>", "inference": "<str>"}, ...],
  "anomalies":        [{"timestamp": "<ISO8601>", "severity": "<low|medium|high>", "description": "<str>"}],
  "trends":           [{"direction": "<up|down|flat|cyclical>", "strength": "<weak|moderate|strong>",
                        "period": "<str or null>", "description": "<str>"}]
}

Rules:
- If you are uncertain, lower the confidence score accordingly.
- Base your reasoning ONLY on what is visible in the image(s).
- Do NOT hallucinate data points that are not visible.
"""

_USER_PROMPT_TEMPLATE = """\
Task: {task}
Series: {series_names}
{question_block}

Analyse the attached plot(s) and provide your step-by-step reasoning followed by the JSON conclusion.
"""


# ── Main service ────────────────────────────────────────────────────────────────

class VLMReasoner:
    """Wraps OpenAI's vision API to reason over rendered time-series images.

    New API:  analyze(PlotArtifact, ...) -> ReasoningOutput
    Legacy:   reason(plot_base64, task, ...) -> VLMResponse  (used by agent.py)
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._client = AsyncOpenAI(api_key=self._cfg.openai_api_key)
        self._model = self._cfg.openai_model
        self._max_tokens = self._cfg.openai_max_tokens
        self._temperature = self._cfg.openai_temperature

    # ── New public API ─────────────────────────────────────────────────────────

    async def analyze(
        self,
        artifact: PlotArtifact,
        domain_context: str | None = None,
        few_shot_library: FewShotLibrary | None = None,
        k: int = 3,
    ) -> ReasoningOutput:
        """Analyse a rendered PlotArtifact and return a structured ReasoningOutput.

        Parameters
        ----------
        artifact:           PlotArtifact from WindowRenderer.render().
        domain_context:     Optional free-text hint, e.g. "HR 60-100 bpm is normal".
        few_shot_library:   Library of labeled examples for in-context learning.
        k:                  Number of similar examples to inject (0 = disabled).
        """
        if self._cfg.dry_run:
            logger.info("DRY_RUN mode — no OpenAI call made (analyze)")
            return _mock_reasoning_output()

        await _check_and_increment()

        examples: list[FewShotExample] = []
        if few_shot_library is not None and k > 0:
            query = _artifact_to_query(artifact, domain_context)
            examples = await few_shot_library.retrieve(query, k=k)
            if examples:
                logger.info("Injecting %d few-shot example(s) into prompt", len(examples))

        messages = self._build_analyze_messages(artifact, domain_context, examples)
        try:
            raw = await self._call_api(messages)
        except Exception as exc:
            raise ReasonerError(f"VLM API call failed after retries: {exc}") from exc

        logger.debug("VLM raw response (%d chars)", len(raw))
        return await self._parse_with_fallback(raw)

    # ── Legacy API (agent.py) ──────────────────────────────────────────────────

    async def reason(
        self,
        plot_base64: str,
        task: AnalysisTask,
        series_names: list[str],
        question: str | None = None,
        chain_of_thought: bool = True,
    ) -> VLMResponse:
        """Call GPT-4o vision with the rendered plot and return a structured result.

        Kept for backward compatibility with agent.py.
        """
        if self._cfg.dry_run:
            logger.info("DRY_RUN mode — no OpenAI call made (reason)")
            return _mock_vlm_response(task, series_names)

        await _check_and_increment()
        return await self._reason_with_retry(
            plot_base64, task, series_names, question, chain_of_thought
        )

    @retry(
        retry=retry_if_exception_type(_RETRIABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _reason_with_retry(
        self,
        plot_base64: str,
        task: AnalysisTask,
        series_names: list[str],
        question: str | None,
        chain_of_thought: bool,
    ) -> VLMResponse:
        user_content = self._build_user_content(
            plot_base64, task, series_names, question, chain_of_thought
        )
        logger.info("VLM call '%s' task=%s series=%s", self._model, task.value, series_names)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _LEGACY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        raw_text = response.choices[0].message.content or ""
        logger.debug("VLM raw response (%d chars)", len(raw_text))
        try:
            result = self._parse_response(raw_text, task, series_names)
        except Exception as exc:
            raise ReasonerError(
                f"Failed to parse VLM response: {exc}\n\nRaw:\n{raw_text}"
            ) from exc
        return VLMResponse(raw_text=raw_text, result=result)

    # ── Internal: API call with targeted retries ───────────────────────────────

    @retry(
        retry=retry_if_exception_type(_RETRIABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_api(self, messages: list[dict]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    # ── JSON parsing with repair fallback ──────────────────────────────────────

    async def _parse_with_fallback(self, raw: str) -> ReasoningOutput:
        try:
            return ReasoningOutput.model_validate(json.loads(self._extract_json_block(raw)))
        except Exception as exc:
            logger.warning("Initial JSON parse failed (%s); sending repair prompt", exc)
            return await self._repair_json(raw)

    async def _repair_json(self, bad_response: str) -> ReasoningOutput:
        messages: list[dict] = [
            {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "The following response contains malformed JSON. "
                    "Fix it and return ONLY the corrected JSON object:\n\n"
                    f"{bad_response}"
                ),
            },
        ]
        try:
            repaired_raw = await self._call_api(messages)
            return ReasoningOutput.model_validate(
                json.loads(self._extract_json_block(repaired_raw))
            )
        except Exception as exc:
            raise ReasonerError(
                f"JSON repair also failed: {exc}\n\nOriginal response:\n{bad_response}"
            ) from exc

    # ── Message builders ───────────────────────────────────────────────────────

    def _build_analyze_messages(
        self,
        artifact: PlotArtifact,
        domain_context: str | None,
        examples: list[FewShotExample],
    ) -> list[dict]:
        user_content: list[dict] = []

        # ── Inject few-shot examples (images detail=low to save tokens) ────────
        if examples:
            user_content.append({
                "type": "text",
                "text": f"Here are {len(examples)} similar case(s) from the library:\n",
            })
            for i, ex in enumerate(examples, 1):
                user_content.append({
                    "type": "text",
                    "text": (
                        f"\n--- Example {i} "
                        f"(domain: {ex.domain}, trend: {ex.output.trend}, "
                        f"confidence: {ex.output.confidence:.2f}) ---\n"
                    ),
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{ex.artifact.base64_string}",
                        "detail": "low",
                    },
                })
                # Exclude raw_reasoning from examples to save tokens
                analysis = ex.output.model_dump(exclude={"raw_reasoning"})
                user_content.append({
                    "type": "text",
                    "text": f"Analysis:\n{json.dumps(analysis, indent=2)}\n",
                })
            user_content.append({"type": "text", "text": "\nNow analyse the new plot:\n"})

        # ── Main query ─────────────────────────────────────────────────────────
        meta = artifact.metadata
        info_parts = [
            f"Domain: {meta.get('domain', 'default')}",
            f"Channels: {meta.get('n_channels', '?')}  ({', '.join(meta.get('channels', []))})",
            f"Window size: {meta.get('window_size', '?')} timesteps",
            f"Normalisation: {meta.get('normalization', 'zscore')}",
        ]
        if domain_context:
            info_parts.append(f"Context: {domain_context}")
        info_parts.append("\nAnalyse this time-series plot following the four-step framework.")

        user_content.append({"type": "text", "text": "\n".join(info_parts)})
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{artifact.base64_string}",
                "detail": "high",
            },
        })

        return [
            {"role": "system", "content": _ANALYZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_user_content(
        self,
        plot_base64: str,
        task: AnalysisTask,
        series_names: list[str],
        question: str | None,
        chain_of_thought: bool,
    ) -> list[dict]:
        question_block = f"Question: {question}" if question else ""
        cot_instruction = (
            "\nThink step-by-step before writing the JSON block." if chain_of_thought else ""
        )
        text_part = _USER_PROMPT_TEMPLATE.format(
            task=task.value.replace("_", " ").title(),
            series_names=", ".join(series_names),
            question_block=question_block,
        ) + cot_instruction
        return [
            {"type": "text", "text": text_part},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{plot_base64}", "detail": "high"},
            },
        ]

    # ── Legacy response parsing ────────────────────────────────────────────────

    def _parse_response(
        self, raw: str, task: AnalysisTask, series_names: list[str]
    ) -> AnalysisResult:
        data = json.loads(self._extract_json_block(raw))
        reasoning_steps = [
            ReasoningStep(
                step=s.get("step", i + 1),
                observation=s.get("observation", ""),
                inference=s.get("inference", ""),
            )
            for i, s in enumerate(data.get("reasoning_steps", []))
        ]
        return AnalysisResult(
            series_names=series_names,
            task=task,
            summary=data.get("summary", "No summary provided."),
            reasoning_steps=reasoning_steps,
            anomalies=data.get("anomalies", []),
            trends=data.get("trends", []),
            confidence=float(data.get("confidence", 0.5)),
            raw_vlm_response=raw,
        )

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """Extract the last ```json ... ``` fence, or fall back to a bare {...} block."""
        matches = re.findall(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            return brace_match.group(0)
        raise ValueError("No JSON block found in VLM response")
