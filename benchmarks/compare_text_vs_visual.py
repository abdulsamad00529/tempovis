"""Benchmark: text-only vs visual (TempoVis) time series analysis via GPT-4o.

Reproduces the core experimental setup from arXiv:2410.02637 — that rendering time
series as images and sending them to a VLM yields significantly better anomaly
detection accuracy and higher confidence than serialising the raw numerical values
as text.

Usage
-----
    # Full run (requires OPENAI_API_KEY and network access for GIFT-Eval download)
    python benchmarks/compare_text_vs_visual.py

    # Dry run with synthetic data (no API calls, no HuggingFace download)
    python benchmarks/compare_text_vs_visual.py --dry-run

    # Custom settings
    python benchmarks/compare_text_vs_visual.py --n-windows 20 --max-series 5

Output
------
    - Markdown summary table printed to stdout
    - benchmarks/results.json — full per-window data
    - benchmarks/summary.json — per-domain aggregates
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.data_loader import GiftEvalLoader, _DOMAIN_MAP  # noqa: E402
from app.services.ingestion import DataIngestionService, TimeSeriesWindow  # noqa: E402
from app.services.renderer import WindowRenderer  # noqa: E402
from app.services.reasoner import (  # noqa: E402
    AnomalyDetail,
    ReasoningOutput,
    VLMReasoner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# 5 subsets, one per representative domain combination
_BENCHMARK_SUBSETS = [
    "m4_monthly",      # financial
    "exchange_rate",   # financial (different pattern type)
    "ett_h1",          # iot
    "weather",         # iot (multivariate)
    "illness",         # clinical
]

# GPT-4o pricing (USD per token, as of 2024-11)
_PRICE_INPUT_PER_TOKEN = 2.50 / 1_000_000
_PRICE_OUTPUT_PER_TOKEN = 10.00 / 1_000_000

_RESULTS_DIR = Path(__file__).parent
_OUTPUT_DIR = _RESULTS_DIR  # write results alongside the script

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ApproachResult:
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    n_anomalies: int
    trend: str
    confidence: float
    raw_output: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class WindowBenchmark:
    window_idx: int
    subset: str
    domain: str
    n_channels: int
    window_size: int
    text: ApproachResult | None = None
    visual: ApproachResult | None = None

    @property
    def trend_agree(self) -> bool | None:
        if self.text is None or self.visual is None:
            return None
        return self.text.trend == self.visual.trend

    @property
    def anomaly_presence_agree(self) -> bool | None:
        if self.text is None or self.visual is None:
            return None
        return (self.text.n_anomalies > 0) == (self.visual.n_anomalies > 0)

    @property
    def confidence_delta(self) -> float | None:
        if self.text is None or self.visual is None:
            return None
        return self.visual.confidence - self.text.confidence

    def to_dict(self) -> dict:
        return {
            "window_idx": self.window_idx,
            "subset": self.subset,
            "domain": self.domain,
            "n_channels": self.n_channels,
            "window_size": self.window_size,
            "text": asdict(self.text) if self.text else None,
            "visual": asdict(self.visual) if self.visual else None,
            "trend_agree": self.trend_agree,
            "anomaly_presence_agree": self.anomaly_presence_agree,
            "confidence_delta": self.confidence_delta,
        }


# ── Prompts ────────────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM_PROMPT = """\
You are a time series analyst. Given time series data, identify patterns, anomalies,
and trends. Return ONLY valid JSON matching this exact schema — no code fence, no explanation:

{
  "description": "2-4 sentence summary of the signal pattern",
  "anomalies": [{"type": "point|contextual|collective", "severity": "low|medium|high",
                 "timestamp_index": <0-based int>}],
  "trend": "up|down|flat|cyclical",
  "forecast_direction": "up|down|flat|uncertain",
  "confidence": <float 0.0-1.0>,
  "raw_reasoning": "step-by-step reasoning before your conclusion"
}
"""

_TEXT_SERIES_TEMPLATE = """\
Analyse the following time series data.
Domain: {domain}
Channels: {channels}
Window size: {window_size} timesteps
Normalization: z-score (mean=0, std=1)

Raw values (one row per channel, comma-separated):
{series_text}

Summary statistics per channel:
{stats_text}
"""


def _format_series_as_text(window: TimeSeriesWindow) -> str:
    """Serialise a TimeSeriesWindow's normalised data as a compact text block."""
    norm = window.normalized_data                # (window_size, n_channels)
    channels: list[str] = window.metadata.get("channels", [])
    n_channels = norm.shape[1] if norm.ndim == 2 else 1

    if norm.ndim == 1:
        norm = norm[:, np.newaxis]

    # Compact value rows: channel_name: v0, v1, v2, ... (4 sig figs)
    series_lines: list[str] = []
    stats_lines: list[str] = []
    for i in range(n_channels):
        ch = channels[i] if i < len(channels) else f"ch{i}"
        vals = norm[:, i]
        # Downsample long series to keep prompt tokens reasonable (max 256 values shown)
        if len(vals) > 256:
            step = len(vals) // 256
            vals_display = vals[::step][:256]
        else:
            vals_display = vals
        series_lines.append(f"{ch}: [{', '.join(f'{v:.4f}' for v in vals_display)}]")
        stats_lines.append(
            f"{ch}: min={vals.min():.3f}, max={vals.max():.3f}, "
            f"mean={vals.mean():.3f}, std={vals.std():.3f}"
        )

    channels_str = ", ".join(channels[:n_channels]) if channels else "value"
    return _TEXT_SERIES_TEMPLATE.format(
        domain=window.domain_hint or "default",
        channels=channels_str,
        window_size=norm.shape[0],
        series_text="\n".join(series_lines),
        stats_text="\n".join(stats_lines),
    )


# ── Core benchmark functions ───────────────────────────────────────────────────

async def run_text_approach(
    window: TimeSeriesWindow,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> ApproachResult:
    """Send raw serialised values as text; return structured result + cost metrics."""
    text_payload = _format_series_as_text(window)

    messages = [
        {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": text_payload},
    ]

    t0 = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_text = response.choices[0].message.content or ""
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        cost = (
            prompt_tokens * _PRICE_INPUT_PER_TOKEN
            + completion_tokens * _PRICE_OUTPUT_PER_TOKEN
        )

        try:
            parsed = _parse_json_output(raw_text)
            output = ReasoningOutput.model_validate(parsed)
        except Exception as exc:
            logger.warning("Text parse error: %s", exc)
            return ApproachResult(
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                n_anomalies=0,
                trend="flat",
                confidence=0.0,
                error=str(exc),
            )

        return ApproachResult(
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            n_anomalies=len(output.anomalies),
            trend=output.trend,
            confidence=output.confidence,
            raw_output=output.model_dump(),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.error("Text approach API error: %s", exc)
        return ApproachResult(
            latency_ms=latency_ms,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            cost_usd=0.0, n_anomalies=0, trend="flat", confidence=0.0,
            error=str(exc),
        )


async def run_visual_approach(
    window: TimeSeriesWindow,
    client: AsyncOpenAI,
    renderer: WindowRenderer,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> ApproachResult:
    """Render the window as a PNG, send as vision message; return metrics."""
    try:
        artifact = renderer.render(window)
    except Exception as exc:
        logger.error("Render error: %s", exc)
        return ApproachResult(
            latency_ms=0.0, prompt_tokens=0, completion_tokens=0, total_tokens=0,
            cost_usd=0.0, n_anomalies=0, trend="flat", confidence=0.0,
            error=f"render failed: {exc}",
        )

    messages = [
        {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Domain: {window.domain_hint or 'default'}\n"
                        f"Channels: {', '.join(artifact.metadata.get('channels', ['value']))}\n"
                        f"Window size: {artifact.metadata.get('window_size', '?')} timesteps\n"
                        "Analyse this time series plot."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{artifact.base64_string}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    t0 = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_text = response.choices[0].message.content or ""
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        cost = (
            prompt_tokens * _PRICE_INPUT_PER_TOKEN
            + completion_tokens * _PRICE_OUTPUT_PER_TOKEN
        )

        try:
            parsed = _parse_json_output(raw_text)
            output = ReasoningOutput.model_validate(parsed)
        except Exception as exc:
            logger.warning("Visual parse error: %s", exc)
            return ApproachResult(
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                n_anomalies=0,
                trend="flat",
                confidence=0.0,
                error=str(exc),
            )

        return ApproachResult(
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            n_anomalies=len(output.anomalies),
            trend=output.trend,
            confidence=output.confidence,
            raw_output=output.model_dump(),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.error("Visual approach API error: %s", exc)
        return ApproachResult(
            latency_ms=latency_ms,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            cost_usd=0.0, n_anomalies=0, trend="flat", confidence=0.0,
            error=str(exc),
        )


def _parse_json_output(raw: str) -> dict:
    """Extract and parse JSON from model response (handles code fences)."""
    import re
    # Strip ```json ... ``` fences
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if match:
        raw = match.group(1)
    return json.loads(raw.strip())


# ── Synthetic data for dry-run ─────────────────────────────────────────────────

def _make_synthetic_windows(
    n_per_subset: int,
    window_size: int = 128,
    seed: int = 42,
) -> list[tuple[str, TimeSeriesWindow]]:
    """Generate synthetic windows without hitting HuggingFace."""
    rng = np.random.default_rng(seed)
    svc = DataIngestionService(window_size=window_size)
    import pandas as pd

    out: list[tuple[str, TimeSeriesWindow]] = []
    for subset in _BENCHMARK_SUBSETS:
        domain = _DOMAIN_MAP.get(subset, "default")
        n_channels = 3 if subset == "weather" else 1
        for i in range(n_per_subset):
            # Vary synthetic patterns per window
            n = window_size * 2
            t = np.linspace(0, 4 * np.pi, n)
            base = np.sin(t) + rng.normal(0, 0.3, n)
            if i % 3 == 0:   # inject a spike anomaly
                spike_idx = n // 2 + rng.integers(-20, 20)
                base[spike_idx] += rng.choice([-4.0, 4.0])
            if i % 4 == 0:   # upward trend
                base += np.linspace(0, 2, n)

            timestamps = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
            df = pd.DataFrame({"timestamp": timestamps})
            if n_channels == 1:
                df["value"] = base
            else:
                for ch in range(n_channels):
                    df[f"ch{ch}"] = base + rng.normal(0, 0.1, n)

            windows = svc.ingest(df, domain_hint=domain)
            if windows:
                out.append((subset, windows[0]))
    return out


# ── Aggregation & reporting ────────────────────────────────────────────────────

def _aggregate_by_domain(results: list[WindowBenchmark]) -> dict[str, dict]:
    from collections import defaultdict

    buckets: dict[str, list[WindowBenchmark]] = defaultdict(list)
    for r in results:
        buckets[r.domain].append(r)

    summary: dict[str, dict] = {}
    for domain, rows in buckets.items():
        def _safe_mean(vals: list) -> float:
            finite = [v for v in vals if v is not None]
            return float(np.mean(finite)) if finite else 0.0

        text_ok = [r for r in rows if r.text and not r.text.error]
        vis_ok = [r for r in rows if r.visual and not r.visual.error]

        summary[domain] = {
            "n_windows": len(rows),
            "text": {
                "avg_latency_ms": _safe_mean([r.text.latency_ms for r in text_ok]),
                "avg_tokens": _safe_mean([r.text.total_tokens for r in text_ok]),
                "avg_cost_usd": _safe_mean([r.text.cost_usd for r in text_ok]),
                "avg_confidence": _safe_mean([r.text.confidence for r in text_ok]),
                "avg_anomalies": _safe_mean([r.text.n_anomalies for r in text_ok]),
                "error_rate": 1.0 - len(text_ok) / len(rows) if rows else 0.0,
            },
            "visual": {
                "avg_latency_ms": _safe_mean([r.visual.latency_ms for r in vis_ok]),
                "avg_tokens": _safe_mean([r.visual.total_tokens for r in vis_ok]),
                "avg_cost_usd": _safe_mean([r.visual.cost_usd for r in vis_ok]),
                "avg_confidence": _safe_mean([r.visual.confidence for r in vis_ok]),
                "avg_anomalies": _safe_mean([r.visual.n_anomalies for r in vis_ok]),
                "error_rate": 1.0 - len(vis_ok) / len(rows) if rows else 0.0,
            },
            "agreement": {
                "trend_agree_rate": _safe_mean(
                    [1.0 if r.trend_agree else 0.0 for r in rows if r.trend_agree is not None]
                ),
                "anomaly_presence_agree_rate": _safe_mean(
                    [1.0 if r.anomaly_presence_agree else 0.0
                     for r in rows if r.anomaly_presence_agree is not None]
                ),
                "avg_confidence_delta": _safe_mean(
                    [r.confidence_delta for r in rows if r.confidence_delta is not None]
                ),
            },
        }
    return summary


def _print_markdown_table(summary: dict[str, dict], results: list[WindowBenchmark]) -> None:
    """Print a GitHub-flavoured markdown comparison table to stdout."""

    total_text_ok = [r for r in results if r.text and not r.text.error]
    total_vis_ok = [r for r in results if r.visual and not r.visual.error]

    def _mean(vals: list) -> float:
        return float(np.mean(vals)) if vals else 0.0

    print()
    print("## TempoVis Benchmark: Text vs Visual Analysis (GPT-4o)")
    print()
    print(
        "Reproducing the core finding from **arXiv:2410.02637**: rendering time series "
        "as images improves VLM reasoning quality over raw numerical text."
    )
    print()

    # --- Per-domain table ---
    header = (
        "| Domain | N | "
        "Text Latency (ms) | Text Tokens | Text Conf | Text Anomalies | "
        "Visual Latency (ms) | Visual Tokens | Visual Conf | Visual Anomalies | "
        "Conf Delta | Trend Agree |"
    )
    sep = "|" + "|".join(["-" * (len(h) + 2) for h in header.split("|")[1:-1]]) + "|"

    print("### Per-Domain Results")
    print()
    print(header)
    print(sep)

    for domain, data in sorted(summary.items()):
        t = data["text"]
        v = data["visual"]
        ag = data["agreement"]
        print(
            f"| {domain} | {data['n_windows']} | "
            f"{t['avg_latency_ms']:.0f} | {t['avg_tokens']:.0f} | "
            f"{t['avg_confidence']:.2f} | {t['avg_anomalies']:.1f} | "
            f"{v['avg_latency_ms']:.0f} | {v['avg_tokens']:.0f} | "
            f"{v['avg_confidence']:.2f} | {v['avg_anomalies']:.1f} | "
            f"{ag['avg_confidence_delta']:+.2f} | "
            f"{ag['trend_agree_rate']:.0%} |"
        )

    print()

    # --- Overall summary ---
    print("### Overall Summary")
    print()
    overall_rows = [
        ("Windows analysed", len(results), len(results)),
        ("Errors", len(results) - len(total_text_ok), len(results) - len(total_vis_ok)),
        ("Avg latency (ms)",
         f"{_mean([r.text.latency_ms for r in total_text_ok]):.0f}",
         f"{_mean([r.visual.latency_ms for r in total_vis_ok]):.0f}"),
        ("Avg prompt tokens",
         f"{_mean([r.text.prompt_tokens for r in total_text_ok]):.0f}",
         f"{_mean([r.visual.prompt_tokens for r in total_vis_ok]):.0f}"),
        ("Avg total tokens",
         f"{_mean([r.text.total_tokens for r in total_text_ok]):.0f}",
         f"{_mean([r.visual.total_tokens for r in total_vis_ok]):.0f}"),
        ("Total cost (USD)",
         f"${sum(r.text.cost_usd for r in total_text_ok):.4f}",
         f"${sum(r.visual.cost_usd for r in total_vis_ok):.4f}"),
        ("Avg confidence",
         f"{_mean([r.text.confidence for r in total_text_ok]):.3f}",
         f"{_mean([r.visual.confidence for r in total_vis_ok]):.3f}"),
        ("Avg anomalies detected",
         f"{_mean([r.text.n_anomalies for r in total_text_ok]):.2f}",
         f"{_mean([r.visual.n_anomalies for r in total_vis_ok]):.2f}"),
        ("Trend agreement rate",
         f"{_mean([1.0 if r.trend_agree else 0.0 for r in results if r.trend_agree is not None]):.0%}",
         "(same metric)"),
        ("Anomaly presence agreement",
         f"{_mean([1.0 if r.anomaly_presence_agree else 0.0 for r in results if r.anomaly_presence_agree is not None]):.0%}",
         "(same metric)"),
    ]

    print("| Metric | Text-only | Visual (TempoVis) |")
    print("|--------|-----------|-------------------|")
    for label, text_val, vis_val in overall_rows:
        print(f"| {label} | {text_val} | {vis_val} |")

    # Confidence lift
    conf_deltas = [r.confidence_delta for r in results if r.confidence_delta is not None]
    if conf_deltas:
        avg_delta = np.mean(conf_deltas)
        pct_positive = np.mean([d > 0 for d in conf_deltas]) * 100
        print()
        print(
            f"> **Visual confidence lift:** {avg_delta:+.3f} on average "
            f"({pct_positive:.0f}% of windows saw higher confidence with the visual approach)"
        )

    print()


# ── Dry-run mode ───────────────────────────────────────────────────────────────

def _run_dry_run(args: argparse.Namespace) -> None:
    """Simulate the benchmark with synthetic data and zero API calls."""
    import random

    logger.info("DRY RUN — generating synthetic windows (no API calls, no HF download)")
    n_per_subset = max(1, args.n_windows // len(_BENCHMARK_SUBSETS))
    pairs = _make_synthetic_windows(n_per_subset, window_size=args.window_size)

    rng = random.Random(42)
    results: list[WindowBenchmark] = []

    for idx, (subset, window) in enumerate(pairs):
        domain = _DOMAIN_MAP.get(subset, "default")
        n_channels = window.metadata.get("n_channels", 1)

        # Simulate plausible text-approach results
        text_res = ApproachResult(
            latency_ms=rng.uniform(800, 2500),
            prompt_tokens=rng.randint(400, 900),
            completion_tokens=rng.randint(150, 350),
            total_tokens=0,
            cost_usd=0.0,
            n_anomalies=rng.randint(0, 2),
            trend=rng.choice(["up", "down", "flat", "cyclical"]),
            confidence=rng.uniform(0.45, 0.70),
        )
        text_res.total_tokens = text_res.prompt_tokens + text_res.completion_tokens
        text_res.cost_usd = (
            text_res.prompt_tokens * _PRICE_INPUT_PER_TOKEN
            + text_res.completion_tokens * _PRICE_OUTPUT_PER_TOKEN
        )

        # Visual typically: higher confidence (+0.05–0.20), lower prompt tokens,
        # but similar completion tokens.
        vis_conf = min(1.0, text_res.confidence + rng.uniform(0.05, 0.25))
        vis_prompt = rng.randint(500, 750)  # image tiles add ~300-500 tokens but text shorter
        vis_completion = rng.randint(150, 350)
        vis_res = ApproachResult(
            latency_ms=rng.uniform(900, 3000),
            prompt_tokens=vis_prompt,
            completion_tokens=vis_completion,
            total_tokens=vis_prompt + vis_completion,
            cost_usd=(
                vis_prompt * _PRICE_INPUT_PER_TOKEN
                + vis_completion * _PRICE_OUTPUT_PER_TOKEN
            ),
            n_anomalies=rng.randint(0, 3),
            trend=rng.choice(["up", "down", "flat", "cyclical"]),
            confidence=vis_conf,
        )

        results.append(WindowBenchmark(
            window_idx=idx,
            subset=subset,
            domain=domain,
            n_channels=n_channels,
            window_size=window.normalized_data.shape[0],
            text=text_res,
            visual=vis_res,
        ))
        logger.info("[%d/%d] subset=%s domain=%s (dry-run)", idx + 1, len(pairs), subset, domain)

    _save_and_report(results, args, dry_run=True)


# ── Main async benchmark loop ──────────────────────────────────────────────────

async def run_benchmark(args: argparse.Namespace) -> None:
    cfg = get_settings()
    client = AsyncOpenAI(api_key=cfg.openai_api_key)
    renderer = WindowRenderer()
    model = args.model or cfg.openai_model

    logger.info(
        "Starting benchmark: model=%s, n_windows=%d, subsets=%s",
        model, args.n_windows, _BENCHMARK_SUBSETS,
    )

    # Load windows from GIFT-Eval (cached after first download)
    n_per_subset = max(1, args.n_windows // len(_BENCHMARK_SUBSETS))
    loader = GiftEvalLoader(
        subsets=_BENCHMARK_SUBSETS,
        window_size=args.window_size,
        cache_dir=Path("data/cache"),
        max_series=args.max_series,
    )

    logger.info("Loading up to %d windows per subset from GIFT-Eval...", n_per_subset)
    windows_by_subset: dict[str, list[TimeSeriesWindow]] = {s: [] for s in _BENCHMARK_SUBSETS}

    for window in loader.iter_windows():
        subset = window.metadata.get("subset", "unknown")
        if subset in windows_by_subset and len(windows_by_subset[subset]) < n_per_subset:
            windows_by_subset[subset].append(window)

    pairs: list[tuple[str, TimeSeriesWindow]] = []
    for subset in _BENCHMARK_SUBSETS:
        subset_windows = windows_by_subset[subset]
        pairs.extend((subset, w) for w in subset_windows)
        logger.info("  %s: %d windows loaded", subset, len(subset_windows))

    if not pairs:
        logger.error("No windows loaded — check network access or use --dry-run")
        return

    results: list[WindowBenchmark] = []
    total = len(pairs)

    for idx, (subset, window) in enumerate(pairs):
        domain = _DOMAIN_MAP.get(subset, "default")
        n_ch = window.metadata.get("n_channels", window.normalized_data.shape[-1] if window.normalized_data.ndim > 1 else 1)
        ws = window.normalized_data.shape[0]

        logger.info("[%d/%d] subset=%s domain=%s n_channels=%d", idx + 1, total, subset, domain, n_ch)

        # Run text approach
        logger.info("  -> text-only...")
        text_res = await run_text_approach(
            window, client, model,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        logger.info(
            "     latency=%.0fms tokens=%d conf=%.2f anomalies=%d%s",
            text_res.latency_ms, text_res.total_tokens, text_res.confidence,
            text_res.n_anomalies, f" [ERROR: {text_res.error}]" if text_res.error else "",
        )

        # Small delay to avoid rate limiting
        await asyncio.sleep(args.request_delay)

        # Run visual approach
        logger.info("  -> visual (TempoVis)...")
        vis_res = await run_visual_approach(
            window, client, renderer, model,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        logger.info(
            "     latency=%.0fms tokens=%d conf=%.2f anomalies=%d%s",
            vis_res.latency_ms, vis_res.total_tokens, vis_res.confidence,
            vis_res.n_anomalies, f" [ERROR: {vis_res.error}]" if vis_res.error else "",
        )

        results.append(WindowBenchmark(
            window_idx=idx,
            subset=subset,
            domain=domain,
            n_channels=n_ch,
            window_size=ws,
            text=text_res,
            visual=vis_res,
        ))

        # Save incrementally in case of interruption
        if (idx + 1) % 5 == 0:
            _save_results(results)
            logger.info("Checkpoint saved (%d windows done)", idx + 1)

        await asyncio.sleep(args.request_delay)

    _save_and_report(results, args, dry_run=False)


def _save_results(results: list[WindowBenchmark]) -> None:
    out = _OUTPUT_DIR / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str),
        encoding="utf-8",
    )


def _save_and_report(
    results: list[WindowBenchmark],
    args: argparse.Namespace,
    dry_run: bool = False,
) -> None:
    # Save per-window results
    _save_results(results)
    logger.info("Results saved to %s", _OUTPUT_DIR / "results.json")

    # Aggregate and save summary
    summary = _aggregate_by_domain(results)
    summary_path = _OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dry_run": dry_run,
                "model": getattr(args, "model", "gpt-4o") or "gpt-4o",
                "n_windows": len(results),
                "subsets": _BENCHMARK_SUBSETS,
                "window_size": args.window_size,
                "domains": summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Summary saved to %s", summary_path)

    # Print markdown table
    _print_markdown_table(summary, results)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark text-only vs visual time series analysis with GPT-4o",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data; no API calls or HuggingFace download",
    )
    p.add_argument(
        "--n-windows", type=int, default=50,
        help="Total windows to analyse (split evenly across subsets)",
    )
    p.add_argument(
        "--max-series", type=int, default=5,
        help="Max series loaded per subset from GIFT-Eval",
    )
    p.add_argument(
        "--window-size", type=int, default=128,
        help="Timesteps per window",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Override the OpenAI model (default: from .env / config)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Max completion tokens per API call",
    )
    p.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature",
    )
    p.add_argument(
        "--request-delay", type=float, default=1.0,
        help="Seconds to sleep between API calls (rate-limit buffer)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.dry_run:
        _run_dry_run(args)
    else:
        asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
