"""Adaptive plot renderer — converts time series data into domain-aware PNG images.

Two APIs live here:

  New API  — used by WindowRenderer / the ingestion pipeline:
    WindowRenderer.render(TimeSeriesWindow) -> PlotArtifact

  Legacy API  — used by agent.py:
    PlotRenderer.render(list[NormalizedSeries], ...) -> str   (base64 PNG)

Design principle: images carry richer perceptual signal than raw numbers for VLMs.
Reference: arXiv:2410.02637 (up to 150% accuracy improvement via visual encoding).
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.models.schemas import AnalysisTask, PlotStyle
from app.services.ingestion import NormalizedSeries, TimeSeriesWindow, align_series

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

matplotlib.use("Agg")  # non-interactive backend — safe in async/server contexts

logger = logging.getLogger(__name__)


# ── Domain style registry ──────────────────────────────────────────────────────

_DOMAIN_STYLES: dict[str, dict[str, Any]] = {
    "clinical": {
        # Dark background — mirrors medical monitor aesthetics (ECG, vitals display)
        "fig_facecolor": "#0d1117",
        "ax_facecolor": "#161b22",
        "text_color": "#e6edf3",
        "grid_color": "#30363d",
        "grid_alpha": 0.75,
        "grid_which": "both",
        "palette": [
            "#00e676",  # green  — primary signal
            "#ff5252",  # red    — alert / secondary
            "#40c4ff",  # cyan
            "#ffab40",  # amber
            "#ce93d8",  # purple
            "#80cbc4",  # teal
            "#fff176",  # yellow
            "#ef9a9a",  # pink
        ],
        "linewidth": 1.7,
        "spine_color": "#30363d",
        "title_tag": "Clinical Monitor",
    },
    "financial": {
        "fig_facecolor": "#ffffff",
        "ax_facecolor": "#f8fafc",
        "text_color": "#111827",
        "grid_color": "#e5e7eb",
        "grid_alpha": 0.85,
        "grid_which": "both",
        "palette": [
            "#1d4ed8",  # blue  — price / primary
            "#16a34a",  # green — positive
            "#dc2626",  # red   — negative / volume
            "#d97706",  # amber
            "#7c3aed",  # violet
            "#0891b2",  # cyan
        ],
        "linewidth": 1.4,
        "spine_color": "#d1d5db",
        "title_tag": "Market Data",
    },
    "iot": {
        # Minimal, multi-channel overlay — sensor stream aesthetic
        "fig_facecolor": "#fafafa",
        "ax_facecolor": "#ffffff",
        "text_color": "#1f2937",
        "grid_color": "#f3f4f6",
        "grid_alpha": 0.6,
        "grid_which": "major",
        "palette": [
            "#2563eb",
            "#dc2626",
            "#16a34a",
            "#d97706",
            "#7c3aed",
            "#0891b2",
            "#db2777",
            "#65a30d",
        ],
        "linewidth": 1.2,
        "spine_color": "#e5e7eb",
        "title_tag": "IoT Sensor Stream",
    },
    "default": {
        "fig_facecolor": "#ffffff",
        "ax_facecolor": "#ffffff",
        "text_color": "#111827",
        "grid_color": "#e5e7eb",
        "grid_alpha": 0.45,
        "grid_which": "major",
        "palette": [
            "#2563eb",
            "#dc2626",
            "#16a34a",
            "#d97706",
            "#7c3aed",
            "#0891b2",
            "#db2777",
            "#65a30d",
        ],
        "linewidth": 1.6,
        "spine_color": "#d1d5db",
        "title_tag": "",
    },
}

# Legacy palette (kept for PlotRenderer below)
_PALETTE = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706",
    "#7C3AED", "#0891B2", "#DB2777", "#65A30D",
]


# ── Output type ────────────────────────────────────────────────────────────────

@dataclass
class PlotArtifact:
    """Rendered PNG image with associated metadata.

    Attributes:
        image_bytes:   Raw PNG bytes (BytesIO-friendly, no disk I/O).
        base64_string: Base64-encoded PNG, ready for GPT-4o vision API.
        metadata:      domain, channels, window_index, duration, image_size_bytes, …
    """
    image_bytes: bytes
    base64_string: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Exceptions ─────────────────────────────────────────────────────────────────

class RenderError(RuntimeError):
    """Raised when plot generation fails."""


# ── New API — WindowRenderer ───────────────────────────────────────────────────

class WindowRenderer:
    """Renders TimeSeriesWindow objects into domain-aware PNG PlotArtifacts.

    Domain awareness:
        "clinical"  — dark monitor background, green/red medical palette, fine grid.
        "financial" — clean white, candlestick-friendly, volume subplot when present.
        "iot"       — minimal style, multi-channel overlay with distinct hues.
        "default"   — professional white background, blue-dominant palette.

    Multivariate series are rendered as shared-x subplots (one per channel).
    """

    def __init__(
        self,
        dpi: int | None = None,
        width_in: float | None = None,
    ) -> None:
        cfg = get_settings()
        self._dpi = dpi or cfg.plot_dpi
        self._width = width_in or cfg.plot_width_in

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(self, window: TimeSeriesWindow) -> PlotArtifact:
        """Render a single TimeSeriesWindow → PlotArtifact."""
        try:
            fig = self._build_figure(window)
            return self._fig_to_artifact(fig, window)
        except Exception as exc:
            raise RenderError(f"Failed to render window: {exc}") from exc

    def render_batch(self, windows: list[TimeSeriesWindow]) -> list[PlotArtifact]:
        """Render multiple windows, one PlotArtifact per window."""
        return [self.render(w) for w in windows]

    # ── Figure construction ────────────────────────────────────────────────────

    def _build_figure(self, window: TimeSeriesWindow) -> Figure:
        domain = (window.domain_hint or "default").lower()
        style = _DOMAIN_STYLES.get(domain, _DOMAIN_STYLES["default"])

        channels: list[str] = window.metadata["channels"]
        n_channels = len(channels)
        window_size: int = window.metadata["window_size"]

        # Reconstruct evenly-spaced timestamps from the window's recorded boundary
        start_ts = pd.Timestamp(window.metadata["start_timestamp"])
        end_ts = pd.Timestamp(window.metadata["end_timestamp"])
        timestamps = pd.date_range(start=start_ts, end=end_ts, periods=window_size)

        # Financial domain: isolate "volume" channel into its own bar subplot
        vol_idx = self._find_volume_channel(channels) if domain == "financial" else None
        price_idxs = [i for i in range(n_channels) if i != vol_idx] if vol_idx is not None else list(range(n_channels))

        n_price = len(price_idxs)
        n_subplots = n_price + (1 if vol_idx is not None else 0)
        height_ratios = ([3] * n_price + [1]) if vol_idx is not None else ([1] * n_subplots)

        subplot_h = 2.4
        total_h = max(3.5, min(14.0, subplot_h * n_subplots))

        fig, axes_2d = plt.subplots(
            n_subplots, 1,
            figsize=(self._width, total_h),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.10},
            squeeze=False,
        )
        axes: list[Axes] = axes_2d.flatten().tolist()

        fig.patch.set_facecolor(style["fig_facecolor"])

        # ── Plot price / signal channels ───────────────────────────────────────
        for subplot_i, ch_idx in enumerate(price_idxs):
            ax = axes[subplot_i]
            ch_name = channels[ch_idx]
            color = style["palette"][ch_idx % len(style["palette"])]
            values = window.normalized_data[:, ch_idx]

            ax.plot(timestamps, values, color=color, linewidth=style["linewidth"], label=ch_name, zorder=3)
            self._style_ax(ax, style, ylabel=ch_name, hide_xticks=True)

            if domain == "clinical":
                ax.axhline(0, color=style["text_color"], linewidth=0.4, alpha=0.35, zorder=1)

        # ── Annotation box — top-right of first subplot ────────────────────────
        self._add_annotation(axes[0], window, style)

        # ── Volume bar subplot (financial only) ────────────────────────────────
        if vol_idx is not None:
            ax_vol = axes[-1]
            vol_values = window.normalized_data[:, vol_idx]
            bar_w = pd.Timedelta(
                seconds=(end_ts - start_ts).total_seconds() / window_size * 0.8
            )
            ax_vol.bar(timestamps, vol_values, width=bar_w, color=style["palette"][2], alpha=0.65, zorder=3)
            self._style_ax(ax_vol, style, ylabel="Volume", hide_xticks=False)

        # ── X-axis formatting on bottom subplot ───────────────────────────────
        bottom_ax = axes[-1]
        locator = mdates.AutoDateLocator()
        bottom_ax.xaxis.set_major_locator(locator)
        bottom_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        bottom_ax.tick_params(axis="x", labelcolor=style["text_color"], labelsize=8)
        for tick in bottom_ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")

        # ── Figure-level title ─────────────────────────────────────────────────
        tag = style["title_tag"]
        sep = "  ·  " if tag else ""
        win_idx = window.metadata.get("window_index", 0)
        fig.suptitle(
            f"Window #{win_idx}  ·  {n_channels} channel(s){sep}{tag}",
            fontsize=11,
            fontweight="bold",
            color=style["text_color"],
            y=1.01,
        )

        return fig

    # ── Axis styling ───────────────────────────────────────────────────────────

    def _style_ax(
        self,
        ax: Axes,
        style: dict[str, Any],
        ylabel: str,
        hide_xticks: bool = False,
    ) -> None:
        ax.set_facecolor(style["ax_facecolor"])
        ax.grid(
            True,
            which=style["grid_which"],
            color=style["grid_color"],
            alpha=style["grid_alpha"],
            linestyle="--",
            linewidth=0.55,
            zorder=0,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(style["spine_color"])
        ax.spines["bottom"].set_color(style["spine_color"])
        ax.set_ylabel(ylabel, fontsize=8, color=style["text_color"], labelpad=4)
        ax.tick_params(colors=style["text_color"], labelsize=7)
        ax.yaxis.label.set_color(style["text_color"])
        if hide_xticks:
            ax.tick_params(axis="x", labelbottom=False)

    # ── Annotation box ─────────────────────────────────────────────────────────

    def _add_annotation(
        self,
        ax: Axes,
        window: TimeSeriesWindow,
        style: dict[str, Any],
    ) -> None:
        meta = window.metadata
        start_ts = pd.Timestamp(meta["start_timestamp"])
        end_ts = pd.Timestamp(meta["end_timestamp"])
        duration_s = (end_ts - start_ts).total_seconds()
        dur_str = f"{duration_s / 60:.1f} min" if duration_s >= 60 else f"{duration_s:.2f} s"

        domain_label = (window.domain_hint or "default").title()
        lines = [
            f"Domain   {domain_label}",
            f"Channels {meta['n_channels']}",
            f"Duration {dur_str}",
            f"Norm     {meta['normalization']}",
            f"Window   #{meta['window_index']}",
        ]
        text = "\n".join(lines)

        is_dark = style["fig_facecolor"] not in ("#ffffff", "#f8fafc", "#fafafa")
        box_fc = "#21262d" if is_dark else "white"
        box_ec = "#30363d" if is_dark else "#d1d5db"

        ax.text(
            0.993, 0.967,
            text,
            transform=ax.transAxes,
            va="top", ha="right",
            fontsize=7.5,
            color=style["text_color"],
            fontfamily="monospace",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": box_fc,
                "edgecolor": box_ec,
                "alpha": 0.90,
            },
            zorder=10,
        )

    # ── Volume channel detection ───────────────────────────────────────────────

    @staticmethod
    def _find_volume_channel(channels: list[str]) -> int | None:
        keywords = {"volume", "vol", "volume_usd", "volume_btc", "trade_vol"}
        for i, ch in enumerate(channels):
            if ch.lower() in keywords:
                return i
        return None

    # ── Serialisation ──────────────────────────────────────────────────────────

    def _fig_to_artifact(self, fig: Figure, window: TimeSeriesWindow) -> PlotArtifact:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=self._dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        buf.seek(0)
        image_bytes = buf.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        meta = window.metadata
        start_ts = pd.Timestamp(meta["start_timestamp"])
        end_ts = pd.Timestamp(meta["end_timestamp"])

        return PlotArtifact(
            image_bytes=image_bytes,
            base64_string=b64,
            metadata={
                "domain": window.domain_hint or "default",
                "window_index": meta["window_index"],
                "n_channels": meta["n_channels"],
                "channels": meta["channels"],
                "window_size": meta["window_size"],
                "start_timestamp": meta["start_timestamp"],
                "end_timestamp": meta["end_timestamp"],
                "duration_s": (end_ts - start_ts).total_seconds(),
                "normalization": meta["normalization"],
                "image_size_bytes": len(image_bytes),
            },
        )


# ── Legacy API — PlotRenderer (used by agent.py) ───────────────────────────────

class PlotRenderer:
    """Renders NormalizedSeries objects to a base64-encoded PNG string.

    This is the legacy API consumed by agent.py and the LangGraph pipeline.
    For new code, prefer WindowRenderer which accepts TimeSeriesWindow directly.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._dpi = cfg.plot_dpi
        self._w = cfg.plot_width_in
        self._h = cfg.plot_height_in

    def render(
        self,
        series: list[NormalizedSeries],
        style: PlotStyle = PlotStyle.line,
        task: AnalysisTask = AnalysisTask.general,
        title: str | None = None,
    ) -> str:
        """Render series to a base64-encoded PNG string."""
        try:
            fig = self._build_figure(series, style, task, title)
            return self._fig_to_base64(fig)
        except Exception as exc:
            raise RenderError(f"Failed to render plot: {exc}") from exc

    def _build_figure(
        self,
        series: list[NormalizedSeries],
        style: PlotStyle,
        task: AnalysisTask,
        title: str | None,
    ) -> Figure:
        dispatch = {
            PlotStyle.line: self._plot_line,
            PlotStyle.area: self._plot_area,
            PlotStyle.multi_panel: self._plot_multi_panel,
            PlotStyle.heatmap: self._plot_heatmap,
            PlotStyle.candlestick: self._plot_line,
        }
        fig = dispatch.get(style, self._plot_line)(series, task, title)
        fig.tight_layout(pad=2.0)
        return fig

    def _plot_line(
        self, series: list[NormalizedSeries], task: AnalysisTask, title: str | None
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(self._w, self._h))
        aligned = align_series(series)
        for i, ns in enumerate(series):
            color = _PALETTE[i % len(_PALETTE)]
            label = f"{ns.name}" + (f" ({ns.unit})" if ns.unit else "")
            ax.plot(aligned.index, aligned[ns.name], label=label, color=color, linewidth=1.8, alpha=0.9)
            if task == AnalysisTask.anomaly_detection:
                self._mark_anomalies(ax, aligned[ns.name], color)
        if task == AnalysisTask.trend_analysis and len(series) == 1:
            self._add_trend_line(ax, aligned.iloc[:, 0])
        self._style_axes(ax, title or f"Time Series — {task.value.replace('_', ' ').title()}")
        return fig

    def _plot_area(
        self, series: list[NormalizedSeries], task: AnalysisTask, title: str | None
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(self._w, self._h))
        aligned = align_series(series)
        for i, ns in enumerate(series):
            color = _PALETTE[i % len(_PALETTE)]
            ax.fill_between(aligned.index, aligned[ns.name], alpha=0.35, color=color, label=ns.name)
            ax.plot(aligned.index, aligned[ns.name], color=color, linewidth=1.4)
        self._style_axes(ax, title or "Area Chart")
        return fig

    def _plot_multi_panel(
        self, series: list[NormalizedSeries], task: AnalysisTask, title: str | None
    ) -> Figure:
        n = len(series)
        fig, axes = plt.subplots(n, 1, figsize=(self._w, self._h * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, ns, color in zip(axes, series, _PALETTE, strict=False):
            ax.plot(ns.df.index, ns.df["value"], color=color, linewidth=1.6)
            ax.set_ylabel(f"{ns.name}" + (f"\n({ns.unit})" if ns.unit else ""), fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if task == AnalysisTask.anomaly_detection:
                self._mark_anomalies(ax, ns.df["value"], color)
        axes[0].set_title(title or "Multi-Panel Time Series", fontsize=13, fontweight="bold", pad=10)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
        fig.autofmt_xdate(rotation=25)
        return fig

    def _plot_heatmap(
        self, series: list[NormalizedSeries], task: AnalysisTask, title: str | None
    ) -> Figure:
        ns = series[0]
        df = ns.df["value"].resample("D").mean().to_frame("value")
        df["weekday"] = df.index.dayofweek
        df["week"] = df.index.isocalendar().week.astype(int)
        pivot = df.pivot_table(index="weekday", columns="week", values="value")
        fig, ax = plt.subplots(figsize=(self._w, self._h))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        plt.colorbar(im, ax=ax, label=ns.unit or "value")
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlabel("ISO Week")
        ax.set_title(title or f"Heatmap — {ns.name}", fontsize=13, fontweight="bold")
        return fig

    def _mark_anomalies(self, ax: Axes, series: pd.Series, color: str, z_thresh: float = 3.0) -> None:
        mean, std = series.mean(), series.std()
        if std == 0:
            return
        z = (series - mean) / std
        idx = z[z.abs() > z_thresh].index
        if len(idx):
            ax.scatter(idx, series[idx], color=color, edgecolors="red", linewidths=1.5, s=60, zorder=5, label="_anomaly")

    def _add_trend_line(self, ax: Axes, series: pd.Series) -> None:
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        ax.plot(series.index, np.polyval(coeffs, x), "--", color="gray", linewidth=1.2, alpha=0.7, label="Trend")

    def _style_axes(self, ax: Axes, title: str) -> None:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Time", fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.figure.autofmt_xdate(rotation=25)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=9, loc="best", framealpha=0.85)

    @staticmethod
    def _fig_to_base64(fig: Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=get_settings().plot_dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def base64_to_bytes(b64: str) -> bytes:
        return base64.b64decode(b64)


# align_series is imported from ingestion at the top and re-exported here for
# agent.py which does: from app.services.renderer import align_series
__all__ = ["PlotArtifact", "PlotRenderer", "RenderError", "WindowRenderer", "align_series"]
