"""Data ingestion and normalization pipeline.

Two APIs live here:

  Legacy API  — used by agent.py / renderer.py:
    DataIngestionService.process(list[TimeSeriesInput]) -> list[NormalizedSeries]

  New API  — full-featured windowed pipeline:
    DataIngestionService.ingest(list[dict] | DataFrame, ...) -> list[TimeSeriesWindow]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from app.models.schemas import TimeSeriesInput

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────────

class NormalizationMethod(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"


# ── Output types ───────────────────────────────────────────────────────────────

@dataclass
class TimeSeriesWindow:
    """A fixed-length window of (possibly multi-channel) time series data.

    Attributes:
        raw_data:        Shape (window_size, n_channels) — original values.
        normalized_data: Shape (window_size, n_channels) — normalised values.
        metadata:        Channel names, timestamps, per-channel stats, window index.
        domain_hint:     Optional semantic label e.g. "clinical", "financial".
    """
    raw_data: np.ndarray
    normalized_data: np.ndarray
    metadata: dict[str, Any]
    domain_hint: str | None = None


# ── Legacy output type (kept for agent.py / renderer.py) ──────────────────────

@dataclass(frozen=True)
class NormalizedSeries:
    name: str
    df: pd.DataFrame          # index: DatetimeIndex; columns: value, value_norm
    unit: str | None
    stats: dict[str, float]   # mean, std, min, max, n_points, n_imputed


# ── Exceptions ─────────────────────────────────────────────────────────────────

class IngestionError(ValueError):
    """Raised when a series cannot be ingested or normalised."""


# ── Main service ───────────────────────────────────────────────────────────────

class DataIngestionService:
    """Validates, cleans, normalises, and windows raw time series data.

    Parameters:
        normalization:        ZSCORE (default) or MINMAX, applied per channel.
        window_size:          Number of timesteps per window (default 128).
        overlap:              Timesteps shared between consecutive windows (default 0).
        interpolate_method:   pandas interpolation method for gap-filling (legacy path).
        outlier_z_threshold:  Clip values beyond this many std-devs (legacy path).
    """

    def __init__(
        self,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
        window_size: int = 128,
        overlap: int = 0,
        interpolate_method: str = "time",
        outlier_z_threshold: float = 4.0,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if not (0 <= overlap < window_size):
            raise ValueError("overlap must satisfy 0 <= overlap < window_size")

        self._norm = normalization
        self._window_size = window_size
        self._overlap = overlap
        self._interpolate_method = interpolate_method
        self._outlier_z = outlier_z_threshold

    # ── New public API ─────────────────────────────────────────────────────────

    def ingest(
        self,
        data: list[dict[str, Any]] | pd.DataFrame,
        domain_hint: str | None = None,
    ) -> list[TimeSeriesWindow]:
        """Ingest raw data and return a list of fixed-length TimeSeriesWindows.

        Accepted input formats:

          1. list of dicts — each dict must have keys:
               "timestamp"    : datetime-parseable string or datetime object
               "value"        : numeric
               "channel_name" : str   (identifies the channel / sensor)

          2. pandas DataFrame — columns must include "timestamp" and at least one
             value column.  If a "channel_name" column is present it is used to
             pivot channels; otherwise every non-timestamp column is a channel.

        Steps:
            parse → pivot channels → fill missing → normalise → segment → wrap
        """
        try:
            df = self._parse(data)
            df = self._fill_missing(df)
            norm_df, stats = self._normalise_channels(df)
            windows = self._segment(df, norm_df, stats, domain_hint)
            logger.info(
                "Ingested %d channel(s), %d row(s) → %d window(s)",
                len(df.columns), len(df), len(windows),
            )
            return windows
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Ingestion failed: {exc}") from exc

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse(self, data: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame indexed by timestamp with one column per channel."""
        if isinstance(data, pd.DataFrame):
            return self._parse_dataframe(data.copy())
        if isinstance(data, list):
            return self._parse_dicts(data)
        raise IngestionError(f"Unsupported input type: {type(data)}")

    def _parse_dicts(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        if not records:
            raise IngestionError("Input list is empty")
        required = {"timestamp", "value", "channel_name"}
        missing_keys = required - records[0].keys()
        if missing_keys:
            raise IngestionError(f"Records missing required keys: {missing_keys}")

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Collect all timestamps before pivoting — pivot_table may drop NaN-valued rows.
        all_timestamps = df["timestamp"].drop_duplicates().sort_values()

        pivoted = (
            df.pivot_table(index="timestamp", columns="channel_name", values="value", aggfunc="mean")
            .sort_index()
        )
        pivoted.columns.name = None
        # Reindex to restore any rows that pivot_table silently dropped (NaN values).
        return pivoted.reindex(all_timestamps)

    def _parse_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            raise IngestionError("DataFrame must have a 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").set_index("timestamp")

        if "channel_name" in df.columns and "value" in df.columns:
            # Long format with channel_name column — pivot it
            pivoted = df.pivot_table(
                index=df.index, columns="channel_name", values="value", aggfunc="mean"
            )
            pivoted.columns.name = None
            return pivoted.sort_index()

        # Wide format — every remaining column is a channel
        value_cols = [c for c in df.columns if c not in ("channel_name",)]
        if not value_cols:
            raise IngestionError("DataFrame has no value columns after removing 'timestamp'")
        return df[value_cols].apply(pd.to_numeric, errors="coerce")

    # ── Missing value handling ─────────────────────────────────────────────────

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then backward-fill all channels."""
        n_before = df.isna().sum().sum()
        df = df.ffill().bfill()
        n_after = df.isna().sum().sum()
        filled = int(n_before - n_after)
        if filled:
            logger.info("Filled %d missing value(s) via ffill→bfill", filled)
        if df.isna().any().any():
            logger.warning("Some NaNs remain after fill (channel may be entirely null)")
            df = df.fillna(0.0)
        return df

    # ── Normalisation ──────────────────────────────────────────────────────────

    def _normalise_channels(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Normalise each channel independently; return (normalised_df, per-channel stats).

        Uses population std (ddof=0) so that z-scored output has std == 1.0
        when measured with numpy's default ddof=0.
        """
        norm = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        stats: dict[str, dict[str, float]] = {}

        for ch in df.columns:
            arr = df[ch].to_numpy(dtype=float)
            ch_mean = float(np.mean(arr))
            ch_std = float(np.std(arr))          # ddof=0  — population std
            ch_min = float(np.min(arr))
            ch_max = float(np.max(arr))

            if self._norm == NormalizationMethod.ZSCORE:
                denom = ch_std if ch_std > 0 else 1.0
                norm[ch] = (arr - ch_mean) / denom
            else:  # MINMAX
                denom = (ch_max - ch_min) if (ch_max - ch_min) > 0 else 1.0
                norm[ch] = (arr - ch_min) / denom

            stats[ch] = {
                "mean": ch_mean,
                "std": ch_std,
                "min": ch_min,
                "max": ch_max,
            }

        return norm, stats

    # ── Windowing ──────────────────────────────────────────────────────────────

    def _segment(
        self,
        raw_df: pd.DataFrame,
        norm_df: pd.DataFrame,
        stats: dict[str, dict[str, float]],
        domain_hint: str | None,
    ) -> list[TimeSeriesWindow]:
        """Slice both DataFrames into fixed-length windows with optional overlap."""
        raw_arr = raw_df.to_numpy(dtype=float)
        norm_arr = norm_df.to_numpy(dtype=float)
        timestamps = raw_df.index.tolist()
        channels = list(raw_df.columns)
        n = len(raw_arr)
        step = self._window_size - self._overlap
        windows: list[TimeSeriesWindow] = []

        start = 0
        window_idx = 0
        while start + self._window_size <= n:
            end = start + self._window_size
            raw_win = raw_arr[start:end]
            norm_win = norm_arr[start:end]

            meta: dict[str, Any] = {
                "window_index": window_idx,
                "window_size": self._window_size,
                "overlap": self._overlap,
                "channels": channels,
                "start_timestamp": str(timestamps[start]),
                "end_timestamp": str(timestamps[end - 1]),
                "n_channels": len(channels),
                "normalization": self._norm.value,
                "channel_stats": stats,
            }

            windows.append(
                TimeSeriesWindow(
                    raw_data=raw_win,
                    normalized_data=norm_win,
                    metadata=meta,
                    domain_hint=domain_hint,
                )
            )
            start += step
            window_idx += 1

        if not windows:
            logger.warning(
                "No complete windows produced: %d rows < window_size %d",
                n, self._window_size,
            )

        return windows

    # ── Legacy API (used by agent.py and renderer.py) ─────────────────────────

    def process(self, series_list: list[TimeSeriesInput]) -> list[NormalizedSeries]:
        """Normalize Pydantic TimeSeriesInput objects → NormalizedSeries for rendering."""
        results: list[NormalizedSeries] = []
        for ts in series_list:
            try:
                results.append(self._process_one(ts))
            except Exception as exc:
                raise IngestionError(f"Failed to process series '{ts.name}': {exc}") from exc
        return results

    def _process_one(self, ts: TimeSeriesInput) -> NormalizedSeries:
        df = pd.DataFrame(
            {"timestamp": [p.timestamp for p in ts.points],
             "value": [p.value for p in ts.points]}
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")

        n_before = df["value"].isna().sum()
        df["value"] = df["value"].interpolate(method=self._interpolate_method)
        df["value"] = df["value"].ffill().bfill()
        n_imputed = int(n_before - df["value"].isna().sum())

        df = self._clip_outliers(df, ts.name)

        mean_, std_ = float(df["value"].mean()), float(df["value"].std())
        df["value_norm"] = (df["value"] - mean_) / std_ if std_ > 0 else 0.0

        stats: dict[str, float] = {
            "mean": mean_,
            "std": std_,
            "min": float(df["value"].min()),
            "max": float(df["value"].max()),
            "n_points": float(len(df)),
            "n_imputed": float(n_imputed),
        }
        logger.debug("Series '%s': %d pts, mean=%.4f, std=%.4f", ts.name, len(df), mean_, std_)
        return NormalizedSeries(name=ts.name, df=df, unit=ts.unit, stats=stats)

    def _clip_outliers(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        z = (df["value"] - df["value"].mean()) / (df["value"].std() + 1e-9)
        n_out = int((z.abs() > self._outlier_z).sum())
        if n_out:
            logger.warning("Series '%s': %d outlier(s) clipped", name, n_out)
            lo = df["value"].mean() - self._outlier_z * df["value"].std()
            hi = df["value"].mean() + self._outlier_z * df["value"].std()
            df = df.copy()
            df["value"] = df["value"].clip(lower=lo, upper=hi)
        return df


# ── Module-level helper (used by renderer.py) ──────────────────────────────────

def align_series(normalized: list[NormalizedSeries]) -> pd.DataFrame:
    """Outer-join multiple NormalizedSeries on a common DatetimeIndex."""
    if len(normalized) == 1:
        df = normalized[0].df[["value"]].copy()
        df.columns = [normalized[0].name]
        return df
    frames = {ns.name: ns.df["value"].rename(ns.name) for ns in normalized}
    combined = pd.concat(frames.values(), axis=1, join="outer").sort_index()
    return combined.ffill().bfill()
