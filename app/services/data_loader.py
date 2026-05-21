"""GIFT-Eval data loader — downloads, caches, and segments HuggingFace benchmark datasets.

Downloads the Salesforce/gift-eval dataset (or any GluonTS-compatible HuggingFace dataset)
and converts each time series into ``TimeSeriesWindow`` objects ready for ingestion into
the TempoVis pipeline.

Usage
-----
    from app.services.data_loader import GiftEvalLoader

    loader = GiftEvalLoader(subsets=["m4_monthly", "ett_h1"], window_size=128)
    for window in loader.iter_windows():
        artifact = renderer.render(window)
        result   = await reasoner.analyze(artifact)

The first call per subset downloads from HuggingFace and writes a local pickle cache
in ``data/cache/``.  Subsequent calls read from cache and are instant.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional dependency — imported at module level so tests can patch it cleanly.
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]

from app.services.ingestion import DataIngestionService, NormalizationMethod, TimeSeriesWindow

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")
_HF_DATASET = "Salesforce/gift-eval"

# Domain labels per subset; unmapped subsets fall back to "default".
_DOMAIN_MAP: dict[str, str] = {
    # Financial / economic
    "m4_yearly": "financial",
    "m4_quarterly": "financial",
    "m4_monthly": "financial",
    "m4_weekly": "financial",
    "m4_daily": "financial",
    "m4_hourly": "financial",
    "exchange_rate": "financial",
    "nn5_daily_without_missing": "financial",
    "nn5_weekly": "financial",
    "car_parts_without_missing": "financial",
    "fred_md": "financial",
    "tourism_monthly": "financial",
    "tourism_quarterly": "financial",
    "tourism_yearly": "financial",
    "cif_2016": "financial",
    "uber_tlc_daily": "financial",
    "uber_tlc_hourly": "financial",
    # IoT / physical sensors
    "ett_h1": "iot",
    "ett_h2": "iot",
    "ett_m1": "iot",
    "ett_m2": "iot",
    "weather": "iot",
    "traffic": "iot",
    "electricity": "iot",
    "kdd_cup_2018_without_missing": "iot",
    "pedestrian_counts": "iot",
    "australian_electricity_demand": "iot",
    "solar_10_minutes": "iot",
    "solar_weekly": "iot",
    "wind_farms_without_missing": "iot",
    # Clinical / health
    "illness": "clinical",
    "hospital": "clinical",
    "covid_deaths": "clinical",
}

# A curated short list used when no subsets are specified
_DEFAULT_SUBSETS = [
    "m4_monthly",
    "ett_h1",
    "weather",
    "electricity",
    "exchange_rate",
    "illness",
]

# GluonTS → pandas frequency alias mapping (handles pandas 2.x renames)
_FREQ_MAP: dict[str, str] = {
    "T": "min",   # minutely
    "t": "min",
    "H": "h",     # hourly
    "h": "h",
    "D": "D",     # daily
    "W": "W",     # weekly
    "M": "ME",    # month-end (pandas 2.2+)
    "Q": "QE",    # quarter-end
    "Y": "YE",    # year-end
    "A": "YE",
    "S": "s",     # secondly
    "min": "min",
    "ME": "ME",
    "QE": "QE",
    "YE": "YE",
}


class GiftEvalLoader:
    """Downloads and segments GIFT-Eval time series as ``TimeSeriesWindow`` objects.

    Parameters
    ----------
    subsets:      Subset names to load.  Defaults to ``_DEFAULT_SUBSETS``.
    window_size:  Timesteps per window (default 128).
    overlap:      Overlap between consecutive windows (default 0).
    cache_dir:    Root directory for pickle caches (default ``data/cache/``).
    max_series:   Max number of series loaded per subset.  ``None`` = all.
    normalization: Z-score or min-max normalisation applied per channel.
    """

    def __init__(
        self,
        subsets: list[str] | None = None,
        window_size: int = 128,
        overlap: int = 0,
        cache_dir: str | Path = _CACHE_DIR,
        max_series: int | None = 10,
        normalization: NormalizationMethod = NormalizationMethod.ZSCORE,
    ) -> None:
        self._subsets = subsets or _DEFAULT_SUBSETS
        self._window_size = window_size
        self._overlap = overlap
        self._cache_dir = Path(cache_dir)
        self._max_series = max_series
        self._ingestion = DataIngestionService(
            normalization=normalization,
            window_size=window_size,
            overlap=overlap,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def iter_windows(self) -> Iterator[TimeSeriesWindow]:
        """Yield ``TimeSeriesWindow`` objects one at a time across all subsets."""
        for subset in self._subsets:
            yield from self._load_subset(subset)

    def load_all(self) -> list[TimeSeriesWindow]:
        """Materialise all windows into a list (use ``iter_windows`` for large sets)."""
        return list(self.iter_windows())

    @property
    def available_subsets(self) -> list[str]:
        """Return a sorted list of all known subset names."""
        return sorted(_DOMAIN_MAP)

    # ── Subset loading ─────────────────────────────────────────────────────────

    def _load_subset(self, subset: str) -> list[TimeSeriesWindow]:
        cache_path = self._cache_path(subset)
        if cache_path.exists():
            logger.info("Cache hit: '%s' <- %s", subset, cache_path)
            return _load_pickle(cache_path)

        logger.info("Downloading '%s' from HuggingFace (%s)...", subset, _HF_DATASET)
        try:
            windows = self._download_and_convert(subset)
        except Exception as exc:
            logger.error("Failed to load subset '%s': %s", subset, exc)
            return []

        _save_pickle(cache_path, windows)
        logger.info("Cached %d window(s) for '%s'", len(windows), subset)
        return windows

    def _download_and_convert(self, subset: str) -> list[TimeSeriesWindow]:
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' package is required for GiftEvalLoader. "
                "Install it with: pip install datasets"
            )

        # Try test split first; fall back to train if unavailable
        for split in ("test", "train"):
            try:
                hf_dataset = load_dataset(
                    _HF_DATASET,
                    name=subset,
                    split=split,
                    trust_remote_code=True,
                )
                logger.info("Loaded subset '%s' split='%s'", subset, split)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not load any split for subset '{subset}'")

        domain = _DOMAIN_MAP.get(subset, "default")
        windows: list[TimeSeriesWindow] = []
        series_seen = 0

        for record in hf_dataset:
            if self._max_series is not None and series_seen >= self._max_series:
                break
            new_windows = self._record_to_windows(record, subset, domain)
            windows.extend(new_windows)
            series_seen += 1

        logger.info(
            "Subset '%s': %d series → %d window(s)", subset, series_seen, len(windows)
        )
        return windows

    # ── Record conversion ──────────────────────────────────────────────────────

    def _record_to_windows(
        self, record: dict[str, Any], subset: str, domain: str
    ) -> list[TimeSeriesWindow]:
        try:
            return self._convert(record, subset, domain)
        except Exception as exc:
            logger.warning("Skipping malformed record in '%s': %s", subset, exc)
            return []

    def _convert(
        self, record: dict[str, Any], subset: str, domain: str
    ) -> list[TimeSeriesWindow]:
        target = record.get("target")
        if target is None or len(target) == 0:
            return []

        start_raw = record.get("start", "2000-01-01")
        freq_raw = str(record.get("freq") or record.get("period") or "H")
        item_id = str(record.get("item_id", "series"))
        freq = _normalise_freq(freq_raw)

        # Handle univariate (1-D) and multivariate (2-D: channels × timesteps)
        arr = np.array(target, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]        # → (1, n_timesteps)
        elif arr.ndim == 2:
            pass                            # already (n_channels, n_timesteps)
        else:
            logger.warning("Unexpected target shape %s in '%s'; skipping", arr.shape, subset)
            return []

        n_channels, n_timesteps = arr.shape
        channel_names = (
            [item_id] if n_channels == 1
            else [f"{item_id}_ch{i}" for i in range(n_channels)]
        )

        timestamps = pd.date_range(
            start=_parse_start(start_raw),
            periods=n_timesteps,
            freq=freq,
            tz="UTC",
        )
        df = pd.DataFrame({"timestamp": timestamps})
        for ch_name, ch_data in zip(channel_names, arr, strict=False):
            df[ch_name] = ch_data

        windows = self._ingestion.ingest(df, domain_hint=domain)
        for w in windows:
            w.metadata["subset"] = subset
            w.metadata["item_id"] = item_id
        return windows

    # ── Cache paths ────────────────────────────────────────────────────────────

    def _cache_path(self, subset: str) -> Path:
        key = f"{subset}__ws{self._window_size}__ov{self._overlap}"
        digest = hashlib.md5(key.encode()).hexdigest()[:8]
        return self._cache_dir / f"{subset}__{digest}.pkl"


# ── Module-level helpers ───────────────────────────────────────────────────────

def _normalise_freq(freq: str) -> str:
    """Map GluonTS / GIFT-Eval frequency strings to pandas 2.x offset aliases."""
    return _FREQ_MAP.get(freq.strip(), freq)


def _parse_start(start: Any) -> str:
    """Return an ISO date string from various GIFT-Eval start formats."""
    if isinstance(start, str):
        return start
    # GluonTS Period objects expose .to_timestamp() but not .isoformat();
    # check this first so we don't accidentally call .isoformat() on a Period.
    if hasattr(start, "to_timestamp"):
        return start.to_timestamp().isoformat()
    if hasattr(start, "isoformat"):
        return start.isoformat()
    return str(start)


def _load_pickle(path: Path) -> list[TimeSeriesWindow]:
    with path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301


def _save_pickle(path: Path, windows: list[TimeSeriesWindow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(windows, fh, protocol=pickle.HIGHEST_PROTOCOL)
