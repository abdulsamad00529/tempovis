"""Tests for app/services/ingestion.py — covers both the new windowed API
and the legacy process() API used by agent.py / renderer.py."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.models.schemas import TimeSeriesInput, TimeSeriesPoint
from app.services.ingestion import (
    DataIngestionService,
    IngestionError,
    NormalizationMethod,
    NormalizedSeries,
    TimeSeriesWindow,
    align_series,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_records(
    n: int,
    channel: str = "sensor_a",
    start_hour: int = 0,
    value_fn=None,
) -> list[dict]:
    """Build a list of {timestamp, value, channel_name} dicts."""
    if value_fn is None:
        value_fn = lambda i: float(i)
    return [
        {
            "timestamp": (BASE + timedelta(hours=start_hour + i)).isoformat(),
            "value": value_fn(i),
            "channel_name": channel,
        }
        for i in range(n)
    ]


def _make_df(n: int, channels: list[str] | None = None) -> pd.DataFrame:
    """Build a wide-format DataFrame with one column per channel."""
    channels = channels or ["ch_a"]
    idx = [BASE + timedelta(hours=i) for i in range(n)]
    data = {ch: [float(i + j) for i, j in enumerate([hash(ch) % 10] * n)] for ch in channels}
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    return df.reset_index()


def _make_pts(values: list[float], start_hour: int = 0) -> list[TimeSeriesPoint]:
    return [
        TimeSeriesPoint(timestamp=BASE + timedelta(hours=start_hour + i), value=v)
        for i, v in enumerate(values)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TimeSeriesWindow dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeSeriesWindow:
    def test_fields_present(self):
        win = TimeSeriesWindow(
            raw_data=np.zeros((128, 2)),
            normalized_data=np.zeros((128, 2)),
            metadata={"channels": ["a", "b"]},
            domain_hint="financial",
        )
        assert win.raw_data.shape == (128, 2)
        assert win.normalized_data.shape == (128, 2)
        assert win.domain_hint == "financial"
        assert win.metadata["channels"] == ["a", "b"]

    def test_domain_hint_optional(self):
        win = TimeSeriesWindow(
            raw_data=np.zeros((10, 1)),
            normalized_data=np.zeros((10, 1)),
            metadata={},
        )
        assert win.domain_hint is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Input parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestParsing:
    def setup_method(self):
        self.svc = DataIngestionService(window_size=10)

    # ── list of dicts ──────────────────────────────────────────────────────────

    def test_list_of_dicts_single_channel(self):
        records = _make_records(20, channel="temp")
        windows = self.svc.ingest(records)
        assert len(windows) == 2
        assert windows[0].metadata["channels"] == ["temp"]

    def test_list_of_dicts_multi_channel(self):
        records = _make_records(20, channel="a") + _make_records(20, channel="b")
        windows = self.svc.ingest(records)
        assert windows[0].metadata["n_channels"] == 2
        assert set(windows[0].metadata["channels"]) == {"a", "b"}

    def test_list_of_dicts_missing_key_raises(self):
        bad = [{"timestamp": "2024-01-01", "value": 1.0}]  # no channel_name
        with pytest.raises(IngestionError, match="missing required keys"):
            self.svc.ingest(bad)

    def test_empty_list_raises(self):
        with pytest.raises(IngestionError):
            self.svc.ingest([])

    def test_unsupported_type_raises(self):
        with pytest.raises(IngestionError):
            self.svc.ingest("not a list or dataframe")  # type: ignore

    # ── DataFrame ─────────────────────────────────────────────────────────────

    def test_wide_dataframe(self):
        df = _make_df(20, channels=["alpha", "beta"])
        windows = self.svc.ingest(df)
        assert windows[0].metadata["n_channels"] == 2

    def test_long_dataframe_with_channel_col(self):
        records = _make_records(20, channel="x") + _make_records(20, channel="y")
        df = pd.DataFrame(records)
        windows = self.svc.ingest(df)
        assert windows[0].metadata["n_channels"] == 2

    def test_dataframe_missing_timestamp_raises(self):
        df = pd.DataFrame({"value": [1.0, 2.0], "channel_name": ["a", "a"]})
        with pytest.raises(IngestionError, match="timestamp"):
            self.svc.ingest(df)

    def test_dataframe_no_value_cols_raises(self):
        df = pd.DataFrame({"timestamp": [BASE.isoformat()]})
        with pytest.raises(IngestionError):
            self.svc.ingest(df)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Missing value handling
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingValues:
    def test_ffill_then_bfill(self):
        records = _make_records(20, channel="s")
        # Inject NaN by using float('nan')
        records[5]["value"] = float("nan")
        records[10]["value"] = float("nan")
        svc = DataIngestionService(window_size=20)
        windows = svc.ingest(records)
        assert not np.isnan(windows[0].raw_data).any(), "NaNs survived fill"

    def test_leading_nan_filled_by_bfill(self):
        records = _make_records(20, channel="s")
        records[0]["value"] = float("nan")  # leading NaN — only bfill can fix
        svc = DataIngestionService(window_size=20)
        windows = svc.ingest(records)
        assert not np.isnan(windows[0].raw_data).any()

    def test_trailing_nan_filled_by_ffill(self):
        records = _make_records(20, channel="s")
        records[-1]["value"] = float("nan")
        svc = DataIngestionService(window_size=20)
        windows = svc.ingest(records)
        assert not np.isnan(windows[0].raw_data).any()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Normalisation
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalisation:

    def test_zscore_mean_zero(self):
        import math as _math
        records = _make_records(128, value_fn=lambda i: float(i))
        svc = DataIngestionService(normalization=NormalizationMethod.ZSCORE, window_size=128)
        windows = svc.ingest(records)
        col = windows[0].normalized_data[:, 0]
        assert abs(col.mean()) < 1e-9

    def test_zscore_std_one(self):
        records = _make_records(128, value_fn=lambda i: float(i))
        svc = DataIngestionService(normalization=NormalizationMethod.ZSCORE, window_size=128)
        windows = svc.ingest(records)
        col = windows[0].normalized_data[:, 0]
        assert abs(col.std() - 1.0) < 1e-6

    def test_minmax_range_zero_to_one(self):
        records = _make_records(128, value_fn=lambda i: float(i * 3 + 7))
        svc = DataIngestionService(normalization=NormalizationMethod.MINMAX, window_size=128)
        windows = svc.ingest(records)
        col = windows[0].normalized_data[:, 0]
        assert col.min() >= 0.0 - 1e-9
        assert col.max() <= 1.0 + 1e-9

    def test_minmax_min_is_zero(self):
        records = _make_records(128, value_fn=lambda i: float(i))
        svc = DataIngestionService(normalization=NormalizationMethod.MINMAX, window_size=128)
        windows = svc.ingest(records)
        col = windows[0].normalized_data[:, 0]
        assert abs(col.min()) < 1e-9

    def test_minmax_max_is_one(self):
        records = _make_records(128, value_fn=lambda i: float(i))
        svc = DataIngestionService(normalization=NormalizationMethod.MINMAX, window_size=128)
        windows = svc.ingest(records)
        col = windows[0].normalized_data[:, 0]
        assert abs(col.max() - 1.0) < 1e-9

    def test_constant_channel_no_crash(self):
        """A constant channel has std=0 / range=0 — must not divide by zero."""
        records = _make_records(128, value_fn=lambda i: 5.0)
        for method in NormalizationMethod:
            svc = DataIngestionService(normalization=method, window_size=128)
            windows = svc.ingest(records)
            assert not np.isnan(windows[0].normalized_data).any()

    def test_normalisation_per_channel_independent(self):
        """Two channels with different scales must each be normalised independently."""
        recs_a = _make_records(128, channel="small", value_fn=lambda i: float(i))
        recs_b = _make_records(128, channel="large", value_fn=lambda i: float(i * 1000))
        svc = DataIngestionService(normalization=NormalizationMethod.ZSCORE, window_size=128)
        windows = svc.ingest(recs_a + recs_b)
        ch_idx = windows[0].metadata["channels"]
        idx_small = ch_idx.index("small")
        idx_large = ch_idx.index("large")
        # Both should have ~zero mean after z-score
        assert abs(windows[0].normalized_data[:, idx_small].mean()) < 1e-9
        assert abs(windows[0].normalized_data[:, idx_large].mean()) < 1e-9

    def test_raw_data_unchanged_by_normalisation(self):
        values = list(range(128))
        records = _make_records(128, value_fn=lambda i: float(values[i]))
        svc = DataIngestionService(normalization=NormalizationMethod.ZSCORE, window_size=128)
        windows = svc.ingest(records)
        raw_vals = windows[0].raw_data[:, 0]
        np.testing.assert_allclose(raw_vals, np.array(values, dtype=float))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Windowing
# ══════════════════════════════════════════════════════════════════════════════

class TestWindowing:

    def test_window_count_no_overlap(self):
        records = _make_records(256)
        svc = DataIngestionService(window_size=128, overlap=0)
        windows = svc.ingest(records)
        assert len(windows) == 2

    def test_window_count_with_overlap(self):
        # step = 128 - 64 = 64; windows = floor((256 - 128) / 64) + 1 = 3
        records = _make_records(256)
        svc = DataIngestionService(window_size=128, overlap=64)
        windows = svc.ingest(records)
        assert len(windows) == 3

    def test_window_shape(self):
        records = _make_records(200, channel="a") + _make_records(200, channel="b")
        svc = DataIngestionService(window_size=50, overlap=0)
        windows = svc.ingest(records)
        assert windows[0].raw_data.shape == (50, 2)
        assert windows[0].normalized_data.shape == (50, 2)

    def test_insufficient_data_returns_empty(self):
        records = _make_records(10)
        svc = DataIngestionService(window_size=128, overlap=0)
        windows = svc.ingest(records)
        assert windows == []

    def test_exact_fit(self):
        records = _make_records(128)
        svc = DataIngestionService(window_size=128, overlap=0)
        windows = svc.ingest(records)
        assert len(windows) == 1

    def test_overlap_data_continuity(self):
        """The tail of window N and the head of window N+1 must share `overlap` rows."""
        records = _make_records(200, value_fn=lambda i: float(i))
        overlap = 20
        svc = DataIngestionService(window_size=50, overlap=overlap)
        windows = svc.ingest(records)
        assert len(windows) >= 2
        tail = windows[0].raw_data[-overlap:]
        head = windows[1].raw_data[:overlap]
        np.testing.assert_array_equal(tail, head)

    def test_window_index_metadata(self):
        records = _make_records(300)
        svc = DataIngestionService(window_size=100, overlap=0)
        windows = svc.ingest(records)
        for i, w in enumerate(windows):
            assert w.metadata["window_index"] == i

    def test_metadata_timestamps_present(self):
        records = _make_records(128)
        svc = DataIngestionService(window_size=128)
        windows = svc.ingest(records)
        assert "start_timestamp" in windows[0].metadata
        assert "end_timestamp" in windows[0].metadata

    def test_domain_hint_propagated(self):
        records = _make_records(128)
        svc = DataIngestionService(window_size=128)
        windows = svc.ingest(records, domain_hint="clinical")
        assert all(w.domain_hint == "clinical" for w in windows)

    def test_domain_hint_none_by_default(self):
        records = _make_records(128)
        svc = DataIngestionService(window_size=128)
        windows = svc.ingest(records)
        assert windows[0].domain_hint is None

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            DataIngestionService(window_size=1)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            DataIngestionService(window_size=10, overlap=10)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            DataIngestionService(window_size=10, overlap=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Legacy process() API
# ══════════════════════════════════════════════════════════════════════════════

class TestLegacyProcessAPI:
    def setup_method(self):
        self.svc = DataIngestionService()

    def test_basic_normalization(self):
        ts = TimeSeriesInput(name="temp", points=_make_pts([10.0, 20.0, 30.0]))
        results = self.svc.process([ts])
        assert len(results) == 1
        ns = results[0]
        assert ns.name == "temp"
        assert len(ns.df) == 3
        assert "value" in ns.df.columns
        assert "value_norm" in ns.df.columns

    def test_zscore_mean_zero(self):
        values = [float(i) for i in range(10)]
        ts = TimeSeriesInput(name="x", points=_make_pts(values))
        ns = self.svc.process([ts])[0]
        assert abs(ns.df["value_norm"].mean()) < 1e-6

    def test_outlier_clipping(self):
        values = [1.0] * 20 + [9999.0]
        ts = TimeSeriesInput(name="outlier", points=_make_pts(values))
        ns = self.svc.process([ts])[0]
        assert ns.df["value"].max() < 9999.0

    def test_stats_populated(self):
        ts = TimeSeriesInput(name="s", points=_make_pts([1.0, 2.0, 3.0, 4.0, 5.0]))
        ns = self.svc.process([ts])[0]
        assert ns.stats["n_points"] == 5.0
        assert math.isclose(ns.stats["mean"], 3.0, rel_tol=1e-5)

    def test_duplicate_timestamps_deduped(self):
        pts = _make_pts([1.0, 2.0, 3.0])
        ts = TimeSeriesInput(name="dup", points=pts + [pts[0]])
        ns = self.svc.process([ts])[0]
        assert len(ns.df) == 3

    def test_multiple_series(self):
        s1 = TimeSeriesInput(name="a", points=_make_pts([1.0, 2.0, 3.0]))
        s2 = TimeSeriesInput(name="b", points=_make_pts([4.0, 5.0, 6.0]))
        results = self.svc.process([s1, s2])
        assert len(results) == 2
        assert {r.name for r in results} == {"a", "b"}

    def test_empty_series_schema_raises(self):
        with pytest.raises(Exception):
            TimeSeriesInput(name="empty", points=[])

    def test_single_point_schema_raises(self):
        with pytest.raises(Exception):
            TimeSeriesInput(name="one", points=_make_pts([42.0])[:1])


# ══════════════════════════════════════════════════════════════════════════════
# 7. align_series helper
# ══════════════════════════════════════════════════════════════════════════════

class TestAlignSeries:
    def test_single_series_passthrough(self):
        svc = DataIngestionService()
        ts = TimeSeriesInput(name="x", points=_make_pts([1.0, 2.0, 3.0]))
        ns = svc.process([ts])
        aligned = align_series(ns)
        assert "x" in aligned.columns
        assert len(aligned) == 3

    def test_multi_series_no_nulls(self):
        svc = DataIngestionService()
        s1 = TimeSeriesInput(name="a", points=_make_pts([1.0, 2.0, 3.0]))
        s2 = TimeSeriesInput(name="b", points=_make_pts([4.0, 5.0, 6.0]))
        ns = svc.process([s1, s2])
        aligned = align_series(ns)
        assert "a" in aligned.columns
        assert "b" in aligned.columns
        assert not aligned.isnull().any().any()
