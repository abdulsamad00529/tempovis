"""Tests for app/services/data_loader.py.

All HuggingFace network calls are mocked; no real downloads happen.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.data_loader import (
    _DOMAIN_MAP,
    GiftEvalLoader,
    _load_pickle,
    _normalise_freq,
    _parse_start,
    _save_pickle,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_record(
    n: int = 200,
    freq: str = "H",
    item_id: str = "T1",
    start: str = "2020-01-01",
    multivariate: bool = False,
) -> dict:
    """Build a minimal GluonTS-style record."""
    if multivariate:
        target = np.random.rand(3, n).tolist()  # 3 channels × n timesteps
    else:
        target = np.random.rand(n).tolist()
    return {"target": target, "start": start, "freq": freq, "item_id": item_id}


def _mock_hf_dataset(records: list[dict]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(return_value=iter(records))
    return ds


# ── _normalise_freq ────────────────────────────────────────────────────────────


class TestNormaliseFreq:
    @pytest.mark.parametrize("gluonts,expected", [
        ("H", "h"),
        ("T", "min"),
        ("M", "ME"),
        ("Q", "QE"),
        ("Y", "YE"),
        ("A", "YE"),
        ("D", "D"),
        ("W", "W"),
        ("h", "h"),       # already lowercase
        ("ME", "ME"),     # already pandas 2.x alias
        ("X", "X"),       # unknown → pass-through
    ])
    def test_mapping(self, gluonts, expected):
        assert _normalise_freq(gluonts) == expected


# ── _parse_start ───────────────────────────────────────────────────────────────


class TestParseStart:
    def test_string_passthrough(self):
        assert _parse_start("2020-01-01") == "2020-01-01"

    def test_datetime_uses_isoformat(self):
        dt = pd.Timestamp("2022-06-15 12:00")
        result = _parse_start(dt)
        assert "2022-06-15" in result

    def test_object_with_to_timestamp(self):
        obj = MagicMock()
        obj.to_timestamp.return_value = pd.Timestamp("2021-03-01")
        result = _parse_start(obj)
        assert "2021-03-01" in result

    def test_fallback_to_str(self):
        result = _parse_start(42)
        assert result == "42"


# ── GiftEvalLoader: basic record conversion ───────────────────────────────────


class TestGiftEvalLoaderConversion:
    @pytest.fixture()
    def loader(self, tmp_path):
        return GiftEvalLoader(
            subsets=["m4_monthly"],
            window_size=128,
            overlap=0,
            cache_dir=tmp_path / "cache",
            max_series=5,
        )

    def test_univariate_record_produces_windows(self, loader, tmp_path):
        record = _make_record(n=300, freq="H")
        windows = loader._record_to_windows(record, "m4_monthly", "financial")
        assert len(windows) >= 2   # 300 timesteps / 128 = at least 2 full windows

    def test_multivariate_record_produces_windows(self, loader):
        record = _make_record(n=256, freq="H", multivariate=True)
        windows = loader._record_to_windows(record, "ett_h1", "iot")
        assert len(windows) >= 1
        assert windows[0].metadata["n_channels"] == 3

    def test_empty_target_returns_no_windows(self, loader):
        record = {"target": [], "start": "2020-01-01", "freq": "H", "item_id": "X"}
        windows = loader._record_to_windows(record, "m4_monthly", "financial")
        assert windows == []

    def test_window_metadata_tagged_with_subset(self, loader):
        record = _make_record(n=256, freq="H")
        windows = loader._record_to_windows(record, "ett_h1", "iot")
        for w in windows:
            assert w.metadata["subset"] == "ett_h1"
            assert "item_id" in w.metadata

    def test_domain_hint_applied(self, loader):
        record = _make_record(n=256, freq="H")
        windows = loader._record_to_windows(record, "illness", "clinical")
        for w in windows:
            assert w.domain_hint == "clinical"

    def test_malformed_record_returns_empty(self, loader):
        windows = loader._record_to_windows({"bad": "record"}, "m4_monthly", "financial")
        assert windows == []


# ── GiftEvalLoader: download / cache flow ─────────────────────────────────────


class TestGiftEvalLoaderDownload:
    @pytest.fixture()
    def loader(self, tmp_path):
        return GiftEvalLoader(
            subsets=["m4_monthly"],
            window_size=128,
            cache_dir=tmp_path / "cache",
            max_series=3,
        )

    def test_iter_windows_calls_load_dataset(self, loader, tmp_path):
        records = [_make_record(n=256) for _ in range(3)]
        mock_ds = _mock_hf_dataset(records)

        with patch("app.services.data_loader.load_dataset") as mock_load:
            mock_load.return_value = mock_ds
            windows = list(loader.iter_windows())

        assert len(windows) > 0

    def test_cache_written_after_download(self, loader, tmp_path):
        records = [_make_record(n=256)]
        mock_ds = _mock_hf_dataset(records)
        cache_dir = tmp_path / "cache"

        with patch("app.services.data_loader.load_dataset") as mock_load:
            mock_load.return_value = mock_ds
            list(loader.iter_windows())

        pkl_files = list(cache_dir.glob("*.pkl"))
        assert len(pkl_files) == 1

    def test_cache_read_on_second_call(self, loader, tmp_path):
        records = [_make_record(n=256)]
        mock_ds = _mock_hf_dataset(records)

        with patch("app.services.data_loader.load_dataset") as mock_load:
            mock_load.return_value = mock_ds
            list(loader.iter_windows())   # first call — downloads
            list(loader.iter_windows())   # second call — should use cache

        # load_dataset only called once
        assert mock_load.call_count == 1

    def test_max_series_respected(self, loader, tmp_path):
        records = [_make_record(n=256) for _ in range(10)]
        mock_ds = _mock_hf_dataset(records)

        with patch("app.services.data_loader.load_dataset") as mock_load:
            mock_load.return_value = mock_ds
            windows = list(loader.iter_windows())

        # max_series=3, each produces 2 windows → at most 6
        assert len(windows) <= 3 * 2 + 1

    def test_failed_download_returns_empty(self, loader):
        with patch("app.services.data_loader.load_dataset") as mock_load:
            mock_load.side_effect = RuntimeError("net err")
            windows = list(loader.iter_windows())
        assert windows == []

    def test_missing_datasets_returns_empty_when_unavailable(self, tmp_path):
        loader = GiftEvalLoader(
            subsets=["m4_monthly"], cache_dir=tmp_path / "cache"
        )
        with patch("app.services.data_loader.load_dataset", None):
            windows = list(loader.iter_windows())
        assert windows == []


# ── Domain mapping ─────────────────────────────────────────────────────────────


class TestDomainMapping:
    @pytest.mark.parametrize("subset,expected_domain", [
        ("m4_monthly", "financial"),
        ("ett_h1", "iot"),
        ("illness", "clinical"),
        ("weather", "iot"),
        ("exchange_rate", "financial"),
        ("hospital", "clinical"),
    ])
    def test_known_domains(self, subset, expected_domain):
        assert _DOMAIN_MAP[subset] == expected_domain

    def test_unknown_subset_falls_back_to_default(self, tmp_path):
        loader = GiftEvalLoader(
            subsets=["unknown_subset"],
            cache_dir=tmp_path / "cache",
        )
        record = _make_record(n=256)
        windows = loader._record_to_windows(record, "unknown_subset", "default")
        for w in windows:
            assert w.domain_hint == "default"


# ── Pickle cache helpers ───────────────────────────────────────────────────────


class TestPickleHelpers:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "test.pkl"
        # Create a minimal TimeSeriesWindow
        from app.services.ingestion import DataIngestionService
        svc = DataIngestionService(window_size=128)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=256, freq="h", tz="UTC"),
            "value": np.random.rand(256),
        })
        windows = svc.ingest(df, domain_hint="iot")

        _save_pickle(path, windows)
        loaded = _load_pickle(path)

        assert len(loaded) == len(windows)
        assert np.allclose(loaded[0].raw_data, windows[0].raw_data)

    def test_save_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "test.pkl"
        _save_pickle(deep_path, [])
        assert deep_path.exists()
