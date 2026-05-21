"""Demo: synthetic PAMAP2-style data rendered across 3 domain styles.

PAMAP2 (Physical Activity Monitoring) is a public dataset captured at 100 Hz
with IMU sensors on the wrist, chest, and ankle plus a chest-worn heart rate
monitor.  This script synthesises realistic walking-activity signals for 5
channels and renders 3 windows (clinical / iot / default) to PNG files.

Usage (from the project root):
    python scripts/demo_renderer.py
Output:
    demo_output/window_00_clinical.png
    demo_output/window_01_iot.png
    demo_output/window_02_default.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.services.ingestion import DataIngestionService, NormalizationMethod
from app.services.renderer import WindowRenderer

# ── Simulation parameters ──────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
SAMPLE_RATE = 100       # Hz — matches real PAMAP2 capture rate
DURATION_S = 5          # seconds of simulated walking data
N_SAMPLES = SAMPLE_RATE * DURATION_S   # 500 samples


# ── Synthetic PAMAP2-like data generator ───────────────────────────────────────

def make_pamap2_dataframe() -> pd.DataFrame:
    """Return a wide-format DataFrame with 5 PAMAP2-style sensor channels.

    Walking activity modelled as:
      heart_rate — slow sinusoidal drift around 82 bpm
      acc_x      — fore-aft gait oscillation at ~2 Hz
      acc_y      — lateral gait oscillation at ~2 Hz (90° phase)
      acc_z      — vertical axis: gravity (9.81) + bounce at ~2 Hz
      wrist_temp — slow upward thermal drift from 34.2 °C
    """
    t = np.linspace(0, DURATION_S, N_SAMPLES, endpoint=False)
    ts = pd.date_range(
        start="2024-01-15 09:00:00",
        periods=N_SAMPLES,
        freq="10ms",   # 100 Hz → 10 ms between samples
        tz="UTC",
    )

    heart_rate = (
        82
        + 4 * np.sin(2 * np.pi * 0.05 * t)
        + RNG.normal(0, 0.5, N_SAMPLES)
    )
    acc_x = (
        0.30 * np.sin(2 * np.pi * 2.0 * t)
        + RNG.normal(0, 0.05, N_SAMPLES)
    )
    acc_y = (
        0.20 * np.sin(2 * np.pi * 2.0 * t + np.pi / 2)
        + RNG.normal(0, 0.04, N_SAMPLES)
    )
    acc_z = (
        9.81
        + 0.5 * np.sin(2 * np.pi * 2.0 * t + np.pi)
        + RNG.normal(0, 0.08, N_SAMPLES)
    )
    wrist_temp = 34.2 + 0.05 * t + RNG.normal(0, 0.02, N_SAMPLES)

    return pd.DataFrame(
        {
            "timestamp":  ts,
            "heart_rate": heart_rate,
            "acc_x":      acc_x,
            "acc_y":      acc_y,
            "acc_z":      acc_z,
            "wrist_temp": wrist_temp,
        }
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path("demo_output")
    out_dir.mkdir(exist_ok=True)

    print("Generating synthetic PAMAP2-style data (walking, 100 Hz)…")
    df = make_pamap2_dataframe()
    n_channels = len(df.columns) - 1   # exclude timestamp
    print(f"  {len(df):,} rows × {n_channels} sensor channels")

    # Ingest → windows of 128 timesteps (= 1.28 s at 100 Hz)
    service = DataIngestionService(
        normalization=NormalizationMethod.ZSCORE,
        window_size=128,
        overlap=0,
    )
    windows = service.ingest(df, domain_hint="clinical")
    print(f"  -> {len(windows)} window(s) of 128 timesteps each")

    if len(windows) < 3:
        print(f"  Only {len(windows)} window(s) produced — need at least 3.")
        print("  Increase DURATION_S or decrease window_size and re-run.")
        sys.exit(1)

    renderer = WindowRenderer()

    # Render the first 3 windows under different domain styles
    demo_configs = [
        (0, "clinical"),
        (1, "iot"),
        (2, "default"),
    ]

    print("\nRendering windows…")
    for win_i, domain in demo_configs:
        window = windows[win_i]
        window.domain_hint = domain          # override for style demo

        artifact = renderer.render(window)

        path = out_dir / f"window_{win_i:02d}_{domain}.png"
        path.write_bytes(artifact.image_bytes)
        kb = artifact.metadata["image_size_bytes"] / 1024
        print(
            f"  [{domain:>8}]  window #{win_i}  ->  {path}  "
            f"({kb:.1f} KB,  {artifact.metadata['n_channels']} channels)"
        )

    print(f"\nDone - open {out_dir}/ to inspect the rendered plots.")


if __name__ == "__main__":
    main()
