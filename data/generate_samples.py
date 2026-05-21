"""Generate synthetic sample datasets for development and testing.

Run: python data/generate_samples.py
Outputs CSV files to data/samples/.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

rng = random.Random(0)


def sinusoidal_with_anomalies(n: int = 720) -> pd.DataFrame:
    """Hourly CPU-like signal with injected spikes."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = base + timedelta(hours=i)
        daily = math.sin(2 * math.pi * i / 24) * 15
        weekly = math.sin(2 * math.pi * i / (24 * 7)) * 8
        noise = rng.gauss(0, 1.5)
        value = 45 + daily + weekly + noise
        if rng.random() < 0.02:
            value += rng.uniform(30, 60)
        rows.append({"timestamp": ts.isoformat(), "value": round(value, 3)})
    return pd.DataFrame(rows)


def trend_with_seasonality(n: int = 365) -> pd.DataFrame:
    """Daily revenue-like signal: upward trend + annual seasonality."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = base + timedelta(days=i)
        trend = i * 120
        season = math.sin(2 * math.pi * i / 365) * 15_000
        noise = rng.gauss(0, 3000)
        value = 200_000 + trend + season + noise
        rows.append({"timestamp": ts.isoformat(), "value": round(max(0, value), 2)})
    return pd.DataFrame(rows)


def two_correlated_series(n: int = 500) -> pd.DataFrame:
    """Two correlated minutely series (temperature + humidity)."""
    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    rows = []
    temp = 20.0
    hum = 60.0
    for i in range(n):
        ts = base + timedelta(minutes=i)
        temp += rng.gauss(0, 0.3)
        hum -= temp * 0.05 + rng.gauss(0, 0.5)
        hum = max(20, min(95, hum))
        temp = max(10, min(40, temp))
        rows.append({
            "timestamp": ts.isoformat(),
            "temperature": round(temp, 2),
            "humidity": round(hum, 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    datasets = {
        "cpu_anomalies.csv": sinusoidal_with_anomalies(),
        "revenue_trend.csv": trend_with_seasonality(),
        "temp_humidity.csv": two_correlated_series(),
    }
    for fname, df in datasets.items():
        path = SAMPLES_DIR / fname
        df.to_csv(path, index=False)
        print(f"Generated {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
