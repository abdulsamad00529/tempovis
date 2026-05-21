# tempovis-sdk

Python client for [TempoVis](https://github.com/abdulsamad00529/tempovis) —
agentic multimodal time series intelligence.

## Install

```bash
pip install tempovis-sdk
```

> **Note:** PyPI publish is on the roadmap. Until then, install from source:
> ```bash
> pip install -e ./sdk
> ```

## Quickstart

```python
from tempovis import TempoVis

client = TempoVis(base_url="http://localhost:8000")

# From a CSV file
result = client.analyze("metrics.csv", domain="ops")
print(result.anomalies)       # list[Anomaly]
print(result.explanation)     # chain-of-thought reasoning
print(result.confidence)      # float 0-1
print(result.trend)           # "up" | "down" | "flat" | "cyclical"
```

```python
import pandas as pd

df = pd.DataFrame({"timestamp": [...], "value": [...], "channel": [...]})
result = client.analyze(df, domain="clinical")
```

```python
# Async
import asyncio

async def run():
    result = await client.analyze_async("metrics.csv", domain="financial")
    return result

asyncio.run(run())
```

## Client Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `analyze` | `(data, domain="ops", use_agent=True) -> AnalysisResult` | Synchronous analysis |
| `analyze_async` | `(data, domain="ops", use_agent=True) -> AnalysisResult` | Async analysis |

**Constructor:**
```python
TempoVis(
    api_key: str | None = None,
    base_url: str = "http://localhost:8000",
    timeout: float = 120.0,
)
```

## Response Model

```python
class AnalysisResult:
    anomalies: list[Anomaly]          # detected anomalies
    explanation: str                   # full chain-of-thought
    confidence: float                  # 0.0 – 1.0
    trend: str                         # "up"|"down"|"flat"|"cyclical"
    forecast_direction: str            # "up"|"down"|"flat"|"uncertain"
    plot_url: str | None               # rendered plot artifact URL
    iterations_taken: int              # agentic loop iterations

class Anomaly:
    type: str        # "point"|"contextual"|"collective"
    severity: str    # "low"|"medium"|"high"
    timestamp_index: int
    description: str
```

## Exceptions

```python
from tempovis.exceptions import (
    TempoVisError,        # base — catch-all
    APIKeyError,          # 401/403
    QuotaExceededError,   # 429 — daily limit hit
    ConnectionError,      # server unreachable
    InvalidDataError,     # bad input file/DataFrame
)
```

## Links

- [Main repo & docs](https://github.com/abdulsamad00529/tempovis)
- [API reference](http://localhost:8000/docs)
- [Contributing](https://github.com/abdulsamad00529/tempovis/blob/main/CONTRIBUTING.md)
