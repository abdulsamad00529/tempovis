# tempovis-sdk

Minimal Python client for the [TempoVis](https://github.com/abdulsamad00529/tempovis) API.

## Install

```bash
pip install tempovis-sdk   # coming soon — use the local path for now
# or from source:
pip install -e ./sdk
```

## Usage

```python
from tempovis import TempoVis

client = TempoVis(base_url="http://localhost:8000")
result = client.analyze("metrics.csv", domain="ops")

print(result["result"]["summary"])
print(result["result"]["anomalies"])
```

The SDK expects the CSV to have at minimum a `timestamp` and `value` column.
An optional `channel` column is used for multi-channel series.

## Status

This SDK is a stub — the full PyPI package (`tempovis-sdk`) is on the roadmap.
