"""TempoVis Python SDK — thin client for the TempoVis API."""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["TempoVis"]


class TempoVis:
    """Minimal client for the TempoVis REST API.

    Parameters
    ----------
    base_url:   Base URL of a running TempoVis backend, e.g. ``http://localhost:8000``.
    api_key:    Optional API key (reserved for future auth support).

    Example
    -------
    >>> client = TempoVis(base_url="http://localhost:8000")
    >>> result = client.analyze("metrics.csv", domain="ops")
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def analyze(self, csv_path: str, domain: str = "default") -> dict:
        """Send a CSV file to /api/v1/analyze and return the structured result.

        This is a synchronous convenience wrapper. For async usage import
        ``httpx`` directly and call the API endpoint.
        """
        import csv
        import json
        import urllib.request
        from datetime import datetime, timezone

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"CSV file is empty: {csv_path}")

        # Expect columns: timestamp, value (and optionally channel)
        series = [
            {
                "timestamp": row.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "value": float(row.get("value", 0)),
                "channel": row.get("channel", "value"),
            }
            for row in rows
        ]

        payload = json.dumps({"series": series, "domain": domain, "use_agent": True}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/api/v1/analyze",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
