"""TempoVis Python SDK — agentic multimodal time series intelligence.

Quickstart
----------
>>> from tempovis import TempoVis
>>> client = TempoVis(api_key="tv_...")
>>> result = client.analyze("metrics.csv", domain="ops")
>>> print(result.anomalies)
>>> print(result.explanation)

The client supports both file paths and ``pandas.DataFrame`` inputs, and
exposes both synchronous (:meth:`TempoVis.analyze`) and async
(:meth:`TempoVis.analyze_async`) interfaces.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["TempoVis", "AnalysisResult", "Anomaly"]

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from tempovis.exceptions import (
    APIKeyError,
    ConnectionError,
    InvalidDataError,
    QuotaExceededError,
    TempoVisError,
)
from tempovis.models import AnalysisResult, Anomaly

if TYPE_CHECKING:
    import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _df_to_series_payload(df: pd.DataFrame, domain: str) -> list[dict]:
    """Convert a DataFrame to the flat list-of-points format the API expects.

    The DataFrame must contain at minimum a ``value`` column. A ``timestamp``
    column is used when present; otherwise an ISO-8601 sequence is generated.
    An optional ``channel`` column is forwarded as-is.

    Parameters
    ----------
    df:     Input data frame.
    domain: Domain hint — forwarded to the API but not used for column mapping.

    Returns
    -------
    list[dict]
        Each element is ``{"timestamp": str, "value": float, "channel": str}``.
    """
    if "value" not in df.columns:
        raise InvalidDataError(
            "DataFrame must contain a 'value' column. "
            f"Columns present: {list(df.columns)}"
        )

    rows = []
    for i, row in df.iterrows():
        ts = (
            row["timestamp"].isoformat()
            if "timestamp" in df.columns
            else datetime.fromtimestamp(int(i), tz=UTC).isoformat()  # type: ignore[arg-type]
        )
        rows.append({
            "timestamp": str(ts),
            "value": float(row["value"]),
            "channel": str(row.get("channel", "value")),
        })
    return rows


def _csv_to_series_payload(path: str) -> list[dict]:
    """Read a CSV file and convert it to the API series-point format.

    Expected columns: ``timestamp`` (optional), ``value`` (required),
    ``channel`` (optional).

    Parameters
    ----------
    path: Absolute or relative path to the CSV file.

    Returns
    -------
    list[dict]
    """
    try:
        import pandas as pd  # optional at import time
    except ImportError as exc:
        raise InvalidDataError(
            "pandas is required to read CSV files. Install it with: pip install pandas"
        ) from exc

    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise InvalidDataError(f"File not found: {path}") from exc
    except Exception as exc:
        raise InvalidDataError(f"Failed to read CSV '{path}': {exc}") from exc

    return _df_to_series_payload(df, domain="")


def _parse_response(raw: dict) -> AnalysisResult:
    """Map the raw API JSON response to an :class:`AnalysisResult`.

    The API may return results under ``result`` (legacy agent path) or directly
    at the top level (new agentic path). Both shapes are handled here.
    """
    # Prefer the top-level result block when present
    data = raw.get("result") or raw

    # Normalise anomalies — the legacy API uses {"timestamp", "severity", "description"}
    # while the new API uses {"type", "severity", "timestamp_index"}.
    raw_anomalies = data.get("anomalies") or []
    anomalies: list[Anomaly] = []
    for a in raw_anomalies:
        anomalies.append(Anomaly(
            type=a.get("type", "point"),
            severity=a.get("severity", "low"),
            timestamp_index=int(a.get("timestamp_index", 0)),
            description=a.get("description", ""),
        ))

    # explanation comes from raw_reasoning (new API) or summary (legacy)
    explanation = (
        data.get("raw_reasoning")
        or data.get("summary")
        or raw.get("final_analysis") or ""
    )

    # trend / forecast
    trends = data.get("trends") or []
    trend_dir = data.get("trend") or (trends[0].get("direction") if trends else "flat") or "flat"
    forecast = data.get("forecast_direction", "uncertain")

    return AnalysisResult(
        anomalies=anomalies,
        explanation=str(explanation),
        confidence=float(data.get("confidence") or 0.0),
        trend=trend_dir,  # type: ignore[arg-type]
        forecast_direction=forecast,  # type: ignore[arg-type]
        plot_url=raw.get("plot_artifact_url"),
        iterations_taken=raw.get("iterations_taken", 1),
    )


def _raise_for_status(response: httpx.Response) -> None:
    """Translate HTTP error codes to typed SDK exceptions."""
    if response.status_code == 401:
        raise APIKeyError(
            "Invalid or missing API key. Pass api_key= to TempoVis().",
            status_code=401,
        )
    if response.status_code == 403:
        raise APIKeyError(
            "API key does not have permission for this request.",
            status_code=403,
        )
    if response.status_code == 429:
        raise QuotaExceededError(
            "Daily API call limit reached. Set DRY_RUN=true on the server "
            "or wait until the next UTC day.",
            status_code=429,
        )
    if response.status_code >= 500:
        raise TempoVisError(
            f"TempoVis server error ({response.status_code}): {response.text[:200]}",
            status_code=response.status_code,
        )
    if response.status_code >= 400:
        raise TempoVisError(
            f"Request error ({response.status_code}): {response.text[:200]}",
            status_code=response.status_code,
        )


# ── Public client ─────────────────────────────────────────────────────────────

class TempoVis:
    """Synchronous and async client for the TempoVis REST API.

    Parameters
    ----------
    api_key:  API key (prefix ``tv_``). Reserved for future auth — the
              current open-source server does not require one.
    base_url: Base URL of the running TempoVis backend.
    timeout:  Request timeout in seconds (default: 120).

    Examples
    --------
    Synchronous usage:

    >>> client = TempoVis(api_key="tv_...")
    >>> result = client.analyze("metrics.csv", domain="ops")
    >>> print(result.anomalies)

    Async usage:

    >>> import asyncio
    >>> async def run():
    ...     client = TempoVis()
    ...     return await client.analyze_async("metrics.csv", domain="clinical")
    >>> result = asyncio.run(run())
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    # ── Sync API ───────────────────────────────────────────────────────────────

    def analyze(
        self,
        data: str | pd.DataFrame,
        domain: str = "ops",
        use_agent: bool = True,
    ) -> AnalysisResult:
        """Analyse a time series and return a structured :class:`AnalysisResult`.

        Parameters
        ----------
        data:       Path to a CSV file **or** a ``pandas.DataFrame``.
                    Required columns: ``value``.
                    Optional columns: ``timestamp``, ``channel``.
        domain:     Domain hint for domain-aware plot rendering.
                    One of ``ops``, ``clinical``, ``financial``, ``iot``, ``energy``,
                    ``default``.
        use_agent:  Whether to run the full LangGraph self-critique loop.
                    Set ``False`` for a single-pass VLM call (faster, less accurate).

        Returns
        -------
        AnalysisResult
            Typed result with ``anomalies``, ``explanation``, ``confidence``,
            ``trend``, ``forecast_direction``, ``plot_url``, and
            ``iterations_taken``.

        Raises
        ------
        InvalidDataError:   Input file not found or missing required columns.
        APIKeyError:        Server rejected the API key (401/403).
        QuotaExceededError: Daily call limit reached (429).
        ConnectionError:    Backend is unreachable.
        TempoVisError:      Any other server-side error.
        """
        series = self._load(data)
        payload = json.dumps({"series": series, "domain": domain, "use_agent": use_agent})
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/api/v1/analyze",
                    content=payload,
                    headers=self._headers,
                )
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach TempoVis at {self._base_url}. "
                "Is the server running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Request timed out after {self._timeout}s."
            ) from exc

        _raise_for_status(response)
        return _parse_response(response.json())

    # ── Async API ──────────────────────────────────────────────────────────────

    async def analyze_async(
        self,
        data: str | pd.DataFrame,
        domain: str = "ops",
        use_agent: bool = True,
    ) -> AnalysisResult:
        """Async version of :meth:`analyze` — identical signature and return type.

        Parameters
        ----------
        data:       Path to a CSV file **or** a ``pandas.DataFrame``.
        domain:     Domain hint (``ops``, ``clinical``, ``financial``, etc.).
        use_agent:  Run the full agentic loop when ``True``.

        Returns
        -------
        AnalysisResult

        Raises
        ------
        Same exceptions as :meth:`analyze`.
        """
        series = self._load(data)
        payload = json.dumps({"series": series, "domain": domain, "use_agent": use_agent})
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/v1/analyze",
                    content=payload,
                    headers=self._headers,
                )
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach TempoVis at {self._base_url}. "
                "Is the server running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(f"Request timed out after {self._timeout}s.") from exc

        _raise_for_status(response)
        return _parse_response(response.json())

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load(self, data: str | pd.DataFrame) -> list[dict]:
        """Normalise ``data`` to the API series-point list format."""
        if isinstance(data, str):
            return _csv_to_series_payload(data)
        try:
            import pandas as pd  # noqa: PLC0415
            if isinstance(data, pd.DataFrame):
                return _df_to_series_payload(data, domain="")
        except ImportError:
            pass
        raise InvalidDataError(
            f"data must be a file path (str) or pandas DataFrame, got {type(data).__name__}"
        )
