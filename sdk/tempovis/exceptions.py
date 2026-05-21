"""Custom exception hierarchy for the TempoVis SDK."""

from __future__ import annotations


class TempoVisError(Exception):
    """Base exception for all TempoVis SDK errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIKeyError(TempoVisError):
    """Raised when the API key is missing, invalid, or rejected (HTTP 401/403)."""


class QuotaExceededError(TempoVisError):
    """Raised when the daily call limit is reached (HTTP 429)."""


class ConnectionError(TempoVisError):  # noqa: A001
    """Raised when the TempoVis backend cannot be reached."""


class InvalidDataError(TempoVisError):
    """Raised when the input data cannot be parsed or is structurally invalid."""
