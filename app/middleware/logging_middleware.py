"""ASGI middleware that logs endpoint, latency, status, and token cost."""

from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("tempovis.access")

# Request-scoped context key used by route handlers to attach token cost
TOKEN_COST_KEY = "token_cost_usd"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured access log: method, path, status, latency_ms, cost_usd."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.monotonic_ns()

        # Stash a mutable dict so route handlers can attach cost metadata
        request.state.token_cost_usd = 0.0
        request.state.input_tokens = 0
        request.state.output_tokens = 0

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.monotonic_ns() - start) // 1_000_000
            logger.error(
                "method=%s path=%s latency_ms=%d error=%s",
                request.method,
                request.url.path,
                elapsed,
                type(exc).__name__,
            )
            raise

        elapsed = (time.monotonic_ns() - start) // 1_000_000
        cost = getattr(request.state, TOKEN_COST_KEY, 0.0)

        logger.info(
            "method=%s path=%s status=%d latency_ms=%d "
            "cost_usd=%.6f in_tokens=%d out_tokens=%d",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            cost,
            getattr(request.state, "input_tokens", 0),
            getattr(request.state, "output_tokens", 0),
        )

        # Expose latency in response header (useful for dashboards)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        return response
