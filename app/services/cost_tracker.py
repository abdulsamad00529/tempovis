"""OpenAI call cost estimator and DB logger."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# USD per 1 million tokens (input, output) as of 2025-05
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-2024-04-09": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

_PER_MILLION = 1_000_000.0


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Return estimated USD cost for a single OpenAI API call."""
    base = model.split(":")[0]  # strip fine-tune suffixes
    input_rate, output_rate = _PRICING.get(base, _PRICING["gpt-4o"])
    return (input_tokens * input_rate + output_tokens * output_rate) / _PER_MILLION


async def log_cost(
    db: AsyncSession,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    endpoint: str,
    analysis_id: uuid.UUID | None = None,
) -> float:
    """Compute cost, persist a CostLog row, and return the cost in USD.

    Best-effort — logs a warning on failure so analysis path is never blocked.
    """
    from app.models.db import CostLog

    cost = estimate_cost(model, input_tokens, output_tokens)
    try:
        row = CostLog(
            analysis_id=analysis_id,
            endpoint=endpoint,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        db.add(row)
        await db.flush()
    except Exception as exc:
        logger.warning("CostLog write failed: %s", exc)

    return cost
