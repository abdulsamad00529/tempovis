"""Async Redis client — thin wrapper around redis-py with lifecycle management."""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_pool: aioredis.Redis | None = None


async def init_redis() -> None:
    global _pool
    cfg = get_settings()
    _pool = aioredis.from_url(
        cfg.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )


async def get_redis() -> aioredis.Redis:
    """FastAPI dependency — yield the shared Redis client."""
    if _pool is None:
        raise RuntimeError("Redis not initialised; call init_redis() first")
    return _pool


async def ping_redis() -> bool:
    """Return True if Redis is reachable, False otherwise."""
    try:
        client = await get_redis()
        return await client.ping()
    except Exception as exc:
        logger.warning("Redis ping failed: %s", exc)
        return False


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
