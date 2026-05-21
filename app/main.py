"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.redis import close_redis, init_redis, ping_redis
from app.middleware.logging_middleware import RequestLoggingMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    logger.info("TempoVis starting up (env=%s)", cfg.app_env)

    if cfg.app_env == "development":
        try:
            from app.core.database import create_tables
            await create_tables()
            logger.info("Database tables verified")
        except Exception as exc:
            logger.warning("Could not connect to DB at startup: %s", exc)

    await init_redis()
    logger.info("Redis connection pool initialised")

    yield

    await close_redis()
    logger.info("TempoVis shutting down")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application with all middleware and routes."""
    cfg = get_settings()

    # Rate limiter — shared instance, also imported by routes.py
    limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

    app = FastAPI(
        title="TempoVis",
        description=(
            "Agentic multimodal time series intelligence — powered by GPT-4o vision. "
            "Renders time series as images and uses Chain-of-Thought VLM reasoning."
        ),
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if cfg.app_env != "production" else None,
        redoc_url="/redoc" if cfg.app_env != "production" else None,
    )

    # Attach limiter to app state so slowapi decorators find it
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Middleware — order matters: logging wraps CORS wraps handler
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if cfg.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health", tags=["ops"], summary="Liveness + dependency probe")
    async def health() -> dict:
        redis_ok = await ping_redis()
        return {
            "status": "ok",
            "service": "tempovis",
            "version": "0.2.0",
            "dependencies": {"redis": "ok" if redis_ok else "unavailable"},
        }

    @app.exception_handler(Exception)
    async def _global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception for %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "app.main:app",
        host=cfg.app_host,
        port=cfg.app_port,
        reload=cfg.app_env == "development",
        log_level=cfg.log_level.lower(),
    )
