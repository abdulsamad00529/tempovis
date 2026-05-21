"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def configure_logging(level: str = "INFO") -> None:
    """Set up root logger with a clean format for both dev and prod."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
    handler = logging.StreamHandler(stream)
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpcore", "httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
