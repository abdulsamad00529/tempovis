"""Plot artifact storage: local filesystem, MinIO, or AWS S3.

STORAGE_BACKEND=local  → saves to local_plots_dir, returns /plots/<key> URL
STORAGE_BACKEND=minio  → MinIO via boto3, returns presigned GET URL
STORAGE_BACKEND=s3     → AWS S3 via boto3, returns presigned GET URL
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    async def upload(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        """Upload data and return a publicly accessible (or presigned) URL."""
        ...


# ── Local filesystem ───────────────────────────────────────────────────────────


class LocalStorage:
    def __init__(self, plots_dir: str) -> None:
        self._dir = Path(plots_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    async def upload(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        dest = self._dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return f"/plots/{key}"


# ── MinIO / S3 (boto3) ────────────────────────────────────────────────────────


class S3Storage:
    """Works for both MinIO (endpoint_url set) and real AWS S3 (endpoint_url=None)."""

    def __init__(
        self,
        bucket: str,
        *,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        secure: bool = True,
        presigned_expiry: int = 3600,
    ) -> None:
        import boto3  # type: ignore[import-untyped]

        scheme = "https" if secure else "http"
        url = f"{scheme}://{endpoint_url}" if endpoint_url else None

        self._bucket = bucket
        self._expiry = presigned_expiry
        self._s3 = boto3.client(
            "s3",
            endpoint_url=url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except Exception:
            try:
                self._s3.create_bucket(Bucket=self._bucket)
            except Exception as exc:
                logger.warning("Could not create bucket %s: %s", self._bucket, exc)

    async def upload(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            ),
        )
        loop2 = asyncio.get_event_loop()
        url: str = await loop2.run_in_executor(
            None,
            lambda: self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=self._expiry,
            ),
        )
        return url


# ── Factory ───────────────────────────────────────────────────────────────────

_instance: LocalStorage | S3Storage | None = None


def get_storage() -> LocalStorage | S3Storage:
    global _instance
    if _instance is not None:
        return _instance

    from app.core.config import get_settings

    cfg = get_settings()

    if cfg.storage_backend == "local":
        _instance = LocalStorage(cfg.local_plots_dir)
    elif cfg.storage_backend == "minio":
        _instance = S3Storage(
            bucket=cfg.minio_bucket,
            access_key=cfg.minio_access_key,
            secret_key=cfg.minio_secret_key,
            endpoint_url=cfg.minio_endpoint,
            secure=cfg.minio_secure,
        )
    else:  # s3
        _instance = S3Storage(
            bucket=cfg.aws_s3_bucket,
            access_key=cfg.aws_access_key_id or "",
            secret_key=cfg.aws_secret_access_key or "",
            region=cfg.aws_region,
        )

    return _instance
