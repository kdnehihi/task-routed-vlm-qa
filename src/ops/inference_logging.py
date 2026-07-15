"""Runtime logging and in-memory metrics for routed VLM serving."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class UploadedImageInfo:
    """Metadata for an uploaded image saved to a temporary file."""

    path: str
    filename: str | None
    content_type: str | None
    sha256: str
    size_bytes: int


class InferenceJsonlLogger:
    """Append one compact JSON record for each inference request."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = Lock()

    def log(self, record: dict[str, Any]) -> None:
        """Write a JSONL record if logging is enabled."""
        if self.path is None:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


class InferenceStats:
    """Small process-local metrics store for health and debugging endpoints."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency_seconds = 0.0
        self.task_counts: Counter[str] = Counter()
        self.backend_counts: Counter[str] = Counter()
        self.adapter_counts: Counter[str] = Counter()

    def record_success(
        self,
        task_type: str,
        backend: str,
        adapter: str | None,
        latency_seconds: float,
    ) -> None:
        """Record one successful prediction."""
        with self._lock:
            self.total_requests += 1
            self.total_latency_seconds += latency_seconds
            self.task_counts[task_type] += 1
            self.backend_counts[backend] += 1
            self.adapter_counts[adapter or "base"] += 1

    def record_error(self) -> None:
        """Record one failed prediction."""
        with self._lock:
            self.total_requests += 1
            self.total_errors += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics snapshot."""
        with self._lock:
            success_count = self.total_requests - self.total_errors
            avg_latency = (
                self.total_latency_seconds / success_count
                if success_count > 0
                else None
            )
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "success_count": success_count,
                "avg_latency_seconds": avg_latency,
                "task_counts": dict(self.task_counts),
                "backend_counts": dict(self.backend_counts),
                "adapter_counts": dict(self.adapter_counts),
            }


def new_request_id() -> str:
    """Return an opaque request identifier for logs and responses."""
    return uuid4().hex


def image_info_from_upload(
    path: str,
    filename: str | None,
    content_type: str | None,
    content: bytes,
) -> UploadedImageInfo:
    """Build metadata for a saved upload without storing image bytes in logs."""
    return UploadedImageInfo(
        path=path,
        filename=filename,
        content_type=content_type,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def dataclass_to_dict(value) -> dict[str, Any]:
    """Serialize dataclasses used in serving logs."""
    return asdict(value)
