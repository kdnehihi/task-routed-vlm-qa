"""Tests for inference logging and serving metrics."""

from __future__ import annotations

import json
from pathlib import Path

from src.ops.inference_logging import (
    InferenceJsonlLogger,
    InferenceStats,
    image_info_from_upload,
)


def test_inference_jsonl_logger_writes_records(tmp_path: Path) -> None:
    log_path = tmp_path / "logs/inference.jsonl"
    logger = InferenceJsonlLogger(log_path)

    logger.log({"request_id": "abc", "answer": "42"})

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["request_id"] == "abc"
    assert record["answer"] == "42"
    assert "timestamp" in record


def test_inference_stats_tracks_successes_and_errors() -> None:
    stats = InferenceStats()

    stats.record_success("chartqa", "chart_backend", "chart_lora", 2.0)
    stats.record_error()
    snapshot = stats.snapshot()

    assert snapshot["total_requests"] == 2
    assert snapshot["total_errors"] == 1
    assert snapshot["success_count"] == 1
    assert snapshot["avg_latency_seconds"] == 2.0
    assert snapshot["task_counts"] == {"chartqa": 1}
    assert snapshot["adapter_counts"] == {"chart_lora": 1}


def test_image_info_from_upload_hashes_content() -> None:
    info = image_info_from_upload(
        path="/tmp/image.png",
        filename="image.png",
        content_type="image/png",
        content=b"abc",
    )

    assert info.size_bytes == 3
    assert info.sha256 == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
