"""Quality gate helpers for routed VLM releases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import (
    build_prediction_records,
    evaluate_routing,
    summarize_quality_records_by_task,
)
from src.ops.model_manifest import ServingManifest


TASK_ALIASES = {
    "chart_qa": "chartqa",
    "chartqa": "chartqa",
    "document_qa": "docvqa",
    "docvqa": "docvqa",
    "image_vqa": "textvqa",
    "textvqa": "textvqa",
}


def load_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL prediction records from an evaluation run."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            clean_line = line.strip()
            if not clean_line:
                continue
            record = json.loads(clean_line)
            if "prediction" not in record:
                raise ValueError(f"Missing prediction on line {line_number}.")
            if "answers" not in record:
                raise ValueError(f"Missing answers on line {line_number}.")
            records.append(record)
    if not records:
        raise ValueError(f"No prediction records found in {path}.")
    return records


def evaluate_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate JSONL prediction records using project task-specific metrics."""
    predictions = [str(record["prediction"]) for record in records]
    references = [normalize_reference(record) for record in records]
    quality_records = build_prediction_records(predictions, references)
    report = summarize_quality_records_by_task(quality_records)

    predicted_tasks = [
        normalize_task_type(record["predicted_task_type"])
        for record in records
        if "predicted_task_type" in record
    ]
    if len(predicted_tasks) == len(records):
        report["routing"] = evaluate_routing(predicted_tasks, references)
    return report


def evaluate_release_gate(
    manifest: ServingManifest,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Compare an evaluation report against manifest quality gates."""
    gates = manifest.quality_gates or {}
    checks = []

    for gate_name, gate in sorted(gates.items()):
        metric = gate["metric"]
        min_score = float(gate["min_score"])
        actual = lookup_metric(report, gate_name, metric)
        passed = actual is not None and actual >= min_score
        checks.append(
            {
                "gate": gate_name,
                "metric": metric,
                "min_score": min_score,
                "actual_score": actual,
                "passed": passed,
            }
        )

    return {
        "manifest": manifest.to_metadata(),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def normalize_reference(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one prediction record into evaluator reference format."""
    reference = dict(record)
    reference["task_type"] = normalize_task_type(str(record.get("task_type", "")))
    reference["answers"] = list(record["answers"])
    return reference


def normalize_task_type(task_type: str) -> str:
    """Map legacy dataset labels to current routed task labels."""
    return TASK_ALIASES.get(task_type, task_type)


def lookup_metric(
    report: dict[str, Any],
    gate_name: str,
    metric: str,
) -> float | None:
    """Read a metric from the standard report shape."""
    if gate_name == "routing":
        value = report.get("routing", {}).get(metric)
    else:
        value = report.get("by_task", {}).get(gate_name, {}).get(metric)
    if value is None:
        return None
    return float(value)
