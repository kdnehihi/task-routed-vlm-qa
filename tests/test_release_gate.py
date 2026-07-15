"""Tests for routed VLM release quality gates."""

from __future__ import annotations

from pathlib import Path

from src.ops.model_manifest import ServingManifest
from src.ops.release_gate import evaluate_prediction_records, evaluate_release_gate


def test_evaluate_prediction_records_normalizes_legacy_task_names() -> None:
    records = [
        {
            "task_type": "chart_qa",
            "question": "Value?",
            "answers": ["19346"],
            "prediction": "19346.08",
            "predicted_task_type": "chartqa",
        },
        {
            "task_type": "document_qa",
            "question": "Brand?",
            "answers": ["Bausch & Lomb"],
            "prediction": "Bausch Lomb",
            "predicted_task_type": "docvqa",
        },
        {
            "task_type": "image_vqa",
            "question": "Wine?",
            "answers": ["beauregard", "beauregard", "chablis"],
            "prediction": "beauregard",
            "predicted_task_type": "textvqa",
        },
    ]

    report = evaluate_prediction_records(records)

    assert report["by_task"]["chartqa"]["chart_hybrid_accuracy"] == 1.0
    assert report["by_task"]["docvqa"]["docvqa_anls"] > 0.5
    assert report["by_task"]["textvqa"]["textvqa_vqa_score"] == 2 / 3
    assert report["routing"]["routing_accuracy"] == 1.0


def test_evaluate_release_gate_compares_manifest_thresholds(tmp_path: Path) -> None:
    manifest = ServingManifest(
        name="test",
        model_name="model",
        router_dir=tmp_path / "router",
        chart_adapter_path=tmp_path / "chart",
        text_adapter_path=tmp_path / "text",
        quality_gates={
            "chartqa": {"metric": "chart_hybrid_accuracy", "min_score": 0.9},
            "routing": {"metric": "routing_accuracy", "min_score": 0.95},
        },
    )
    report = {
        "by_task": {"chartqa": {"chart_hybrid_accuracy": 1.0}},
        "routing": {"routing_accuracy": 0.8},
    }

    gate_report = evaluate_release_gate(manifest, report)

    assert gate_report["passed"] is False
    assert gate_report["checks"][0]["passed"] is True
    assert gate_report["checks"][1]["passed"] is False
