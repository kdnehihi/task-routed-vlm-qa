"""Tests for serving manifest loading and artifact validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ops.model_manifest import ServingManifest


def write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def base_payload() -> dict:
    return {
        "name": "test-deployment",
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "router_dir": "router",
        "chart_adapter_path": "adapters/chart",
        "text_adapter_path": "adapters/text",
    }


def test_serving_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "manifest.json", base_payload())

    manifest = ServingManifest.load(manifest_path, project_root=tmp_path)

    assert manifest.name == "test-deployment"
    assert manifest.router_dir == tmp_path / "router"
    assert manifest.chart_adapter_path == tmp_path / "adapters/chart"
    assert manifest.require_adapters is True
    assert manifest.local_files_only is False
    assert manifest.load_in_4bit is False


def test_serving_manifest_loads_local_files_only_flag(tmp_path: Path) -> None:
    payload = base_payload()
    payload["local_files_only"] = True
    payload["load_in_4bit"] = True
    manifest_path = write_manifest(tmp_path / "manifest.json", payload)

    manifest = ServingManifest.load(manifest_path, project_root=tmp_path)

    assert manifest.local_files_only is True
    assert manifest.load_in_4bit is True


def test_serving_manifest_reports_missing_artifacts(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "manifest.json", base_payload())
    manifest = ServingManifest.load(manifest_path, project_root=tmp_path)

    missing = manifest.validate_local_artifacts()

    assert f"Missing router directory: {tmp_path / 'router'}" in missing
    assert f"Missing chart_adapter_path: {tmp_path / 'adapters/chart'}" in missing
    assert f"Missing text_adapter_path: {tmp_path / 'adapters/text'}" in missing


def test_serving_manifest_accepts_complete_local_artifacts(tmp_path: Path) -> None:
    (tmp_path / "router").mkdir()
    (tmp_path / "router/multimodal_logreg.joblib").write_text("fake")
    (tmp_path / "adapters/chart").mkdir(parents=True)
    (tmp_path / "adapters/text").mkdir(parents=True)
    manifest_path = write_manifest(tmp_path / "manifest.json", base_payload())
    manifest = ServingManifest.load(manifest_path, project_root=tmp_path)

    assert manifest.validate_local_artifacts() == []


def test_serving_manifest_can_skip_adapter_validation(tmp_path: Path) -> None:
    payload = base_payload()
    payload["require_adapters"] = False
    (tmp_path / "router").mkdir()
    (tmp_path / "router/multimodal_logreg.joblib").write_text("fake")
    manifest_path = write_manifest(tmp_path / "manifest.json", payload)
    manifest = ServingManifest.load(manifest_path, project_root=tmp_path)

    assert manifest.validate_local_artifacts() == []


def test_serving_manifest_requires_core_fields(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "manifest.json", {"name": "broken"})

    with pytest.raises(ValueError, match="missing required fields"):
        ServingManifest.load(manifest_path, project_root=tmp_path)
