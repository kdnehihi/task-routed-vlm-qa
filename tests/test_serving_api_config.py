"""Tests for API service configuration without loading model weights."""

from __future__ import annotations

import json
from pathlib import Path

from src.serving.config import build_service_from_environment


def test_build_service_from_manifest_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "api-test",
                "model_name": "custom/qwen",
                "router_dir": "router",
                "chart_adapter_path": "chart",
                "text_adapter_path": "text",
                "require_adapters": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROUTED_VLM_MANIFEST", str(manifest_path))

    service = build_service_from_environment()

    assert service.manifest is not None
    assert service.manifest.name == "api-test"
    assert service.model_name == "custom/qwen"
    assert service.require_adapters is False


def test_require_adapters_environment_overrides_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "api-test",
                "model_name": "custom/qwen",
                "router_dir": "router",
                "chart_adapter_path": "chart",
                "text_adapter_path": "text",
                "require_adapters": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROUTED_VLM_MANIFEST", str(manifest_path))
    monkeypatch.setenv("REQUIRE_ADAPTERS", "0")

    service = build_service_from_environment()

    assert service.require_adapters is False


def test_local_files_only_environment_overrides_service(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "api-test",
                "model_name": "custom/qwen",
                "router_dir": "router",
                "chart_adapter_path": "chart",
                "text_adapter_path": "text",
                "local_files_only": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROUTED_VLM_MANIFEST", str(manifest_path))
    monkeypatch.setenv("HF_LOCAL_FILES_ONLY", "1")

    service = build_service_from_environment()

    assert service.local_files_only is True


def test_device_and_quantization_environment_override_service(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROUTED_VLM_MANIFEST", raising=False)
    monkeypatch.setenv("ROUTED_VLM_DEVICE", "cuda")
    monkeypatch.setenv("ROUTED_VLM_LOAD_IN_4BIT", "1")

    service = build_service_from_environment()

    assert service.device == "cuda"
    assert service.load_in_4bit is True
