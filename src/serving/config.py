"""Configuration helpers for routed VLM serving."""

from __future__ import annotations

import os

from src.ops.model_manifest import PROJECT_ROOT, ServingManifest
from src.serving.routed_vlm import RoutedVLMService


def build_service_from_environment() -> RoutedVLMService:
    """Build the service from env-configured artifacts or repository defaults."""
    manifest_path = os.getenv("ROUTED_VLM_MANIFEST")
    if manifest_path:
        manifest = ServingManifest.load(manifest_path, project_root=PROJECT_ROOT)
        service = RoutedVLMService.from_manifest(manifest)
    else:
        service = RoutedVLMService()

    if "REQUIRE_ADAPTERS" in os.environ:
        service.require_adapters = os.getenv("REQUIRE_ADAPTERS") not in {
            "0",
            "false",
            "False",
        }
    if "HF_LOCAL_FILES_ONLY" in os.environ:
        service.local_files_only = os.getenv("HF_LOCAL_FILES_ONLY") in {
            "1",
            "true",
            "True",
        }
    if "ROUTED_VLM_DEVICE" in os.environ:
        service.device = os.getenv("ROUTED_VLM_DEVICE") or None
    if "ROUTED_VLM_LOAD_IN_4BIT" in os.environ:
        service.load_in_4bit = os.getenv("ROUTED_VLM_LOAD_IN_4BIT") in {
            "1",
            "true",
            "True",
        }
    return service
