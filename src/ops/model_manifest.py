"""Serving manifest utilities for routed VLM deployments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServingManifest:
    """Artifact contract used to configure one routed VLM deployment."""

    name: str
    model_name: str
    router_dir: Path
    chart_adapter_path: Path
    text_adapter_path: Path
    min_confidence: float = 0.55
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 384 * 28 * 28
    require_adapters: bool = True
    require_local_router_encoders: bool = False
    local_files_only: bool = False
    load_in_4bit: bool = False
    description: str | None = None
    training_source: str | None = None
    release_version: str | None = None
    python_version: str | None = None
    scikit_learn_version: str | None = None
    quality_gates: dict[str, Any] | None = None
    owner: str | None = None
    stage: str | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        project_root: str | Path = PROJECT_ROOT,
    ) -> "ServingManifest":
        """Load a JSON manifest and resolve repo-relative artifact paths."""
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = Path(project_root)

        required_fields = {
            "name",
            "model_name",
            "router_dir",
            "chart_adapter_path",
            "text_adapter_path",
        }
        missing = sorted(required_fields.difference(payload))
        if missing:
            raise ValueError(f"Manifest is missing required fields: {missing}")

        return cls(
            name=str(payload["name"]),
            model_name=str(payload["model_name"]),
            router_dir=_resolve_path(payload["router_dir"], root),
            chart_adapter_path=_resolve_path(payload["chart_adapter_path"], root),
            text_adapter_path=_resolve_path(payload["text_adapter_path"], root),
            min_confidence=float(payload.get("min_confidence", 0.55)),
            min_pixels=int(payload.get("min_pixels", 256 * 28 * 28)),
            max_pixels=int(payload.get("max_pixels", 384 * 28 * 28)),
            require_adapters=bool(payload.get("require_adapters", True)),
            require_local_router_encoders=bool(
                payload.get("require_local_router_encoders", False)
            ),
            local_files_only=bool(payload.get("local_files_only", False)),
            load_in_4bit=bool(payload.get("load_in_4bit", False)),
            description=payload.get("description"),
            training_source=payload.get("training_source"),
            release_version=payload.get("release_version"),
            python_version=payload.get("python_version"),
            scikit_learn_version=payload.get("scikit_learn_version"),
            quality_gates=payload.get("quality_gates"),
            owner=payload.get("owner"),
            stage=payload.get("stage"),
        )

    def validate_local_artifacts(self) -> list[str]:
        """Return missing artifact messages for this local deployment."""
        missing: list[str] = []

        if not self.router_dir.exists():
            missing.append(f"Missing router directory: {self.router_dir}")
        elif not (self.router_dir / "multimodal_logreg.joblib").exists():
            missing.append(
                "Missing router classifier: "
                f"{self.router_dir / 'multimodal_logreg.joblib'}"
            )
        elif self.require_local_router_encoders:
            for artifact_dir in (
                "text_tokenizer",
                "text_encoder",
                "image_processor",
                "image_encoder",
            ):
                path = self.router_dir / artifact_dir
                if not path.exists():
                    missing.append(f"Missing router artifact directory: {path}")

        if self.require_adapters:
            for label, path in (
                ("chart_adapter_path", self.chart_adapter_path),
                ("text_adapter_path", self.text_adapter_path),
            ):
                if not path.exists():
                    missing.append(f"Missing {label}: {path}")

        return missing

    def to_metadata(self) -> dict[str, Any]:
        """Serialize deployment metadata without loading model weights."""
        payload = asdict(self)
        for key in ("router_dir", "chart_adapter_path", "text_adapter_path"):
            payload[key] = str(payload[key])
        return payload


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path
