"""Prefetch HuggingFace assets used by one routed VLM serving manifest."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.model_manifest import ServingManifest


DEFAULT_ROUTER_TEXT_MODEL = "microsoft/deberta-v3-small"
DEFAULT_ROUTER_IMAGE_MODEL = "openai/clip-vit-base-patch32"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/serving_manifest.json",
        help="Serving manifest whose model assets should be prefetched.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HuggingFace cache directory. Defaults to HF_HOME/cache.",
    )
    parser.add_argument(
        "--skip-backbone",
        action="store_true",
        help="Only prefetch router encoder assets, not the Qwen backbone.",
    )
    parser.add_argument(
        "--skip-router",
        action="store_true",
        help="Only prefetch the Qwen backbone, not router encoder assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ServingManifest.load(args.manifest, project_root=PROJECT_ROOT)
    model_ids = []

    if not args.skip_router:
        router_config = load_router_config(manifest.router_dir)
        model_ids.extend(
            [
                router_config.get("text_model") or DEFAULT_ROUTER_TEXT_MODEL,
                router_config.get("image_model") or DEFAULT_ROUTER_IMAGE_MODEL,
            ]
        )

    if not args.skip_backbone:
        model_ids.append(manifest.model_name)

    seen = set()
    unique_model_ids = [model_id for model_id in model_ids if not (model_id in seen or seen.add(model_id))]
    for index, model_id in enumerate(unique_model_ids, start=1):
        print(f"[{index}/{len(unique_model_ids)}] Prefetching {model_id}", flush=True)
        prefetch_model(model_id, cache_dir=args.cache_dir)

    print("Prefetch complete.", flush=True)


def load_router_config(router_dir: Path) -> dict:
    """Read router config if it exists; otherwise return defaults."""
    for filename in ("router_config.json", "router_metadata.json"):
        path = router_dir / filename
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def prefetch_model(model_id: str, cache_dir: str | None = None) -> None:
    """Download a model snapshot into the HuggingFace cache."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
    )


if __name__ == "__main__":
    main()
