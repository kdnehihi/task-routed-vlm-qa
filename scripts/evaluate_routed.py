"""Generate routed VLM predictions for release-gate evaluation."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import VQADataset
from src.ops.model_manifest import ServingManifest
from src.serving.routed_vlm import RoutedVLMService


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/serving_manifest.json",
        help="Serving manifest to evaluate.",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/processed/multitask/validation.jsonl",
        help="Validation metadata JSONL.",
    )
    parser.add_argument(
        "--predictions-path",
        default="outputs/predictions/routed_validation.jsonl",
        help="Where routed prediction JSONL will be written.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load router, backbone, and adapters, then exit before generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ServingManifest.load(args.manifest, project_root=PROJECT_ROOT)
    service = RoutedVLMService.from_manifest(manifest)
    if os.getenv("HF_LOCAL_FILES_ONLY") in {"1", "true", "True"}:
        service.local_files_only = True
    if os.getenv("ROUTED_VLM_LOAD_IN_4BIT") in {"1", "true", "True"}:
        service.load_in_4bit = True
    if args.device:
        service.device = args.device
    service.load()
    if args.load_only:
        print(
            "Loaded routed service: "
            f"device={service.device}, adapters={sorted(service.loaded_adapters)}",
            flush=True,
        )
        return

    dataset = VQADataset(args.metadata_path)
    examples = list(dataset)
    if args.limit is not None:
        examples = examples[: args.limit]

    output_path = Path(args.predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            prediction = service.predict(
                image_path=example["image_path"],
                question=example["question"],
                min_confidence=manifest.min_confidence,
            )
            decision = prediction.decision
            record = {
                **example,
                "prediction": prediction.answer,
                "predicted_task_type": decision.task_type,
                "backend": decision.backend_name,
                "use_adapter": decision.use_adapter,
                "adapter": decision.adapter_name,
                "confidence": decision.confidence,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(examples)}] "
                f"{example['task_type']} -> {decision.task_type} "
                f"backend={decision.backend_name} prediction={prediction.answer!r}",
                flush=True,
            )

    print(f"Saved routed predictions to {output_path}", flush=True)


if __name__ == "__main__":
    main()
