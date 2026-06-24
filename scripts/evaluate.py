"""Run baseline evaluation for multi-task vision-language QA."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import VQADataset
from src.evaluation.evaluator import build_prediction_records, evaluate_predictions_by_task
from src.models.baseline_vlm import create_baseline_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--metadata-path",
        default="data/processed/multitask/validation.jsonl",
    )
    parser.add_argument(
        "--model",
        default="dummy",
        choices=("dummy", "blip", "blip_lora", "qwen2vl", "qwen2vl_chart_lora"),
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--predictions-path",
        default="outputs/predictions/baseline_predictions.jsonl",
    )
    return parser.parse_args()


def select_examples(dataset: VQADataset, limit: int | None) -> list[dict]:
    examples = list(dataset)

    if limit is None:
        return examples

    return examples[:limit]


def write_predictions(records: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    dataset = VQADataset(args.metadata_path)
    references = select_examples(dataset, args.limit)
    model = create_baseline_model(
        model_name=args.model,
        model_id=args.model_id,
        device=args.device,
        adapter_path=args.adapter_path,
    )

    predictions = []

    for index, reference in enumerate(references):
        prediction = model.predict(
            image_path=reference["image_path"],
            question=reference["question"],
        )
        predictions.append(prediction)
        print(
            f"[{index + 1}/{len(references)}] "
            f"{reference['dataset']} | prediction={prediction!r}",
            flush=True,
        )

    report = evaluate_predictions_by_task(predictions, references)
    prediction_records = build_prediction_records(predictions, references)
    write_predictions(prediction_records, args.predictions_path)

    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved predictions to {args.predictions_path}", flush=True)


if __name__ == "__main__":
    main()
