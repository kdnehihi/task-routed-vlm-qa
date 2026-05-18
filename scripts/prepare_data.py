"""Prepare datasets for multi-task vision-language QA."""

from argparse import ArgumentParser
from pathlib import Path
import json

from datasets import load_dataset


DATASET_NAME = "lmms-lab/DocVQA"
DATASET_CONFIG = "DocVQA"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default="data/raw/docvqa/sample")
    return parser.parse_args()


def save_docvqa_sample(split: str, limit: int, output_dir: str) -> None:
    output_path = Path(output_dir)
    image_dir = output_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_path / f"{split}.jsonl"

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
    )

    with metadata_path.open("w", encoding="utf-8") as f:
        for index, example in enumerate(dataset):
            if index >= limit:
                break

            image = example["image"]
            image_filename = f"{split}_{index:05d}.png"
            image_path = image_dir / image_filename
            image.save(image_path)

            record = {
                "dataset": "docvqa",
                "split": split,
                "question_id": example.get("questionId"),
                "question": example.get("question"),
                "answers": example.get("answers", []),
                "image_path": str(image_path),
                "task_type": "document_qa",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {limit} examples to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    save_docvqa_sample(args.split, args.limit, args.output_dir)


if __name__ == "__main__":
    main()
