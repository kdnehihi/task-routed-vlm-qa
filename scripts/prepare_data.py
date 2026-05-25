"""Prepare datasets for multi-task vision-language QA."""

from argparse import ArgumentParser
from pathlib import Path
import json


DOCVQA_DATASET_NAME = "lmms-lab/DocVQA"
DOCVQA_DATASET_CONFIG = "DocVQA"
CHARTQA_DATASET_NAME = "HuggingFaceM4/ChartQA"
TEXTVQA_DATASET_NAME = "lmms-lab/textvqa"
SUPPORTED_DATASETS = ("docvqa", "chartqa", "textvqa")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="docvqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream examples from Hugging Face instead of caching full splits.",
    )
    return parser.parse_args()


def get_default_output_dir(dataset_name: str) -> str:
    return f"data/raw/{dataset_name}/sample"


def prepare_output_paths(output_dir: str, split: str) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    image_dir = output_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_path / f"{split}.jsonl"
    return output_path, image_dir, metadata_path


def as_answer_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def save_image(image, image_dir: Path, split: str, index: int) -> str:
    image_filename = f"{split}_{index:05d}.png"
    image_path = image_dir / image_filename
    image = image.convert("RGB")
    image.save(image_path)
    return str(image_path)


def save_docvqa_sample(
    split: str,
    limit: int,
    output_dir: str,
    streaming: bool = False,
) -> None:
    from datasets import load_dataset

    output_path, image_dir, metadata_path = prepare_output_paths(output_dir, split)

    dataset = load_dataset(
        DOCVQA_DATASET_NAME,
        DOCVQA_DATASET_CONFIG,
        split=split,
        streaming=streaming,
    )

    saved_count = 0

    with metadata_path.open("w", encoding="utf-8") as f:
        for index, example in enumerate(dataset):
            if index >= limit:
                break

            record = {
                "dataset": "docvqa",
                "split": split,
                "question_id": example.get("questionId"),
                "question": example.get("question"),
                "answers": as_answer_list(example.get("answers")),
                "image_path": save_image(example["image"], image_dir, split, index),
                "task_type": "document_qa",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved_count += 1

    print(f"Saved {saved_count} examples to {output_path}", flush=True)


def save_chartqa_sample(
    split: str,
    limit: int,
    output_dir: str,
    streaming: bool = False,
) -> None:
    from datasets import load_dataset

    hf_split = "val" if split == "validation" else split
    output_path, image_dir, metadata_path = prepare_output_paths(output_dir, split)

    dataset = load_dataset(
        CHARTQA_DATASET_NAME,
        split=hf_split,
        streaming=streaming,
    )

    saved_count = 0

    with metadata_path.open("w", encoding="utf-8") as f:
        for index, example in enumerate(dataset):
            if index >= limit:
                break

            record = {
                "dataset": "chartqa",
                "split": split,
                "question_id": example.get("id") or example.get("id_image"),
                "question": example.get("query") or example.get("question"),
                "answers": as_answer_list(example.get("label") or example.get("answer")),
                "image_path": save_image(example["image"], image_dir, split, index),
                "task_type": "chart_qa",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved_count += 1

    print(f"Saved {saved_count} examples to {output_path}", flush=True)


def save_textvqa_sample(
    split: str,
    limit: int,
    output_dir: str,
    streaming: bool = False,
) -> None:
    from datasets import load_dataset

    output_path, image_dir, metadata_path = prepare_output_paths(output_dir, split)

    dataset = load_dataset(
        TEXTVQA_DATASET_NAME,
        split=split,
        streaming=streaming,
    )

    saved_count = 0

    with metadata_path.open("w", encoding="utf-8") as f:
        for index, example in enumerate(dataset):
            if index >= limit:
                break

            record = {
                "dataset": "textvqa",
                "split": split,
                "question_id": example.get("question_id"),
                "question": example.get("question"),
                "answers": as_answer_list(example.get("answers")),
                "image_path": save_image(example["image"], image_dir, split, index),
                "task_type": "image_vqa",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved_count += 1

    print(f"Saved {saved_count} examples to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or get_default_output_dir(args.dataset)

    if args.dataset == "docvqa":
        save_docvqa_sample(args.split, args.limit, output_dir, args.streaming)
    elif args.dataset == "chartqa":
        save_chartqa_sample(args.split, args.limit, output_dir, args.streaming)
    elif args.dataset == "textvqa":
        save_textvqa_sample(args.split, args.limit, output_dir, args.streaming)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
