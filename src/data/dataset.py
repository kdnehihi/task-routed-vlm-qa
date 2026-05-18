"""Dataset abstractions for multi-task vision-language QA."""

from pathlib import Path
import json

from torch.utils.data import Dataset

REQUIRED_FIELDS = {
    "dataset",
    "split",
    "question",
    "answers",
    "image_path",
    "task_type",
}


class VQADataset(Dataset):
    """Read normalized dataset examples from a JSONL file."""

    def __init__(self, metadata_path: str) -> None:
        self.metadata_path = Path(metadata_path)
        self.examples = self._load_examples()

    def _load_examples(self) -> list[dict]:
        examples = []

        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                example = json.loads(line)
                self._validate_example(example, line_number)
                examples.append(example)
        return examples

    def _validate_example(self, example: dict, line_number: int) -> None:
        missing_fields = REQUIRED_FIELDS - set(example)

        if missing_fields:
            raise ValueError(
                f"Missing fields at line {line_number}: {sorted(missing_fields)}"
            )
        if not Path(example["image_path"]).exists():
            raise FileNotFoundError(
                f"Image not found at line {line_number}: {example['image_path']}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]
