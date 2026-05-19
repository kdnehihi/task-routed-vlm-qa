"""Tests for normalized VQA data format."""

import json
from pathlib import Path

import pytest

from src.data.dataset import VQADataset


def test_vqa_dataset_loads_valid_jsonl(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake image bytes")

    metadata_path = tmp_path / "sample.jsonl"
    record = {
        "dataset": "docvqa",
        "split": "validation",
        "question": "What is the date?",
        "answers": ["2021"],
        "image_path": str(image_path),
        "task_type": "document_qa",
    }

    metadata_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = VQADataset(str(metadata_path))

    assert len(dataset) == 1
    assert dataset[0]["question"] == "What is the date?"
    assert dataset[0]["task_type"] == "document_qa"


def test_vqa_dataset_rejects_missing_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake image bytes")

    metadata_path = tmp_path / "bad_sample.jsonl"
    record = {
        "dataset": "docvqa",
        "split": "validation",
        "answers": ["2021"],
        "image_path": str(image_path),
        "task_type": "document_qa",
    }

    metadata_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        VQADataset(str(metadata_path))