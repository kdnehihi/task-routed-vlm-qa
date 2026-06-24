"""Tests for baseline VLM interfaces."""

import pytest

from src.models.baseline_vlm import (
    DEFAULT_BLIP_MODEL_NAME,
    DEFAULT_CHARTQA_ADAPTER_PATH,
    DEFAULT_QWEN25VL_MODEL_NAME,
    BlipLoRAVQABaselineVLM,
    BlipVQABaselineVLM,
    DummyBaselineVLM,
    Qwen2VLVQABaselineVLM,
    create_baseline_model,
)


def test_dummy_baseline_returns_default_answer() -> None:
    model = DummyBaselineVLM(default_answer="unknown")

    prediction = model.predict(
        image_path="fake/path.png",
        question="What is shown?",
    )

    assert prediction == "unknown"


def test_create_baseline_model_returns_dummy_model() -> None:
    model = create_baseline_model("dummy")

    assert isinstance(model, DummyBaselineVLM)


def test_create_baseline_model_returns_blip_model_without_loading_weights() -> None:
    model = create_baseline_model("blip", device="cpu")

    assert isinstance(model, BlipVQABaselineVLM)
    assert model.model_name == DEFAULT_BLIP_MODEL_NAME
    assert model.device == "cpu"
    assert model.processor is None
    assert model.model is None


def test_create_baseline_model_accepts_custom_blip_model_id() -> None:
    model = create_baseline_model(
        "blip",
        model_id="custom/blip-model",
        device="cpu",
    )

    assert isinstance(model, BlipVQABaselineVLM)
    assert model.model_name == "custom/blip-model"


def test_create_baseline_model_returns_blip_lora_model_without_loading_weights() -> None:
    model = create_baseline_model(
        "blip_lora",
        adapter_path="outputs/checkpoints/blip_lora_baseline_sample",
        device="cpu",
    )

    assert isinstance(model, BlipLoRAVQABaselineVLM)
    assert model.model_name == DEFAULT_BLIP_MODEL_NAME
    assert model.adapter_path == "outputs/checkpoints/blip_lora_baseline_sample"
    assert model.device == "cpu"
    assert model.processor is None
    assert model.model is None


def test_create_baseline_model_requires_adapter_path_for_blip_lora() -> None:
    with pytest.raises(ValueError, match="adapter-path"):
        create_baseline_model("blip_lora")


def test_create_baseline_model_returns_qwen2vl_without_loading_weights() -> None:
    model = create_baseline_model("qwen2vl", device="cpu")

    assert isinstance(model, Qwen2VLVQABaselineVLM)
    assert model.model_name == DEFAULT_QWEN25VL_MODEL_NAME
    assert model.device == "cpu"
    assert model.adapter_path is None
    assert model.processor is None
    assert model.model is None


def test_create_baseline_model_returns_local_chart_lora_without_loading_weights() -> None:
    model = create_baseline_model("qwen2vl_chart_lora", device="cpu")

    assert isinstance(model, Qwen2VLVQABaselineVLM)
    assert model.model_name == DEFAULT_QWEN25VL_MODEL_NAME
    assert model.adapter_path == DEFAULT_CHARTQA_ADAPTER_PATH
    assert model.adapter_name == "chart_lora"
    assert model.processor is None
    assert model.model is None


def test_create_baseline_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        create_baseline_model("not-a-real-model")
