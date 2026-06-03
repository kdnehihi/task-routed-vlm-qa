"""Tests for planned Qwen2-VL LoRA expert registry."""

import pytest

from src.models.lora_adapter import (
    QWEN2VL_BACKBONE_ID,
    QWEN2VL_LORA_EXPERTS,
    get_lora_expert,
)
from src.routing.task_router import route_task_from_instruction


def test_qwen2vl_lora_expert_registry_contains_three_task_experts() -> None:
    assert QWEN2VL_BACKBONE_ID == "Qwen/Qwen2-VL-2B-Instruct"
    assert set(QWEN2VL_LORA_EXPERTS) == {"chartqa", "docvqa", "textvqa"}


def test_get_lora_expert_returns_adapter_metadata() -> None:
    expert = get_lora_expert("chartqa")

    assert expert.adapter_name == "LoRA_chartqa"
    assert expert.checkpoint_dir == "outputs/checkpoints/qwen2vl_lora_chartqa"
    assert expert.target_modules == ("q_proj", "v_proj")


def test_get_lora_expert_rejects_unknown_task_type() -> None:
    with pytest.raises(ValueError, match="Unsupported LoRA expert task type"):
        get_lora_expert("medical_qa")


def test_placeholder_router_selects_symbolic_lora_task_type() -> None:
    assert route_task_from_instruction("What is the revenue value in 2021?") == "chartqa"
    assert route_task_from_instruction("What is the expiration date?") == "docvqa"
    assert route_task_from_instruction("What word is written on the sign?") == "textvqa"
