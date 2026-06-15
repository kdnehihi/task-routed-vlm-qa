"""Tests for planned Qwen2-VL LoRA expert registry."""

import pytest

from src.models.lora_adapter import (
    PLANNED_SHARED_LORA_EXPERTS,
    QWEN2VL_BACKBONE_ID,
    QWEN2VL_HYBRID_LORA_ADAPTERS,
    QWEN2VL_LORA_EXPERTS,
    get_lora_expert,
)
from src.routing.task_router import (
    format_router_decision,
    route_task_from_instruction,
    select_lora_expert,
    summarize_router_decisions,
)


def test_qwen2vl_lora_expert_registry_contains_three_task_experts() -> None:
    assert QWEN2VL_BACKBONE_ID == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert set(QWEN2VL_LORA_EXPERTS) == {"chartqa", "docvqa", "textvqa"}
    assert QWEN2VL_LORA_EXPERTS["chartqa"].expert_id == 1
    assert QWEN2VL_LORA_EXPERTS["docvqa"].expert_id == 2
    assert QWEN2VL_LORA_EXPERTS["textvqa"].expert_id == 3


def test_get_lora_expert_returns_adapter_metadata() -> None:
    expert = get_lora_expert("chartqa")

    assert expert.expert_id == 1
    assert expert.adapter_name == "LoRA_chartqa"
    assert expert.checkpoint_dir == "outputs/checkpoints/qwen2vl_lora_chartqa"
    assert expert.target_modules == ("q_proj", "v_proj")


def test_get_lora_expert_accepts_legacy_task_alias() -> None:
    expert = get_lora_expert("document_qa")

    assert expert.task_type == "docvqa"


def test_get_lora_expert_rejects_unknown_task_type() -> None:
    with pytest.raises(ValueError, match="Unsupported LoRA expert task type"):
        get_lora_expert("medical_qa")


def test_placeholder_router_selects_symbolic_lora_task_type() -> None:
    assert route_task_from_instruction("What is the revenue value in 2021?") == "chartqa"
    assert route_task_from_instruction("What is the expiration date?") == "docvqa"
    assert route_task_from_instruction("What word is written on the sign?") == "textvqa"


def test_router_decision_can_be_logged_with_expert_id() -> None:
    decision = select_lora_expert("What is the revenue value in 2021?")
    log_line = format_router_decision(
        decision,
        sample_id=7,
        true_task_type="chartqa",
    )

    assert decision.expert_id == 1
    assert decision.adapter_name == "LoRA_chartqa"
    assert "sample=7" in log_line
    assert "expert=1" in log_line
    assert "task=chartqa" in log_line
    assert "adapter=LoRA_chartqa" in log_line
    assert "correct=True" in log_line


def test_router_decision_summary_counts_expert_usage() -> None:
    decisions = [
        select_lora_expert("What is the revenue value in 2021?"),
        select_lora_expert("What is the expiration date?"),
        select_lora_expert("What word is written on the sign?"),
        select_lora_expert("What is the total on this receipt?"),
    ]

    assert summarize_router_decisions(decisions) == {1: 1, 2: 2, 3: 1}


def test_planned_shared_experts_are_separate_from_task_experts() -> None:
    assert set(PLANNED_SHARED_LORA_EXPERTS) == {"shared_ocr", "shared_reasoning"}
    assert PLANNED_SHARED_LORA_EXPERTS["shared_ocr"].is_shared is True
    assert PLANNED_SHARED_LORA_EXPERTS["shared_ocr"].expert_id == 101


def test_hybrid_lora_adapter_registry_contains_chart_and_doc_text_paths() -> None:
    assert set(QWEN2VL_HYBRID_LORA_ADAPTERS) == {
        "chart_lora",
        "shared_doc_text_lora",
        "shared_lora_all_tasks",
    }
    assert QWEN2VL_HYBRID_LORA_ADAPTERS["shared_doc_text_lora"].checkpoint_dir.endswith(
        "qwen2vl/shared_doc_text_lora"
    )
    assert QWEN2VL_HYBRID_LORA_ADAPTERS["chart_lora"].checkpoint_dir.endswith(
        "qwen2vl/chart_lora"
    )
