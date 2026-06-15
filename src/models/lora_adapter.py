"""LoRA expert definitions for the planned Qwen2-VL adapter system.

This module intentionally does not train or load adapters yet. It only defines
the symbolic expert registry that later training and routing code can share.

Main design:
- one frozen Qwen2-VL backbone
- one LoRA adapter per task family
- router predicts a task type and selects the matching LoRA adapter
"""

from dataclasses import dataclass

from src.data.answers import canonicalize_task_type


QWEN2VL_BACKBONE_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@dataclass(frozen=True)
class LoRAExpertConfig:
    """Configuration metadata for one task-specific LoRA expert."""

    expert_id: int
    task_type: str
    adapter_name: str
    checkpoint_dir: str
    description: str
    is_shared: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


QWEN2VL_LORA_EXPERTS: dict[str, LoRAExpertConfig] = {
    "chartqa": LoRAExpertConfig(
        expert_id=1,
        task_type="chartqa",
        adapter_name="LoRA_chartqa",
        checkpoint_dir="outputs/checkpoints/qwen2vl_lora_chartqa",
        description="Expert adapter for chart reading and numerical visual QA.",
    ),
    "docvqa": LoRAExpertConfig(
        expert_id=2,
        task_type="docvqa",
        adapter_name="LoRA_docvqa",
        checkpoint_dir="outputs/checkpoints/qwen2vl_lora_docvqa",
        description="Expert adapter for document QA, forms, receipts, and layout-heavy images.",
    ),
    "textvqa": LoRAExpertConfig(
        expert_id=3,
        task_type="textvqa",
        adapter_name="LoRA_textvqa",
        checkpoint_dir="outputs/checkpoints/qwen2vl_lora_textvqa",
        description="Expert adapter for natural images that require reading scene text.",
    ),
}


PLANNED_SHARED_LORA_EXPERTS: dict[str, LoRAExpertConfig] = {
    "shared_ocr": LoRAExpertConfig(
        expert_id=101,
        task_type="shared_ocr",
        adapter_name="LoRA_shared_ocr",
        checkpoint_dir="outputs/checkpoints/qwen2vl_lora_shared_ocr",
        description="Planned shared adapter for OCR-heavy behavior common to DocVQA and TextVQA.",
        is_shared=True,
    ),
    "shared_reasoning": LoRAExpertConfig(
        expert_id=102,
        task_type="shared_reasoning",
        adapter_name="LoRA_shared_reasoning",
        checkpoint_dir="outputs/checkpoints/qwen2vl_lora_shared_reasoning",
        description="Planned shared adapter for general visual reasoning common across tasks.",
        is_shared=True,
    ),
}


QWEN2VL_HYBRID_LORA_ADAPTERS: dict[str, LoRAExpertConfig] = {
    "shared_doc_text_lora": LoRAExpertConfig(
        expert_id=201,
        task_type="docvqa,textvqa",
        adapter_name="shared_doc_text_lora",
        checkpoint_dir="outputs/checkpoints/qwen2vl/shared_doc_text_lora",
        description="Shared adapter for OCR/span extraction on DocVQA and TextVQA.",
        is_shared=True,
        rank=4,
        alpha=8,
    ),
    "chart_lora": LoRAExpertConfig(
        expert_id=202,
        task_type="chartqa",
        adapter_name="chart_lora",
        checkpoint_dir="outputs/checkpoints/qwen2vl/chart_lora",
        description="Chart-specific adapter for chart reading and numeric grounding.",
        rank=8,
        alpha=16,
    ),
    "shared_lora_all_tasks": LoRAExpertConfig(
        expert_id=203,
        task_type="chartqa,docvqa,textvqa",
        adapter_name="shared_lora_all_tasks",
        checkpoint_dir="outputs/checkpoints/qwen2vl/shared_lora_all_tasks",
        description="Backward-compatible shared adapter trained on all three tasks.",
        is_shared=True,
        rank=4,
        alpha=8,
    ),
}


def get_lora_expert(task_type: str) -> LoRAExpertConfig:
    """Return the LoRA expert config for a predicted task type."""
    normalized_task_type = canonicalize_task_type(task_type)

    if normalized_task_type not in QWEN2VL_LORA_EXPERTS:
        known_tasks = ", ".join(sorted(QWEN2VL_LORA_EXPERTS))
        raise ValueError(
            f"Unsupported LoRA expert task type: {task_type!r}. "
            f"Expected one of: {known_tasks}"
        )

    return QWEN2VL_LORA_EXPERTS[normalized_task_type]
