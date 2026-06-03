"""Task router placeholders for LoRA expert selection.

The router's job is to predict which LoRA adapter should be attached to the
shared frozen Qwen2-VL backbone. It should not choose between separate full
models.

TODO:
- Replace the keyword heuristic with a supervised classifier.
- Track routing accuracy separately from answer accuracy.
- Calibrate confidence before using router decisions in inference.
"""

from dataclasses import dataclass

from src.models.lora_adapter import get_lora_expert


TASK_TYPES = ("chartqa", "docvqa", "textvqa")


@dataclass(frozen=True)
class RouterDecision:
    """A router output that can be logged during training or inference."""

    task_type: str
    expert_id: int
    adapter_name: str
    checkpoint_dir: str
    confidence: float | None = None


def route_task_from_instruction(question: str) -> str:
    """Return a rough task type from a question string.

    This is a lightweight placeholder so downstream adapter-selection code can
    be developed before the learned router exists.
    """
    normalized_question = question.lower()

    chart_keywords = (
        "chart",
        "graph",
        "axis",
        "bar",
        "line",
        "plot",
        "value",
        "revenue",
        "percentage",
    )
    doc_keywords = (
        "document",
        "receipt",
        "invoice",
        "form",
        "date",
        "expiration",
        "total",
        "address",
    )

    if any(keyword in normalized_question for keyword in chart_keywords):
        return "chartqa"
    if any(keyword in normalized_question for keyword in doc_keywords):
        return "docvqa"

    return "textvqa"


def select_lora_expert(question: str, confidence: float | None = None) -> RouterDecision:
    """Route a question to one symbolic LoRA expert.

    The returned decision is intentionally explicit so training logs can show
    whether examples are being routed to expert 1, 2, or 3.
    """
    task_type = route_task_from_instruction(question)
    expert = get_lora_expert(task_type)

    return RouterDecision(
        task_type=task_type,
        expert_id=expert.expert_id,
        adapter_name=expert.adapter_name,
        checkpoint_dir=expert.checkpoint_dir,
        confidence=confidence,
    )


def format_router_decision(
    decision: RouterDecision,
    sample_id: int | str | None = None,
    true_task_type: str | None = None,
) -> str:
    """Format one router decision for readable training logs."""
    parts = ["[router]"]

    if sample_id is not None:
        parts.append(f"sample={sample_id}")

    parts.extend(
        [
            f"expert={decision.expert_id}",
            f"task={decision.task_type}",
            f"adapter={decision.adapter_name}",
        ]
    )

    if true_task_type is not None:
        is_correct = decision.task_type == true_task_type
        parts.append(f"true_task={true_task_type}")
        parts.append(f"correct={is_correct}")

    if decision.confidence is not None:
        parts.append(f"confidence={decision.confidence:.3f}")

    return " ".join(parts)


def summarize_router_decisions(decisions: list[RouterDecision]) -> dict[int, int]:
    """Count how often each expert was selected."""
    counts: dict[int, int] = {}

    for decision in decisions:
        counts[decision.expert_id] = counts.get(decision.expert_id, 0) + 1

    return counts
