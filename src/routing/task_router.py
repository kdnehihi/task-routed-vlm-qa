"""Task router placeholders for LoRA expert selection.

The router's job is to predict which LoRA adapter should be attached to the
shared frozen Qwen2-VL backbone. It should not choose between separate full
models.

TODO:
- Replace the keyword heuristic with a supervised classifier.
- Track routing accuracy separately from answer accuracy.
- Calibrate confidence before using router decisions in inference.
"""


TASK_TYPES = ("chartqa", "docvqa", "textvqa")


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
