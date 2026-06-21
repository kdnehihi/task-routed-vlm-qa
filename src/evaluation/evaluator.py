"""Evaluation helpers for vision-language QA predictions."""

import re

from src.evaluation.metrics import (
    anls,
    chart_hybrid_accuracy,
    containment,
    docvqa_anls,
    docvqa_exact_match,
    loose_containment,
    maybe_extract_docvqa_short_span,
    mean_score,
    normalize_answer,
    normalize_docvqa_for_match,
    normalized_exact_match,
    output_token_length,
    postprocess_docvqa_answer,
    raw_exact_match,
    relaxed_numeric_accuracy,
    routing_accuracy,
    strict_containment,
    token_f1,
    vqa_soft_score,
)


def evaluate_predictions(
    predictions: list[str],
    references: list[dict],
) -> dict:
    """Evaluate answer predictions against reference examples."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    records = build_prediction_records(predictions, references)

    return {
        "num_examples": len(predictions),
        **summarize_quality_records(records),
    }


def evaluate_predictions_by_task(
    predictions: list[str],
    references: list[dict],
) -> dict:
    """Evaluate answer predictions overall and grouped by task type."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    records = build_prediction_records(predictions, references)
    task_records = {}

    for record, reference in zip(records, references):
        task_type = reference["task_type"]
        task_records.setdefault(task_type, []).append(record)

    by_task = {
        task_type: summarize_quality_records(records)
        for task_type, records in sorted(task_records.items())
    }

    return {
        "overall": {
            "num_examples": len(records),
            **summarize_quality_records(records),
        },
        "by_task": by_task,
    }


def compute_exact_match_scores(
    predictions: list[str],
    references: list[dict],
) -> list[float]:
    """Compute per-example exact match scores."""
    return [
        normalized_exact_match(prediction, reference["answers"])
        for prediction, reference in zip(predictions, references)
    ]


def build_prediction_records(
    predictions: list[str],
    references: list[dict],
    cleaned_predictions: list[str] | None = None,
) -> list[dict]:
    """Build per-sample records with answer quality metrics."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")
    if cleaned_predictions is not None and len(cleaned_predictions) != len(predictions):
        raise ValueError("cleaned_predictions and predictions must have the same length.")

    records = []
    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        task_type = reference.get("task_type", "")
        question = reference.get("question", "")
        cleaned_prediction = (
            cleaned_predictions[index]
            if cleaned_predictions is not None
            else str(prediction).strip()
        )
        answers = reference["answers"]
        old_cleaned_prediction = cleaned_prediction
        docvqa_metric_prediction = None
        docvqa_postprocess_reason = None
        if task_type == "docvqa":
            cleaned_prediction = postprocess_docvqa_answer(cleaned_prediction, question)
            docvqa_metric_prediction = maybe_extract_docvqa_short_span(
                cleaned_prediction,
                question,
                answers,
            )
            docvqa_postprocess_reason = infer_docvqa_postprocess_reason(
                old_cleaned_prediction,
                cleaned_prediction,
                docvqa_metric_prediction,
                answers,
            )

        metric_prediction = docvqa_metric_prediction or cleaned_prediction
        normalized_prediction = (
            normalize_docvqa_for_match(metric_prediction)
            if task_type == "docvqa"
            else normalize_answer(cleaned_prediction)
        )
        normalized_references = (
            [normalize_docvqa_for_match(answer) for answer in answers]
            if task_type == "docvqa"
            else [normalize_answer(answer) for answer in answers]
        )
        normalized_em_score = (
            docvqa_exact_match(metric_prediction, answers)
            if task_type == "docvqa"
            else normalized_exact_match(cleaned_prediction, answers)
        )
        records.append(
            {
                "index": index,
                "question": question,
                "task_type": task_type,
                "answers": answers,
                "chosen_training_answer": reference.get("chosen_training_answer"),
                "raw_prediction": prediction,
                "cleaned_prediction": cleaned_prediction,
                "normalized_prediction": normalized_prediction,
                "normalized_references": normalized_references,
                "output_length": output_token_length(cleaned_prediction),
                "raw_exact_match": raw_exact_match(prediction, answers),
                "normalized_em": normalized_em_score,
                "token_f1": token_f1(cleaned_prediction, answers),
                "strict_containment": strict_containment(cleaned_prediction, answers),
                "old_containment_if_available": loose_containment(cleaned_prediction, answers),
                "containment": containment(cleaned_prediction, answers),
                "old_cleaned_prediction_if_available": (
                    old_cleaned_prediction if task_type == "docvqa" else None
                ),
                "old_normalized_match_if_available": (
                    normalized_exact_match(old_cleaned_prediction, answers)
                    if task_type == "docvqa"
                    else None
                ),
                "docvqa_metric_prediction": (
                    docvqa_metric_prediction if task_type == "docvqa" else None
                ),
                "docvqa_normalized_prediction": (
                    normalized_prediction if task_type == "docvqa" else None
                ),
                "docvqa_postprocess_reason": (
                    docvqa_postprocess_reason if task_type == "docvqa" else None
                ),
                "chart_relaxed_accuracy": task_metric(
                    task_type,
                    "chartqa",
                    relaxed_numeric_accuracy(cleaned_prediction, answers),
                ),
                "chart_hybrid_accuracy": task_metric(
                    task_type,
                    "chartqa",
                    chart_hybrid_accuracy(cleaned_prediction, answers),
                ),
                "anls": task_metric(
                    task_type,
                    ("docvqa", "textvqa"),
                    docvqa_anls(metric_prediction, answers)
                    if task_type == "docvqa"
                    else anls(cleaned_prediction, answers),
                ),
                "vqa_score": task_metric(
                    task_type,
                    "textvqa",
                    vqa_soft_score(cleaned_prediction, answers),
                ),
            }
        )
    return records


def infer_docvqa_postprocess_reason(
    old_prediction: str,
    new_prediction: str,
    metric_prediction: str | None,
    answers: list[str],
) -> str | None:
    """Infer a compact debug label for DocVQA postprocess changes."""
    if metric_prediction and metric_prediction != new_prediction:
        return "short_span"
    if old_prediction == new_prediction:
        return None
    joined = " ".join([old_prediction, new_prediction, *answers])
    if re_contains(r"\$\s*\d|\$\d{1,3},\d{3}", joined):
        return "currency_format"
    if re_contains(r"\d{1,3},\d{3}", joined):
        return "numeric_comma"
    if re_contains(r"\d\s*-\s*\d", joined):
        return "phone_hyphen"
    if re_contains(r"\b[A-Za-z]\.\s+[A-Za-z]\.", joined):
        return "initials_spacing"
    if re_contains(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", joined.lower()):
        return "date_cleanup"
    if re_contains(r"\b(?:mg|kg|ml|mls|copies|waves)\b", joined.lower()):
        return "unit_suffix"
    return "punctuation_spacing"


def re_contains(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text))


def build_docvqa_postprocess_debug_records(records: list[dict]) -> list[dict]:
    """Return DocVQA records whose cleaned or metric-normalized output changed."""
    debug_records = []
    for record in records:
        if record.get("task_type") != "docvqa":
            continue
        old_cleaned = record.get("old_cleaned_prediction_if_available")
        new_cleaned = record.get("cleaned_prediction")
        metric_prediction = record.get("docvqa_metric_prediction")
        if old_cleaned == new_cleaned and metric_prediction == new_cleaned:
            continue
        debug_records.append(
            {
                "question": record.get("question"),
                "references": record.get("answers"),
                "raw_prediction": record.get("raw_prediction"),
                "old_cleaned_prediction": old_cleaned,
                "new_cleaned_prediction": new_cleaned,
                "docvqa_metric_prediction": metric_prediction,
                "docvqa_normalized_prediction": record.get("docvqa_normalized_prediction"),
                "old_normalized_match": record.get("old_normalized_match_if_available"),
                "new_normalized_match": record.get("normalized_em"),
                "reason": record.get("docvqa_postprocess_reason"),
            }
        )
    return debug_records


def task_metric(task_type: str, expected: str | tuple[str, ...], score: float) -> float | None:
    """Return score only for matching task types."""
    if isinstance(expected, str):
        expected = (expected,)
    return score if task_type in expected else None


def summarize_quality_records(records: list[dict]) -> dict:
    """Summarize per-sample answer quality records."""
    lengths = [record["output_length"] for record in records]
    return {
        "num_examples": len(records),
        "raw_exact_match": mean_score([record["raw_exact_match"] for record in records]),
        "normalized_exact_match": mean_score(
            [record["normalized_em"] for record in records]
        ),
        "exact_match": mean_score([record["normalized_em"] for record in records]),
        "token_f1": mean_score([record["token_f1"] for record in records]),
        "strict_containment": mean_score([record["strict_containment"] for record in records]),
        "containment": mean_score([record["strict_containment"] for record in records]),
        "old_containment_if_available": mean_score(
            [record["old_containment_if_available"] for record in records]
        ),
        "chart_relaxed_accuracy": mean_optional(
            [record["chart_relaxed_accuracy"] for record in records]
        ),
        "chart_hybrid_accuracy": mean_optional(
            [record["chart_hybrid_accuracy"] for record in records]
        ),
        "docvqa_anls": mean_optional(
            [record["anls"] for record in records if record["task_type"] == "docvqa"]
        ),
        "textvqa_anls": mean_optional(
            [record["anls"] for record in records if record["task_type"] == "textvqa"]
        ),
        "textvqa_vqa_score": mean_optional(
            [record["vqa_score"] for record in records if record["task_type"] == "textvqa"]
        ),
        "avg_output_token_length": mean_score(lengths),
        "pct_outputs_gt_10_tokens": mean_score(
            [1.0 if length > 10 else 0.0 for length in lengths]
        ),
        "pct_outputs_gt_20_tokens": mean_score(
            [1.0 if length > 20 else 0.0 for length in lengths]
        ),
    }


def mean_optional(scores: list[float | None]) -> float | None:
    """Mean over non-null task-specific scores."""
    present_scores = [score for score in scores if score is not None]
    if not present_scores:
        return None
    return mean_score(present_scores)


def summarize_quality_records_by_task(records: list[dict]) -> dict:
    """Summarize quality records overall and grouped by task type."""
    task_records = {}
    for record in records:
        task_records.setdefault(record["task_type"], []).append(record)

    return {
        "overall": summarize_quality_records(records),
        "by_task": {
            task_type: summarize_quality_records(task_records_for_type)
            for task_type, task_records_for_type in sorted(task_records.items())
        },
    }


def evaluate_routing(
    predicted_tasks: list[str],
    references: list[dict],
) -> dict:
    """Evaluate task routing predictions against reference task labels."""
    target_tasks = [reference["task_type"] for reference in references]

    return {
        "num_examples": len(predicted_tasks),
        "routing_accuracy": routing_accuracy(predicted_tasks, target_tasks),
    }
