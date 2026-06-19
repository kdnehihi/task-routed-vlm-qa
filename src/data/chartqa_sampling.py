"""Rule-based ChartQA metadata and stratified sampling helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import random
import re
from typing import Iterable


DEFAULT_CHARTQA_2K_QUOTAS = {
    "lookup_value": 400,
    "extreme": 250,
    "difference": 300,
    "average": 150,
    "sum_total": 150,
    "ratio": 200,
    "yes_no_compare": 200,
    "counting": 100,
    "label_text": 150,
    "time_year": 100,
    "percent_decimal": 200,
}

CHARTQA_QUESTION_TYPES = (
    "lookup_value",
    "extreme",
    "difference",
    "average",
    "sum_total",
    "ratio",
    "yes_no_compare",
    "counting",
    "label_text",
    "time_year",
    "percent_decimal",
    "other",
)


YES_NO_START_RE = re.compile(
    r"^\s*(is|are|was|were|does|do|did|can|could|has|have|had|will|would|"
    r"should|may|might)\b",
    re.IGNORECASE,
)


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def classify_chartqa_question(question: str) -> str:
    """Assign a coarse ChartQA reasoning type from question text."""

    text = _clean_text(question).lower()
    if not text:
        return "other"

    if YES_NO_START_RE.search(text) and re.search(
        r"\b(greater|less|more|lower|higher|exceed|exceeds|exceeded|above|below|than|"
        r"larger|smaller|at least|at most|over|under)\b",
        text,
    ):
        return "yes_no_compare"
    if re.search(r"\b(ratio|proportion|divided by)\b|[a-z0-9][\s-]*:[\s-]*[a-z0-9]", text):
        return "ratio"
    if re.search(r"\b(difference|gap|how much more|how much less|subtract|minus|"
                 r"decrease from|increase from|change from)\b", text):
        return "difference"
    if re.search(r"\b(average|mean)\b", text):
        return "average"
    if re.search(r"\b(sum|total|combined|altogether|overall|in all)\b", text):
        return "sum_total"
    if re.search(r"\b(highest|lowest|maximum|minimum|max|min|largest|smallest|peak|least|most)\b", text):
        return "extreme"
    if re.search(r"\bhow many\b", text) and re.search(
        r"\b(bars?|lines?|countries|regions|categories|values|items|groups|years|months|quarters)\b",
        text,
    ):
        return "counting"
    if re.search(r"\b(percent|percentage|share)\b|%", text):
        return "percent_decimal"
    if re.search(r"\b(year|month|quarter|previous quarter|when|date)\b", text):
        return "time_year"
    if re.search(
        r"\b(which|what country|what region|what category|what color|title|label|legend|"
        r"sector|industry|brand|company|state|city)\b",
        text,
    ):
        return "label_text"
    if re.search(r"\b(what|how much|value|amount|number|rate)\b", text):
        return "lookup_value"
    return "other"


def classify_answer_type(answers: list[str] | tuple[str, ...] | None) -> str:
    """Assign a coarse answer type from the first non-empty reference."""

    answer = ""
    for candidate in answers or []:
        candidate = _clean_text(candidate)
        if candidate:
            answer = candidate
            break
    if not answer:
        return "other"

    lowered = answer.lower()
    if lowered in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"\d{4}", lowered) or re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|q[1-4])\b",
        lowered,
    ):
        return "date_or_year"
    if re.fullmatch(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", lowered):
        return "numeric"
    if re.search(r"[a-z]", lowered):
        return "text_label"
    return "other"


def attach_chartqa_metadata(example: dict) -> dict:
    """Attach question_type and answer_type to one ChartQA example in place."""

    example["question_type"] = classify_chartqa_question(example.get("question", ""))
    example["answer_type"] = classify_answer_type(example.get("answers", []))
    return example


def attach_chartqa_metadata_to_examples(examples: Iterable[dict]) -> list[dict]:
    """Attach ChartQA sampling metadata to each example and return a list."""

    return [attach_chartqa_metadata(example) for example in examples]


def stratified_chartqa_sample(
    examples: list[dict],
    quotas: dict[str, int],
    seed: int = 42,
    sample_limit: int | None = None,
    allow_duplicates: bool = False,
) -> list[dict]:
    """Sample ChartQA examples by question_type quota without duplicates by default."""

    rng = random.Random(seed)
    prepared = attach_chartqa_metadata_to_examples(list(examples))
    grouped = defaultdict(list)
    for example in prepared:
        grouped[example.get("question_type", "other")].append(example)

    selected = []
    selected_ids = set()
    for question_type, quota in quotas.items():
        bucket = list(grouped.get(question_type, []))
        rng.shuffle(bucket)
        if len(bucket) < quota:
            print(
                f"Warning: ChartQA bucket {question_type!r} has {len(bucket)} examples, "
                f"requested {quota}; using all available."
            )
        take_count = min(len(bucket), quota)
        for example in bucket[:take_count]:
            selected.append(example)
            selected_ids.add(id(example))

    rng.shuffle(selected)

    if sample_limit is not None and len(selected) > sample_limit:
        selected = selected[:sample_limit]
        selected_ids = {id(example) for example in selected}

    if sample_limit is not None and len(selected) < sample_limit:
        remaining = [
            example
            for example in prepared
            if allow_duplicates or id(example) not in selected_ids
        ]
        rng.shuffle(remaining)
        needed = sample_limit - len(selected)
        selected.extend(remaining[:needed])
        rng.shuffle(selected)

    return selected


def chartqa_sampling_stats(
    examples: list[dict],
    quotas: dict[str, int],
    seed: int,
    sample_limit: int | None,
) -> dict:
    """Build serializable distribution stats for a sampled ChartQA subset."""

    return {
        "total_examples": len(examples),
        "random_seed": seed,
        "sample_limit": sample_limit,
        "quota_used": dict(quotas),
        "count_by_question_type": dict(Counter(example.get("question_type", "other") for example in examples)),
        "count_by_answer_type": dict(Counter(example.get("answer_type", "other") for example in examples)),
    }


def save_chartqa_sampling_stats(stats: dict, path: str | Path) -> None:
    """Save ChartQA sampling stats as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def print_chartqa_sampling_debug(examples: list[dict], examples_per_bucket: int = 3) -> None:
    """Print a few sampled questions per question_type for sanity checking."""

    grouped = defaultdict(list)
    for example in examples:
        grouped[example.get("question_type", "other")].append(example)

    for question_type in CHARTQA_QUESTION_TYPES:
        bucket = grouped.get(question_type, [])
        if not bucket:
            continue
        print(f"\n[{question_type}] sampled examples")
        for example in bucket[:examples_per_bucket]:
            print(f"[{question_type}] {example.get('question')} -> {example.get('answers', [])}")
