"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    anls,
    chart_hybrid_accuracy,
    containment,
    docvqa_exact_match,
    exact_match,
    maybe_extract_docvqa_short_span,
    mean_score,
    normalize_answer,
    normalize_docvqa_for_match,
    output_token_length,
    postprocess_docvqa_answer,
    raw_exact_match,
    relaxed_numeric_accuracy,
    routing_accuracy,
    strict_containment,
    token_f1,
    vqa_soft_score,
)


def test_normalize_answer() -> None:
    assert normalize_answer(" The Revenue, 2021! ") == "revenue 2021"


def test_docvqa_postprocess_preserves_full_dates() -> None:
    assert postprocess_docvqa_answer("September 3 to 9 ,1972 .") == "September 3 to 9, 1972"


def test_docvqa_normalization_matches_formatting_equivalents() -> None:
    assert normalize_docvqa_for_match("14,000") == normalize_docvqa_for_match("14000")
    assert normalize_docvqa_for_match("$2,000") == normalize_docvqa_for_match("$2000")
    assert normalize_docvqa_for_match("33,600") == normalize_docvqa_for_match("33600")
    assert normalize_docvqa_for_match("E. G. Farrier") == normalize_docvqa_for_match("E.G.Farrier")
    assert normalize_docvqa_for_match("713 - 792 - 3493") == normalize_docvqa_for_match("713-792-3493")
    assert normalize_docvqa_for_match("The Coca-Cola Company") == normalize_docvqa_for_match("Coca-Cola Company")


def test_docvqa_normalization_keeps_wrong_answers_wrong() -> None:
    assert normalize_docvqa_for_match("0.1") != normalize_docvqa_for_match("0.15 mls")
    assert normalize_docvqa_for_match("0") != normalize_docvqa_for_match("50")
    assert normalize_docvqa_for_match("50557 9766") != normalize_docvqa_for_match("503781642")
    assert normalize_docvqa_for_match("Dr. Gio Batta Gori") != normalize_docvqa_for_match("Gloria Geri")


def test_docvqa_short_span_extraction_is_conservative() -> None:
    assert maybe_extract_docvqa_short_span("10 mg/kg", "What is the recommended maximum residue limit?", ["10"]) == "10"
    assert maybe_extract_docvqa_short_span("$36,000 (4 waves)", "What is the amount?", ["$36,000"]) == "$36,000"
    assert maybe_extract_docvqa_short_span("George Washington", "Where?", ["Washington"]) == "Washington"
    assert maybe_extract_docvqa_short_span("0.15 mls", "What is the value?", ["0.1"]) == "0.15 mls"
    assert docvqa_exact_match("$2,000", ["$2000"]) == 1.0


def test_exact_match_accepts_any_ground_truth_answer() -> None:
    assert exact_match("2021.", ["2020", "2021"]) == 1.0


def test_raw_exact_match_preserves_surface_form() -> None:
    assert raw_exact_match("New York", ["New York"]) == 1.0
    assert raw_exact_match("new york", ["New York"]) == 0.0


def test_token_f1_uses_best_reference() -> None:
    assert token_f1("new york city", ["new york"]) == pytest.approx(0.8)


def test_strict_containment_detects_answer_span_without_substring_matches() -> None:
    assert strict_containment("the answer is New York City", ["New York"]) == 1.0
    assert strict_containment("CMROJOURNAL", ["CMRO"]) == 0.0
    assert strict_containment("notebook", ["no"]) == 0.0
    assert strict_containment("Gray (light grey)", ["gray"]) == 1.0
    assert strict_containment("64%", ["64"]) == 0.0
    assert strict_containment("19346.08", ["19346"]) == 0.0
    assert containment("CMROJOURNAL", ["CMRO"]) == 0.0


def test_relaxed_numeric_accuracy() -> None:
    assert relaxed_numeric_accuracy("19346.08", ["19346"]) == 1.0
    assert relaxed_numeric_accuracy("15.08647", ["15"]) == 1.0
    assert relaxed_numeric_accuracy("64%", ["64"]) == 1.0
    assert relaxed_numeric_accuracy("2.465833", ["10.13"]) == 0.0


def test_chart_hybrid_accuracy_handles_numeric_and_label_answers() -> None:
    assert chart_hybrid_accuracy("19346.08", ["19346"]) == 1.0
    assert chart_hybrid_accuracy("15.08647", ["15"]) == 1.0
    assert chart_hybrid_accuracy("64%", ["64"]) == 1.0
    assert chart_hybrid_accuracy("2.465833", ["10.13"]) == 0.0
    assert chart_hybrid_accuracy("Yes", ["Yes"]) == 1.0
    assert chart_hybrid_accuracy("Gray (light grey)", ["gray"]) == 1.0


def test_anls_thresholds_low_similarity() -> None:
    assert anls("Bausch Lomb", ["Bausch & Lomb"]) > 0.5
    assert anls("zzzz", ["Bausch & Lomb"]) == 0.0


def test_vqa_soft_score_counts_repeated_references() -> None:
    assert vqa_soft_score("beauregard", ["beauregard", "beauregard", "chablis"]) == pytest.approx(2 / 3)
    assert vqa_soft_score("unanswerable", ["no answer", "cannot be determined", "blue"]) == pytest.approx(2 / 3)


def test_output_token_length() -> None:
    assert output_token_length("New York City") == 3


def test_exact_match_rejects_wrong_answer() -> None:
    assert exact_match("2022", ["2021"]) == 0.0


def test_mean_score() -> None:
    assert mean_score([1.0, 0.0, 1.0]) == pytest.approx(2 / 3)


def test_mean_score_empty_list() -> None:
    assert mean_score([]) == 0.0


def test_routing_accuracy() -> None:
    predicted_tasks = ["chart_qa", "document_qa", "image_vqa"]
    target_tasks = ["chart_qa", "document_qa", "document_qa"]

    assert routing_accuracy(predicted_tasks, target_tasks) == pytest.approx(2 / 3)


def test_routing_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        routing_accuracy(["chart_qa"], ["chart_qa", "document_qa"])
