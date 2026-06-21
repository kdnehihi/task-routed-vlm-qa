"""Evaluation metrics for vision-language QA."""

import re
import string

NO_ANSWER_NORMALIZED_LABELS = {
    "unanswerable",
    "no answer",
    "not answerable",
    "answer not available",
    "not question",
    "cannot be determined",
    "cannot determine",
    "n/a",
    "na",
    "answering does not require reading text in image",
}


def normalize_answer(text: str) -> str:
    """Normalize answer text for exact-match style evaluation."""
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_articles(text)
    text = fix_whitespace(text)
    return text


def normalize_text(text: str) -> str:
    """Return the normalized string used for diagnostic outputs."""
    return normalize_answer(text)


def postprocess_docvqa_answer(raw_answer: str, question: str | None = None) -> str:
    """Clean DocVQA predictions without destroying valid document spans."""
    del question
    text = str(raw_answer).strip()
    if not text:
        return ""

    text = normalize_docvqa_surface(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(?<=\D)([,;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"(?<=\d),(?=\d{4}\b)", ", ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\$\s+", "$", text)
    text = re.sub(r"\s+", " ", text).strip()

    if safe_remove_docvqa_trailing_punctuation(text):
        text = text[:-1].rstrip()
    return text


def normalize_docvqa_surface(text: str) -> str:
    """Normalize harmless Unicode and whitespace surface differences."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def safe_remove_docvqa_trailing_punctuation(text: str) -> bool:
    """Return whether final sentence punctuation can be dropped safely."""
    if not text or text[-1] not in ".!?":
        return False
    if re.search(r"\b[A-Z]\.$", text):
        return False
    if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|No|Inc|Ltd|Co)\.$", text):
        return False
    return True


def normalize_docvqa_for_match(text: str) -> str:
    """Normalize DocVQA answers for conservative formatting-equivalent matches."""
    normalized = postprocess_docvqa_answer(text).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\bthe\s+", "", normalized, count=1)
    normalized = re.sub(r"\$\s*", "$", normalized)
    normalized = re.sub(r"(?<=\d),\s*(?=\d{3}\b)", "", normalized)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = normalize_docvqa_initials(normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"(?<=\D)([,;:])(?=\S)", r"\1 ", normalized)
    normalized = re.sub(r"(?<=\d),(?=\d{4}\b)", ", ", normalized)
    normalized = normalized.strip()
    if safe_remove_docvqa_trailing_punctuation(normalized):
        normalized = normalized[:-1].rstrip()
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_docvqa_initials(text: str) -> str:
    """Collapse spacing in initials without deleting letters."""
    text = re.sub(r"\b([a-z])\.\s+(?=[a-z]\.)", r"\1.", text)
    text = re.sub(r"((?:\b[a-z]\.){2,})\s+(?=[a-z])", r"\1", text)
    return text


def docvqa_exact_match(prediction: str, answers: list[str]) -> float:
    """Return exact match using DocVQA-specific normalization."""
    normalized_prediction = normalize_docvqa_for_match(prediction)
    for answer in answers:
        if normalized_prediction == normalize_docvqa_for_match(answer):
            return 1.0
    return 0.0


def docvqa_anls(prediction: str, answers: list[str]) -> float:
    """Return ANLS using DocVQA-specific formatting normalization."""
    normalized_prediction = normalize_docvqa_for_match(prediction)
    if not normalized_prediction:
        return 0.0

    best_score = 0.0
    for answer in answers:
        normalized_answer = normalize_docvqa_for_match(answer)
        if not normalized_answer:
            continue
        distance = levenshtein_distance(normalized_prediction, normalized_answer)
        similarity = 1.0 - distance / max(len(normalized_prediction), len(normalized_answer))
        if similarity >= 0.5:
            best_score = max(best_score, similarity)
    return best_score


def maybe_extract_docvqa_short_span(
    pred: str,
    question: str,
    references: list[str],
) -> str:
    """Return a conservative reference span for DocVQA metric debugging."""
    cleaned_pred = postprocess_docvqa_answer(pred, question)
    normalized_pred = normalize_docvqa_for_match(cleaned_pred)
    if not normalized_pred:
        return cleaned_pred

    for reference in references:
        normalized_ref = normalize_docvqa_for_match(reference)
        if not normalized_ref or normalized_ref == normalized_pred:
            continue
        if safe_docvqa_span_match(normalized_ref, normalized_pred):
            return postprocess_docvqa_answer(reference, question)
    return cleaned_pred


def safe_docvqa_span_match(normalized_ref: str, normalized_pred: str) -> bool:
    """Allow only full-token span matches that are unlikely to flip meaning."""
    if normalized_ref not in normalized_pred:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_ref)}(?![a-z0-9])"
    if not re.search(pattern, normalized_pred):
        return False
    if is_decimal_like(normalized_ref):
        return False
    if is_plain_number(normalized_ref):
        return True
    if len(normalized_ref) >= 4 and len(normalized_ref.split()) <= 3:
        return True
    return False


def is_plain_number(text: str) -> bool:
    return bool(re.fullmatch(r"\$?\d+(?:\.\d+)?", text))


def is_decimal_like(text: str) -> bool:
    return bool(re.fullmatch(r"\d*\.\d+", text))


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from text."""
    return "".join(char for char in text if char not in string.punctuation)


def remove_articles(text: str) -> str:
    """Remove English articles from text."""
    return re.sub(r"\b(a|an|the)\b", " ", text)


def fix_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if prediction matches any normalized answer."""
    return normalized_exact_match(prediction, answers)


def raw_exact_match(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if prediction exactly matches any raw answer after trimming."""
    clean_prediction = str(prediction).strip()
    for answer in answers:
        if clean_prediction == str(answer).strip():
            return 1.0
    return 0.0


def normalized_exact_match(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if prediction matches any normalized answer."""
    normalized_prediction = normalize_answer(prediction)

    for answer in answers:
        if normalized_prediction == normalize_answer(answer):
            return 1.0

    return 0.0


def token_f1(prediction: str, answers: list[str]) -> float:
    """Return the best token F1 against normalized reference answers."""
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0

    best_score = 0.0
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens:
            continue
        overlap = 0
        remaining = answer_tokens.copy()
        for token in prediction_tokens:
            if token in remaining:
                overlap += 1
                remaining.remove(token)
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best_score = max(best_score, 2 * precision * recall / (precision + recall))
    return best_score


def containment(prediction: str, answers: list[str]) -> float:
    """Return token/span-aware containment without substring matches."""
    return strict_containment(prediction, answers)


def loose_containment(prediction: str, answers: list[str]) -> float:
    """Return legacy substring containment for regression diagnostics."""
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0

    for answer in answers:
        normalized_answer = normalize_answer(answer)
        if normalized_answer and normalized_answer in normalized_prediction:
            return 1.0
    return 0.0


def strict_containment(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if an answer appears as a full token or phrase."""
    prediction_tokens = containment_tokens(prediction)
    if not prediction_tokens:
        return 0.0

    for answer in answers:
        answer_tokens = containment_tokens(answer)
        if not answer_tokens:
            continue
        if contains_token_span(prediction_tokens, answer_tokens):
            return 1.0
    return 0.0


def containment_tokens(text: str) -> list[str]:
    """Tokenize for containment while preserving percent-attached numbers."""
    return re.findall(r"[a-z]+|\d+(?:[,.]\d+)*%?", str(text).lower())


def contains_token_span(tokens: list[str], span: list[str]) -> bool:
    """Return whether span appears as a contiguous token sequence."""
    if len(span) > len(tokens):
        return False
    for start in range(len(tokens) - len(span) + 1):
        if tokens[start:start + len(span)] == span:
            return True
    return False


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values, handling commas and percent signs."""
    values = []
    for match in re.finditer(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?", str(text)):
        value = match.group(0).replace(",", "").rstrip("%")
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def relaxed_numeric_accuracy(
    prediction: str,
    answers: list[str],
    tolerance: float = 0.05,
) -> float:
    """Return 1.0 if prediction/reference numeric values match within tolerance."""
    prediction_values = extract_numbers(prediction)
    if not prediction_values:
        return 0.0

    for answer in answers:
        reference_values = extract_numbers(answer)
        for predicted_value in prediction_values:
            for reference_value in reference_values:
                if reference_value == 0:
                    if abs(predicted_value - reference_value) <= tolerance:
                        return 1.0
                elif abs(predicted_value - reference_value) / abs(reference_value) <= tolerance:
                    return 1.0
    return 0.0


def chart_hybrid_accuracy(prediction: str, answers: list[str]) -> float:
    """Return ChartQA accuracy for numeric, yes/no, and label answers."""
    prediction_has_number = bool(extract_numbers(prediction))
    numeric_answers = [answer for answer in answers if extract_numbers(answer)]

    if prediction_has_number and numeric_answers:
        return relaxed_numeric_accuracy(prediction, numeric_answers)

    if normalized_exact_match(prediction, answers):
        return 1.0

    return strict_containment(prediction, answers)


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls(prediction: str, answers: list[str]) -> float:
    """Return best ANLS-like normalized edit similarity."""
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0

    best_score = 0.0
    for answer in answers:
        normalized_answer = normalize_answer(answer)
        if not normalized_answer:
            continue
        distance = levenshtein_distance(normalized_prediction, normalized_answer)
        similarity = 1.0 - distance / max(len(normalized_prediction), len(normalized_answer))
        if similarity >= 0.5:
            best_score = max(best_score, similarity)
    return best_score


def vqa_soft_score(prediction: str, answers: list[str]) -> float:
    """Return TextVQA/VQA-style soft accuracy from repeated references."""
    normalized_prediction = normalize_vqa_label(prediction)
    if not normalized_prediction:
        return 0.0
    match_count = sum(
        1
        for answer in answers
        if normalize_vqa_label(answer) == normalized_prediction
    )
    return min(1.0, match_count / 3.0)


def normalize_vqa_label(text: str) -> str:
    """Normalize labels for VQA-style soft scoring."""
    normalized = normalize_answer(text)
    if normalized in NO_ANSWER_NORMALIZED_LABELS:
        return "unanswerable"
    return normalized


def output_token_length(prediction: str) -> int:
    """Count whitespace tokens in a generated answer."""
    return len(str(prediction).split())


def mean_score(scores: list[float]) -> float:
    """Return the mean score, or 0.0 for an empty list."""
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def routing_accuracy(predicted_tasks: list[str], target_tasks: list[str]) -> float:
    """Compute task routing accuracy."""
    if len(predicted_tasks) != len(target_tasks):
        raise ValueError("predicted_tasks and target_tasks must have the same length.")

    if not predicted_tasks:
        return 0.0

    correct = 0
    for predicted, target in zip(predicted_tasks, target_tasks):
        if predicted == target:
            correct += 1

    return correct / len(predicted_tasks)
