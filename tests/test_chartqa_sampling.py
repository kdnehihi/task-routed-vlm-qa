from src.data.chartqa_sampling import (
    classify_answer_type,
    classify_chartqa_question,
    stratified_chartqa_sample,
)


def test_classify_chartqa_question_rules():
    assert classify_chartqa_question("What is the ratio of A to B?") == "ratio"
    assert classify_chartqa_question("What is the difference between 2020 and 2021?") == "difference"
    assert classify_chartqa_question("What is the average sales value?") == "average"
    assert classify_chartqa_question("Is Europe higher than Asia?") == "yes_no_compare"
    assert classify_chartqa_question("How many bars are above 10?") == "counting"
    assert classify_chartqa_question("Which country has the largest value?") == "extreme"
    assert classify_chartqa_question("What percentage is shown for 2019?") == "percent_decimal"


def test_classify_answer_type_rules():
    assert classify_answer_type(["42.5%"]) == "numeric"
    assert classify_answer_type(["No"]) == "yes_no"
    assert classify_answer_type(["2021"]) == "date_or_year"
    assert classify_answer_type(["South Korea"]) == "text_label"
    assert classify_answer_type([]) == "other"


def test_stratified_chartqa_sample_respects_quotas_and_fills():
    examples = [
        {"question": "What is the value for A?", "answers": ["1"]},
        {"question": "What is the value for B?", "answers": ["2"]},
        {"question": "What is the difference between A and B?", "answers": ["1"]},
        {"question": "Is A higher than B?", "answers": ["No"]},
    ]

    sampled = stratified_chartqa_sample(
        examples,
        quotas={"lookup_value": 1, "difference": 1},
        seed=7,
        sample_limit=3,
    )

    assert len(sampled) == 3
    assert len({id(example) for example in sampled}) == 3
    assert sum(example["question_type"] == "lookup_value" for example in sampled) >= 1
    assert sum(example["question_type"] == "difference" for example in sampled) == 1
