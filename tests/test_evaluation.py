from lib.evaluation import aggregate_metrics, exact_match, extract_answer, token_f1


def test_answer_metrics_ignore_reasoning_wrapper() -> None:
    prediction = "Reasoning <ANSWER>: Patch monthly </ANSWER>"
    reference = "<ANSWER>Patch monthly</ANSWER>"

    assert extract_answer(prediction) == "Patch monthly"
    assert exact_match(prediction, reference) == 1.0
    assert token_f1(prediction, reference) == 1.0


def test_aggregate_metrics() -> None:
    metrics = aggregate_metrics(
        [{"prediction": "monthly", "reference": "monthly", "latency_ms": 10}]
    )
    assert metrics["answer_token_f1"] == 1.0
    assert metrics["mean_latency_ms"] == 10.0
