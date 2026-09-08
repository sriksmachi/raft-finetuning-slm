from lib.evaluation import (
    aggregate_evaluation_metrics,
    aggregate_metrics,
    evaluate_predictions,
    exact_match,
    extract_answer,
    token_f1,
)


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


def test_evaluate_predictions_supports_offline_and_online_rows() -> None:
    received = {}

    def judge(metric, score):
        def evaluate(**arguments):
            received.update(arguments)
            return {f"gpt_{metric}": score, f"{metric}_reason": "Supported"}

        return evaluate

    judges = {
        "relevancy": judge("relevance", 4),
        "groundedness": judge("groundedness", 5),
        "coherence": judge("coherence", 4),
    }
    online_row = {
        "question": "How often?",
        "prediction": "<ANSWER>Monthly</ANSWER>",
        "reference": "<ANSWER>Monthly</ANSWER>",
        "context": {"sentences": [["Patch monthly."]]},
        "latency_ms": 10,
    }

    evaluated = evaluate_predictions([online_row], judges=judges)

    assert evaluated[0]["relevancy"] == 4.0
    assert evaluated[0]["relevancy_reason"] == "Supported"
    assert received["query"] == "How often?"
    assert received["context"] == "Patch monthly."

    offline_row = dict(online_row, query="When?", response="Monthly")
    assert evaluate_predictions([offline_row], judges=judges)[0]["coherence"] == 4.0


def test_aggregate_evaluation_metrics_includes_judge_means() -> None:
    metrics = aggregate_evaluation_metrics(
        [
            {
                "prediction": "monthly",
                "reference": "monthly",
                "latency_ms": 10,
                "relevancy": 4,
                "groundedness": 5,
                "coherence": 3,
            }
        ]
    )

    assert metrics["relevancy"] == 4.0
    assert metrics["groundedness"] == 5.0
    assert metrics["coherence"] == 3.0
