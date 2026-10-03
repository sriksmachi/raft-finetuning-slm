import pytest

from lib.release import DeploymentPolicy, assess_deployment_rows


def _test_records(count: int = 3) -> list[dict]:
    return [
        {
            "id": f"row-{index}",
            "question": f"Question {index}?",
            "cot_answer": f"<ANSWER>answer {index}</ANSWER>",
        }
        for index in range(count)
    ]


def _deployment_rows(records: list[dict], *, token_f1: float = 0.8) -> list[dict]:
    return [
        {
            "id": record["id"],
            "question": record["question"],
            "reference": record["cot_answer"],
            "prediction": f"<ANSWER>answer {index}</ANSWER>",
            "latency_ms": 100 + index,
            "answer_token_f1": token_f1,
        }
        for index, record in enumerate(records)
    ]


def test_deployment_gate_passes_complete_bound_evidence() -> None:
    records = _test_records()
    result = assess_deployment_rows(
        _deployment_rows(records),
        records,
        DeploymentPolicy(
            min_token_f1=0.75,
            max_mean_latency_ms=200,
            max_p95_latency_ms=200,
            min_examples=3,
        ),
    )
    assert result["passed"] is True
    assert result["reasons"] == []
    assert result["mean_token_f1"] == pytest.approx(0.8)


def test_deployment_gate_reports_quality_latency_and_sample_failures() -> None:
    records = _test_records(2)
    rows = _deployment_rows(records, token_f1=0.4)
    rows[1]["latency_ms"] = 1_000
    result = assess_deployment_rows(
        rows,
        records,
        DeploymentPolicy(
            min_token_f1=0.7,
            max_mean_latency_ms=200,
            max_p95_latency_ms=500,
            min_examples=3,
        ),
    )
    assert result["passed"] is False
    assert len(result["reasons"]) == 4


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda rows: rows.pop(), "complete sealed test set"),
        (lambda rows: rows[0].update(question="other"), "mismatched question"),
        (lambda rows: rows[0].update(reference="other"), "mismatched reference"),
        (lambda rows: rows[0].update(prediction=""), "no endpoint prediction"),
        (lambda rows: rows[0].update(latency_ms=0), "latency must be positive"),
    ],
)
def test_deployment_gate_rejects_invalid_evidence(mutation, error) -> None:
    records = _test_records()
    rows = _deployment_rows(records)
    mutation(rows)
    with pytest.raises(ValueError, match=error):
        assess_deployment_rows(rows, records, DeploymentPolicy(min_examples=1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_token_f1": -0.1},
        {"min_token_f1": True},
        {"max_mean_latency_ms": 0},
        {"max_p95_latency_ms": float("inf")},
        {"min_examples": 0},
        {"min_examples": True},
    ],
)
def test_deployment_policy_rejects_invalid_thresholds(kwargs) -> None:
    with pytest.raises(ValueError):
        DeploymentPolicy(**kwargs)
