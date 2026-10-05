from types import SimpleNamespace

import pytest

from lib.release import (
    DeploymentPolicy,
    assess_deployment_rows,
    is_training_output_path,
    validate_test_rows,
    validate_cloud_evaluation,
)


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


@pytest.mark.parametrize(
    "path",
    [
        "azureml://jobs/training-job/outputs/model",
        "azureml://datastores/workspaceblobstore/paths/azureml/training-job/model/",
        (
            "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/"
            "datastores/workspaceblobstore/paths/azureml/training-job/model/"
        ),
    ],
)
def test_training_output_path_accepts_supported_immutable_references(path) -> None:
    assert is_training_output_path(path, "training-job") is True


@pytest.mark.parametrize(
    "path",
    [
        "azureml://jobs/other-job/outputs/model",
        "azureml://datastores/workspaceblobstore/paths/azureml/other-job/model/",
        "azureml://datastores/workspaceblobstore/paths/azureml/training-job/model/extra",
    ],
)
def test_training_output_path_rejects_other_artifacts(path) -> None:
    assert is_training_output_path(path, "training-job") is False


def test_cloud_evaluation_accepts_concrete_training_output_datastore_uri() -> None:
    base_model = "azureml://registries/azureml-meta/models/model/versions/1"
    training = SimpleNamespace(
        status="Completed",
        inputs={"base_model": SimpleNamespace(path=base_model)},
    )
    evaluation = SimpleNamespace(
        status="Completed",
        tags={"finetuning_job_name": "training-job"},
        inputs={
            "base_model": SimpleNamespace(path=base_model),
            "fine_tuned_model": SimpleNamespace(
                path=(
                    "azureml://datastores/workspaceblobstore/paths/"
                    "azureml/training-job/model/"
                )
            ),
        },
    )
    jobs = {
        "training-job": training,
        "evaluation-job": evaluation,
    }
    client = SimpleNamespace(
        jobs=SimpleNamespace(get=lambda name: jobs[name])
    )

    assert validate_cloud_evaluation(
        client, "training-job", "evaluation-job"
    ) == (training, evaluation)


def _evaluation_row(source_index: int, record: dict) -> dict:
    row = {
        "source_index": str(source_index),
        "id": record["id"],
        "type": record["type"],
        "question": record["question"],
        "context": record["context"]["sentences"][0][0],
        "cot_answer": record["cot_answer"],
        "evaluation_framework": "deepeval",
    }
    for prefix in ("oob", "fine_tuned", "gpt"):
        row[f"{prefix}_inference_error"] = ""
        row[f"{prefix}_rag_error"] = ""
        row[f"{prefix}_slm_response"] = "Answer"
        row[f"{prefix}_slm_latency_ms"] = "100"
        for metric in ("faithfulness", "relevancy", "correctness"):
            row[f"{prefix}_{metric}_score"] = "0.8"
            row[f"{prefix}_{metric}_reason"] = "Supported"
    return row


def test_evaluation_rows_bind_duplicate_ids_by_source_index() -> None:
    records = [
        {
            "id": "shared-id",
            "type": "oracle",
            "question": "First?",
            "context": {"sentences": [["First context"]]},
            "cot_answer": "<ANSWER>first</ANSWER>",
        },
        {
            "id": "shared-id",
            "type": "distractor",
            "question": "Second?",
            "context": {"sentences": [["Second context"]]},
            "cot_answer": "<ANSWER>second</ANSWER>",
        },
    ]
    rows = [_evaluation_row(index, record) for index, record in enumerate(records)]

    assert len(validate_test_rows(rows, records)) == 2


def test_evaluation_rows_reject_duplicate_source_indices() -> None:
    records = [
        {
            "id": "shared-id",
            "type": "oracle",
            "question": f"Question {index}?",
            "context": {"sentences": [[f"Context {index}"]]},
            "cot_answer": f"<ANSWER>{index}</ANSWER>",
        }
        for index in range(2)
    ]
    rows = [_evaluation_row(index, record) for index, record in enumerate(records)]
    rows[1]["source_index"] = "0"

    with pytest.raises(ValueError, match="complete sealed test set by source_index"):
        validate_test_rows(rows, records)
