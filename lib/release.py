"""Candidate-bound evidence, approval, and release records for stages 04 and 05."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from lib.data import read_jsonl
    from lib.evaluation import RAG_METRICS, context_text
else:
    from evaluation import RAG_METRICS, context_text

    from data import read_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_release_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("Release ID must contain only letters, digits, underscores, or hyphens")
    return value


def write_json(value: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return target


@dataclass(frozen=True)
class ReleasePolicy:
    min_faithfulness: float = 0.7
    min_relevancy: float = 0.7
    min_correctness: float = 0.7
    correctness_tolerance: float = 0.0
    max_mean_latency_ms: float = 20_000
    max_p95_latency_ms: float = 40_000
    min_examples: int = 30
    min_token_f1: float | None = None

    def __post_init__(self):
        for name in ("min_faithfulness", "min_relevancy", "min_correctness", "correctness_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        for name in ("max_mean_latency_ms", "max_p95_latency_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.min_examples, bool) or not isinstance(self.min_examples, int) or self.min_examples < 1:
            raise ValueError("min_examples must be a positive integer")
        if self.min_token_f1 is not None and (
            not math.isfinite(self.min_token_f1) or not 0 <= self.min_token_f1 <= 1
        ):
            raise ValueError("min_token_f1 must be between 0 and 1")


@dataclass(frozen=True)
class DeploymentPolicy:
    min_token_f1: float = 0.7
    max_mean_latency_ms: float = 20_000
    max_p95_latency_ms: float = 40_000
    min_examples: int = 30

    def __post_init__(self):
        if (
            isinstance(self.min_token_f1, bool)
            or not math.isfinite(self.min_token_f1)
            or not 0 <= self.min_token_f1 <= 1
        ):
            raise ValueError("min_token_f1 must be a finite number between 0 and 1")
        for name in ("max_mean_latency_ms", "max_p95_latency_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.min_examples, bool)
            or not isinstance(self.min_examples, int)
            or self.min_examples < 1
        ):
            raise ValueError("min_examples must be a positive integer")


def _number(value: Any, field: str, *, score: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or (score and not 0 <= parsed <= 1):
        raise ValueError(f"{field} is not a finite {'0-1 score' if score else 'number'}")
    return parsed


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_test_rows(
    rows: Iterable[Mapping[str, Any]], test_records: Iterable[Mapping[str, Any]],
    *, prefixes: tuple[str, ...] = ("oob", "fine_tuned", "gpt"),
) -> list[dict[str, Any]]:
    records = list(test_records)
    if not records or any(not str(record.get("id", "")).strip() for record in records):
        raise ValueError("Sealed test records must have non-empty IDs")
    observed = [dict(row) for row in rows]
    source_indices = []
    for row in observed:
        value = row.get("source_index")
        if isinstance(value, bool):
            raise ValueError("Evaluation source_index must be an integer")
        try:
            source_index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Evaluation source_index must be an integer") from exc
        if str(source_index) != str(value).strip():
            raise ValueError("Evaluation source_index must be an integer")
        source_indices.append(source_index)
    if (
        len(source_indices) != len(records)
        or len(set(source_indices)) != len(source_indices)
        or set(source_indices) != set(range(len(records)))
    ):
        raise ValueError(
            "Evaluation must cover exactly the complete sealed test set by source_index"
        )
    for row in observed:
        source = records[int(row["source_index"])]
        for name, value in (
            ("id", str(source["id"])), ("question", source["question"]),
            ("cot_answer", source["cot_answer"]),
            ("context", context_text(source["context"])), ("type", str(source.get("type", ""))),
        ):
            if row.get(name) != value:
                raise ValueError(f"Record {row['id']} has mismatched {name}")
        if row.get("evaluation_framework") != "deepeval":
            raise ValueError("Release evidence must be evaluated with DeepEval")
        for prefix in prefixes:
            if row.get(f"{prefix}_inference_error") or row.get(f"{prefix}_rag_error"):
                raise ValueError(f"Record {row['id']} has failed {prefix} inference/evaluation")
            if not str(row.get(f"{prefix}_slm_response", "")).strip():
                raise ValueError(f"Record {row['id']} has no {prefix} response")
            latency = _number(row.get(f"{prefix}_slm_latency_ms"), f"{prefix}_latency_ms")
            if latency <= 0:
                raise ValueError("Model latency must be positive")
            for metric in RAG_METRICS:
                field = f"{prefix}_{metric}_score"
                row[field] = _number(row.get(field), field, score=True)
                if not str(row.get(f"{prefix}_{metric}_reason", "")).strip():
                    raise ValueError(f"Record {row['id']} has no {prefix} {metric} reason")
    return observed


def assess_rows(
    rows: list[dict[str, Any]], policy: ReleasePolicy, *, compare_gpt: bool,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Release gate requires evaluated rows")
    reasons = []
    if len(rows) < policy.min_examples:
        reasons.append(f"Only {len(rows)} examples; at least {policy.min_examples} required")
    means = {
        metric: statistics.mean(_number(row[f"fine_tuned_{metric}_score"], metric, score=True)
                                for row in rows)
        for metric in RAG_METRICS
    }
    slices = {}
    for sample_type in sorted({str(row.get("type", "")) for row in rows}):
        group = [row for row in rows if str(row.get("type", "")) == sample_type]
        slices[sample_type or "unspecified"] = {
            "count": len(group),
            **{metric: statistics.mean(row[f"fine_tuned_{metric}_score"] for row in group)
               for metric in RAG_METRICS},
        }
    for metric in RAG_METRICS:
        threshold = getattr(policy, f"min_{metric}")
        if means[metric] < threshold:
            reasons.append(f"{metric}={means[metric]:.4f} is below {threshold:.4f}")
        for label, values in slices.items():
            if values[metric] < threshold:
                reasons.append(f"{label} slice {metric}={values[metric]:.4f} is below {threshold:.4f}")
    if compare_gpt:
        baseline = statistics.mean(row["gpt_correctness_score"] for row in rows)
        if means["correctness"] + policy.correctness_tolerance < baseline:
            reasons.append("Fine-tuned correctness falls below the GPT baseline")
    else:
        baseline = None
    latencies = [_number(row["fine_tuned_slm_latency_ms"], "latency") for row in rows]
    mean_latency, p95_latency = statistics.mean(latencies), _p95(latencies)
    if mean_latency > policy.max_mean_latency_ms:
        reasons.append("Mean latency exceeds the approved budget")
    if p95_latency > policy.max_p95_latency_ms:
        reasons.append("p95 latency exceeds the approved budget")
    if policy.min_token_f1 is not None:
        token_f1_mean = statistics.mean(_number(row["answer_token_f1"], "answer_token_f1", score=True)
                                       for row in rows)
        if token_f1_mean < policy.min_token_f1:
            reasons.append("Answer token-F1 falls below the approved budget")
    return {
        "passed": not reasons, "reasons": reasons, "row_count": len(rows), "means": means,
        "slices": slices, "gpt_correctness": baseline, "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency, "policy": asdict(policy),
    }


def assess_deployment_rows(
    rows: Iterable[Mapping[str, Any]],
    test_records: Iterable[Mapping[str, Any]],
    policy: DeploymentPolicy,
) -> dict[str, Any]:
    """Assess live endpoint evidence against the complete sealed test set."""
    source = list(test_records)
    expected_ids = [str(record["id"]) for record in source]
    if not source or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Sealed test records must have non-empty, unique IDs")

    observed = [dict(row) for row in rows]
    observed_ids = [str(row.get("id", "")) for row in observed]
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != set(expected_ids):
        raise ValueError(
            "Deployment evaluation must cover exactly the complete sealed test set"
        )
    expected = {str(record["id"]): record for record in source}
    for row in observed:
        source_row = expected[str(row["id"])]
        if row.get("question") != source_row["question"]:
            raise ValueError(f"Record {row['id']} has a mismatched question")
        if row.get("reference") != source_row["cot_answer"]:
            raise ValueError(f"Record {row['id']} has a mismatched reference")

    token_f1_scores = [
        _number(row.get("answer_token_f1"), "answer_token_f1", score=True)
        for row in observed
    ]
    latencies = []
    for row in observed:
        if not str(row.get("prediction", "")).strip():
            raise ValueError(f"Record {row.get('id')} has no endpoint prediction")
        latency = _number(row.get("latency_ms"), "latency_ms")
        if latency <= 0:
            raise ValueError("Endpoint latency must be positive")
        latencies.append(latency)

    mean_token_f1 = statistics.mean(token_f1_scores)
    mean_latency = statistics.mean(latencies)
    p95_latency = _p95(latencies)
    reasons = []
    if len(observed) < policy.min_examples:
        reasons.append(
            f"Only {len(observed)} examples; at least {policy.min_examples} required"
        )
    if mean_token_f1 < policy.min_token_f1:
        reasons.append(
            f"answer_token_f1={mean_token_f1:.4f} is below {policy.min_token_f1:.4f}"
        )
    if mean_latency > policy.max_mean_latency_ms:
        reasons.append("Mean latency exceeds the approved budget")
    if p95_latency > policy.max_p95_latency_ms:
        reasons.append("p95 latency exceeds the approved budget")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "row_count": len(observed),
        "mean_token_f1": mean_token_f1,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "policy": asdict(policy),
    }


def write_evaluation_manifest(
    predictions_csv: str | Path, test_path: str | Path,
    training_job_name: str | None, evaluation_run_id: str,
) -> Path:
    predictions = Path(predictions_csv)
    return write_json({
        "schema_version": 1, "stage": 3, "framework": "deepeval",
        "training_job_name": training_job_name, "evaluation_run_id": evaluation_run_id,
        "test_sha256": file_sha256(test_path),
        "predictions_file": predictions.name, "predictions_sha256": file_sha256(predictions),
        "created_at": utc_now(),
    }, predictions.parent / "evaluation_evidence.json")


def load_inference_evidence(
    evidence_dir: str | Path, test_path: str | Path, training_job_name: str,
    policy: ReleasePolicy,
) -> dict[str, Any]:
    directory = Path(evidence_dir)
    manifests = list(directory.rglob("evaluation_evidence.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one downloaded evaluation_evidence.json")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("stage") != 3
            or manifest.get("framework") != "deepeval"
            or manifest.get("training_job_name") != training_job_name):
        raise ValueError("Evaluation evidence is not bound to the requested training candidate")
    if manifest.get("test_sha256") != file_sha256(test_path):
        raise ValueError("Evaluation evidence uses a different sealed test file")
    filename = manifest.get("predictions_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Evidence predictions_file must be a single filename")
    predictions = manifest_path.parent / filename
    if manifest.get("predictions_sha256") != file_sha256(predictions):
        raise ValueError("Evaluation CSV has changed since its manifest was produced")
    with predictions.open(encoding="utf-8", newline="") as stream:
        rows = validate_test_rows(csv.DictReader(stream), read_jsonl(test_path))
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "test_sha256": manifest["test_sha256"],
        "predictions_sha256": manifest["predictions_sha256"],
        "predictions_file": str(predictions),
        "gate": assess_rows(rows, policy, compare_gpt=True),
    }


def approval_record(approved: bool, actor: str | None, reason: str | None, evidence_sha: str) -> dict:
    if not approved or not actor or not actor.strip() or not reason or not reason.strip():
        raise ValueError("Explicit --approve, --approved-by, and --approval-reason are required")
    return {
        "approved_by": actor.strip(), "reason": reason.strip(),
        "approved_at": utc_now(), "evidence_sha256": evidence_sha,
    }


def save_release(record: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, suffix=".yaml", delete=False,
    ) as stream:
        yaml.safe_dump(dict(record), stream, sort_keys=False)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_release(path: str | Path) -> dict[str, Any]:
    record = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Unsupported release receipt")
    validate_release_id(record["release_id"])
    return record


def input_path(job, name: str) -> str:
    value = job.inputs[name]
    path = value if isinstance(value, str) else value.path
    if not isinstance(path, str) or not path:
        raise ValueError(f"Job input {name} has no immutable source reference")
    return path


def is_training_output_path(path: str, training_job_name: str) -> bool:
    if path == f"azureml://jobs/{training_job_name}/outputs/model":
        return True
    suffix = f"/paths/azureml/{training_job_name}/model/"
    return (
        path.startswith("azureml://")
        and "/datastores/" in path
        and path.endswith(suffix)
    )


def validate_cloud_evaluation(ml_client, training_job_name: str, evaluation_job_name: str) -> tuple[Any, Any]:
    training = ml_client.jobs.get(training_job_name)
    evaluation = ml_client.jobs.get(evaluation_job_name)
    if training.status != "Completed" or evaluation.status != "Completed":
        raise ValueError("Training and inference evaluation jobs must both be Completed")
    if (evaluation.tags or {}).get("finetuning_job_name") != training_job_name:
        raise ValueError("Inference job tags refer to a different training candidate")
    if not is_training_output_path(
        input_path(evaluation, "fine_tuned_model"), training_job_name
    ):
        raise ValueError("Inference did not mount the requested candidate's adapter output")
    if input_path(training, "base_model") != input_path(evaluation, "base_model"):
        raise ValueError("Inference used different base weights from training")
    return training, evaluation
