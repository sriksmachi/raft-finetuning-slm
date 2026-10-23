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

MODELS = ("fine_tuned", "oob", "gpt")


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


def read_candidate_evidence(
    evidence_dir: str | Path, test_path: str | Path, training_job_name: str,
    evaluation_job_name: str | None = None,
) -> tuple[dict, list[dict], dict]:
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
    if evaluation_job_name is not None and manifest.get("evaluation_run_id") != evaluation_job_name:
        raise ValueError("Evidence belongs to a different evaluation job")
    if manifest.get("test_sha256") != file_sha256(test_path):
        raise ValueError("Evaluation evidence uses a different sealed test file")
    filename = manifest.get("predictions_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Evidence predictions_file must be a single filename")
    predictions = manifest_path.parent / filename
    if manifest_path.is_symlink() or predictions.is_symlink():
        raise ValueError("Symlink release evidence is not supported")
    if manifest.get("predictions_sha256") != file_sha256(predictions):
        raise ValueError("Evaluation CSV has changed since its manifest was produced")
    with predictions.open(encoding="utf-8", newline="") as stream:
        rows = validate_test_rows(csv.DictReader(stream), read_jsonl(test_path))
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "test_sha256": manifest["test_sha256"],
        "predictions_sha256": manifest["predictions_sha256"],
        "predictions_file": str(predictions),
    }, rows, manifest


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


def policy_description() -> dict:
    return {
        "version": "paired-quality-v2",
        "metrics": list(RAG_METRICS),
        "minimum_examples": 30,
        "minimum_absolute_uplift": 0.02,
        "perfect_baseline_exception": {
            "baseline_score": 1.0,
            "candidate_score": 1.0,
            "waived_checks": ["uplift", "positive_confidence_interval"],
        },
        "maximum_relative_gpt_shortfall_exclusive": 0.05,
        "confidence": 0.95,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 42,
        "method": "Paired nonparametric percentile bootstrap, resampling complete sealed-test rows.",
        "excluded": ["latency", "token_usage", "token_cost"],
        "editable": False,
    }


def assess_candidate_rows(rows: list[dict]) -> dict:
    import numpy as np

    policy = policy_description()
    if not rows:
        raise ValueError("Readiness requires per-example sealed-test scores.")
    values = {}
    for model in MODELS:
        columns = []
        for metric in RAG_METRICS:
            scores = []
            for row in rows:
                raw = row.get(f"{model}_{metric}_score")
                if raw is None or isinstance(raw, bool):
                    raise ValueError(f"Missing {model} {metric} per-example score.")
                score = float(raw)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid {model} {metric} score.")
                scores.append(score)
            columns.append(scores)
        values[model] = np.array(columns, dtype=float).T
    differences = values["fine_tuned"] - values["oob"]
    rng = np.random.default_rng(policy["bootstrap_seed"])
    remaining = policy["bootstrap_replicates"]
    replicates = []
    # Bound temporary arrays for large sealed sets without changing the sampling method.
    batch_size = max(1, min(256, 500_000 // len(rows)))
    while remaining:
        count = min(batch_size, remaining)
        indexes = rng.integers(0, len(rows), size=(count, len(rows)))
        replicates.append(differences[indexes].mean(axis=1))
        remaining -= count
    intervals = np.quantile(np.concatenate(replicates), [0.025, 0.975], axis=0)
    reasons = []
    if len(rows) < policy["minimum_examples"]:
        reasons.append(f"At least 30 sealed-test examples required; observed {len(rows)}.")
    metrics = []
    for index, metric in enumerate(RAG_METRICS):
        means = {model: statistics.mean(values[model][:, index].tolist()) for model in MODELS}
        uplift = means["fine_tuned"] - means["oob"]
        gap = max(0.0, (means["gpt"] - means["fine_tuned"]) / means["gpt"]) if means["gpt"] else 0.0
        ceiling = policy["perfect_baseline_exception"]
        baseline_at_ceiling = bool(np.all(values["oob"][:, index] == ceiling["baseline_score"]))
        perfect_tie = baseline_at_ceiling and bool(
            np.all(values["fine_tuned"][:, index] == ceiling["candidate_score"])
        )
        checks = {
            "uplift": perfect_tie if baseline_at_ceiling else (
                uplift >= 0.02 or math.isclose(uplift, 0.02, rel_tol=0, abs_tol=1e-12)
            ),
            "gpt_gap": gap < 0.05 and not math.isclose(gap, 0.05, rel_tol=0, abs_tol=1e-12),
            "positive_confidence_interval": perfect_tie if baseline_at_ceiling else float(intervals[0, index]) > 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            reasons.append(
                f"{metric}: original SLM is 100%; fine-tuned scores must remain exactly 100% on every sealed-test example."
                if baseline_at_ceiling and not perfect_tie
                else f"{metric}: failed {', '.join(failed)}."
            )
        metrics.append({
            "metric": metric, **means, "absolute_uplift": uplift,
            "relative_gpt_shortfall": gap,
            "uplift_ci95": [float(intervals[0, index]), float(intervals[1, index])],
            "baseline_at_ceiling": baseline_at_ceiling,
            "ceiling_exception_applied": perfect_tie,
            "checks": checks, "passed": not failed and len(rows) >= 30,
        })
    return {
        "passed": not reasons, "status": "ready" if not reasons else "blocked",
        "sample_count": len(rows), "policy": policy, "metrics": metrics, "reasons": reasons,
    }
