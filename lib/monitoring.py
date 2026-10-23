"""Feature-profile drift detection for SLM Studio requests."""

from __future__ import annotations

import hashlib
import math
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import numpy as np

FEATURE_NAMES = ("instruction_chars", "question_chars", "document_count")


def _positive_threshold(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("PSI threshold must be finite and positive")


def _probabilities(values: list[float]) -> np.ndarray:
    try:
        raw = np.asarray(values)
        probabilities = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be a finite positive vector") from exc
    if (
        probabilities.ndim != 1
        or raw.dtype.kind not in "iuf"
        or not probabilities.size
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0)
        or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-8)
    ):
        raise ValueError("Probabilities must be finite, positive, and sum to one")
    return probabilities


def _features(record: dict[str, Any]) -> dict[str, float]:
    if not isinstance(record, dict) or any(
        not isinstance(record.get(name), str) for name in ("instruction", "question")
    ):
        raise ValueError("Request records must contain instruction and question strings")
    instruction = record["instruction"]
    question = record["question"]
    return {
        "instruction_chars": float(len(instruction)),
        "question_chars": float(len(question)),
        "document_count": float(instruction.count("<DOCUMENT>")),
    }


def create_reference_profile(records: Iterable[dict[str, Any]], bins: int = 10) -> dict:
    """Create a reference profile for feature-profile drift detection."""
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    rows = list(records)
    if not rows:
        raise ValueError("Reference records cannot be empty")
    features = [_features(row) for row in rows]
    profile: dict[str, Any] = {"row_count": len(rows), "features": {}}
    for name in features[0]:
        values = np.asarray([row[name] for row in features], dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))).tolist()
        if len(edges) == 1:
            edges = [edges[0] - 0.5, edges[0] + 0.5]
        counts, _ = np.histogram(values, bins=np.asarray(edges))
        probabilities = ((counts + 1e-6) / (counts.sum() + len(counts) * 1e-6)).tolist()
        profile["features"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "edges": edges,
            "probabilities": probabilities,
        }
    return profile


def population_stability_index(reference: list[float], observed: list[float]) -> float:
    expected = _probabilities(reference)
    actual = _probabilities(observed)
    if len(expected) != len(actual):
        raise ValueError("Reference and observed probability vectors must have matching lengths")
    # Subtract logs rather than dividing: finite positive inputs can have an infinite ratio.
    return float(np.sum((actual - expected) * (np.log(actual) - np.log(expected))))


def _validate_profile(profile: dict) -> None:
    if not isinstance(profile, dict):
        raise ValueError("Reference profile must be an object")
    count = profile.get("row_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("Reference profile row_count must be a positive integer")
    baselines = profile.get("features")
    if not isinstance(baselines, dict) or set(baselines) != set(FEATURE_NAMES):
        raise ValueError(f"Reference profile must contain features: {FEATURE_NAMES}")
    for name, baseline in baselines.items():
        if not isinstance(baseline, dict):
            raise ValueError(f"Invalid reference feature: {name}")
        for statistic in ("mean", "std"):
            value = baseline.get(statistic)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Reference {name}.{statistic} must be finite and nonnegative")
        try:
            raw_edges = np.asarray(baseline.get("edges"))
            edges = np.asarray(baseline.get("edges"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reference edges for {name}") from exc
        if (
            edges.ndim != 1
            or raw_edges.dtype.kind not in "iuf"
            or len(edges) < 2
            or not np.all(np.isfinite(edges))
            or not np.all(edges[1:] > edges[:-1])
        ):
            raise ValueError(f"Reference edges for {name} must be finite and strictly increasing")
        probabilities = _probabilities(baseline.get("probabilities"))
        if len(probabilities) != len(edges) - 1:
            raise ValueError(f"Reference edges and probabilities for {name} do not match")


def detect_drift(
    reference_profile: dict,
    production_records: Iterable[dict[str, Any]],
    psi_threshold: float = 0.2,
) -> dict:
    """Compare normalized request features, folding observations outside the baseline into tail bins."""
    _positive_threshold(psi_threshold)
    _validate_profile(reference_profile)
    rows = list(production_records)
    if not rows:
        raise ValueError("Production records cannot be empty")
    features = [_features(row) for row in rows]
    report: dict[str, Any] = {
        "row_count": len(rows),
        "psi_threshold": psi_threshold,
        "features": {},
    }
    for name, baseline in reference_profile["features"].items():
        edges = np.asarray(baseline["edges"], dtype=float)
        values = np.asarray([row[name] for row in features], dtype=float)
        # Histogram otherwise silently drops out-of-range production observations.
        values = np.clip(values, edges[0], edges[-1])
        counts, _ = np.histogram(values, bins=edges)
        observed = ((counts + 1e-6) / (counts.sum() + len(counts) * 1e-6)).tolist()
        psi = population_stability_index(baseline["probabilities"], observed)
        report["features"][name] = {
            "psi": psi,
            "drift_detected": psi >= psi_threshold,
            "production_mean": float(values.mean()),
        }
    report["drift_detected"] = any(
        value["drift_detected"] for value in report["features"].values()
    )
    return report


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def validate_production_uri(value: str) -> str:
    uri = _required_text("production_data_uri", value)
    if re.search(
        r"[<>{}\s]|placeholder|replace[-_]|your[-_]|changeme|example|(?:^|[/.:_-])todo(?:$|[/.:_-])",
        uri,
        re.IGNORECASE,
    ):
        raise ValueError("production_data_uri must be a nonplaceholder governed Azure ML or HTTPS URI")
    parsed = urlsplit(uri)
    if parsed.scheme == "https":
        valid = bool(parsed.hostname and parsed.path.strip("/")) and not (
            parsed.username or parsed.password or parsed.fragment
        )
    elif uri.startswith("azureml://"):
        valid = bool(parsed.netloc and parsed.path.strip("/")) and not (
            parsed.query or parsed.fragment
        )
    else:
        valid = bool(re.fullmatch(
            r"azureml:[A-Za-z0-9][A-Za-z0-9_.-]*:[A-Za-z0-9][A-Za-z0-9_.-]*"
            r"|azureml:/subscriptions/[^/]+/resourceGroups/[^/]+/providers/"
            r"Microsoft\.MachineLearningServices/workspaces/[^/]+/data/[^/]+/versions/[^/?#]+",
            uri,
        ))
    if not valid:
        raise ValueError("production_data_uri must be a governed Azure ML or HTTPS URI")
    return uri


def create_drift_schedule(
    ml_client,
    *,
    schedule_name: str,
    reference_profile: Path,
    production_data_uri: str,
    environment: str,
    compute: str,
    psi_threshold: float = 0.2,
    frequency: str = "week",
    interval: int = 1,
    hour: int = 2,
    managed_identity_client_id: str | None = None,
):
    """Create and await a UTC request-feature drift schedule on caller-selected CPU compute.

    Production must be a governed folder of normalized JSONL request records containing
    ``instruction`` and ``question`` strings, not a training/test split or raw endpoint
    telemetry. The existing inference environment must supply numpy and mlflow.
    The baseline content hash is recorded in both job and schedule tags.
    SDK imports are lazy; validation never contacts Azure.
    """
    schedule_name = _required_text("schedule_name", schedule_name)
    environment = _required_text("environment", environment)
    compute = _required_text("compute", compute)
    production_data_uri = validate_production_uri(production_data_uri)
    reference_profile = Path(reference_profile).resolve()
    if not reference_profile.is_file():
        raise ValueError("reference_profile must be an existing file")
    _positive_threshold(psi_threshold)
    if frequency not in {"minute", "hour", "day", "week", "month"}:
        raise ValueError("frequency must be minute, hour, day, week, or month")
    if isinstance(interval, bool) or not isinstance(interval, Integral) or interval <= 0:
        raise ValueError("interval must be a positive integer")
    if isinstance(hour, bool) or not isinstance(hour, Integral) or not 0 <= hour <= 23:
        raise ValueError("hour must be an integer between 0 and 23")
    if managed_identity_client_id is not None:
        managed_identity_client_id = _required_text(
            "managed_identity_client_id", managed_identity_client_id
        )

    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.dsl import pipeline
    from azure.ai.ml.entities import (
        JobSchedule,
        ManagedIdentityConfiguration,
        RecurrencePattern,
        RecurrenceTrigger,
    )

    tags = {
        "monitoring_schedule": schedule_name,
        "baseline_sha256": hashlib.sha256(reference_profile.read_bytes()).hexdigest(),
        "monitoring_type": "request_feature_psi",
    }
    job_options = {}
    if managed_identity_client_id is not None:
        job_options["identity"] = ManagedIdentityConfiguration(
            client_id=managed_identity_client_id
        )
    job = command(
        code=str(Path(__file__).resolve().parents[1]),
        command=(
            'python -m lib.monitor_drift'
            ' --reference-profile "${{inputs.reference_profile}}"'
            ' --production-data "${{inputs.production_data}}"'
            ' --output "${{outputs.report}}"'
            ' --psi-threshold ${{inputs.psi_threshold}}'
        ),
        inputs={
            "reference_profile": Input(type=AssetTypes.URI_FILE, path=str(reference_profile)),
            "production_data": Input(type=AssetTypes.URI_FOLDER, path=production_data_uri),
            "psi_threshold": psi_threshold,
        },
        outputs={"report": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
        compute=compute,
        display_name=f"{schedule_name}-drift",
        experiment_name="request-feature-monitoring",
        is_deterministic=False,
        tags=tags,
        **job_options,
    )

    # Azure ML schedules accept pipeline jobs, not standalone command jobs.
    @pipeline(name=f"{schedule_name}-pipeline")
    def monitoring_pipeline():
        drift = job()
        return {"report": drift.outputs.report}

    scheduled_job = monitoring_pipeline()
    scheduled_job.tags = dict(tags)
    scheduled_job.settings.force_rerun = True
    trigger = RecurrenceTrigger(
        frequency=frequency,
        interval=interval,
        time_zone="UTC",
        schedule=RecurrencePattern(hours=[hour], minutes=[0]),
    )
    schedule = JobSchedule(
        name=schedule_name,
        trigger=trigger,
        create_job=scheduled_job,
        tags=tags,
        description="PSI monitoring of governed normalized production request features",
    )
    return ml_client.schedules.begin_create_or_update(schedule).result()
