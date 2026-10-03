"""Self-contained Azure ML schedules for governed production request monitoring."""

from __future__ import annotations

import hashlib
import re
from numbers import Integral
from pathlib import Path
from urllib.parse import urlsplit

from lib.monitoring import _positive_threshold


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def _production_uri(value: str) -> str:
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
    production_data_uri = _production_uri(production_data_uri)
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
