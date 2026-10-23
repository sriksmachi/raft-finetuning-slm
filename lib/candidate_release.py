"""Shared candidate planning, registration and authenticated endpoint execution."""

import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lib.release import (
    approval_record,
    assess_candidate_rows,
    file_sha256,
    input_path,
    read_candidate_evidence,
    save_release,
    utc_now,
    validate_cloud_evaluation,
)


def plan_hash(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(plan: dict, expected_hash: str) -> None:
    if plan_hash(plan) != expected_hash:
        raise ValueError("Release plan changed after review.")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(plan["expires_at"]):
        raise ValueError("Release plan expired; obtain a fresh plan and review the hourly price.")
    if not plan["readiness"]["passed"]:
        raise ValueError("Release policy is blocked; overrides are not supported.")


def cloud_context(source: dict, directory: Path, report):
    from azureml.fsspec import AzureMachineLearningFileSystem
    from dotenv import load_dotenv

    from lib.config import AzureMLConfig

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    configuration = AzureMLConfig.from_env()
    credential = configuration.create_credential()
    client = configuration.create_ml_client(credential)
    evaluation = source["evaluation"]
    training, evaluated = validate_cloud_evaluation(client, evaluation["candidate_id"], evaluation["job_name"])
    dataset_id = evaluation["dataset_id"]
    _, name, version = dataset_id.split(":")
    asset = client.data.get(name=name, version=version)
    accepted = {dataset_id, asset.id, f"azureml:{asset.id}", asset.path}
    if input_path(training, "training_data") not in accepted or input_path(evaluated, "test_data") not in accepted:
        raise ValueError("Cloud training/evaluation dataset lineage differs from the selected pinned dataset.")
    directory.mkdir(parents=True, exist_ok=True)
    report("Verifying Azure job lineage and sealed-test artifact hashes.")
    evidence_dir = directory / "evidence"
    # Fresh downloads are essential: never register from a persisted UI readiness result.
    evidence_dir.mkdir(exist_ok=False)
    client.jobs.download(name=evaluation["job_name"], download_path=str(evidence_dir), output_name="predictions")
    test_path = directory / "test.jsonl"
    filesystem = AzureMachineLearningFileSystem(asset.path)
    with filesystem.open("test.jsonl", "rb") as remote, test_path.open("wb") as local:
        for chunk in iter(lambda: remote.read(1024 * 1024), b""):
            local.write(chunk)
    evidence, rows, manifest = read_candidate_evidence(
        evidence_dir, test_path, evaluation["candidate_id"], evaluation["job_name"],
    )
    if evidence["test_sha256"] != evaluation["data_fingerprint"]:
        raise ValueError("Sealed-test hash differs from the selected evaluation.")
    snapshot = evaluation.get("evidence")
    if snapshot and any(evidence[key] != snapshot[key] for key in ("manifest_sha256", "predictions_sha256")):
        raise ValueError("Cloud evaluation artifacts changed since evaluation was displayed.")
    judges = {row.get("rag_judge_deployment") for row in rows}
    if judges != {evaluation["judge_deployment"]}:
        raise ValueError("Release evidence was scored by a different judge.")
    gate = assess_candidate_rows(rows)
    if not gate["passed"]:
        raise ValueError(f"Scientific release policy blocked: {' '.join(gate['reasons'])}")
    prompt_counts = [row.get("fine_tuned_prompt_token_count") for row in rows]
    evidence["maximum_prompt_tokens"] = max(int(value) for value in prompt_counts) if all(value not in (None, "") for value in prompt_counts) else None
    return client, credential, training, evidence, gate, test_path, manifest


def prepare_plan(source: dict, options: dict, directory: Path, report) -> dict:
    client, credential, training, evidence, gate, _, _ = cloud_context(source, directory, report)
    plan = {
        "schema_version": 1, "action": options["action"],
        "evaluation_run_id": source["run_id"], "training_job_name": source["evaluation"]["candidate_id"],
        "evaluation_job_name": source["evaluation"]["job_name"],
        "dataset_id": source["evaluation"]["dataset_id"], "evidence": evidence,
        "readiness": gate, "model_name": options["model_name"],
        "model_version": hashlib.sha256((evidence["predictions_sha256"] + gate["policy"]["version"]).encode()).hexdigest()[:20],
        "base_model_reference": input_path(training, "base_model"),
        "created_at": utc_now(), "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
    }
    if options["action"] == "register_and_deploy":
        from azure.core.exceptions import ResourceNotFoundError

        from lib.deployment_planning import recommend_gpu
        from lib.monitoring import validate_production_uri

        uri = validate_production_uri(options["production_data_uri"])
        if "?" in uri:
            raise ValueError("Use a credential-free governed production URI, not a SAS URL.")
        if uri in {plan["dataset_id"], plan["dataset_id"].removeprefix("azureml:")}:
            raise ValueError("Drift production data cannot be the training/evaluation dataset.")
        compute = client.compute.get(options["monitor_compute"])
        if str(compute.type).lower() != "amlcompute":
            raise ValueError("Drift monitoring requires an existing Azure ML CPU compute cluster.")
        from azureml.fsspec import AzureMachineLearningFileSystem

        from lib.monitoring import create_reference_profile

        production_path = uri
        if uri.startswith("azureml:") and not uri.startswith("azureml://"):
            _, name, version = uri.split(":")
            production_path = client.data.get(name=name, version=version).path
        _, name, version = plan["dataset_id"].split(":")
        dataset_path = client.data.get(name=name, version=version).path.rstrip("/")
        if production_path.rstrip("/") == dataset_path or production_path.startswith(dataset_path + "/"):
            raise ValueError("Production monitoring must not reuse training or sealed-test data.")
        filesystem = AzureMachineLearningFileSystem(production_path)
        files = filesystem.glob("**/*.jsonl")
        if not files:
            raise ValueError("Governed production URI has no JSONL request data; configure the normalized request stream before deploying.")
        with filesystem.open(files[0], "rb") as stream:
            first_line = stream.readline()
        create_reference_profile([json.loads(first_line)])
        endpoint_name = options["endpoint_name"]
        try:
            client.online_endpoints.get(endpoint_name)
        except ResourceNotFoundError:
            pass
        else:
            raise ValueError("Use a new dedicated endpoint name; existing endpoints and their routing are never modified.")
        base = plan["base_model_reference"]
        base_match = re.fullmatch(r"azureml://registries/([^/]+)/models/([^/]+)/versions/([^/]+)", base)
        if base_match is None:
            raise ValueError("Deployment requires an immutable registry base-model version from training.")
        workspace = client.workspaces.get(os.environ["AZUREML_WORKSPACE_NAME"])
        available = {size.name for size in client.compute.list_sizes()}
        report("Requesting GPT GPU sizing and fetching regional Azure retail hourly prices. No GPU allocated.")
        model = base_match[2]
        trainer = (source.get("training") or {}).get("config", {})
        maximum = evidence["maximum_prompt_tokens"]
        input_budget = max(2048, math.ceil((maximum + 256) / 1024) * 1024) if maximum is not None else 8192
        plan.update(
            endpoint_name=endpoint_name, deployment_name="candidate",
            production_data_uri=uri, monitor_compute=options["monitor_compute"],
            gpu=recommend_gpu(model, trainer, available, workspace.location, credential, input_budget),
        )
    return plan


def execute_plan(source: dict, plan: dict, approved_by: str, reason: str, directory: Path, report, persist) -> dict:
    from azure.core.exceptions import ResourceNotFoundError

    from lib.azureml_ops import register_job_model

    client, _, training, evidence, gate, test_path, _ = cloud_context(source, directory, report)
    if (evidence["manifest_sha256"] != plan["evidence"]["manifest_sha256"]
            or input_path(training, "base_model") != plan["base_model_reference"]
            or gate != plan["readiness"]):
        raise ValueError("Candidate evidence or lineage changed after approval; obtain a fresh plan.")
    approval = approval_record(True, approved_by, reason, evidence["manifest_sha256"])
    try:
        model = client.models.get(name=plan["model_name"], version=plan["model_version"])
    except ResourceNotFoundError:
        report("Registering candidate adapter and tokenizer. Registration alone allocates no GPU.")
        model = register_job_model(
            client, plan["training_job_name"], plan["model_name"], plan["model_version"],
            metadata_tags={"evaluation_job": plan["evaluation_job_name"],
                           "evidence_sha256": evidence["manifest_sha256"],
                           "release_policy": gate["policy"]["version"]},
        )
    else:
        if ((model.tags or {}).get("source_job") != plan["training_job_name"]
                or (model.tags or {}).get("evidence_sha256") != evidence["manifest_sha256"]):
            raise ValueError("Existing model version is not bound to this candidate evidence.")
    result = {
        "registered_model": {"id": model.id, "name": model.name, "version": model.version},
        "approval": approval, "status": "registered",
    }
    persist(result)
    receipt_path = directory.parent / ("candidate-release.yaml" if plan["action"] == "register" else "endpoint-release.yaml")
    save_release({"schema_version": 1, "release_id": directory.parent.name, **result, "plan": plan, "source": source},
                 receipt_path)
    if plan["action"] == "register":
        return result
    from lib.azureml_ops import (
        create_inference_environment,
        deploy_managed_endpoint,
        evaluate_endpoint,
    )
    from lib.data import read_jsonl
    from lib.monitoring import create_drift_schedule, create_reference_profile
    from lib.release import write_json

    report(f"Provisioning approved GPU {plan['gpu']['sku']} at estimated ${plan['gpu']['price']['hourly_usd']}/hour. Billing continues while allocated.")
    environment = create_inference_environment(client, "slm-studio-inference", "2")
    result.update(status="provisioning", endpoint={"name": plan["endpoint_name"], "deployment": plan["deployment_name"]})
    persist(result)
    endpoint = deploy_managed_endpoint(
        client, plan["endpoint_name"], plan["deployment_name"], model.id, environment.id,
        plan["gpu"]["sku"], base_model=plan["base_model_reference"],
        max_input_tokens=plan["gpu"]["max_input_tokens"],
        max_output_tokens=plan["gpu"]["max_output_tokens"],
        key_vault_name=os.getenv("AZURE_KEY_VAULT_NAME"),
        hf_token_secret_name=os.getenv("AZURE_KEY_VAULT_HF_TOKEN_SECRET_NAME"),
    )
    deployment = client.online_deployments.get(endpoint_name=plan["endpoint_name"], name=plan["deployment_name"])
    if str(getattr(deployment.model, "id", deployment.model)) != model.id or deployment.provisioning_state != "Succeeded":
        raise ValueError("Deployed model identity or provisioning state does not match the approved candidate.")
    records = read_jsonl(test_path)
    reference = write_json(create_reference_profile(records), directory / "drift-reference.json")
    report("Creating weekly request-feature PSI monitoring on governed production data.")
    schedule = create_drift_schedule(
        client, schedule_name=f"{plan['endpoint_name']}-request-drift", reference_profile=reference,
        production_data_uri=plan["production_data_uri"], environment=environment.id, compute=plan["monitor_compute"],
    )
    checked = client.schedules.get(schedule.name)
    if not checked.is_enabled:
        raise ValueError("Azure drift schedule was not enabled.")
    report("Validating scoring through the authenticated endpoint URI.")
    # No deployment header: validate the exact default endpoint users will invoke.
    probes = evaluate_endpoint(client, endpoint.name, records[:3], batch_size=1, max_new_tokens=128)
    if len(probes) != min(3, len(records)) or not endpoint.scoring_uri:
        raise ValueError("Endpoint scoring smoke validation did not return all expected responses.")
    result.update(
        status="deployed",
        endpoint={"name": endpoint.name, "deployment": plan["deployment_name"], "uri": endpoint.scoring_uri,
                  "authentication": "aml_token", "validated": True, "validated_at": utc_now()},
        monitoring={"schedule": schedule.name, "status": "enabled", "psi_threshold": 0.2,
                    "production_data_uri": plan["production_data_uri"], "reference_sha256": file_sha256(reference),
                    "note": "Weekly request-feature PSI; not outcome/concept drift. Governed normalized production JSONL must be maintained."},
    )
    persist(result)
    save_release({"schema_version": 1, "release_id": directory.parent.name, **result, "plan": plan, "source": source},
                 receipt_path)
    report("Endpoint validated; model registration and drift schedule are recorded. No canary or rollout policy was configured.")
    return result
