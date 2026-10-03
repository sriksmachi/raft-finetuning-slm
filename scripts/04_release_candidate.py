"""Approve, register, and deploy an evaluated model candidate with zero traffic.

Stage 04 consumes candidate-bound Stage 03 evidence. It verifies Azure ML job
lineage, evaluates the complete sealed test evidence against release policy,
records explicit human approval, registers the adapter, deploys it without
changing endpoint traffic, and creates the drift reference profile.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.azureml_ops import (
    create_inference_environment,
    deploy_managed_endpoint,
    register_job_model,
)
from lib.config import AzureMLConfig
from lib.data import read_jsonl
from lib.monitoring import create_reference_profile, write_json_artifact
from lib.release import (
    ReleasePolicy,
    approval_record,
    input_path,
    load_inference_evidence,
    save_release,
    utc_now,
    validate_cloud_evaluation,
    validate_release_id,
)
from lib.run_registry import latest_run, record_run

log = logging.getLogger("release_candidate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--training-job-name")
    parser.add_argument("--evaluation-job-name")
    parser.add_argument("--training-experiment-name", default="slm-studio-finetuning")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--test-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data_raft" / "test.jsonl",
    )
    parser.add_argument("--release-root", type=Path, default=PROJECT_ROOT / "output" / "releases")

    parser.add_argument("--registered-model-name", default="slm-studio-llama32-3b")
    parser.add_argument("--registered-model-version")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--deployment-name")
    parser.add_argument("--inference-environment-name", default="slm-studio-inference")
    parser.add_argument("--inference-environment-version", default="2")
    parser.add_argument("--skip-environment", action="store_true")
    parser.add_argument("--instance-type", default="Standard_NC4as_T4_v3")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument(
        "--base-model-reference",
        required=True,
        help="Deployable base checkpoint path or Hugging Face model ID for the LoRA adapter.",
    )
    parser.add_argument("--key-vault-name", default=os.getenv("AZURE_KEY_VAULT_NAME"))
    parser.add_argument(
        "--hf-token-secret-name",
        default=os.getenv("AZURE_KEY_VAULT_HF_TOKEN_SECRET_NAME"),
    )

    parser.add_argument("--min-faithfulness", type=float, default=0.7)
    parser.add_argument("--min-relevancy", type=float, default=0.7)
    parser.add_argument("--min-correctness", type=float, default=0.7)
    parser.add_argument("--correctness-tolerance", type=float, default=0.0)
    parser.add_argument("--max-mean-latency-ms", type=float, default=20_000)
    parser.add_argument("--max-p95-latency-ms", type=float, default=40_000)
    parser.add_argument("--min-examples", type=int, default=30)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--approved-by")
    parser.add_argument("--approval-reason")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_jobs(args: argparse.Namespace) -> tuple[str, str]:
    training_job = args.training_job_name
    evaluation_job = args.evaluation_job_name
    if not training_job:
        run = latest_run(args.training_experiment_name)
        if not run:
            raise RuntimeError("No recorded training run; pass --training-job-name")
        training_job = run["job_name"]
        evaluation_job = evaluation_job or run.get("evaluation_job_name")
    if not evaluation_job:
        raise RuntimeError("No recorded evaluation run; pass --evaluation-job-name")
    return training_job, evaluation_job


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    release_id = validate_release_id(args.release_id)
    model_version = args.registered_model_version or release_id
    deployment_name = args.deployment_name or f"candidate-{release_id.lower().replace('_', '-')}"
    release_dir = args.release_root.resolve() / release_id
    receipt_path = release_dir / "candidate-release.yaml"
    if receipt_path.exists():
        raise FileExistsError(f"Release receipt already exists: {receipt_path}")

    policy = ReleasePolicy(
        min_faithfulness=args.min_faithfulness,
        min_relevancy=args.min_relevancy,
        min_correctness=args.min_correctness,
        correctness_tolerance=args.correctness_tolerance,
        max_mean_latency_ms=args.max_mean_latency_ms,
        max_p95_latency_ms=args.max_p95_latency_ms,
        min_examples=args.min_examples,
    )
    training_job_name, evaluation_job_name = resolve_jobs(args)
    config = AzureMLConfig.from_env()
    ml_client = config.create_ml_client()
    training_job, _ = validate_cloud_evaluation(
        ml_client, training_job_name, evaluation_job_name
    )

    evidence_dir = (args.evidence_dir or release_dir / "stage03-evidence").resolve()
    if args.evidence_dir is None:
        evidence_dir.mkdir(parents=True, exist_ok=False)
        ml_client.jobs.download(
            name=evaluation_job_name,
            download_path=str(evidence_dir),
            output_name="predictions",
        )
    evidence = load_inference_evidence(
        evidence_dir, args.test_path.resolve(), training_job_name, policy
    )
    if not evidence["gate"]["passed"]:
        raise RuntimeError(f"Offline release gate failed: {evidence['gate']['reasons']}")
    approval = approval_record(
        args.approve, args.approved_by, args.approval_reason, evidence["manifest_sha256"]
    )

    registered_model = register_job_model(
        ml_client,
        training_job_name,
        args.registered_model_name,
        model_version,
        validation_metric=evidence["gate"]["means"]["correctness"],
    )
    if args.skip_environment:
        environment_id = (
            f"azureml:{args.inference_environment_name}:{args.inference_environment_version}"
        )
    else:
        environment_id = create_inference_environment(
            ml_client,
            args.inference_environment_name,
            args.inference_environment_version,
        ).id
    endpoint = deploy_managed_endpoint(
        ml_client=ml_client,
        endpoint_name=args.endpoint_name,
        deployment_name=deployment_name,
        model=registered_model.id,
        environment=environment_id,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        traffic_percent=0,
        base_model=args.base_model_reference,
        key_vault_name=args.key_vault_name,
        hf_token_secret_name=args.hf_token_secret_name,
    )

    reference_path = write_json_artifact(
        create_reference_profile(read_jsonl(args.test_path.resolve())),
        release_dir / "drift-reference.json",
    )
    receipt = {
        "schema_version": 1,
        "release_id": release_id,
        "stage": 4,
        "status": "candidate_deployed_zero_traffic",
        "created_at": utc_now(),
        "training_job_name": training_job_name,
        "evaluation_job_name": evaluation_job_name,
        "training_base_model": input_path(training_job, "base_model"),
        "registered_model": {
            "name": args.registered_model_name,
            "version": str(registered_model.version),
            "id": registered_model.id,
        },
        "endpoint": {
            "name": args.endpoint_name,
            "deployment": deployment_name,
            "scoring_uri": endpoint.scoring_uri,
            "traffic_percent": 0,
            "environment": environment_id,
        },
        "offline_evidence": evidence,
        "release_policy": asdict(policy),
        "approval": approval,
        "sealed_test_path": str(args.test_path.resolve()),
        "drift_reference_path": str(reference_path.resolve()),
    }
    save_release(receipt, receipt_path)
    record_run(
        training_job_name,
        registered_model_name=args.registered_model_name,
        registered_model_version=str(registered_model.version),
        release_id=release_id,
        release_receipt=str(receipt_path),
        endpoint_name=args.endpoint_name,
        deployment_name=deployment_name,
        status="candidate_deployed_zero_traffic",
    )
    log.info("Candidate deployed with zero traffic. Receipt: %s", receipt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
