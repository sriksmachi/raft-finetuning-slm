"""Test a zero-traffic candidate, route traffic, and enable drift monitoring.

Stage 05 invokes the named candidate directly on the complete sealed test set.
Traffic changes only when token-F1 and latency policy pass. A partial rollout
preserves one incumbent deployment; a first release must receive 100 percent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.azureml_ops import promote_deployment
from lib.config import AzureMLConfig
from lib.data import read_jsonl
from lib.drift_schedule import create_drift_schedule
from lib.endpoint_inference import evaluate_endpoint
from lib.evaluation import token_f1
from lib.release import (
    DeploymentPolicy,
    assess_deployment_rows,
    file_sha256,
    load_release,
    save_release,
    utc_now,
)
from lib.run_registry import record_run

log = logging.getLogger("deployment_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--traffic-percent", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-token-f1", type=float, default=0.7)
    parser.add_argument("--max-mean-latency-ms", type=float, default=20_000)
    parser.add_argument("--max-p95-latency-ms", type=float, default=40_000)
    parser.add_argument("--min-examples", type=int, default=30)
    parser.add_argument("--production-data-uri")
    parser.add_argument("--skip-drift-schedule", action="store_true")
    parser.add_argument("--monitor-compute", default="cpu-cluster")
    parser.add_argument("--schedule-name")
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--schedule-frequency", default="week")
    parser.add_argument("--schedule-interval", type=int, default=1)
    parser.add_argument("--schedule-hour-utc", type=int, default=2)
    parser.add_argument("--managed-identity-client-id")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    release = load_release(args.release_receipt.resolve())
    if release.get("stage") != 4 or release.get("status") != "candidate_deployed_zero_traffic":
        raise ValueError("Stage 05 requires a Stage 04 zero-traffic candidate receipt")
    if not release.get("offline_evidence", {}).get("gate", {}).get("passed"):
        raise ValueError("Stage 04 receipt does not contain passing offline evidence")

    endpoint_name = release["endpoint"]["name"]
    deployment_name = release["endpoint"]["deployment"]
    promotion_path = args.release_receipt.resolve().with_name("promotion-release.yaml")
    evidence_path = args.release_receipt.resolve().with_name("deployment-evidence.json")
    if promotion_path.exists() or evidence_path.exists():
        raise FileExistsError(
            "Deployment evidence or promotion receipt already exists for this release"
        )
    test_path = Path(release["sealed_test_path"])
    if release["offline_evidence"]["test_sha256"] != file_sha256(test_path):
        raise ValueError("Sealed test data changed after Stage 04 approval")

    policy = DeploymentPolicy(
        min_token_f1=args.min_token_f1,
        max_mean_latency_ms=args.max_mean_latency_ms,
        max_p95_latency_ms=args.max_p95_latency_ms,
        min_examples=args.min_examples,
    )
    config = AzureMLConfig.from_env()
    ml_client = config.create_ml_client()
    deployment = ml_client.online_deployments.get(
        endpoint_name=endpoint_name, name=deployment_name
    )
    deployed_model = getattr(deployment, "model", None)
    expected_model = release["registered_model"]["id"]
    if deployed_model and str(deployed_model) != expected_model:
        raise ValueError("Candidate deployment no longer references the approved model")

    test_records = read_jsonl(test_path)
    rows = evaluate_endpoint(
        ml_client,
        endpoint_name,
        test_records,
        batch_size=args.batch_size,
        deployment_name=deployment_name,
        max_new_tokens=args.max_new_tokens,
    )
    for row in rows:
        row["answer_token_f1"] = token_f1(row["prediction"], row["reference"])
    gate = assess_deployment_rows(rows, test_records, policy)

    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": release["release_id"],
                "created_at": utc_now(),
                "endpoint": endpoint_name,
                "deployment": deployment_name,
                "sealed_test_sha256": file_sha256(test_path),
                "gate": gate,
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    if not gate["passed"]:
        raise RuntimeError(
            f"Deployment gate failed; traffic remains unchanged: {gate['reasons']}. "
            f"Evidence: {evidence_path}"
        )

    schedule = None
    if not args.skip_drift_schedule:
        if not args.production_data_uri:
            raise ValueError(
                "--production-data-uri is required unless --skip-drift-schedule is used"
            )
        schedule = create_drift_schedule(
            ml_client,
            schedule_name=args.schedule_name or f"{endpoint_name}-request-drift",
            reference_profile=Path(release["drift_reference_path"]),
            production_data_uri=args.production_data_uri,
            environment=release["endpoint"]["environment"],
            compute=args.monitor_compute,
            psi_threshold=args.psi_threshold,
            frequency=args.schedule_frequency,
            interval=args.schedule_interval,
            hour=args.schedule_hour_utc,
            managed_identity_client_id=args.managed_identity_client_id,
        )

    endpoint = promote_deployment(
        ml_client,
        endpoint_name,
        deployment_name,
        traffic_percent=args.traffic_percent,
    )
    promotion = {
        "schema_version": 1,
        "release_id": release["release_id"],
        "stage": 5,
        "status": "traffic_routed",
        "promoted_at": utc_now(),
        "candidate_receipt": str(args.release_receipt.resolve()),
        "deployment_evidence": str(evidence_path),
        "deployment_evidence_sha256": file_sha256(evidence_path),
        "deployment_policy": asdict(policy),
        "gate": gate,
        "endpoint": {
            "name": endpoint_name,
            "deployment": deployment_name,
            "traffic": dict(endpoint.traffic or {}),
        },
        "drift_schedule": getattr(schedule, "name", None),
    }
    save_release(promotion, promotion_path)
    record_run(
        release["training_job_name"],
        promotion_receipt=str(promotion_path),
        traffic=dict(endpoint.traffic or {}),
        drift_schedule=getattr(schedule, "name", None),
        status="traffic_routed",
    )
    log.info("Traffic routed after passing deployment gate: %s", endpoint.traffic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
