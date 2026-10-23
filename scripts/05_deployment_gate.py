"""Plan, explicitly approve, deploy and validate a registered candidate endpoint.

First run without --approve to obtain the GPT-selected SKU and Azure hourly price.
Then approve that exact plan hash. Existing endpoint routing is never modified.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.candidate_release import execute_plan, plan_hash, prepare_plan, validate_plan
from lib.release import load_release


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-name")
    parser.add_argument("--production-data-uri")
    parser.add_argument("--monitor-compute", default="cpu-cluster")
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--accept-plan-sha256")
    parser.add_argument("--approved-by")
    parser.add_argument("--approval-reason")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    receipt = load_release(args.release_receipt.resolve())
    if receipt.get("status") != "registered" or receipt.get("plan", {}).get("action") != "register":
        raise ValueError("Stage 05 requires a register-only Stage 04 receipt.")
    source = receipt["source"]
    directory = args.release_receipt.resolve().parent
    report = logging.getLogger("deployment_gate").info
    if not args.approve:
        if not args.endpoint_name or not args.production_data_uri:
            raise ValueError("Planning requires --endpoint-name and --production-data-uri.")
        plan = prepare_plan(source, {
            "action": "register_and_deploy", "model_name": receipt["registered_model"]["name"],
            "endpoint_name": args.endpoint_name, "production_data_uri": args.production_data_uri,
            "monitor_compute": args.monitor_compute,
        }, directory / "deployment-plan-evidence", report)
        path = args.plan_file or directory / "deployment-plan.json"
        if path.exists():
            raise FileExistsError("Deployment plan already exists; use a fresh registered release for a new plan.")
        path.write_text(json.dumps({"plan": plan, "sha256": plan_hash(plan)}, indent=2, allow_nan=False), encoding="utf-8")
        report("No GPU allocated. Review %s: SKU=%s; USD/hour=%s; approval hash=%s",
               path, plan["gpu"]["sku"], plan["gpu"]["price"]["hourly_usd"], plan_hash(plan))
        return 0
    if not args.plan_file or not args.accept_plan_sha256 or not args.approved_by or not args.approval_reason:
        raise ValueError("Approval requires --plan-file, --accept-plan-sha256, --approved-by and --approval-reason.")
    plan = json.loads(args.plan_file.read_text(encoding="utf-8"))["plan"]
    validate_plan(plan, args.accept_plan_sha256)
    if (plan["action"] != "register_and_deploy"
            or plan["model_name"] != receipt["registered_model"]["name"]
            or plan["model_version"] != receipt["registered_model"]["version"]
            or plan["evidence"]["manifest_sha256"] != receipt["plan"]["evidence"]["manifest_sha256"]
            or plan["training_job_name"] != receipt["plan"]["training_job_name"]):
        raise ValueError("Deployment plan is not bound to the registered candidate receipt.")
    if (directory / "deployment-execution").exists():
        raise FileExistsError("Deployment already attempted; inspect cloud resources rather than automatically retrying.")
    execute_plan(source, plan, args.approved_by, args.approval_reason,
                 directory / "deployment-execution", report, lambda result: None)
    report("Validated endpoint receipt: %s", directory / "endpoint-release.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
