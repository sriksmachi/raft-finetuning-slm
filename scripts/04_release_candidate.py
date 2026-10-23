"""Verify fixed scientific policy and register a candidate, without allocating GPUs."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.candidate_release import execute_plan, plan_hash, prepare_plan
from lib.release import validate_release_id


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-record", type=Path, required=True,
                        help="Completed Studio evaluation YAML in runs/production.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--registered-model-name", default="slm-studio-candidate")
    parser.add_argument("--release-root", type=Path, default=PROJECT_ROOT / "output" / "releases")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approval-reason", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.approve or not args.approved_by.strip() or not args.approval_reason.strip():
        raise ValueError("Explicit --approve, approver and reason are required.")
    source = yaml.safe_load(args.evaluation_record.read_text(encoding="utf-8"))
    if (not isinstance(source, dict) or source.get("mode") != "production"
            or source.get("run_type") != "evaluation"
            or source["stages"]["evaluation"]["status"] != "succeeded"):
        raise ValueError("Select a completed production evaluation record.")
    directory = args.release_root.resolve() / validate_release_id(args.release_id)
    if directory.exists():
        raise FileExistsError("Use a new release directory; previous cloud operations are never automatically repeated.")
    report = logging.getLogger("release_candidate").info
    plan = prepare_plan(source, {"action": "register", "model_name": args.registered_model_name}, directory / "plan", report)
    report("Fixed policy passed. Registering only; no endpoint or GPU will be allocated.")
    execute_plan(source, plan, args.approved_by, args.approval_reason, directory / "execution", report, lambda result: None)
    report("Registered candidate receipt: %s (plan %s)", directory / "candidate-release.yaml", plan_hash(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
