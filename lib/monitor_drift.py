"""Azure ML command-job entry point for scheduled request drift checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from lib.data import read_jsonl
from lib.monitoring import detect_drift, write_json_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-profile", required=True)
    parser.add_argument("--production-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    profile_path = Path(args.reference_profile)
    if profile_path.is_dir():
        profile_path = profile_path / "reference_profile.json"
    production_path = Path(args.production_data)
    if production_path.is_dir():
        candidates = sorted(production_path.rglob("*.jsonl"))
        if not candidates:
            raise ValueError("No JSONL request data found in production-data input")
        records = [row for path in candidates for row in read_jsonl(path)]
    else:
        records = read_jsonl(production_path)

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    report = detect_drift(profile, records, args.psi_threshold)
    output_path = write_json_artifact(report, Path(args.output) / "drift_report.json")
    with mlflow.start_run():
        mlflow.log_metric("drift_detected", float(report["drift_detected"]))
        for feature, values in report["features"].items():
            mlflow.log_metric(f"psi_{feature}", values["psi"])
        mlflow.log_artifact(str(output_path), artifact_path="monitoring")


if __name__ == "__main__":
    main(parse_args())
