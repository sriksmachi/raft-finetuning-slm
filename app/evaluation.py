"""Saved training-job discovery and the production Stage 03 adapter."""

import asyncio
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.models import ProductionEvaluation, ProductionTraining
from app.preparation import append_job_status
from app.store import RunStore, StoreError
from app.training import TERMINAL_JOB_STATUSES
from lib.job_receipts import read_receipt

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,254}:[A-Za-z0-9_.-]{1,30}\Z")

def evaluation_needs_refresh(record: dict) -> bool:
    evaluation = record.get("evaluation") or {}
    receipt = evaluation.get("jobs", {}).get("slm_evaluation") or {}
    return (
        record.get("run_type") == "evaluation"
        and record["status"] in {"running", "failed"}
        and record["stages"]["evaluation"]["status"] in {"running", "failed"}
        and bool(receipt.get("job_name"))
        and receipt.get("status") not in TERMINAL_JOB_STATUSES
    )


def validate_evaluation_receipt(receipt: dict, job: dict, mode: str) -> None:
    if (receipt.get("mode") != mode
            or receipt.get("dataset_id") != job["dataset_id"]
            or receipt.get("training_job_name") != (job["job_name"] if mode == "slm_evaluation" else None)
            or not all(isinstance(receipt.get(key), str) and receipt[key]
                       for key in ("job_name", "job_id", "status", "judge_deployment", "baseline_id"))
            or not BASELINE_PATTERN.fullmatch(receipt["baseline_id"])):
        raise ValueError("Evaluation receipt does not match the selected job and dataset.")


def refresh_evaluation(job: dict, directory: Path, receipt: dict) -> tuple[dict, dict | None]:
    from dotenv import load_dotenv

    from lib.config import AzureMLConfig

    validate_evaluation_receipt(receipt, job, "slm_evaluation")
    load_dotenv(ROOT / ".env")
    config = AzureMLConfig.from_env()
    with config.create_credential() as credential:
        client = config.create_ml_client(credential=credential)
        status = client.jobs.get(receipt["job_name"]).status
        if not isinstance(status, str) or not status:
            raise ValueError("Azure ML returned an empty evaluation job status.")
        updated = {**receipt, "status": status}
        if status != "Completed":
            return updated, None
        outputs = directory / "outputs"
        if not any(outputs.rglob("evaluation_evidence.json")):
            client.jobs.download(receipt["job_name"], download_path=str(outputs), output_name="predictions")
    return updated, evaluation_result(outputs, job, updated)


def training_jobs(ledger_dir: Path, store: RunStore) -> list[dict]:
    jobs = {}
    for path in sorted(ledger_dir.glob("*.yaml")):
        try:
            if path.is_symlink():
                raise ValueError("Symlink training ledgers are not supported.")
            record = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise TypeError("Training ledger must contain a mapping.")
            if not record.get("job_name"):
                continue
            ProductionEvaluation(job_name=record["job_name"])
            if record["job_name"] != path.stem:
                raise ValueError("Training job does not match its ledger filename.")
            if str(record.get("status", "")).lower() not in {
                "completed", "candidate_deployed_zero_traffic", "traffic_routed",
            }:
                continue
            jobs[record["job_name"]] = {
                "job_name": record["job_name"], "training_date": record.get("submitted_at"),
                "base_model": record.get("base_model"), "base_model_source": record.get("base_model_source"),
                "base_model_registry": record.get("base_model_registry", "azureml-meta"),
                "base_model_label": record.get("base_model_label", "latest"),
                "dataset_id": f"azureml:{record['data_asset']}" if record.get("data_asset") else None,
                "sample_count": record.get("sample_count"),
                "config": record.get("training_parameters") or {},
                "baseline_id": record.get("llm_baseline_data_asset"), "training_run_id": None,
            }
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            raise StoreError(f"Unable to read training ledger {path.name}: {exc}") from exc
    records = store.list()
    datasets = {
        record["data_card"].get("asset_reference"): record["data_card"]
        for record in records if record.get("data_card")
    }
    for record in records:
        training = record.get("training")
        if (record.get("run_type") != "training" or not training
                or record["stages"]["training"]["status"] != "succeeded"
                or training.get("job_status") != "Completed"):
            continue
        name = training.get("job_name")
        try:
            ProductionEvaluation(job_name=name)
        except ValidationError as exc:
            raise StoreError(f"Invalid saved training job in {record['run_id']}.") from exc
        previous = jobs.get(name, {})
        jobs[name] = {
            **previous, "job_name": name,
            "training_date": previous.get("training_date") or record["stages"]["training"]["started_at"],
            "base_model": training["config"]["base_model"], "base_model_source": "foundry",
            "base_model_registry": previous.get("base_model_registry", "azureml-meta"),
            "base_model_label": previous.get("base_model_label", "latest"),
            "dataset_id": training["dataset_id"], "config": training["config"],
            "baseline_id": previous.get("baseline_id"), "training_run_id": record["run_id"],
            "sample_count": (record.get("data_card") or {}).get("record_count"),
        }
    for job in jobs.values():
        config = job.pop("config")
        if not isinstance(config, dict):
            raise StoreError(f"Invalid saved training parameters for {job['job_name']}.")
        job["epochs"] = config.get("epochs")
        job["adapter_name"] = config.get("lora_adapter_name")
        job["sample_count"] = job.get("sample_count") or datasets.get(job["dataset_id"], {}).get("record_count")
        job["dataset_version"] = job["dataset_id"].rsplit(":", 1)[-1] if job["dataset_id"] else None
        try:
            ProductionTraining(dataset_id=job["dataset_id"])
            if job["base_model_source"] not in {"foundry", "huggingface"}:
                raise ValueError("Training model source is not recorded.")
            for key in ("base_model", "base_model_registry", "base_model_label"):
                if not isinstance(job.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,199}", job[key]):
                    raise ValueError(f"Training {key} is missing or invalid.")
            if job["baseline_id"] and not BASELINE_PATTERN.fullmatch(job["baseline_id"]):
                raise ValueError("Recorded baseline must be a pinned name:version reference.")
        except (ValueError, TypeError) as exc:
            job.update(eligible=False, unavailable_reason=str(exc))
        else:
            job.update(eligible=True, unavailable_reason=None)
    return sorted(jobs.values(), key=lambda job: str(job["training_date"] or ""), reverse=True)


def evaluation_result(directory: Path, job: dict, receipt: dict, price_snapshot: dict | None = None) -> dict:
    from lib.token_costs import load_prices, summarize_usage
    from scripts.export_evaluation_showcase import MODELS, SCORES, build_showcase

    manifests = list(directory.rglob("evaluation_evidence.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one downloaded evaluation evidence manifest.")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("stage") != 3
            or manifest.get("framework") != "deepeval"
            or manifest.get("training_job_name") != job["job_name"]
            or manifest.get("evaluation_run_id") != receipt["job_name"]):
        raise ValueError("Evaluation evidence does not match the selected training and evaluation jobs.")
    filename = manifest.get("predictions_file")
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.csv", filename):
        raise ValueError("Evaluation predictions must be a local CSV filename.")
    predictions = manifest_path.parent / filename
    if manifest_path.is_symlink() or predictions.is_symlink():
        raise ValueError("Symlink evaluation artifacts are not supported.")
    with predictions.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Evaluation returned no sealed test records.")
    for row in rows:
        if row.get("evaluation_framework") != "deepeval":
            raise ValueError("Evaluation scores must come from the configured DeepEval judge.")
        if row.get("rag_judge_deployment") != receipt["judge_deployment"]:
            raise ValueError("Evaluation judge does not match the backend-configured deployment.")
        for model, (_, response, latency) in MODELS.items():
            if row.get(f"{model}_rag_error") or row.get(f"{model}_inference_error"):
                raise ValueError(f"Incomplete evaluation: {model} inference or judging failed.")
            if not row.get(response, "").strip():
                raise ValueError(f"Evaluation returned an empty {model} response.")
            for field in [latency, *(f"{model}_{score}_score" for score in SCORES)]:
                value = float(row[field])
                if not math.isfinite(value) or (value <= 0 if field == latency else not 0 <= value <= 1):
                    raise ValueError(f"Evaluation returned invalid {field}.")
    if {int(row["source_index"]) for row in rows} != set(range(len(rows))):
        raise ValueError("Evaluation did not cover the complete sealed test split.")
    payload = build_showcase(predictions, manifest_path)
    for model, (_, _, latency) in MODELS.items():
        values = sorted(float(row[latency]) for row in rows)
        position = .95 * (len(values) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        payload["summary"]["models"][model]["latency_p95_ms"] = (
            values[lower] + (values[upper] - values[lower]) * (position - lower)
        )
    metrics = []
    for key, label in (("faithfulness", "Faithfulness"), ("relevancy", "Relevancy"), ("correctness", "Correctness")):
        metrics.append({"key": key, "label": label, "unit": "score",
                        **{model: payload["summary"]["models"][model]["scores"][key] for model in MODELS}})
    for key, label in (("latency_avg_ms", "Mean latency"), ("latency_p95_ms", "P95 latency")):
        metrics.append({"key": key, "label": label, "unit": "ms",
                        **{model: payload["summary"]["models"][model][key] for model in MODELS}})
    usage = summarize_usage(rows, price_snapshot if price_snapshot is not None else load_prices(),
                            baseline_reused=bool(job.get("baseline_id")))
    for direction in ("input", "output", "total"):
        metrics.append({"key": f"{direction}_tokens", "label": f"{direction.title()} tokens", "unit": "tokens",
                        **{model: usage["models"][model][f"{direction}_tokens"] for model in MODELS}})
    metrics.append({"key": "token_cost", "label": "Frontier API token cost (estimate)", "unit": "USD",
                    **{model: usage["models"][model]["estimated_cost_usd"] for model in MODELS}})
    return {
        **payload, "simulated": False, "candidate_id": job["job_name"], "dataset_id": job["dataset_id"],
        "test_split": f"{job['dataset_id']}/test.jsonl", "data_fingerprint": manifest["test_sha256"],
        "judge_deployment": receipt["judge_deployment"], "metrics": metrics,
        "job_name": receipt["job_name"], "job_uri": receipt.get("job_uri"), "job_status": "Completed",
        "baseline_id": receipt["baseline_id"],
        "token_usage": usage,
        "evidence": {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                     "predictions_sha256": manifest["predictions_sha256"]},
    }


async def evaluate_model(job: dict, directory: Path, on_submitted) -> dict:
    from lib.token_costs import load_prices

    price_snapshot = load_prices()
    directory.mkdir(parents=True, exist_ok=True)
    _, name, version = job["dataset_id"].split(":")
    base = [
        sys.executable, "-u", str(ROOT / "scripts" / "03_azureml_slm_vs_gpt_evaluation.py"),
        "--test-data-asset-name", name, "--test-data-asset-version", version,
        "--base-model-source", job["base_model_source"], "--base-model-name", job["base_model"],
        "--base-model-registry", job["base_model_registry"], "--base-model-label", job["base_model_label"],
        "--wait",
    ]
    if job["base_model_source"] == "huggingface":
        base.extend(["--hf-base-model", job["base_model"]])

    async def phase(mode, flags):
        path = directory / f"{mode}_receipt.json"
        append_job_status(directory, f"Starting {mode.replace('_', ' ')}.", "evaluation.log")
        with (directory / "evaluation.log").open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                *base, "--mode", mode, "--receipt-path", str(path), *flags,
                cwd=ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            waiter = asyncio.create_task(process.wait())
            latest = None

            def update():
                nonlocal latest
                receipt = read_receipt(path)
                if receipt is None:
                    return
                validate_evaluation_receipt(receipt, job, mode)
                if receipt != latest:
                    latest = receipt
                    on_submitted(receipt)
            try:
                while not waiter.done():
                    await asyncio.wait({waiter}, timeout=1)
                    update()
                code = await waiter
                update()
                if code or not latest or latest["status"] != "Completed":
                    raise RuntimeError(f"{mode} did not complete successfully (exit {code}). See evaluation job logs.")
                return latest
            finally:
                if process.returncode is None:
                    process.terminate()
                if not waiter.done():
                    await asyncio.gather(waiter, return_exceptions=True)

    baseline = job.get("baseline_id")
    if not baseline:
        baseline = (await phase("llm_baseline", []))["baseline_id"]
    baseline_name, baseline_version = baseline.split(":")
    receipt = await phase("slm_evaluation", [
        "--finetuning-job-name", job["job_name"], "--llm-baseline-data-asset-name", baseline_name,
        "--llm-baseline-data-asset-version", baseline_version, "--result-dir", str(directory / "outputs"),
    ])
    if receipt["baseline_id"] != baseline:
        raise ValueError("Evaluation returned a different GPT baseline than requested.")
    return evaluation_result(directory / "outputs", job, receipt, price_snapshot)
