"""Background adapter for the existing Azure ML fine-tuning CLI."""

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path

from app.models import ProductionTraining
from lib.job_receipts import read_receipt

ROOT = Path(__file__).resolve().parents[1]
TERMINAL_JOB_STATUSES = {"Completed", "Failed", "Canceled", "Cancelled"}


def training_needs_refresh(record: dict) -> bool:
    training = record.get("training") or {}
    return (
        record.get("run_type") == "training"
        and record["status"] in {"running", "failed"}
        and record["stages"]["training"]["status"] in {"running", "failed"}
        and bool(training.get("job_name"))
        and training.get("job_status") not in TERMINAL_JOB_STATUSES
    )


def training_status(job_name: str) -> str:
    from dotenv import load_dotenv

    from lib.config import AzureMLConfig

    load_dotenv(ROOT / ".env")
    config = AzureMLConfig.from_env()
    with config.create_credential() as credential:
        client = config.create_ml_client(credential=credential)
        status = client.jobs.get(job_name).status
    if not isinstance(status, str) or not status:
        raise ValueError("Azure ML returned an empty training job status.")
    return status


async def train_model(
    body: ProductionTraining,
    dataset_id: str,
    directory: Path,
    on_submitted: Callable[[dict], None],
) -> dict:
    _, name, version = dataset_id.split(":")
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "training_receipt.json"
    command = [
        sys.executable, "-u", str(ROOT / "scripts" / "02_azureml_fine_tuning.py"),
        "--data-asset-name", name, "--data-asset-version", version,
        "--base-model-name", body.base_model, "--receipt-path", str(receipt_path), "--wait",
    ]
    for field, flag in (
        ("max_seq_length", "--max-seq-length"), ("learning_rate", "--learning-rate"),
        ("epochs", "--epochs"), ("batch_size", "--batch-size"),
        ("gradient_accumulation", "--gradient-accumulation-steps"),
        ("lora_rank", "--lora-rank"), ("lora_adapter_name", "--lora-adapter-name"),
        ("chat_template", "--chat-template"), ("seed", "--seed"),
    ):
        command.extend([flag, str(getattr(body, field))])
    command.append("--load-in-4bit" if body.load_in_4bit else "--no-load-in-4bit")
    receipt = None

    def update_receipt():
        nonlocal receipt
        latest = read_receipt(receipt_path)
        if latest is None:
            return
        if (not isinstance(latest, dict)
                or not all(isinstance(latest.get(key), str) and latest[key]
                           for key in ("job_name", "job_id", "status", "model_artifact"))
                or latest.get("dataset_id") != dataset_id):
            raise ValueError("Training job receipt is incomplete or does not match the selected dataset.")
        if latest != receipt:
            receipt = latest
            on_submitted(latest)

    with (directory / "training.log").open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        waiter = asyncio.create_task(process.wait())
        try:
            while not waiter.done():
                await asyncio.wait({waiter}, timeout=1)
                update_receipt()
            code = await waiter
            update_receipt()
            if code:
                raise RuntimeError(f"Training command exited with code {code}. See the recent training job logs.")
            if receipt is None or receipt["status"] != "Completed":
                raise RuntimeError("Training did not return a completed Azure ML job receipt.")
            return receipt
        finally:
            if process.returncode is None:
                process.terminate()
            if not waiter.done():
                await asyncio.gather(waiter, return_exceptions=True)
