"""Production adapter for the existing preparation CLI, isolated per run."""

import asyncio
import os
import sys
from pathlib import Path

from app.fixtures import now
from app.models import ProductionPreparation

ROOT = Path(__file__).resolve().parents[1]


def append_job_status(directory: Path, message: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "preparation.log").open("a", encoding="utf-8") as log:
        log.write(f"\n{now()} [STATUS] {message}\n")
        log.flush()


def build_data_card(body: ProductionPreparation, directory: Path, version: str) -> dict:
    from lib.data import SPLITS, dataset_fingerprint, read_jsonl, validate_dataset

    dataset_dir = directory / "dataset"
    validation = validate_dataset(dataset_dir)
    counts = {split: validation[split]["rows"] for split in SPLITS}
    if sum(counts.values()) != body.num_samples:
        raise ValueError("Prepared dataset does not match the requested sample count")
    samples = []
    for split in SPLITS:
        for row in read_jsonl(dataset_dir / f"{split}.jsonl")[:2]:
            samples.append({"split": split, **{
                key: row[key] for key in ("question", "cot_answer", "type", "instruction")
            }})
    fingerprint = dataset_fingerprint(dataset_dir)
    reference = f"azureml:{body.data_asset_name}:{version}"
    return {
        "simulated": False, "source_folder": body.source_folder, "recipe": body.recipe,
        "record_count": sum(counts.values()), "requested_samples": body.num_samples,
        "split_counts": counts, "splits": counts,
        "fingerprint": fingerprint, "data_fingerprint": fingerprint,
        "validation": {"status": "passed", "checked_records": sum(counts.values()),
                       "invalid_records": 0, "splits": validation},
        "asset_reference": reference,
        "assets": {split: f"{reference}/{split}.jsonl" for split in SPLITS},
        "samples": samples,
    }


async def prepare_data(body: ProductionPreparation, directory: Path, version: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(ROOT / "scripts" / "01_prepare_data.py"),
        "--source-folder", body.source_folder,
        "--images-dir", str(directory / "images"),
        "--chunks-dir", str(directory / "chunks"),
        "--dataset-dir", str(directory / "dataset"),
        "--num-samples", str(body.num_samples), "--num-questions", str(body.num_questions),
        "--oracle-probability", str(body.oracle_probability),
        "--data-asset-name", body.data_asset_name, "--data-asset-version", version,
    ]
    with (directory / "preparation.log").open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        try:
            code = await process.wait()
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
            await process.wait()
            raise
    if code:
        raise RuntimeError(
            f"Preparation pipeline exited with code {code}. "
            f"See the server-side log: {directory / 'preparation.log'}"
        )
    return build_data_card(body, directory, version)
