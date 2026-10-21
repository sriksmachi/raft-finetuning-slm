"""Production adapter for the existing preparation CLI, isolated per run."""

import asyncio
import json
import os
import sys
from pathlib import Path

from app.fixtures import now
from app.models import ProductionPreparation

ROOT = Path(__file__).resolve().parents[1]


def append_job_status(directory: Path, message: str, log_name: str = "preparation.log") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / log_name).open("a", encoding="utf-8") as log:
        log.write(f"\n{now()} [STATUS] {message}\n")
        log.flush()


def load_data_samples(dataset_dir: Path) -> list[dict]:
    samples = []
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        if path.is_symlink():
            raise ValueError("Symlink dataset samples are not supported.")
        with path.open(encoding="utf-8") as stream:
            count = 0
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"{split} sample must be a JSON object.")
                samples.append({**row, "split": split})
                count += 1
                if count == 2:
                    break
    return samples


def build_data_card(body: ProductionPreparation, directory: Path, version: str) -> dict:
    from lib.data import SPLITS, dataset_fingerprint, validate_dataset

    dataset_dir = directory / "dataset"
    validation = validate_dataset(dataset_dir)
    counts = {split: validation[split]["rows"] for split in SPLITS}
    if sum(counts.values()) != body.num_samples:
        raise ValueError("Prepared dataset does not match the requested sample count")
    samples = load_data_samples(dataset_dir)
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
