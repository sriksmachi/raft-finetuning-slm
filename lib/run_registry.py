"""YAML-backed ledger of Azure ML fine-tuning runs.

``scripts/02_azureml_fine_tuning.py`` writes one YAML record per submitted
training job to ``runs/<job_name>.yaml`` (created/updated via
:func:`record_run`). Stages 03 and 04 resolve and progressively enrich the
same record, so day-to-day usage does not require copying Azure ML job names
between scripts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def record_run(job_name: str, **fields: Any) -> Path:
    """Create or update the YAML record for ``job_name`` under ``runs/``.

    Existing fields are preserved and merged with ``fields``, so this can be
    called multiple times for the same job (e.g. once at submission, again
    after registration) to progressively enrich the same record.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{job_name}.yaml"

    record: dict[str, Any] = {}
    if run_path.exists():
        record = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}

    record["job_name"] = job_name
    record.setdefault("submitted_at", datetime.now(timezone.utc).isoformat())
    record.update(fields)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()

    run_path.write_text(
        yaml.safe_dump(record, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return run_path


def latest_run(experiment_name: str | None = None) -> dict[str, Any] | None:
    """Return the most recently submitted run record under ``runs/``.

    If ``experiment_name`` is given, only records with a matching
    ``experiment_name`` field are considered. Returns ``None`` if no (matching)
    records exist.
    """
    if not RUNS_DIR.exists():
        return None

    candidates = []
    for run_path in RUNS_DIR.glob("*.yaml"):
        record = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}
        if not record.get("job_name"):
            continue
        if experiment_name and record.get("experiment_name") != experiment_name:
            continue
        candidates.append(record)

    if not candidates:
        return None

    candidates.sort(key=lambda record: record.get("submitted_at", ""), reverse=True)
    return candidates[0]
