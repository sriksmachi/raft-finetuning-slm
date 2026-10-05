"""Read-only Azure evaluation evidence exposed by the demo application."""

from __future__ import annotations

import json
from pathlib import Path


def load_evaluation_showcase(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Evaluation showcase file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Evaluation showcase file is invalid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Evaluation showcase must be a JSON object.")
    summary = payload.get("summary")
    records = payload.get("records")
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise RuntimeError("Evaluation showcase requires summary and records.")
    if summary.get("sample_count") != len(records):
        raise RuntimeError("Evaluation showcase sample_count does not match its records.")
    required_models = {"oob", "fine_tuned", "gpt"}
    models = summary.get("models")
    if not isinstance(models, dict) or set(models) != required_models:
        raise RuntimeError("Evaluation showcase requires oob, fine_tuned, and gpt summaries.")
    return payload
