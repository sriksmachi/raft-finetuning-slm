"""Feature-profile drift detection for RAFT requests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _features(record: dict[str, Any]) -> dict[str, float]:
    instruction = str(record.get("instruction", ""))
    question = str(record.get("question", ""))
    return {
        "instruction_chars": float(len(instruction)),
        "question_chars": float(len(question)),
        "document_count": float(instruction.count("<DOCUMENT>")),
    }


def create_reference_profile(records: Iterable[dict[str, Any]], bins: int = 10) -> dict:
    rows = list(records)
    if not rows:
        raise ValueError("Reference records cannot be empty")
    features = [_features(row) for row in rows]
    profile: dict[str, Any] = {"row_count": len(rows), "features": {}}
    for name in features[0]:
        values = np.asarray([row[name] for row in features], dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))).tolist()
        if len(edges) == 1:
            edges = [edges[0] - 0.5, edges[0] + 0.5]
        counts, _ = np.histogram(values, bins=np.asarray(edges))
        probabilities = ((counts + 1e-6) / (counts.sum() + len(counts) * 1e-6)).tolist()
        profile["features"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "edges": edges,
            "probabilities": probabilities,
        }
    return profile


def population_stability_index(reference: list[float], observed: list[float]) -> float:
    return float(
        sum((actual - expected) * math.log(actual / expected) for expected, actual in zip(reference, observed))
    )


def detect_drift(
    reference_profile: dict,
    production_records: Iterable[dict[str, Any]],
    psi_threshold: float = 0.2,
) -> dict:
    rows = list(production_records)
    if not rows:
        raise ValueError("Production records cannot be empty")
    features = [_features(row) for row in rows]
    report: dict[str, Any] = {
        "row_count": len(rows),
        "psi_threshold": psi_threshold,
        "features": {},
    }
    for name, baseline in reference_profile["features"].items():
        edges = np.asarray(baseline["edges"], dtype=float)
        values = np.asarray([row[name] for row in features], dtype=float)
        values = np.clip(values, edges[0], edges[-1])
        counts, _ = np.histogram(values, bins=edges)
        observed = ((counts + 1e-6) / (counts.sum() + len(counts) * 1e-6)).tolist()
        psi = population_stability_index(baseline["probabilities"], observed)
        report["features"][name] = {
            "psi": psi,
            "drift_detected": psi >= psi_threshold,
            "production_mean": float(values.mean()),
        }
    report["drift_detected"] = any(
        value["drift_detected"] for value in report["features"].values()
    )
    return report


def write_json_artifact(value: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output
