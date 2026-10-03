"""Feature-profile drift detection for SLM Studio requests."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable
import numpy as np

FEATURE_NAMES = ("instruction_chars", "question_chars", "document_count")


def _positive_threshold(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("PSI threshold must be finite and positive")


def _probabilities(values: list[float]) -> np.ndarray:
    try:
        raw = np.asarray(values)
        probabilities = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be a finite positive vector") from exc
    if (
        probabilities.ndim != 1
        or raw.dtype.kind not in "iuf"
        or not probabilities.size
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0)
        or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-8)
    ):
        raise ValueError("Probabilities must be finite, positive, and sum to one")
    return probabilities


def _features(record: dict[str, Any]) -> dict[str, float]:
    if not isinstance(record, dict) or any(
        not isinstance(record.get(name), str) for name in ("instruction", "question")
    ):
        raise ValueError("Request records must contain instruction and question strings")
    instruction = record["instruction"]
    question = record["question"]
    return {
        "instruction_chars": float(len(instruction)),
        "question_chars": float(len(question)),
        "document_count": float(instruction.count("<DOCUMENT>")),
    }


def create_reference_profile(records: Iterable[dict[str, Any]], bins: int = 10) -> dict:
    """Create a reference profile for feature-profile drift detection."""
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins <= 0:
        raise ValueError("bins must be a positive integer")
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
    expected = _probabilities(reference)
    actual = _probabilities(observed)
    if len(expected) != len(actual):
        raise ValueError("Reference and observed probability vectors must have matching lengths")
    # Subtract logs rather than dividing: finite positive inputs can have an infinite ratio.
    return float(np.sum((actual - expected) * (np.log(actual) - np.log(expected))))


def _validate_profile(profile: dict) -> None:
    if not isinstance(profile, dict):
        raise ValueError("Reference profile must be an object")
    count = profile.get("row_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("Reference profile row_count must be a positive integer")
    baselines = profile.get("features")
    if not isinstance(baselines, dict) or set(baselines) != set(FEATURE_NAMES):
        raise ValueError(f"Reference profile must contain features: {FEATURE_NAMES}")
    for name, baseline in baselines.items():
        if not isinstance(baseline, dict):
            raise ValueError(f"Invalid reference feature: {name}")
        for statistic in ("mean", "std"):
            value = baseline.get(statistic)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Reference {name}.{statistic} must be finite and nonnegative")
        try:
            raw_edges = np.asarray(baseline.get("edges"))
            edges = np.asarray(baseline.get("edges"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reference edges for {name}") from exc
        if (
            edges.ndim != 1
            or raw_edges.dtype.kind not in "iuf"
            or len(edges) < 2
            or not np.all(np.isfinite(edges))
            or not np.all(edges[1:] > edges[:-1])
        ):
            raise ValueError(f"Reference edges for {name} must be finite and strictly increasing")
        probabilities = _probabilities(baseline.get("probabilities"))
        if len(probabilities) != len(edges) - 1:
            raise ValueError(f"Reference edges and probabilities for {name} do not match")


def detect_drift(
    reference_profile: dict,
    production_records: Iterable[dict[str, Any]],
    psi_threshold: float = 0.2,
) -> dict:
    """Compare normalized request features, folding observations outside the baseline into tail bins."""
    _positive_threshold(psi_threshold)
    _validate_profile(reference_profile)
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
        # Histogram otherwise silently drops out-of-range production observations.
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
    output.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return output
