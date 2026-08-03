"""Lightweight, deterministic metrics for grounded generation."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_ANSWER_PATTERN = re.compile(r"<ANSWER>\s*:?\s*(.*?)\s*</ANSWER>", re.I | re.S)
_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def extract_answer(text: str) -> str:
    match = _ANSWER_PATTERN.search(text or "")
    return (match.group(1) if match else text or "").strip()


def token_f1(prediction: str, reference: str) -> float:
    predicted = _TOKEN_PATTERN.findall(extract_answer(prediction).lower())
    expected = _TOKEN_PATTERN.findall(extract_answer(reference).lower())
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    normalize = lambda value: " ".join(_TOKEN_PATTERN.findall(extract_answer(value).lower()))
    return float(normalize(prediction) == normalize(reference))


def aggregate_metrics(rows: Iterable[dict]) -> dict[str, float]:
    records = list(rows)
    if not records:
        raise ValueError("At least one prediction is required")
    f1_values = [token_f1(row["prediction"], row["reference"]) for row in records]
    exact_values = [exact_match(row["prediction"], row["reference"]) for row in records]
    latencies = [float(row["latency_ms"]) for row in records if "latency_ms" in row]
    metrics = {
        "answer_token_f1": sum(f1_values) / len(f1_values),
        "answer_exact_match": sum(exact_values) / len(exact_values),
    }
    if latencies:
        metrics["mean_latency_ms"] = sum(latencies) / len(latencies)
    return metrics
