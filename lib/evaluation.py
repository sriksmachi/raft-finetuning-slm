"""Deterministic and LLM-as-judge metrics for grounded generation."""

from __future__ import annotations

import re
import time
from collections import Counter
from importlib import import_module
from numbers import Real
from typing import Any, Callable, Iterable, Mapping

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
    latencies = [
        float(row["latency_ms"])
        for row in records
        if row.get("latency_ms") is not None
    ]
    metrics = {
        "answer_token_f1": sum(f1_values) / len(f1_values),
        "answer_exact_match": sum(exact_values) / len(exact_values),
    }
    if latencies:
        metrics["mean_latency_ms"] = sum(latencies) / len(latencies)
    return metrics


def _context_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return "\n\n".join(filter(None, (_context_text(item) for item in value.values())))
    if isinstance(value, (list, tuple)):
        return "\n\n".join(filter(None, (_context_text(item) for item in value)))
    return ""


def extract_judge_score(result: Mapping[str, Any], metric: str) -> float:
    candidate_keys = (
        metric,
        f"{metric}_score",
        f"gpt_{metric}",
        f"gpt_{metric}_score",
    )
    for key in candidate_keys:
        value = result.get(key)
        if isinstance(value, Real) and not isinstance(value, bool):
            return float(value)
    for key, value in result.items():
        if metric in key.lower() and isinstance(value, Real) and not isinstance(value, bool):
            return float(value)
    raise KeyError(f"No numeric {metric} score in evaluator result keys: {sorted(result)}")


def _create_judges(judge_model_config: Mapping[str, Any], credential: Any) -> dict[str, Callable]:
    evaluation = import_module("azure.ai.evaluation")

    return {
        "relevancy": evaluation.RelevanceEvaluator(judge_model_config, credential=credential),
        "groundedness": evaluation.GroundednessEvaluator(judge_model_config, credential=credential),
        "coherence": evaluation.CoherenceEvaluator(judge_model_config, credential=credential),
    }


def evaluate_predictions(
    predictions: Iterable[Mapping[Any, Any]],
    judge_model_config: Mapping[str, Any] | None = None,
    credential: Any = None,
    *,
    delay_seconds: float = 0.0,
    judges: Mapping[str, Callable] | None = None,
) -> list[dict[str, Any]]:
    """Score offline or endpoint predictions with Azure AI Evaluation judges."""
    if judges is None:
        if judge_model_config is None:
            raise ValueError("judge_model_config is required when judges are not provided")
        judges = _create_judges(judge_model_config, credential)

    evaluated_rows = []
    for source_row in predictions:
        row = dict(source_row)
        query = str(row.get("query", row.get("question", "")))
        response = str(row.get("response", row.get("prediction", "")))
        context = _context_text(row.get("context", ""))
        inference_error = str(row.get("inference_error", ""))
        errors = []

        arguments = {
            "relevancy": {"query": query, "response": response, "context": context},
            "groundedness": {"query": query, "response": response, "context": context},
            "coherence": {"query": query, "response": response},
        }
        for metric in ("relevancy", "groundedness", "coherence"):
            row[metric] = None
            row[f"{metric}_reason"] = ""
            if inference_error or not response:
                errors.append(f"{metric}: skipped after inference failure")
                continue
            try:
                result = judges[metric](**arguments[metric])
                score_name = "relevance" if metric == "relevancy" else metric
                row[metric] = extract_judge_score(result, score_name)
                reason_key = next((key for key in result if "reason" in key.lower()), None)
                if reason_key:
                    row[f"{metric}_reason"] = str(result[reason_key])
            except Exception as exc:
                errors.append(f"{metric}: {type(exc).__name__}: {exc}")
            if delay_seconds:
                time.sleep(delay_seconds)
        row["metric_error"] = " | ".join(errors)
        evaluated_rows.append(row)
    return evaluated_rows


def aggregate_evaluation_metrics(rows: Iterable[dict]) -> dict[str, float]:
    records = list(rows)
    metrics = aggregate_metrics(records)
    for metric in ("relevancy", "groundedness", "coherence"):
        values = [float(row[metric]) for row in records if row.get(metric) is not None]
        if not values:
            raise ValueError(f"No successful {metric} judge scores are available")
        metrics[metric] = sum(values) / len(values)
    return metrics
