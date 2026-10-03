"""Deterministic and LLM-as-judge metrics for grounded generation."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from typing import Any

_ANSWER_PATTERN = re.compile(r"<ANSWER>\s*:?\s*(.*?)\s*</ANSWER>", re.IGNORECASE | re.DOTALL)
_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
log = logging.getLogger(__name__)
RAG_METRICS = ("faithfulness", "relevancy", "correctness")
RAG_MODEL_PREFIXES = ("oob", "fine_tuned", "gpt")


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


def context_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return "\n\n".join(filter(None, (context_text(item) for item in value.values())))
    if isinstance(value, (list, tuple)):
        return "\n\n".join(filter(None, (context_text(item) for item in value)))
    return ""


def evaluate_rag_predictions(
    predictions: Iterable[Mapping[str, Any]],
    judge: Callable[..., Mapping[str, Any]],
    *,
    model_prefixes: Iterable[str] = RAG_MODEL_PREFIXES,
) -> list[dict[str, Any]]:
    """Attach RAG scores/reasons to each model's response without dropping failed rows."""
    prefixes = tuple(model_prefixes)
    if not prefixes or any(prefix not in RAG_MODEL_PREFIXES for prefix in prefixes):
        raise ValueError(f"Unsupported RAG model prefixes: {prefixes}")
    evaluated = []
    for source in predictions:
        row = dict(source)
        question, reference = row["question"], row["cot_answer"]
        context = context_text(row["context"])
        if not all(isinstance(value, str) and value.strip()
                   for value in (question, reference, context)):
            raise ValueError("RAG evaluation requires question, context, and cot_answer")
        for prefix in prefixes:
            error_column = f"{prefix}_rag_error"
            row[error_column] = ""
            for metric in RAG_METRICS:
                row[f"{prefix}_{metric}_score"] = None
                row[f"{prefix}_{metric}_reason"] = ""
            response = row.get(f"{prefix}_slm_response", "")
            inference_error = row.get(f"{prefix}_inference_error", "")
            if inference_error or not isinstance(response, str) or not response.strip():
                error = f"RAG evaluation skipped: {inference_error or 'empty model response'}"
                row[error_column] = error
                for metric in RAG_METRICS:
                    row[f"{prefix}_{metric}_reason"] = error
                log.error("Record %s model %s: %s", row.get("id"), prefix, error)
                continue
            try:
                result = judge(
                    question=question, context=context,
                    response=response, ground_truth=reference,
                )
            except Exception as exc:
                log.exception("RAG evaluation failed for record %s model %s", row.get("id"), prefix)
                error = f"{type(exc).__name__}: {exc}"
                row[error_column] = error
                for metric in RAG_METRICS:
                    row[f"{prefix}_{metric}_reason"] = error
                continue
            errors = []
            for metric in RAG_METRICS:
                try:
                    item = result[metric]
                    if item.get("error"):
                        raise RuntimeError(item["error"])
                    score, reason = item["score"], item["reason"]
                    if (not isinstance(score, Real) or isinstance(score, bool)
                            or not math.isfinite(score) or not 0 <= score <= 1):
                        raise ValueError(f"{metric} score must be a finite number between 0 and 1")
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError(f"{metric} reason must be non-empty text")
                    row[f"{prefix}_{metric}_score"] = float(score)
                    row[f"{prefix}_{metric}_reason"] = reason.strip()
                except Exception as exc:
                    log.exception("RAG %s failed for record %s model %s", metric, row.get("id"), prefix)
                    error = f"{metric}: {type(exc).__name__}: {exc}"
                    errors.append(error)
                    row[f"{prefix}_{metric}_reason"] = error
            row[error_column] = " | ".join(errors)
        evaluated.append(row)
    if not evaluated:
        raise ValueError("At least one test prediction is required for RAG evaluation")
    return evaluated
