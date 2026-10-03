"""Managed endpoint invocation and evaluation helpers."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Iterable

from lib.evaluation import context_text
from lib.prompts import question_context_instruction


def invoke_endpoint(
    ml_client,
    endpoint_name: str,
    items: list[dict[str, Any]],
    deployment_name: str | None = None,
) -> list[dict]:
    payload = {"input_data": items}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream)
        request_path = Path(stream.name)
    try:
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            request_file=str(request_path),
        )
        parsed = json.loads(response)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if not isinstance(parsed, dict) or parsed.get("error"):
            raise RuntimeError(f"Endpoint invocation failed: {parsed}")
        predictions = parsed.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(items):
            raise RuntimeError("Endpoint returned an invalid prediction count")
        for prediction in predictions:
            if not isinstance(prediction, dict) or not str(prediction.get("prediction", "")).strip():
                raise RuntimeError("Endpoint returned an empty or invalid prediction")
            latency = prediction.get("latency_ms")
            if (isinstance(latency, bool) or not isinstance(latency, (int, float))
                    or not math.isfinite(latency) or latency <= 0):
                raise RuntimeError("Endpoint returned invalid latency evidence")
        return predictions
    finally:
        request_path.unlink(missing_ok=True)


def evaluate_endpoint(
    ml_client,
    endpoint_name: str,
    records: Iterable[dict[str, Any]],
    batch_size: int = 1,
    deployment_name: str | None = None,
    max_new_tokens: int = 512,
) -> list[dict]:
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("Endpoint batch size and output token budget must be positive")
    source = list(records)
    if not source:
        raise ValueError("Endpoint evaluation requires sealed test records")
    evaluated: list[dict] = []
    for offset in range(0, len(source), batch_size):
        batch = source[offset : offset + batch_size]
        predictions = invoke_endpoint(
            ml_client,
            endpoint_name,
            [{
                "id": row["id"], "question": row["question"],
                "instruction": question_context_instruction(row["question"], context_text(row["context"])),
                "max_new_tokens": max_new_tokens,
            } for row in batch],
            deployment_name=deployment_name,
        )
        if len(predictions) != len(batch):
            raise RuntimeError("Endpoint returned a different number of predictions than inputs")
        for row, result in zip(batch, predictions):
            evaluated.append(
                {
                    "id": row.get("id"),
                    "type": row.get("type"),
                    "question": row["question"],
                    "context": row.get("context", ""),
                    "prediction": result["prediction"],
                    "reference": row["cot_answer"],
                    "latency_ms": result.get("latency_ms"),
                }
            )
    return evaluated
