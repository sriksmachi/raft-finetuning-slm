"""Managed endpoint invocation and evaluation helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

from lib.evaluation import aggregate_metrics


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
        if "error" in parsed:
            raise RuntimeError(parsed["error"])
        return parsed["predictions"]
    finally:
        request_path.unlink(missing_ok=True)


def evaluate_endpoint(
    ml_client,
    endpoint_name: str,
    records: Iterable[dict[str, Any]],
    batch_size: int = 1,
    deployment_name: str | None = None,
) -> tuple[list[dict], dict[str, float]]:
    source = list(records)
    evaluated: list[dict] = []
    for offset in range(0, len(source), batch_size):
        batch = source[offset : offset + batch_size]
        predictions = invoke_endpoint(
            ml_client,
            endpoint_name,
            [{"instruction": row["instruction"]} for row in batch],
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
                    "prediction": result["prediction"],
                    "reference": row["cot_answer"],
                    "latency_ms": result.get("latency_ms"),
                }
            )
    return evaluated, aggregate_metrics(evaluated)
