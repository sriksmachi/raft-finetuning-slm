"""Convert an Azure evaluation predictions CSV into the app showcase payload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


MODELS = {
    "oob": ("Original SLM", "oob_slm_response", "oob_slm_latency_ms"),
    "fine_tuned": (
        "Fine-tuned SLM",
        "fine_tuned_slm_response",
        "fine_tuned_slm_latency_ms",
    ),
    "gpt": ("GPT baseline", "gpt_slm_response", "gpt_slm_latency_ms"),
}
SCORES = ("faithfulness", "relevancy", "correctness")
REQUIRED_COLUMNS = {
    "id",
    "type",
    "question",
    "cot_answer",
    *(column for _, column, _ in MODELS.values()),
    *(column for _, _, column in MODELS.values()),
    *(f"{model}_{score}_score" for model in MODELS for score in SCORES),
}


def number(row: dict[str, str], column: str, row_number: int) -> float | None:
    value = row[column].strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number} has a non-numeric {column}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Row {row_number} has a non-finite {column}: {value!r}")
    return parsed


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def build_showcase(predictions_path: Path, evidence_path: Path) -> dict:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected_hash = evidence.get("predictions_sha256")
    actual_hash = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("Predictions CSV hash does not match evaluation evidence.")

    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Predictions CSV is missing columns: {sorted(missing)}")
        source_rows = list(reader)

    records = []
    aggregates = {
        model: {"scores": {score: [] for score in SCORES}, "latencies": []}
        for model in MODELS
    }
    for row_number, row in enumerate(source_rows, start=2):
        answers = {}
        for model, (_, response_column, latency_column) in MODELS.items():
            scores = {
                score: number(row, f"{model}_{score}_score", row_number)
                for score in SCORES
            }
            latency = number(row, latency_column, row_number)
            answers[model] = {
                "text": row[response_column],
                "latency_ms": latency,
                "scores": scores,
            }
            for score, value in scores.items():
                if value is not None:
                    aggregates[model]["scores"][score].append(value)
            if latency is not None:
                aggregates[model]["latencies"].append(latency)
        records.append(
            {
                "id": row["id"],
                "type": row["type"],
                "query": row["question"],
                "cot_answer": row["cot_answer"],
                "answers": answers,
            }
        )

    model_summaries = {}
    for model, (label, _, _) in MODELS.items():
        model_summaries[model] = {
            "label": label,
            "scores": {
                score: average(aggregates[model]["scores"][score])
                for score in SCORES
            },
            "latency_avg_ms": average(aggregates[model]["latencies"]),
            "latency_p95_ms": percentile_95(aggregates[model]["latencies"]),
        }

    return {
        "schema_version": 1,
        "summary": {
            "evaluation_run_id": evidence["evaluation_run_id"],
            "training_job_name": evidence["training_job_name"],
            "created_at": evidence["created_at"],
            "sample_count": len(records),
            "models": model_summaries,
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/evaluation_evidence.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_showcase(args.predictions, args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payload['records'])} evaluation records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
