import csv
import hashlib
import json

import pytest

from app.evaluation_showcase import load_evaluation_showcase
from scripts.export_evaluation_showcase import MODELS, SCORES, build_showcase


def write_inputs(tmp_path, row_count=2):
    predictions = tmp_path / "predictions.csv"
    fieldnames = ["id", "type", "question", "cot_answer"]
    for model, (_, response_column, latency_column) in MODELS.items():
        fieldnames.extend(
            [
                response_column,
                latency_column,
                *(f"{model}_{score}_score" for score in SCORES),
            ]
        )
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(row_count):
            row = {
                "id": f"record-{index}",
                "type": "oracle" if index == 0 else "distractor",
                "question": f"Question {index}?",
                "cot_answer": f"Reference {index}",
            }
            for model, (_, response_column, latency_column) in MODELS.items():
                row[response_column] = f"{model} answer {index}"
                row[latency_column] = str(100 + index * 100)
                for score in SCORES:
                    row[f"{model}_{score}_score"] = str(0.8 + index * 0.1)
            writer.writerow(row)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "evaluation_run_id": "evaluation-job",
                "training_job_name": "training-job",
                "created_at": "2026-10-04T00:00:00Z",
                "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return predictions, evidence


def test_build_showcase_summarizes_scores_and_latencies(tmp_path):
    predictions, evidence = write_inputs(tmp_path)

    payload = build_showcase(predictions, evidence)

    assert payload["summary"]["sample_count"] == 2
    assert payload["summary"]["models"]["fine_tuned"]["scores"]["correctness"] == pytest.approx(0.85)
    assert payload["summary"]["models"]["fine_tuned"]["latency_avg_ms"] == 150
    assert payload["summary"]["models"]["fine_tuned"]["latency_p95_ms"] == 200
    assert payload["records"][1]["answers"]["gpt"]["text"] == "gpt answer 1"

    output = tmp_path / "showcase.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert load_evaluation_showcase(output) == payload


def test_build_showcase_rejects_predictions_that_do_not_match_evidence(tmp_path):
    predictions, evidence = write_inputs(tmp_path)
    predictions.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        build_showcase(predictions, evidence)
