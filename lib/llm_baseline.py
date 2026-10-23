"""Immutable GPT baseline evidence for repeatable Stage 03 evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASELINE_FILE = "llm_baseline.csv"
MANIFEST_FILE = "llm_baseline_manifest.json"
IDENTITY_COLUMNS = ("source_index", "id", "type", "question", "context", "cot_answer")
if __package__:
    from lib.evaluation import RAG_METRICS
    from lib.release import file_sha256
else:
    from evaluation import RAG_METRICS
    from release import file_sha256


def required_gpt_columns() -> set[str]:
    columns = {
        "gpt_slm_response",
        "gpt_inference_error",
        "gpt_generated_token_count",
        "gpt_output_reached_limit",
        "gpt_slm_latency_ms",
        "gpt_response_is_valid_json",
        "gpt_json_error",
        "gpt_rag_error",
    }
    for metric in RAG_METRICS:
        columns.update(
            {f"gpt_{metric}_score", f"gpt_{metric}_reason"}
        )
    return columns


def validate_gpt_evidence(frame: pd.DataFrame) -> list[str]:
    required = required_gpt_columns() | set(IDENTITY_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"GPT baseline is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("GPT baseline cannot be empty")
    if frame["source_index"].duplicated().any():
        duplicates = frame.loc[
            frame["source_index"].duplicated(), "source_index"
        ].tolist()
        raise ValueError(f"GPT baseline has duplicate source_index values: {duplicates[:10]}")
    if frame["gpt_slm_response"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("GPT baseline contains empty responses")
    if frame["gpt_inference_error"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("GPT baseline contains inference errors")
    if frame["gpt_rag_error"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("GPT baseline contains RAG evaluation errors")
    for metric in RAG_METRICS:
        scores = pd.to_numeric(frame[f"gpt_{metric}_score"], errors="coerce")
        if scores.isna().any() or not scores.between(0.0, 1.0).all():
            raise ValueError(f"GPT baseline has invalid {metric} scores")
    return sorted(column for column in frame.columns if column.startswith("gpt_"))


def write_llm_baseline(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    test_path: str | Path,
    *,
    gpt_deployment: str,
    judge_deployment: str,
    gpt_api_version: str,
    max_new_tokens: int,
) -> tuple[Path, Path]:
    gpt_columns = validate_gpt_evidence(predictions)
    baseline = predictions[[*IDENTITY_COLUMNS, *gpt_columns]].copy()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    baseline_path = root / BASELINE_FILE
    baseline.to_csv(baseline_path, index=False)
    manifest = {
        "schema_version": 1,
        "asset_name": "llm_baseline_dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_sha256": file_sha256(test_path),
        "record_count": len(baseline),
        "source_indexes": sorted(int(value) for value in baseline["source_index"]),
        "baseline_file": baseline_path.name,
        "baseline_sha256": file_sha256(baseline_path),
        "gpt_deployment": gpt_deployment,
        "judge_deployment": judge_deployment,
        "gpt_api_version": gpt_api_version,
        "max_new_tokens": max_new_tokens,
    }
    manifest_path = root / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return baseline_path, manifest_path


def apply_llm_baseline(
    predictions: pd.DataFrame,
    baseline_dir: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    root = Path(baseline_dir)
    manifests = list(root.rglob(MANIFEST_FILE))
    if len(manifests) != 1:
        raise ValueError(
            f"Expected exactly one {MANIFEST_FILE} in GPT baseline asset, "
            f"found {len(manifests)}"
        )
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported GPT baseline schema version")
    if manifest.get("asset_name") != "llm_baseline_dataset":
        raise ValueError("GPT baseline manifest has an unexpected asset name")
    for field in (
        "gpt_deployment",
        "judge_deployment",
        "gpt_api_version",
        "record_count",
        "source_indexes",
    ):
        if field not in manifest:
            raise ValueError(f"GPT baseline manifest is missing {field}")
    if manifest.get("test_sha256") != file_sha256(test_path):
        raise ValueError("GPT baseline was generated from a different test dataset")
    filename = manifest.get("baseline_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("GPT baseline manifest contains an invalid baseline_file")
    baseline_path = manifests[0].parent / filename
    if manifest.get("baseline_sha256") != file_sha256(baseline_path):
        raise ValueError("GPT baseline CSV hash does not match its manifest")

    baseline = pd.read_csv(baseline_path, keep_default_na=False)
    gpt_columns = validate_gpt_evidence(baseline)
    if manifest.get("record_count") != len(baseline):
        raise ValueError("GPT baseline record count does not match its manifest")
    observed_source_indexes = sorted(
        int(value) for value in baseline["source_index"]
    )
    if manifest.get("source_indexes") != observed_source_indexes:
        raise ValueError("GPT baseline source indexes do not match its manifest")
    baseline_by_source = {
        int(row["source_index"]): row for _, row in baseline.iterrows()
    }

    result = predictions.copy()
    for row_index, current in result.iterrows():
        source_index = int(current["source_index"])
        cached = baseline_by_source.get(source_index)
        if cached is None:
            raise ValueError(
                f"GPT baseline has no row for source_index={source_index}"
            )
        for column in IDENTITY_COLUMNS[1:]:
            if str(current[column]) != str(cached[column]):
                raise ValueError(
                    "GPT baseline row identity mismatch for "
                    f"source_index={source_index}, column={column}"
                )
        for column in gpt_columns:
            result.at[row_index, column] = cached[column]
    return result, manifest
