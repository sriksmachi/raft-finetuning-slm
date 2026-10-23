import json

import pandas as pd
import pytest

from lib.llm_baseline import (
    apply_llm_baseline,
    file_sha256,
    write_llm_baseline,
)


def _predictions() -> pd.DataFrame:
    rows = []
    for source_index in range(2):
        row = {
            "sample_index": source_index,
            "source_index": source_index,
            "id": f"sample-{source_index}",
            "type": "oracle",
            "question": f"Question {source_index}?",
            "context": f"Evidence {source_index}",
            "cot_answer": f"Reference {source_index}",
            "gpt_slm_response": f"GPT answer {source_index}",
            "gpt_inference_error": "",
            "gpt_generated_token_count": 8,
            "gpt_output_reached_limit": False,
            "gpt_slm_latency_ms": 100.0 + source_index,
            "gpt_response_is_valid_json": False,
            "gpt_json_error": "not JSON",
            "gpt_rag_error": "",
        }
        for metric in ("faithfulness", "relevancy", "correctness"):
            row[f"gpt_{metric}_score"] = 0.8
            row[f"gpt_{metric}_reason"] = f"{metric} reason"
        rows.append(row)
    return pd.DataFrame(rows)


def _write_test_file(path):
    path.write_text(
        "\n".join(json.dumps({"id": index}) for index in range(2)) + "\n",
        encoding="utf-8",
    )


def test_baseline_round_trip_reuses_gpt_evidence_without_sample_index_binding(tmp_path):
    test_path = tmp_path / "test.jsonl"
    _write_test_file(test_path)
    baseline_dir = tmp_path / "baseline"
    source = _predictions()
    baseline_file, manifest_file = write_llm_baseline(
        source,
        baseline_dir,
        test_path,
        gpt_deployment="gpt-5.4-mini",
        judge_deployment="gpt-5.4-mini",
        gpt_api_version="2024-12-01-preview",
        max_new_tokens=4096,
    )

    candidate = source.drop(
        columns=[column for column in source.columns if column.startswith("gpt_")]
    ).iloc[[1]].copy()
    candidate["sample_index"] = 0
    reused, manifest = apply_llm_baseline(candidate, baseline_dir, test_path)

    assert baseline_file.is_file()
    assert manifest_file.is_file()
    assert manifest["test_sha256"] == file_sha256(test_path)
    assert reused.iloc[0]["source_index"] == 1
    assert reused.iloc[0]["sample_index"] == 0
    assert reused.iloc[0]["gpt_slm_response"] == "GPT answer 1"
    assert reused.iloc[0]["gpt_correctness_score"] == pytest.approx(0.8)


def test_baseline_rejects_different_test_dataset(tmp_path):
    test_path = tmp_path / "test.jsonl"
    _write_test_file(test_path)
    baseline_dir = tmp_path / "baseline"
    source = _predictions()
    write_llm_baseline(
        source,
        baseline_dir,
        test_path,
        gpt_deployment="gpt",
        judge_deployment="judge",
        gpt_api_version="version",
        max_new_tokens=32,
    )
    test_path.write_text('{"id": "changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="different test dataset"):
        apply_llm_baseline(source, baseline_dir, test_path)


def test_baseline_rejects_row_identity_mismatch(tmp_path):
    test_path = tmp_path / "test.jsonl"
    _write_test_file(test_path)
    baseline_dir = tmp_path / "baseline"
    source = _predictions()
    write_llm_baseline(
        source,
        baseline_dir,
        test_path,
        gpt_deployment="gpt",
        judge_deployment="judge",
        gpt_api_version="version",
        max_new_tokens=32,
    )
    candidate = source.copy()
    candidate.loc[0, "question"] = "Different question"

    with pytest.raises(ValueError, match="identity mismatch"):
        apply_llm_baseline(candidate, baseline_dir, test_path)


def test_baseline_rejects_incomplete_scores(tmp_path):
    test_path = tmp_path / "test.jsonl"
    _write_test_file(test_path)
    predictions = _predictions().drop(columns=["gpt_correctness_score"])

    with pytest.raises(ValueError, match="missing required columns"):
        write_llm_baseline(
            predictions,
            tmp_path / "baseline",
            test_path,
            gpt_deployment="gpt",
            judge_deployment="judge",
            gpt_api_version="version",
            max_new_tokens=32,
        )
