import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from lib.evaluation import RAG_METRICS, context_text, evaluate_rag_predictions
from lib.prompts import question_context_messages
from lib.release import write_evaluation_manifest
from test_lora_artifacts import ROOT, _functions


def _record(index=0):
    return {
        "id": f"sample-{index}", "type": "oracle", "question": f"Question {index}?",
        "context": {"sentences": [["First evidence.", "Second evidence."]]},
        "cot_answer": "<ANSWER>GROUND_TRUTH_ONLY</ANSWER>",
        "messages": [{"role": "user", "content": "Incorrect old prompt"}],
    }


def _scores(**kwargs):
    return {metric: {"score": 0.8, "reason": f"{metric}: supported"} for metric in RAG_METRICS}


def _wide_row():
    row = dict(_record(), oob_slm_response="Base answer", fine_tuned_slm_response="Adapter answer",
               gpt_slm_response="GPT answer")
    for prefix in ("oob", "fine_tuned", "gpt"):
        row[f"{prefix}_response_is_valid_json"] = False
        row[f"{prefix}_output_reached_limit"] = False
    return row


def _inference_functions(**overrides):
    namespace = {
        "Path": Path, "json": json, "random": random, "time": time, "os": os,
        "pd": pd, "argparse": argparse, "DTYPE": None, "DEVICE": "cpu",
        "log": logging.getLogger("test"), "question_context_messages": question_context_messages,
        "context_text": context_text, "evaluate_rag_predictions": evaluate_rag_predictions,
        "RAG_METRICS": RAG_METRICS, "mlflow": Mock(), "tqdm": lambda items, **kwargs: items,
        "write_evaluation_manifest": write_evaluation_manifest,
        "ThreadPoolExecutor": ThreadPoolExecutor, "as_completed": as_completed,
        "torch": SimpleNamespace(inference_mode=nullcontext),
        "build_azure_openai_client": Mock(return_value=Mock()),
        "call_gpt_with_retry": Mock(return_value={
            "text": "GPT answer", "finish_reason": "stop", "completion_tokens": 3,
        }),
        "get_az_client": Mock(return_value=Mock()), "download_model": Mock(),
        "upload_to_blobstore": Mock(return_value="mock://saved"),
    }
    namespace.update(overrides)
    return _functions(ROOT / "lib" / "inference.py", {
        "normalize_evaluation_record", "load_data", "generate_predictions", "run_gpt_inference", "main",
    }, namespace)


def test_actual_test_dataset_uses_question_context_without_ground_truth():
    functions = _inference_functions()
    records = functions["load_data"](
        {"test": ROOT / "data" / "training_data_raft" / "test.jsonl"},
        "test", None, None, 42,
    )
    assert records
    for record in records:
        messages = record["messages"]
        assert [message["role"] for message in messages] == ["system", "user"]
        assert record["question"].strip() in messages[1]["content"]
        assert context_text(record["context"]) in messages[1]["content"]
        assert record["cot_answer"] not in messages[1]["content"]


def test_record_normalization_ignores_old_messages_and_never_leaks_reference():
    functions = _inference_functions()
    record = functions["normalize_evaluation_record"](_record())
    prompt = json.dumps(record["messages"])
    assert "Question 0?" in prompt and "First evidence." in prompt
    assert "GROUND_TRUTH_ONLY" not in prompt and "Incorrect old prompt" not in prompt


@pytest.mark.parametrize("field", ["question", "context", "cot_answer", "id"])
def test_missing_test_fields_fail_explicitly(field):
    source = _record()
    del source[field]
    with pytest.raises(ValueError, match="missing fields"):
        _inference_functions()["normalize_evaluation_record"](source)


def test_test_sampling_is_reproducible_and_rejects_invalid_limit(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(_record(i)) + "\n" for i in range(5)), encoding="utf-8")
    load = _inference_functions()["load_data"]
    first = load({"test": path}, "test", None, 3, 42)
    second = load({"test": path}, "test", None, 3, 42)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    with pytest.raises(ValueError, match="max-records"):
        load({"test": path}, "test", 0, None, 42)


def test_gpt_call_uses_max_completion_tokens_for_reasoning_models():
    completion = Mock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(completion_tokens=4),
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )

    call_gpt_with_retry = _functions(
        ROOT / "lib" / "inference.py",
        {"call_gpt_with_retry"},
        {"random": random, "time": time, "log": logging.getLogger("test")},
    )["call_gpt_with_retry"]
    result = call_gpt_with_retry(
        client,
        "gpt-5.4-mini",
        [{"role": "user", "content": "Question"}],
        max_new_tokens=128,
        max_retries=1,
    )

    assert result["text"] == "answer"
    assert completion.call_args.kwargs["max_completion_tokens"] == 128
    assert "max_tokens" not in completion.call_args.kwargs


def test_inference_suppresses_successful_http_request_chatter():
    source = (ROOT / "lib" / "inference.py").read_text(encoding="utf-8")
    assert '"httpx2",' in source
    assert 'logging.getLogger(logger_name).setLevel(logging.WARNING)' in source


def test_all_three_responses_get_scores_reasons_and_exact_reference():
    judge = Mock(side_effect=_scores)
    result = evaluate_rag_predictions([_wide_row()], judge)[0]
    assert judge.call_count == 3
    assert [call.kwargs["response"] for call in judge.call_args_list] == [
        "Base answer", "Adapter answer", "GPT answer",
    ]
    for call in judge.call_args_list:
        assert call.kwargs["question"] == "Question 0?"
        assert call.kwargs["ground_truth"] == _record()["cot_answer"]
        assert call.kwargs["context"] == "First evidence.\n\nSecond evidence."
    for prefix in ("oob", "fine_tuned", "gpt"):
        assert result[f"{prefix}_rag_error"] == ""
        for metric in RAG_METRICS:
            assert result[f"{prefix}_{metric}_score"] == 0.8
            assert result[f"{prefix}_{metric}_reason"]


@pytest.mark.parametrize("bad_score", [True, math.nan, math.inf, -0.1, 1.1, "0.8"])
def test_invalid_judge_scores_are_errors_not_successful_defaults(bad_score):
    def judge(**kwargs):
        result = _scores()
        result["correctness"]["score"] = bad_score
        return result

    result = evaluate_rag_predictions([_wide_row()], judge)[0]
    assert result["oob_correctness_score"] is None
    assert "finite number" in result["oob_rag_error"]
    assert result["oob_correctness_reason"]


def test_per_model_inference_errors_do_not_skip_other_models():
    source = dict(_wide_row(), oob_inference_error="Out of memory", inference_error="oob failed")
    judge = Mock(side_effect=_scores)
    result = evaluate_rag_predictions([source], judge)[0]
    assert judge.call_count == 2
    assert result["oob_faithfulness_score"] is None
    assert "Out of memory" in result["oob_rag_error"]
    assert result["fine_tuned_faithfulness_score"] == 0.8
    assert result["gpt_correctness_score"] == 0.8


class _Matrix:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, columns = index
            return self.rows[row][columns]
        return self.rows[index]


class _Batch(dict):
    def to(self, device):
        return self


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0
    unk_token_id = -1

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages)
        return json.dumps(messages)

    def __call__(self, prompt, **kwargs):
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, **kwargs):
        return "<ANSWER>Generated answer</ANSWER>" if kwargs.get("skip_special_tokens") else "input"

    def pad(self, features, **kwargs):
        return _Batch(input_ids=_Matrix([row["input_ids"] for row in features]),
                      attention_mask=_Matrix([row["attention_mask"] for row in features]))


def test_generation_runs_oob_adapter_and_gpt_on_identical_prompts():
    functions = _inference_functions()
    records = [dict(functions["normalize_evaluation_record"](_record(i)), _source_index=i)
               for i in range(2)]
    tokenizers = [_Tokenizer(), _Tokenizer()]
    models = [Mock(), Mock()]
    for model in models:
        model.generate.return_value = _Matrix([[1, 2, 3, 0], [1, 2, 3, 0]])
    predictions = functions["generate_predictions"](
        models[0], tokenizers[0], models[1], tokenizers[1], records, "base/model",
        128, 32, 160, "both", 2, compare_with_gpt=True, az_client=Mock(),
        gpt_endpoint="https://example.invalid", gpt_deployment="gpt", gpt_api_version="version",
        gpt_max_workers=2, gpt_max_retries=1,
    )
    assert len(predictions) == 2
    assert tokenizers[0].prompts == tokenizers[1].prompts
    for model in models:
        model.generate.assert_called_once()
        assert "logits_processor" not in model.generate.call_args.kwargs
    for call in functions["call_gpt_with_retry"].call_args_list:
        assert call.args[2] in tokenizers[0].prompts
        assert "GROUND_TRUTH_ONLY" not in json.dumps(call.args[2])
    for column in ("oob_slm_response", "fine_tuned_slm_response", "gpt_slm_response"):
        assert predictions[column].str.len().gt(0).all()
    assert predictions["cot_answer"].tolist() == [record["cot_answer"] for record in records]


def _run_main(tmp_path, monkeypatch, judge):
    data = tmp_path / "data"
    data.mkdir()
    (data / "test.jsonl").write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    monkeypatch.setenv("AZUREML_RUN_ID", "test-run")
    monkeypatch.setitem(sys.modules, "deepeval_metrics",
                        SimpleNamespace(build_deepeval_judge=Mock(return_value=judge)))
    functions = _inference_functions()
    functions["download_model"].return_value = (Mock(max_seq_length=160), Mock(),
                                                Mock(max_seq_length=160), Mock())
    functions["generate_predictions"] = Mock(return_value=pd.DataFrame([_wide_row()]))
    args = argparse.Namespace(
        model="base/model", fine_tuned_model="adapter", data=str(data),
        output=str(tmp_path / "output"), mode="both", max_seq_length=160,
        max_input_tokens=128, max_new_tokens=32, inference_batch_size=1,
        max_records=None, sample_records=None, sample_seed=42,
        azure_client_id="client", hf_token="", load_in_4bit=False,
        log_packages=False, analyze_token_lengths=False, compare_with_gpt=True,
        gpt_endpoint="https://example.invalid", gpt_api_version="version",
        gpt_deployment="response-model", judge_deployment="independent-judge",
        gpt_max_workers=1, gpt_max_retries=1, storage_account_name="storage",
        finetuning_job_name="training-run",
    )
    return functions, args


def test_final_csv_is_evaluated_and_logged_as_artifact(tmp_path, monkeypatch):
    functions, args = _run_main(tmp_path, monkeypatch, _scores)
    functions["main"](args)
    output = tmp_path / "output" / "test-run_predictions.csv"
    saved = pd.read_csv(output, keep_default_na=False)
    assert saved["cot_answer"][0] == _record()["cot_answer"]
    assert saved["evaluation_framework"][0] == "deepeval"
    assert saved["rag_judge_deployment"][0] == "independent-judge"
    for prefix in ("oob", "fine_tuned", "gpt"):
        for metric in RAG_METRICS:
            assert saved[f"{prefix}_{metric}_score"][0] == 0.8
            assert saved[f"{prefix}_{metric}_reason"][0]
    functions["mlflow"].log_artifact.assert_any_call(str(output))
    manifest_path = output.parent / "evaluation_evidence.json"
    functions["mlflow"].log_artifact.assert_any_call(str(manifest_path))
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["training_job_name"] == "training-run"
    functions["upload_to_blobstore"].assert_called_once()
    assert any(call.args[0].get("fine_tuned_correctness_avg") == 0.8
               for call in functions["mlflow"].log_metrics.call_args_list)


def test_failed_evaluation_saves_csv_before_failing_job(tmp_path, monkeypatch):
    def judge(**kwargs):
        raise RuntimeError("Judge unavailable")

    functions, args = _run_main(tmp_path, monkeypatch, judge)
    with pytest.raises(RuntimeError, match="Incomplete RAG evaluation"):
        functions["main"](args)
    saved = pd.read_csv(tmp_path / "output" / "test-run_predictions.csv", keep_default_na=False)
    assert "Judge unavailable" in saved["gpt_rag_error"][0]
    assert functions["mlflow"].log_artifact.call_count == 2
    assert not any("fine_tuned_correctness_avg" in call.args[0]
                   for call in functions["mlflow"].log_metrics.call_args_list)


def test_deepeval_custom_judge_supports_schema_and_async():
    import asyncio
    from pydantic import BaseModel
    from lib.deepeval_metrics import AzureDeepEvalJudge

    class Result(BaseModel):
        reason: str

    completion = Mock(return_value='{"reason": "supported"}')
    judge = AzureDeepEvalJudge(completion, "azure-deployment")
    assert judge.generate("Prompt", Result).reason == "supported"
    assert asyncio.run(judge.a_generate("Prompt", Result)).reason == "supported"
    assert "Return JSON matching this schema" in completion.call_args.args[0]
    assert judge.get_model_name() == "azure-deployment"


def test_real_deepeval_metrics_return_scores_and_reasons_without_network():
    from lib.deepeval_metrics import build_deepeval_judge

    def completion(prompt):
        schema = json.loads(prompt.split("\nReturn JSON matching this schema:\n")[-1])
        keys = set(schema["properties"])
        if keys == {"statements"}:
            return json.dumps({"statements": ["The answer is supported."]})
        if keys == {"claims"}:
            return json.dumps({"claims": ["The answer is supported."]})
        if keys == {"truths"}:
            return json.dumps({"truths": ["The answer is supported."]})
        if keys == {"verdicts"}:
            return json.dumps({"verdicts": [{"verdict": "yes", "reason": "Supported"}]})
        if keys == {"steps"}:
            return json.dumps({"steps": ["Compare the response with the reference."]})
        if keys == {"score", "reason"}:
            return json.dumps({"score": 8, "reason": "Correct reference answer"})
        if keys == {"reason"}:
            return json.dumps({"reason": "Supported by evidence"})
        raise AssertionError(f"Unexpected DeepEval schema: {keys}")

    judge = build_deepeval_judge(completion, "mock-azure")
    results = judge(question="Question?", context="The answer is supported.",
                    response="The answer is supported.", ground_truth="The answer is supported.")
    assert set(results) == set(RAG_METRICS)
    for result in results.values():
        assert 0 <= result["score"] <= 1
        assert result["reason"]
