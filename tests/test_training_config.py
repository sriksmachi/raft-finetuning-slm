import argparse
import ast
import logging
import math
import os
import statistics
import sys
from pathlib import Path

import pytest

from lib.lora_artifacts import DEFAULT_ADAPTER_NAME, validate_adapter_name

ROOT = Path(__file__).resolve().parents[1]


def _functions(path: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def test_stage_02_training_defaults(monkeypatch):
    namespace = _functions(
        ROOT / "scripts" / "02_azureml_fine_tuning.py",
        {"parse_args"},
        {
            "argparse": argparse,
            "get_training_environment_version": lambda: "4",
            "os": os,
            "__doc__": "",
        },
    )
    monkeypatch.setattr(sys, "argv", ["02_azureml_fine_tuning.py"])

    args = namespace["parse_args"]()

    assert args.max_seq_length == 2048
    assert args.load_in_4bit is False
    assert args.lora_rank == 16
    assert args.lora_adapter_name is None
    assert args.learning_rate == 2e-5
    assert args.batch_size == 2
    assert args.gradient_accumulation == 16
    assert args.seed == 42


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("Llama-3.1-8B-Instruct", "llama-3.1"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama-3.2"),
    ],
)
def test_chat_template_matches_llama_model(model_name, expected):
    namespace = _functions(
        ROOT / "scripts" / "02_azureml_fine_tuning.py",
        {"chat_template_for_model"},
        {},
    )
    assert namespace["chat_template_for_model"](model_name) == expected


def test_chat_template_requires_override_for_unknown_family():
    namespace = _functions(
        ROOT / "scripts" / "02_azureml_fine_tuning.py",
        {"chat_template_for_model"},
        {},
    )
    with pytest.raises(ValueError, match="--chat-template"):
        namespace["chat_template_for_model"]("mistralai/Mistral-7B-Instruct")


def test_remote_training_defaults(monkeypatch):
    namespace = _functions(
        ROOT / "lib" / "train.py",
        {"parse_args"},
        {
            "argparse": argparse,
            "DEFAULT_ADAPTER_NAME": DEFAULT_ADAPTER_NAME,
            "validate_adapter_name": validate_adapter_name,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--data", "data", "--model-output", "model"],
    )

    args = namespace["parse_args"]()

    assert args.max_seq_length == 2048
    assert args.load_in_4bit is False
    assert args.lora_rank == 16
    assert args.learning_rate == 2e-5
    assert args.batch_size == 2
    assert args.gradient_accumulation == 16


def test_sequence_length_uses_combined_chat_template_p95():
    namespace = _functions(
        ROOT / "lib" / "train.py",
        {"record_messages", "percentile_sequence_length"},
        {
            "SYSTEM_PROMPT": "system",
            "user_prompt": lambda value: value,
            "math": math,
            "statistics": statistics,
            "log": logging.getLogger(__name__),
        },
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            length = int(messages[1]["content"])
            return list(range(length))

    dataset = {
        "train": [
            {"instruction": str(length), "cot_answer": "answer"}
            for length in range(1, 11)
        ],
        "validation": [
            {"instruction": str(length), "cot_answer": "answer"}
            for length in range(11, 21)
        ],
    }

    selected, summary = namespace["percentile_sequence_length"](
        dataset, Tokenizer(), 2048
    )

    assert selected == 19
    assert summary["p95_token_length"] == 19
    assert summary["train_p95_token_length"] == 10
    assert summary["validation_p95_token_length"] == 20
    assert summary["fallback_max_seq_length"] == 2048


def test_stage_03_defaults_to_env_gpt_configuration(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.services.ai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-env")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2026-01-01-preview")
    namespace = _functions(
        ROOT / "scripts" / "03_azureml_slm_vs_gpt_evaluation.py",
        {"parse_args"},
        {
            "argparse": argparse,
            "datetime": __import__("datetime").datetime,
            "timezone": __import__("datetime").timezone,
            "os": os,
            "RUNS_DIR": Path("runs"),
            "__doc__": "",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["03_azureml_slm_vs_gpt_evaluation.py", "--mode", "llm_baseline"],
    )

    args = namespace["parse_args"]()

    assert args.gpt_endpoint == "https://example.services.ai.azure.com/"
    assert args.gpt_deployment == "gpt-env"
    assert args.gpt_api_version == "2026-01-01-preview"
    assert args.storage_account_name is None
    assert args.mode == "llm_baseline"


def test_stage_03_uses_workspace_default_storage_account():
    namespace = _functions(
        ROOT / "scripts" / "03_azureml_slm_vs_gpt_evaluation.py",
        {"resolve_storage_account_name"},
        {"log": logging.getLogger(__name__)},
    )
    client = type(
        "Client",
        (),
        {
            "datastores": type(
                "Datastores",
                (),
                {
                    "get_default": lambda self: type(
                        "Datastore",
                        (),
                        {"name": "workspaceblobstore", "account_name": "workspaceaccount"},
                    )()
                },
            )()
        },
    )()

    assert (
        namespace["resolve_storage_account_name"](client, None)
        == "workspaceaccount"
    )
    assert (
        namespace["resolve_storage_account_name"](client, "override-account")
        == "override-account"
    )
