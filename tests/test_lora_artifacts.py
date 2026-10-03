import ast
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lib.lora_artifacts import (
    DEFAULT_ADAPTER_NAME,
    load_lora_adapter,
    resolve_lora_adapter,
    save_lora_adapter,
    validate_adapter_name,
)

ROOT = Path(__file__).resolve().parents[1]


def _adapter(root: Path, name: str = DEFAULT_ADAPTER_NAME) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"adapter")
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return directory


def _functions(path: Path, names: set[str], namespace: dict) -> dict:
    # Exercise GPU entry-point functions without importing Unsloth on a CPU host.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    exec(  # noqa: S102 - isolates selected functions from heavyweight optional imports
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("name", [DEFAULT_ADAPTER_NAME, "candidate-1", "security_adapter"])
def test_adapter_names(name):
    assert validate_adapter_name(name) == name


@pytest.mark.parametrize("name", ["", ".", "..", "../other", "a/b", "a\\b", "a.b", "name space"])
def test_invalid_adapter_names(name):
    with pytest.raises(ValueError, match="Adapter name"):
        validate_adapter_name(name)


def test_save_exports_only_named_adapter_and_tokenizer(tmp_path):
    def save_model(directory, **kwargs):
        path = Path(directory)
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"adapter")

    def save_tokenizer(directory):
        (Path(directory) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    model = Mock(save_pretrained=Mock(side_effect=save_model))
    tokenizer = Mock(save_pretrained=Mock(side_effect=save_tokenizer))
    directory = save_lora_adapter(model, tokenizer, tmp_path, "candidate-1")
    assert directory == tmp_path / "candidate-1"
    model.save_pretrained.assert_called_once_with(
        str(directory), selected_adapters=["default"], safe_serialization=True,
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained.assert_called_once_with(str(directory))
    assert {p.name for p in directory.iterdir()} == {
        "adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"
    }


def test_resolve_wrapped_adapter(tmp_path):
    directory = _adapter(tmp_path / "asset" / "outputs" / "model")
    assert resolve_lora_adapter(tmp_path) == directory
    assert resolve_lora_adapter(directory) == directory


@pytest.mark.parametrize("missing", ["adapter_model.safetensors", "tokenizer_config.json"])
def test_reject_incomplete_adapter(tmp_path, missing):
    directory = _adapter(tmp_path)
    (directory / missing).unlink()
    with pytest.raises(ValueError, match="missing"):
        resolve_lora_adapter(tmp_path)


def test_reject_ambiguous_adapters(tmp_path):
    _adapter(tmp_path, "first")
    _adapter(tmp_path, "second")
    with pytest.raises(ValueError, match="found 2"):
        resolve_lora_adapter(tmp_path)


def test_reject_missing_adapter(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        resolve_lora_adapter(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_lora_adapter(tmp_path / "missing")


def test_load_attaches_named_frozen_adapter_to_supplied_base(tmp_path, monkeypatch):
    directory = _adapter(tmp_path, "candidate-1")
    base = Mock()
    adapted = Mock()
    peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=Mock(return_value=adapted)))
    monkeypatch.setitem(sys.modules, "peft", peft)
    assert load_lora_adapter(base, tmp_path) is adapted
    peft.PeftModel.from_pretrained.assert_called_once_with(
        base, str(directory), adapter_name="candidate-1", is_trainable=False
    )
    adapted.set_adapter.assert_called_once_with("candidate-1")
    adapted.requires_grad_.assert_called_once_with(False)
    adapted.eval.assert_called_once()


@pytest.mark.parametrize("mode,base_loads", [("oob", 1), ("fine-tuned", 2), ("both", 2)])
def test_inference_modes_load_base_then_adapter(tmp_path, mode, base_loads):
    directory = _adapter(tmp_path)
    bases = [Mock(name="oob_base"), Mock(name="adapter_base")]
    tokenizers = [Mock(), Mock()]
    adapted = Mock()
    fast = SimpleNamespace(
        from_pretrained=Mock(side_effect=list(zip(bases, tokenizers))),
        for_inference=Mock(),
    )
    attach = Mock(return_value=adapted)
    saved_tokenizer = Mock()
    auto_tokenizer = SimpleNamespace(from_pretrained=Mock(return_value=saved_tokenizer))
    functions = _functions(ROOT / "lib" / "inference.py", {"download_model", "resolve_base_model"}, {
        "Path": Path, "DTYPE": None, "log": logging.getLogger("test"),
        "FastLanguageModel": fast, "configure_tokenizer": lambda value, **kwargs: value,
        "resolve_lora_adapter": resolve_lora_adapter, "load_lora_adapter": attach,
        "AutoTokenizer": auto_tokenizer,
    })
    model, tokenizer, fine_model, fine_tokenizer = functions["download_model"](
        "base/model", tmp_path, mode=mode, max_seq_length=16384, load_in_4bit=True,
    )
    assert fast.from_pretrained.call_count == base_loads
    for call in fast.from_pretrained.call_args_list:
        assert call.kwargs == {
            "model_name": "base/model", "max_seq_length": 16384,
            "dtype": None, "load_in_4bit": True,
        }
    if mode == "oob":
        assert model is bases[0] and tokenizer is tokenizers[0]
        assert fine_model is None and fine_tokenizer is None
        attach.assert_not_called()
    else:
        attach.assert_called_once_with(bases[1], directory)
        assert fine_model is adapted and fine_tokenizer is saved_tokenizer
        auto_tokenizer.from_pretrained.assert_called_once_with(str(directory))
        assert model is bases[0]


def test_inference_resolves_foundry_base_before_attaching_adapter(tmp_path):
    checkpoint = tmp_path / "base" / "mlflow" / "data"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    directory = _adapter(tmp_path / "output")
    base, adapter_base, tokenizer, adapted = Mock(), Mock(), Mock(), Mock()
    fast = SimpleNamespace(
        from_pretrained=Mock(side_effect=[(base, tokenizer), (adapter_base, tokenizer)]),
        for_inference=Mock(),
    )
    attach = Mock(return_value=adapted)
    functions = _functions(ROOT / "lib" / "inference.py", {"download_model", "resolve_base_model"}, {
        "Path": Path, "DTYPE": None, "log": logging.getLogger("test"),
        "FastLanguageModel": fast, "configure_tokenizer": lambda value, **kwargs: value,
        "resolve_lora_adapter": resolve_lora_adapter, "load_lora_adapter": attach,
        "AutoTokenizer": SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
    })
    functions["download_model"](tmp_path / "base", tmp_path / "output", mode="fine-tuned")
    assert fast.from_pretrained.call_args.kwargs["model_name"] == str(checkpoint)
    attach.assert_called_once_with(adapter_base, directory)


def test_training_checkpoints_are_not_exported():
    tree = ast.parse((ROOT / "lib" / "train.py").read_text(encoding="utf-8"))
    training_args = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "TrainingArguments")
    output_dir = next(keyword.value for keyword in training_args.keywords if keyword.arg == "output_dir")
    assert isinstance(output_dir, ast.Constant) and output_dir.value == "training_checkpoints"
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "save_lora_adapter" for node in ast.walk(tree))


def test_adapter_tokenizer_preserves_training_template():
    tokenizer = SimpleNamespace(
        chat_template="saved-training-template", pad_token=None, eos_token="eos",
        padding_side="right",
    )
    replace_template = Mock()
    functions = _functions(ROOT / "lib" / "inference.py", {"configure_tokenizer"}, {
        "get_chat_template": replace_template, "CHAT_TEMPLATE": "base-template",
        "log": logging.getLogger("test"),
    })
    assert functions["configure_tokenizer"](tokenizer, preserve_chat_template=True) is tokenizer
    replace_template.assert_not_called()
    assert tokenizer.chat_template == "saved-training-template"
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "eos"
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="missing its training chat template"):
        functions["configure_tokenizer"](tokenizer, preserve_chat_template=True)


def _scoring_functions():
    base = Mock()
    namespace = {
        "Path": Path, "os": os, "model": None, "tokenizer": None,
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), float32="float32"),
        "AutoModelForCausalLM": SimpleNamespace(from_pretrained=Mock(return_value=base)),
        "AutoTokenizer": SimpleNamespace(from_pretrained=Mock()),
        "load_lora_adapter": Mock(return_value=Mock()),
        "resolve_lora_adapter": resolve_lora_adapter,
    }
    return _functions(
        ROOT / "lib" / "score.py",
        {"_resolve_checkpoint", "_resolve_base_reference", "init"},
        namespace,
    )


def test_endpoint_scoring_loads_explicit_base_and_adapter(tmp_path, monkeypatch):
    directory = _adapter(tmp_path)
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_BASE_MODEL", "base/model")
    functions = _scoring_functions()
    functions["init"]()
    base_loader = functions["AutoModelForCausalLM"].from_pretrained
    base_loader.assert_called_once_with("base/model", torch_dtype="float32", device_map=None)
    functions["AutoTokenizer"].from_pretrained.assert_called_once_with(str(directory))
    functions["load_lora_adapter"].assert_called_once_with(base_loader.return_value, directory)
    assert functions["model"] is functions["load_lora_adapter"].return_value


def test_endpoint_adapter_requires_deployable_base_reference(tmp_path, monkeypatch):
    _adapter(tmp_path)
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("SLM_BASE_MODEL", raising=False)
    functions = _scoring_functions()
    with pytest.raises(ValueError, match="LoRA deployment requires SLM_BASE_MODEL"):
        functions["init"]()
    functions["AutoModelForCausalLM"].from_pretrained.assert_not_called()


def test_endpoint_preserves_legacy_merged_model_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "asset"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
    functions = _scoring_functions()
    functions["init"]()
    functions["AutoModelForCausalLM"].from_pretrained.assert_called_once_with(
        str(checkpoint), torch_dtype="float32", device_map=None
    )
    functions["load_lora_adapter"].assert_not_called()
