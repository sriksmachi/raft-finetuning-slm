"""Base checkpoint resolution and named LoRA adapters shared by GPU entry points."""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)
DEFAULT_ADAPTER_NAME = "finetuned_adapter"


def validate_adapter_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Adapter name must contain only letters, digits, underscores, or hyphens")
    return name


def adapter_name_from_model(model_name: str) -> str:
    normalized = model_name.rstrip("/\\").split("/")[-1].replace(".", "_")
    return validate_adapter_name(f"{normalized}-adapter")


def resolve_lora_adapter(artifact_path: str | Path) -> Path:
    root = Path(artifact_path)
    if not root.is_dir():
        raise ValueError(f"LoRA adapter directory does not exist: {root}")
    configs = list(root.rglob("adapter_config.json"))
    if len(configs) != 1:
        raise ValueError(
            f"Expected exactly one LoRA adapter under {root}, found {len(configs)}: {configs}"
        )
    adapter_dir = configs[0].parent
    if not any((adapter_dir / name).is_file() for name in (
        "adapter_model.safetensors", "adapter_model.bin"
    )):
        raise ValueError(f"LoRA adapter weights are missing from {adapter_dir}")
    if not (adapter_dir / "tokenizer_config.json").is_file():
        raise ValueError(f"Saved adapter tokenizer is missing from {adapter_dir}")
    return adapter_dir


def save_lora_adapter(model, tokenizer, output_dir: str | Path, adapter_name: str) -> Path:
    adapter_dir = Path(output_dir) / validate_adapter_name(adapter_name)
    model.save_pretrained(
        str(adapter_dir),
        selected_adapters=["default"],
        safe_serialization=True,
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained(str(adapter_dir))
    return resolve_lora_adapter(adapter_dir)


def load_lora_adapter(base_model, artifact_path: str | Path):
    from peft import PeftModel

    adapter_dir = resolve_lora_adapter(artifact_path)
    adapter_name = validate_adapter_name(adapter_dir.name)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        adapter_name=adapter_name,
        is_trainable=False,
    )
    model.set_adapter(adapter_name)
    model.requires_grad_(False)
    model.eval()
    return model


def resolve_base_model(base_model: str) -> str:
    """Return a loadable model reference.

    Foundry catalog assets are mounted/downloaded as a folder whose Transformers
    checkpoint (config.json + tokenizer + weights) is nested under an MLflow model
    layout, so ``from_pretrained`` on the mount root fails. When the argument is a
    local directory, locate the single Transformers checkpoint inside it; otherwise
    pass the value through unchanged (e.g. a Hugging Face repo id).
    """
    path = Path(base_model)
    if not path.is_dir():
        return base_model

    candidates = [
        config.parent
        for config in path.rglob("config.json")
        if (config.parent / "tokenizer_config.json").exists()
        and (
            any(config.parent.glob("*.safetensors"))
            or any(config.parent.glob("pytorch_model*.bin"))
        )
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one Transformers checkpoint under {path}, "
            f"found {len(candidates)}: {candidates}"
        )
    resolved = candidates[0]
    log.info("Resolved base model checkpoint: %s", resolved)
    return str(resolved)
