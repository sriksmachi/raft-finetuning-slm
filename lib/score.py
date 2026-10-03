"""Managed online endpoint scoring for LoRA adapters and legacy merged models."""

from __future__ import annotations

import json
import logging
import os
import time
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# score.py and prompts.py sit side by side in lib/. AML puts the scoring
# script's own directory on sys.path, so `from lib.prompts import ...` fails
# in the container; a sibling import always resolves.
from prompts import SYSTEM_PROMPT, user_prompt
from lora_artifacts import load_lora_adapter, resolve_lora_adapter

model = None
tokenizer = None
inputs_collector = None
outputs_collector = None
log = logging.getLogger(__name__)


def _resolve_checkpoint(model_dir: str) -> str:
    """Return the folder inside ``AZUREML_MODEL_DIR`` that holds the checkpoint.

    Registered AML model assets frequently nest the Transformers artifacts one
    or more folders below the mount root (e.g. ``.../1/outputs/model``), so
    ``from_pretrained`` on the mount itself raises ``OSError``. Walk the tree
    for the single directory that contains both ``config.json`` and
    ``tokenizer_config.json``.
    """
    root = Path(model_dir)
    if (root / "config.json").exists() and (root / "tokenizer_config.json").exists():
        return str(root)
    candidates = [
        config.parent
        for config in root.rglob("config.json")
        if (config.parent / "tokenizer_config.json").exists()
    ]
    if len(candidates) == 1:
        print(f"[score] resolved checkpoint at {candidates[0]}", flush=True)
        return str(candidates[0])
    listing = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())[:50]
    raise RuntimeError(
        f"Could not locate a Transformers checkpoint under {root!r}. "
        f"Expected exactly one folder containing config.json + tokenizer_config.json; "
        f"found {len(candidates)}. Mount contents (first 50): {listing}"
    )


def _resolve_base_reference(reference: str) -> str:
    if reference.startswith("azureml://registries/"):
        import re
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        match = re.fullmatch(r"azureml://registries/([^/]+)/models/([^/]+)/versions/([^/]+)", reference)
        if not match:
            raise ValueError("Base registry model must use an immutable /versions/<version> URI")
        registry, name, version = match.groups()
        directory = Path(tempfile.mkdtemp(prefix="slm-studio-base-"))
        registry_client = MLClient(
            credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
            registry_name=registry,
        )
        registry_client.models.download(name=name, version=version, download_path=str(directory))
        return _resolve_checkpoint(str(directory))
    if Path(reference).is_dir():
        return _resolve_checkpoint(reference)
    if reference.startswith("azureml:"):
        raise ValueError("Use an immutable registry URI or an accessible Hugging Face base model ID")
    return reference


def init() -> None:
    global model, tokenizer, inputs_collector, outputs_collector
    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])
    if any(model_dir.rglob("config.json")):
        checkpoint_dir = _resolve_checkpoint(str(model_dir))
        tokenizer_dir = checkpoint_dir
        adapter_dir = None
    else:
        adapter_dir = resolve_lora_adapter(model_dir)
        checkpoint_dir = os.environ.get("SLM_BASE_MODEL")
        if not checkpoint_dir:
            raise ValueError(
                "LoRA deployment requires SLM_BASE_MODEL to reference the original base "
                "weights (a provisioned checkpoint path or accessible Hugging Face model ID). "
                "A training-job-local path in adapter_config.json cannot be reused."
            )
        checkpoint_dir = _resolve_base_reference(checkpoint_dir)
        tokenizer_dir = str(adapter_dir)
        vault = os.environ.get("SLM_KEY_VAULT_NAME")
        secret_name = os.environ.get("SLM_HF_TOKEN_SECRET_NAME")
        if bool(vault) != bool(secret_name):
            raise ValueError("Both endpoint Key Vault settings must be configured")
        if vault:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient

            secret_client = SecretClient(
                vault_url=f"https://{vault}.vault.azure.net",
                credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
            )
            os.environ["HF_TOKEN"] = secret_client.get_secret(secret_name).value
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_dir is not None:
        model = load_lora_adapter(model, adapter_dir)
    model.eval()
    inputs_collector, outputs_collector = None, None
    if os.environ.get("SLM_ENABLE_DATA_COLLECTION") == "true":
        from azureml.ai.monitoring import Collector

        def collection_error(exc):
            log.error("Endpoint data collection failed: %s", exc)

        inputs_collector = Collector(name="model_inputs", on_error=collection_error)
        outputs_collector = Collector(name="model_outputs", on_error=collection_error)


def _generate(item: dict) -> dict:
    instruction = str(item.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("Each input requires a non-empty 'instruction'")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(instruction)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=int(item.get("max_new_tokens", 512)),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, encoded["input_ids"].shape[1] :]
    return {
        "prediction": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def run(raw_data: str) -> str:
    try:
        payload = json.loads(raw_data)
        items = payload.get("input_data", payload)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            raise ValueError("Request must be an object or a list under 'input_data'")
        correlation = None
        if inputs_collector is not None:
            import pandas as pd

            correlation = inputs_collector.collect(pd.DataFrame([
                {key: item.get(key, "") for key in ("id", "question", "instruction")}
                for item in items
            ]))
        predictions = [_generate(item) for item in items]
        if outputs_collector is not None:
            import pandas as pd

            outputs_collector.collect(pd.DataFrame(predictions), correlation)
        return json.dumps({"predictions": predictions})
    except Exception as exc:
        log.exception("Endpoint scoring failed")
        return json.dumps({"error": str(exc)})
