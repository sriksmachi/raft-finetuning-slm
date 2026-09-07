"""Managed online endpoint scoring script for the merged RAFT model."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# score.py and prompts.py sit side by side in lib/. AML puts the scoring
# script's own directory on sys.path, so `from lib.prompts import ...` fails
# in the container; a sibling import always resolves.
from prompts import SYSTEM_PROMPT, user_prompt

model = None
tokenizer = None


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


def init() -> None:
    global model, tokenizer
    checkpoint_dir = _resolve_checkpoint(os.environ["AZUREML_MODEL_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()


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
        return json.dumps({"predictions": [_generate(item) for item in items]})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
