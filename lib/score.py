"""Managed online endpoint scoring script for the merged RAFT model."""

from __future__ import annotations

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.prompts import SYSTEM_PROMPT, user_prompt

model = None
tokenizer = None


def init() -> None:
    global model, tokenizer
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
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
