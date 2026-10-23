import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlflow
import pandas as pd
import torch
import unsloth  # noqa: F401 - must preload before importing Transformers
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient
from evaluation import RAG_METRICS, context_text, evaluate_rag_predictions
from llm_baseline import apply_llm_baseline, write_llm_baseline
from lora_artifacts import load_lora_adapter, resolve_base_model, resolve_lora_adapter
from prompts import question_context_messages
from release import write_evaluation_manifest
from tqdm import tqdm
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from data import read_jsonl

# ==========================================================================
# Logging
# ==========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

log = logging.getLogger("slm_inference")

# Suppress Azure SDK logs
for logger_name in [
    "azure",
    "azure.core",
    "azure.identity",
    "azure.ai.ml",
    "azureml",
    "azureml.fsspec",
    "azure.core.pipeline",
    "azure.core.pipeline.policies",
    "azure.core.pipeline.policies.http_logging_policy",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Suppress HF and network chatter
for logger_name in [
    "transformers",
    "huggingface_hub",
    "accelerate",
    "datasets",
    "httpx",
    "httpx2",
    "openai",
    "urllib3",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


# ==========================================================================
# Constants
# ==========================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.bfloat16
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

CHAT_TEMPLATE = "llama-3.1"


def configure_tokenizer(tokenizer, preserve_chat_template=False):
    """Apply the same chat template used during fine-tuning."""
    if preserve_chat_template:
        if not tokenizer.chat_template:
            raise ValueError("Saved LoRA adapter tokenizer is missing its training chat template")
    else:
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=CHAT_TEMPLATE,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    log.info(
        "Configured tokenizer with chat template %s",
        "saved training template" if preserve_chat_template else CHAT_TEMPLATE,
    )
    return tokenizer


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--azure-client-id",
        required=True,
    )
    parser.add_argument(
        "--data",
        required=True,
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Use the first N test records",
    )
    parser.add_argument(
        "--sample-records",
        type=int,
        default=None,
        help="Randomly sample N test records for a reproducible test run",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used with --sample-records",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=12288,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--hf-token",
        default="",
    )
    parser.add_argument(
        "--storage-account-name",
        required=True,
    )
    parser.add_argument(
        "--fine-tuned-model",
        required=False,
        default=None,
        help="Path to the mounted named LoRA adapter output from lib/train.py. "
        "Required when --mode is 'fine-tuned' or 'both'.",
    )
    parser.add_argument("--finetuning-job-name", default=None)
    parser.add_argument(
        "--workflow-mode",
        choices=["llm_baseline", "slm_evaluation"],
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["oob", "fine-tuned", "both"],
        help="Inference mode: 'oob' or 'fine-tuned' or 'both'",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model in 4-bit precision for memory efficiency",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=2,
        help="Number of similar-length prompts generated together",
    )
    parser.add_argument(
        "--analyze-token-lengths",
        action="store_true",
        help="Analyze every dataset split before inference",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=16384,
        help="Maximum sequence length for the model",
    )
    parser.add_argument("--log-packages", action="store_true")
    parser.add_argument(
        "--compare-with-gpt",
        action="store_true",
        help="Additionally run inference against an Azure OpenAI/Azure AI Foundry "
        "GPT deployment for the same records, alongside whatever --mode selects "
        "for the local SLM (e.g. --mode fine-tuned --compare-with-gpt compares "
        "the fine-tuned SLM against GPT).",
    )
    parser.add_argument(
        "--gpt-endpoint",
        default=None,
        help="Azure OpenAI / Azure AI Foundry endpoint. Required with --compare-with-gpt.",
    )
    parser.add_argument("--gpt-deployment", default="gpt-4.1")
    parser.add_argument(
        "--judge-deployment", default=None,
        help="RAG judge deployment on --gpt-endpoint; defaults to --gpt-deployment.",
    )
    parser.add_argument("--gpt-api-version", default="2024-10-21")
    parser.add_argument(
        "--gpt-max-workers",
        type=int,
        default=16,
        help="Concurrent Azure OpenAI requests for the GPT comparison pass",
    )
    parser.add_argument("--gpt-max-retries", type=int, default=4)
    parser.add_argument(
        "--gpt-baseline",
        default=None,
        help="Mounted llm_baseline_dataset asset. Reuses cached GPT responses and scores.",
    )
    parser.add_argument(
        "--llm-baseline-output",
        default=None,
        help="Registered output directory for a newly generated GPT baseline.",
    )
    return parser.parse_args()


# ==========================================================================
# Model Utilities
# ==========================================================================

def get_az_client(client_id) -> ManagedIdentityCredential:
    """Create authenticated Azure ML client."""
    client = ManagedIdentityCredential(client_id=client_id)
    return client


def download_model(model_name: str,
                   fine_tuned_model_path: str | None = None,
                   mode: str = "oob",
                   max_seq_length: int = 16384,
                   dtype: str = DTYPE,
                   load_in_4bit: bool = False,
    ):
    if (mode == "fine-tuned" or mode == "both") and fine_tuned_model_path is None:
        raise ValueError(
            "For fine-tuned mode, provide a mounted fine-tuned SLM path using --fine-tuned-model"
        )

    model_name = resolve_base_model(str(model_name))
    log.info("Downloading or loading base model: %s", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_length,
        dtype=dtype, load_in_4bit=load_in_4bit
    )
    FastLanguageModel.for_inference(model)
    tokenizer = configure_tokenizer(tokenizer)
    fine_tuned_model, fine_tuned_tokenizer = None, None
    if mode == "fine-tuned" or mode == "both":
        adapter_dir = resolve_lora_adapter(fine_tuned_model_path)
        # The OOB model must never receive the fine-tuned adapter.
        adapter_base_model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        log.info("Loading fine-tuned LoRA adapter '%s' from %s", adapter_dir.name, adapter_dir)
        fine_tuned_model = load_lora_adapter(adapter_base_model, adapter_dir)
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        FastLanguageModel.for_inference(fine_tuned_model)
        fine_tuned_tokenizer = configure_tokenizer(
            fine_tuned_tokenizer, preserve_chat_template=True
        )
        log.info("Loaded base model with fine-tuned adapter '%s'", adapter_dir.name)

    return model, tokenizer, fine_tuned_model, fine_tuned_tokenizer


# ==========================================================================
# Data Utilities
# ==========================================================================

def normalize_evaluation_record(record: dict) -> dict:
    missing = {"id", "question", "context", "cot_answer"}.difference(record)
    if missing:
        raise ValueError(f"Test record is missing fields: {sorted(missing)}")
    if record["id"] is None or not str(record["id"]).strip():
        raise ValueError("Test record id must be non-empty")
    if not isinstance(record["cot_answer"], str) or not record["cot_answer"].strip():
        raise ValueError(f"Test record {record['id']} cot_answer must be non-empty text")
    normalized = dict(record)
    normalized["messages"] = question_context_messages(
        record["question"], context_text(record["context"])
    )
    return normalized


def load_data(
    files: dict,
    split: str,
    max_records: int,
    sample_records: int,
    sample_seed: int,
):
    split_path = Path(files[split])
    if max_records is not None and max_records <= 0:
        raise ValueError("--max-records must be greater than zero")
    records = []
    with (split_path).open(encoding="utf-8") as stream:
        for source_index, line in enumerate(stream):
            if not line.strip():
                continue
            record = normalize_evaluation_record(json.loads(line))
            record["_source_index"] = source_index
            records.append(record)
            if max_records and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No evaluation records found in {split_path}")

    if sample_records is not None:
        if sample_records <= 0:
            raise ValueError("--sample-records must be greater than zero")
        if sample_records > len(records):
            raise ValueError(
                f"--sample-records={sample_records} exceeds the "
                f"{len(records)} available test records"
            )
        records = random.Random(sample_seed).sample(records, sample_records)
        log.info(
            "Sampled %d records using seed %d; source indices=%s",
            sample_records,
            sample_seed,
            [record["_source_index"] for record in records],
        )

    log.info("Loaded %s records from %s", len(records), split_path)
    return records


# ==========================================================================
# GPT comparison (Azure OpenAI / Azure AI Foundry)
# ==========================================================================

def build_azure_openai_client(endpoint: str, api_version: str, az_client):
    """Build an Azure OpenAI client authenticated with the job's managed identity.

    Reuses the same ManagedIdentityCredential already used for blob storage
    access in this script, rather than introducing a second credential type.
    """
    from openai import AzureOpenAI

    if not endpoint:
        raise ValueError("--gpt-endpoint is required when --compare-with-gpt is set")

    token_provider = get_bearer_token_provider(
        az_client, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )


def call_gpt_with_retry(
    client,
    deployment: str,
    messages: list,
    max_new_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float = 2.0,
    response_format: dict | None = None,
) -> dict:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0,
                max_completion_tokens=max_new_tokens,
                **({"response_format": response_format} if response_format else {}),
            )
            choice = response.choices[0]
            usage = response.usage
            return {
                "text": choice.message.content or "",
                "finish_reason": choice.finish_reason,
                "completion_tokens": usage.completion_tokens if usage else None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "model_id": getattr(response, "model", None),
            }
        except Exception as exc:  # noqa: BLE001 - SDK exposes multiple transient exception types
            last_error = exc
            sleep_seconds = (
                retry_backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            )
            log.warning(
                "GPT call failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"GPT call failed after {max_retries} attempts") from last_error


def run_gpt_inference(
    rows: list,
    prepared_records: list,
    az_client,
    gpt_endpoint: str,
    gpt_deployment: str,
    gpt_api_version: str,
    gpt_max_workers: int,
    gpt_max_retries: int,
    max_new_tokens: int,
) -> None:
    """Populate gpt_* columns on `rows` in place, by `sample_index`.

    Runs independently of the local SLM generation path: GPT receives the
    same `prompt_messages` directly via the Chat Completions API (no Llama
    chat template, no tokenization/batching -- those are SLM-specific
    concerns). Calls run concurrently since they are I/O-bound.
    """
    client = build_azure_openai_client(gpt_endpoint, gpt_api_version, az_client)
    log.info(
        "Starting GPT inference for %d records against deployment=%s "
        "with max_workers=%d",
        len(prepared_records),
        gpt_deployment,
        gpt_max_workers,
    )

    def _run_one(sample_index, prompt_messages):
        started_at = time.perf_counter()
        result = call_gpt_with_retry(
            client, gpt_deployment, prompt_messages, max_new_tokens, gpt_max_retries
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        return sample_index, result, latency_ms

    outputs_reaching_limit = 0
    with ThreadPoolExecutor(max_workers=gpt_max_workers) as executor:
        futures = {
            executor.submit(_run_one, sample_index, prompt_messages): sample_index
            for sample_index, prompt_messages in prepared_records
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating gpt"
        ):
            sample_index = futures[future]
            try:
                _, result, latency_ms = future.result()
            except Exception as exc:
                log.exception("GPT inference failed for sample %s", sample_index)
                existing_error = rows[sample_index]["inference_error"]
                rows[sample_index]["inference_error"] = (
                    f"{existing_error}; gpt: {exc}" if existing_error else f"gpt: {exc}"
                )
                rows[sample_index]["gpt_inference_error"] = str(exc)
                continue

            response_text = result["text"].strip()
            reached_limit = result["finish_reason"] == "length"
            if reached_limit:
                outputs_reaching_limit += 1

            rows[sample_index]["gpt_slm_response"] = response_text
            rows[sample_index]["gpt_generated_token_count"] = result["completion_tokens"]
            rows[sample_index]["gpt_prompt_token_count"] = result.get("prompt_tokens")
            rows[sample_index]["gpt_model_id"] = result.get("model_id")
            rows[sample_index]["gpt_output_reached_limit"] = reached_limit
            rows[sample_index]["gpt_slm_latency_ms"] = latency_ms
            try:
                json.loads(response_text)
                rows[sample_index]["gpt_response_is_valid_json"] = True
            except json.JSONDecodeError as exc:
                rows[sample_index]["gpt_json_error"] = str(exc)

    log.info(
        "GPT outputs reaching max_tokens limit: %d",
        outputs_reaching_limit,
    )


# ==========================================================================
# Inference
# ==========================================================================

def generate_predictions(
    model,
    tokenizer,
    fine_tuned_model,
    fine_tuned_tokenizer,
    source_records,
    model_label,
    max_input_tokens,
    max_new_tokens,
    max_seq_length,
    inference_mode,
    inference_batch_size,
    compare_with_gpt=False,
    az_client=None,
    gpt_endpoint=None,
    gpt_deployment=None,
    gpt_api_version=None,
    gpt_max_workers=16,
    gpt_max_retries=4,
):
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be greater than zero")
    artifact_attempt = f"{os.getpid()}-{time.time_ns()}"

    available_input_tokens = max_seq_length - max_new_tokens
    input_limit = min(max_input_tokens, available_input_tokens)
    if input_limit <= 0:
        raise ValueError(
            "max_new_tokens must be smaller than max_seq_length"
        )
    log.info(
        "Token budget: context=%d, input_limit=%d, output_limit=%d, "
        "unused_context_buffer=%d",
        max_seq_length,
        input_limit,
        max_new_tokens,
        max_seq_length - input_limit - max_new_tokens,
    )

    rows = []
    prepared_records = []
    for sample_index, record in enumerate(source_records):
        row = {
            "sample_index": sample_index,
            "source_index": record["_source_index"],
            "id": str(record["id"]),
            "type": str(record.get("type", "")),
            "question": record["question"],
            "context": context_text(record["context"]),
            "cot_answer": record["cot_answer"],
            "query_id": str(record["id"]),
            "model": model_label,
            "query": record["question"],
            "llm_raw_response": record["cot_answer"],
            "stratum_key": str(record.get("type", "")),
            "top_ai_search_results": json.dumps(record["context"], ensure_ascii=False),
            "inference_error": "",
            "oob_inference_error": "",
            "fine_tuned_inference_error": "",
            "gpt_inference_error": "",
            "system_prompt": "",
            "user_prompt": "",
            "oob_slm_response": "",
            "fine_tuned_slm_response": "",
            "gpt_slm_response": "",
            "prompt_token_count": 0,
            "oob_generated_token_count": 0,
            "fine_tuned_generated_token_count": 0,
            "gpt_generated_token_count": 0,
            "oob_output_reached_limit": False,
            "fine_tuned_output_reached_limit": False,
            "gpt_output_reached_limit": False,
            "oob_response_is_valid_json": False,
            "fine_tuned_response_is_valid_json": False,
            "gpt_response_is_valid_json": False,
            "oob_json_error": "",
            "fine_tuned_json_error": "",
            "gpt_json_error": "",
        }
        rows.append(row)

        messages = record["messages"]
        assistant_index = next(
            (
                index
                for index, item in enumerate(messages)
                if item["role"] == "assistant"
            ),
            len(messages),
        )
        prompt_messages = messages[:assistant_index]
        if not prompt_messages:
            row["inference_error"] = (
                f"Record {sample_index} has no messages before the assistant"
            )
            continue

        row["system_prompt"] = "\n\n".join(
            item["content"]
            for item in prompt_messages
            if item["role"] == "system"
        )
        row["user_prompt"] = "\n\n".join(
            item["content"]
            for item in prompt_messages
            if item["role"] == "user"
        )
        prepared_records.append((sample_index, prompt_messages))

    modes = (
        []
        if inference_mode == "none"
        else ["oob", "fine-tuned"]
        if inference_mode == "both"
        else [inference_mode]
    )
    records_over_input_limit = 0
    outputs_reaching_limit = 0

    log.info(
        "Starting prediction generation for %s records in mode=%s with batch_size=%d",
        len(prepared_records),
        inference_mode,
        inference_batch_size,
    )

    for mode in modes:
        if mode == "oob":
            active_model = model
            active_tokenizer = tokenizer
        else:
            active_model = fine_tuned_model
            active_tokenizer = fine_tuned_tokenizer

        if active_model is None or active_tokenizer is None:
            raise ValueError(f"Model for mode {mode!r} is not loaded")

        terminators = [active_tokenizer.eos_token_id]
        eot_token_id = active_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if (
            eot_token_id is not None
            and eot_token_id != active_tokenizer.unk_token_id
            and eot_token_id not in terminators
        ):
            terminators.append(eot_token_id)

        tokenized_records = []
        for sample_index, prompt_messages in prepared_records:
            try:
                prompt = active_tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = active_tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=False,
                )
                prompt_token_count = len(encoded["input_ids"])
                rows[sample_index][f"{mode.replace('-', '_')}_prompt_token_count"] = prompt_token_count
                rows[sample_index]["prompt_token_count"] = max(
                    rows[sample_index]["prompt_token_count"],
                    prompt_token_count,
                )
                if prompt_token_count > input_limit:
                    records_over_input_limit += 1
                    raise ValueError(
                        f"Record {sample_index} mode {mode} prompt has "
                        f"{prompt_token_count} tokens, exceeding the "
                        f"no-truncation input limit of {input_limit}. "
                        f"Context={max_seq_length}, reserved "
                        f"output={max_new_tokens}."
                    )
                tokenized_records.append(
                    (sample_index, prompt_token_count, encoded, prompt)
                )
            except Exception as exc:
                log.exception(
                    "Prompt preparation failed for sample %s mode %s",
                    sample_index,
                    mode,
                )
                existing_error = rows[sample_index]["inference_error"]
                rows[sample_index]["inference_error"] = (
                    f"{existing_error}; {mode}: {exc}"
                    if existing_error
                    else f"{mode}: {exc}"
                )
                rows[sample_index][f"{mode.replace('-', '_')}_inference_error"] = str(exc)

        # Similar-length batches minimize left-padding compute.
        tokenized_records.sort(key=lambda item: item[1])
        if tokenized_records:
            audit_record = max(tokenized_records, key=lambda item: item[1])
            audit_sample_index = audit_record[0]
            audit_prompt_token_count = audit_record[1]
            audit_input_ids = audit_record[2]["input_ids"]
            audit_rendered_prompt = audit_record[3]
            audit_decoded_prompt = active_tokenizer.decode(
                audit_input_ids,
                skip_special_tokens=False,
            )
            assistant_header = (
                "<|start_header_id|>assistant<|end_header_id|>"
            )
            audit_payload = {
                "mode": mode,
                "sample_index": audit_sample_index,
                "prompt_token_count": audit_prompt_token_count,
                "input_token_limit": input_limit,
                "max_sequence_length": max_seq_length,
                "reserved_output_tokens": max_new_tokens,
                "remaining_input_capacity": (
                    input_limit - audit_prompt_token_count
                ),
                "total_context_headroom": (
                    max_seq_length
                    - audit_prompt_token_count
                    - max_new_tokens
                ),
                "truncation_enabled": False,
                "assistant_generation_header_present": (
                    assistant_header in audit_rendered_prompt[-256:]
                ),
                "rendered_prompt": audit_rendered_prompt,
                "decoded_input_ids": audit_decoded_prompt,
            }
            mlflow.log_text(
                json.dumps(
                    audit_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                artifact_file=(
                    f"inference_inputs/{mode}_longest_input_{artifact_attempt}.json"
                ),
            )
            log.info(
                "Logged %s longest accepted input artifact for sample %d "
                "(%d/%d input tokens)",
                mode,
                audit_sample_index,
                audit_prompt_token_count,
                input_limit,
            )

        batches = [
            tokenized_records[start:start + inference_batch_size]
            for start in range(0, len(tokenized_records), inference_batch_size)
        ]
        progress = tqdm(
            batches,
            desc=f"Generating {mode}: {model_label}",
        )

        for batch_records in progress:
            batch_indices = [item[0] for item in batch_records]
            batch_features = [
                {
                    "input_ids": item[2]["input_ids"],
                    "attention_mask": item[2]["attention_mask"],
                }
                for item in batch_records
            ]

            try:
                inputs = active_tokenizer.pad(
                    batch_features,
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                ).to(DEVICE)
                padded_input_length = inputs["input_ids"].shape[1]
                if padded_input_length + max_new_tokens > max_seq_length:
                    raise ValueError(
                        f"Batch padded input length {padded_input_length} plus "
                        f"output budget {max_new_tokens} exceeds context "
                        f"{max_seq_length}"
                    )

                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                with torch.inference_mode():
                    output_ids = active_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=active_tokenizer.pad_token_id,
                        eos_token_id=terminators,
                    )

                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                batch_latency_ms = (time.perf_counter() - start_time) * 1000

                for batch_position, sample_index in enumerate(batch_indices):
                    generated_ids = output_ids[
                        batch_position,
                        padded_input_length:,
                    ]
                    terminator_position = next(
                        (
                            index
                            for index, token_id in enumerate(generated_ids)
                            if int(token_id) in terminators
                        ),
                        None,
                    )
                    if terminator_position is not None:
                        generated_ids = generated_ids[:terminator_position + 1]

                    generated_token_count = len(generated_ids)
                    reached_output_limit = (
                        terminator_position is None
                        and generated_token_count >= max_new_tokens
                    )
                    if reached_output_limit:
                        outputs_reaching_limit += 1
                        log.warning(
                            "Record %s mode %s reached the %s-token output limit",
                            sample_index,
                            mode,
                            max_new_tokens,
                        )

                    response = active_tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    ).strip()
                    if mode == "oob":
                        prefix = "oob"
                        response_column = "oob_slm_response"
                    else:
                        prefix = "fine_tuned"
                        response_column = "fine_tuned_slm_response"

                    rows[sample_index][response_column] = response
                    rows[sample_index][
                        f"{prefix}_generated_token_count"
                    ] = generated_token_count
                    rows[sample_index][
                        f"{prefix}_output_reached_limit"
                    ] = reached_output_limit
                    rows[sample_index][
                        f"{prefix}_slm_latency_ms"
                    ] = batch_latency_ms

                    try:
                        json.loads(response)
                        rows[sample_index][
                            f"{prefix}_response_is_valid_json"
                        ] = True
                    except json.JSONDecodeError as exc:
                        rows[sample_index][f"{prefix}_json_error"] = str(exc)
                        log.warning(
                            "Record %s mode %s returned invalid JSON: %s",
                            sample_index,
                            mode,
                            exc,
                        )
            except Exception as exc:
                log.exception(
                    "Inference failed for samples %s in mode %s",
                    batch_indices,
                    mode,
                )
                for sample_index in batch_indices:
                    existing_error = rows[sample_index]["inference_error"]
                    rows[sample_index]["inference_error"] = (
                        f"{existing_error}; {mode}: {exc}"
                        if existing_error
                        else f"{mode}: {exc}"
                    )
                    rows[sample_index][f"{mode.replace('-', '_')}_inference_error"] = str(exc)

    log.info(
        "Records rejected instead of truncated for exceeding input limit: %s",
        records_over_input_limit,
    )
    log.info(
        "Generated outputs that reached max_new_tokens: %s",
        outputs_reaching_limit,
    )

    if compare_with_gpt:
        run_gpt_inference(
            rows,
            prepared_records,
            az_client=az_client,
            gpt_endpoint=gpt_endpoint,
            gpt_deployment=gpt_deployment,
            gpt_api_version=gpt_api_version,
            gpt_max_workers=gpt_max_workers,
            gpt_max_retries=gpt_max_retries,
            max_new_tokens=max_new_tokens,
        )

    return pd.DataFrame(rows)

def get_token_lengths(tokenizer, data_dir, split):
    lengths = {
        "system": [],
        "user": [],
        "assistant": [],
        "inference_prompt": [],
        "full_chat": [],
        "full_chat_characters": [],
    }
    for record in read_jsonl(data_dir / f"{split}.jsonl"):
        messages = normalize_evaluation_record(record)["messages"]
        for message in messages:
            role = message["role"]
            if role not in lengths:
                continue
            token_ids = tokenizer.encode(
                message["content"],
                add_special_tokens=False,
                truncation=False,
            )
            lengths[role].append(len(token_ids))

        assistant_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message["role"] == "assistant"
            ),
            len(messages),
        )
        prompt_messages = messages[:assistant_index]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_dict=False,
        )

        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
            return_dict=False,
        )

        full_chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            truncation=False,
            return_dict=False,
        )


        lengths["inference_prompt"].append(len(prompt_ids))
        lengths["full_chat"].append(len(full_ids))
        lengths["full_chat_characters"].append(len(full_chat_text))
    return lengths

def get_length_summary(lengths, output_file="length_summary.csv"):
    summary_rows = []
    for role, vals in lengths.items():
        if not vals:
            continue
        s = pd.Series(vals)
        summary_rows.append({
            "role": role,
            "count": len(s),
            "mean": float(s.mean()),
            "p50": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "max": int(s.max()),
        })

    df = pd.DataFrame(summary_rows)
    # save as mlflow artifact
    df.to_csv(output_file, index=False)
    mlflow.log_artifact(output_file)

def upload_to_blobstore(
    output_file,
    run_id,
    az_client,
    storage_account_name,
    container_name="inference-output"
):
    log.info("Uploading %s for run ID %s", output_file, run_id)
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=az_client,
    )
    log.info("Ensuring container %s exists in storage account %s", container_name, storage_account_name)
    from azure.core.exceptions import ResourceExistsError
    try:
        blob_service_client.create_container(container_name)
    except ResourceExistsError:
        pass
    container_client = blob_service_client.get_container_client(container_name)
    log.info("Obtained container client for container %s", container_name)
    blob_path = f"predictions/{run_id}_predictions.csv"
    with open(output_file, "rb") as f:
        log.info("Uploading to %s/%s", container_name, blob_path)
        container_client.upload_blob(name=blob_path,data=f,overwrite=True)
    return f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_path}"

# ==========================================================================
# Main
# ==========================================================================

def main(args: argparse.Namespace):
    log.info("Starting inference job")
    log.info("Using device: %s", DEVICE)
    log.info("Using Azure client ID: %s", args.azure_client_id)
    if DEVICE == "cuda":
        gpu_count = torch.cuda.device_count()
        for gpu_index in range(gpu_count):
            properties = torch.cuda.get_device_properties(gpu_index)
            log.info(
                "GPU %d: %s, memory=%.1f GiB",
                gpu_index,
                properties.name,
                properties.total_memory / (1024 ** 3),
            )
        if gpu_count > 1:
            log.warning(
                "Detected %d GPUs, but this process uses only cuda:0. "
                "Use separate inference workers or distributed execution to "
                "use the full Azure SKU.",
                gpu_count,
            )

    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be greater than zero")
    if args.max_input_tokens <= 0:
        raise ValueError("--max-input-tokens must be greater than zero")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")
    if args.max_input_tokens + args.max_new_tokens > args.max_seq_length:
        raise ValueError(
            "--max-input-tokens + --max-new-tokens must not exceed "
            "--max-seq-length"
        )
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be greater than zero")
    if args.max_records is not None and args.sample_records is not None:
        raise ValueError(
            "Use either --max-records for the first N records or "
            "--sample-records for a random N records, not both"
        )
    gpt_baseline = getattr(args, "gpt_baseline", None)
    llm_baseline_output = getattr(args, "llm_baseline_output", None)
    if args.workflow_mode == "llm_baseline":
        if not args.compare_with_gpt or gpt_baseline:
            raise ValueError(
                "llm_baseline requires GPT generation and cannot consume a baseline"
            )
        if not llm_baseline_output:
            raise ValueError("llm_baseline requires --llm-baseline-output")
    elif args.workflow_mode == "slm_evaluation":
        if args.compare_with_gpt or not gpt_baseline:
            raise ValueError(
                "slm_evaluation requires --gpt-baseline and must not generate GPT"
            )
        if not args.fine_tuned_model:
            raise ValueError("slm_evaluation requires --fine-tuned-model")
    if args.compare_with_gpt and gpt_baseline:
        raise ValueError("Use either --compare-with-gpt or --gpt-baseline, not both")
    if (args.compare_with_gpt or gpt_baseline) and not args.gpt_endpoint:
        raise ValueError(
            "--gpt-endpoint is required for GPT generation or RAG judging"
        )

    # print installed packages
    if args.log_packages:
        import pkg_resources
        installed_packages = pkg_resources.working_set
        log.info("Installed packages:")
        log.info("========================================")
        for package in installed_packages:
            log.info(package)
        log.info("========================================")

    # TODO: Move HF token to Azure Key Vault
    os.environ["HF_TOKEN"] = args.hf_token
    az_client = get_az_client(args.azure_client_id)
    model = tokenizer = fine_tuned_model = fine_tuned_tokenizer = None
    try:
        if args.workflow_mode == "llm_baseline":
            log.info("Skipping SLM model loading in llm_baseline mode")
        else:
            log.info("Downloading model: %s", Path(args.model))
            model, tokenizer, fine_tuned_model, fine_tuned_tokenizer = download_model(
                model_name=Path(args.model),
                fine_tuned_model_path=Path(args.fine_tuned_model),
                mode="both",
                max_seq_length=args.max_seq_length,
                load_in_4bit=args.load_in_4bit,
                dtype=DTYPE,
            )
            log.info("Model loaded successfully: %s", args.model)
            loaded_models = [("oob", model), ("fine-tuned", fine_tuned_model)]
            for loaded_model_name, loaded_model in loaded_models:
                loaded_context = getattr(loaded_model, "max_seq_length", None)
                log.info("%s model max_seq_length=%s", loaded_model_name, loaded_context)
                if loaded_context is not None and loaded_context != args.max_seq_length:
                    raise ValueError(
                        f"{loaded_model_name} model context mismatch: requested "
                        f"{args.max_seq_length}, loaded {loaded_context}"
                    )
        log.info(
            "Configured token budget: context=%d, input=%d, output=%d, "
            "unused_buffer=%d",
            args.max_seq_length,
            args.max_input_tokens,
            args.max_new_tokens,
            (
                args.max_seq_length
                - args.max_input_tokens
                - args.max_new_tokens
            ),
        )
    except Exception:
        log.exception("Failed to resolve or load model")
        raise
    data_path = Path(args.data)
    log.info("Using dataset path: %s", data_path)
    files = {
        "train": data_path / "train.jsonl",
        "test": data_path / "test.jsonl",
        "validation": data_path / "validation.jsonl",
    }
    records = load_data(
        files=files,
        split="test",
        max_records=args.max_records,
        sample_records=args.sample_records,
        sample_seed=args.sample_seed,
    )
    log.info("Loaded %s records for inference", len(records))
    if args.analyze_token_lengths and tokenizer is not None:
        splits = ["train", "test", "validation"]
        for split in splits:
            token_lengths = get_token_lengths(
                tokenizer,
                data_path,
                split=split,
            )
            get_length_summary(
                token_lengths,
                output_file=f"{split}_length_summary.csv",
            )
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        fine_tuned_model=fine_tuned_model,
        fine_tuned_tokenizer=fine_tuned_tokenizer,
        source_records=records,
        model_label=args.model,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        inference_mode=(
            "none" if args.workflow_mode == "llm_baseline" else "both"
        ),
        inference_batch_size=args.inference_batch_size,
        compare_with_gpt=args.compare_with_gpt,
        az_client=az_client,
        gpt_endpoint=args.gpt_endpoint,
        gpt_deployment=args.gpt_deployment,
        gpt_api_version=args.gpt_api_version,
        gpt_max_workers=args.gpt_max_workers,
        gpt_max_retries=args.gpt_max_retries,
    )
    if gpt_baseline:
        predictions, baseline_manifest = apply_llm_baseline(
            predictions, gpt_baseline, data_path / "test.jsonl"
        )
        requested_judge = args.judge_deployment or args.gpt_deployment
        if baseline_manifest["judge_deployment"] != requested_judge:
            raise ValueError(
                "GPT baseline judge deployment does not match this evaluation: "
                f"{baseline_manifest['judge_deployment']!r} != {requested_judge!r}"
            )
        log.info(
            "Reused GPT baseline with %d records generated by deployment=%s",
            baseline_manifest["record_count"],
            baseline_manifest["gpt_deployment"],
        )

    evaluated_modes = (
        [] if args.workflow_mode == "llm_baseline" else ["oob", "fine_tuned"]
    )
    has_gpt_evidence = bool(args.compare_with_gpt or gpt_baseline)
    reported_modes = list(evaluated_modes)
    if has_gpt_evidence:
        reported_modes.append("gpt")
        judge_client = build_azure_openai_client(args.gpt_endpoint, args.gpt_api_version, az_client)
        judge_deployment = args.judge_deployment or args.gpt_deployment
        modes_to_judge = list(evaluated_modes)
        if args.compare_with_gpt:
            modes_to_judge.append("gpt")
        log.info("Evaluating %s responses with RAG judge deployment=%s", modes_to_judge, judge_deployment)

        from deepeval_metrics import build_deepeval_judge

        judge_usage = {"input": 0, "output": 0, "complete": True, "model_ids": set(), "identity_complete": True}

        def completion(prompt):
            result = call_gpt_with_retry(
                judge_client,
                judge_deployment,
                [
                    {"role": "system", "content": "You are an evaluation judge. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=2048,
                max_retries=args.gpt_max_retries,
                response_format={"type": "json_object"},
            )
            for direction, key in (("input", "prompt_tokens"), ("output", "completion_tokens")):
                if result.get(key) is None:
                    judge_usage["complete"] = False
                else:
                    judge_usage[direction] += result[key]
            if isinstance(result.get("model_id"), str) and result["model_id"].strip():
                judge_usage["model_ids"].add(result["model_id"])
            else:
                judge_usage["identity_complete"] = False
            return result["text"]

        raw_judge = build_deepeval_judge(completion, judge_deployment)

        def tracked_judge(**kwargs):
            judge_usage.update(input=0, output=0, complete=True, model_ids=set(), identity_complete=True)
            result = raw_judge(**kwargs)
            result["_usage"] = {
                direction: judge_usage[direction] if judge_usage["complete"] else None
                for direction in ("input", "output")
            }
            result["_usage"]["model_id"] = next(iter(judge_usage["model_ids"])) if judge_usage["identity_complete"] and len(judge_usage["model_ids"]) == 1 else None
            return result

        predictions = pd.DataFrame(evaluate_rag_predictions(
            predictions.to_dict("records"),
            tracked_judge,
            model_prefixes=modes_to_judge,
        ))
        predictions["rag_judge_deployment"] = judge_deployment
        predictions["evaluation_framework"] = "deepeval"
    json_quality_metrics = {}
    for mode_prefix in reported_modes:
        valid_column = f"{mode_prefix}_response_is_valid_json"
        response_column = f"{mode_prefix}_slm_response"
        output_limit_column = f"{mode_prefix}_output_reached_limit"

        total_count = len(predictions)
        valid_count = int(
            predictions[valid_column].fillna(False).astype(bool).sum()
        )
        nonempty_count = int(
            predictions[response_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        output_limit_count = int(
            predictions[output_limit_column]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        valid_rate = (
            100.0 * valid_count / total_count
            if total_count
            else 0.0
        )

        log.info(
            "%s JSON validity: %.2f%% (%d/%d); non-empty responses=%d; "
            "outputs reaching token limit=%d",
            mode_prefix,
            valid_rate,
            valid_count,
            total_count,
            nonempty_count,
            output_limit_count,
        )
        json_quality_metrics[f"{mode_prefix}_json_valid_rate_pct"] = valid_rate
        json_quality_metrics[f"{mode_prefix}_json_valid_count"] = valid_count
        json_quality_metrics[f"{mode_prefix}_response_count"] = total_count
        json_quality_metrics[
            f"{mode_prefix}_nonempty_response_count"
        ] = nonempty_count
        json_quality_metrics[
            f"{mode_prefix}_output_limit_count"
        ] = output_limit_count

    run_id = os.environ.get("AZUREML_RUN_ID")
    if run_id is None:
            log.warning("AZUREML_RUN_ID environment variable is not set")
            run_id = time.strftime("%Y%m%d-%H%M%S")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_id}_predictions.csv"
    predictions.to_csv(output_file, index=False)
    log.info("Saved %s predictions to %s", len(predictions), output_file)
    log.info(f"Current Run ID: {run_id}")

    # this is a workaround since I do not have access to datastores attached to AML workspace
    # Ideal approach is to upload to the Azure ML datastore associated with the workspace
    mlflow.log_artifact(str(output_file))
    evidence_file = write_evaluation_manifest(
        output_file, data_path / "test.jsonl",
        getattr(args, "finetuning_job_name", None), run_id,
    )
    mlflow.log_artifact(str(evidence_file))
    path = upload_to_blobstore(output_file, run_id, az_client, storage_account_name=args.storage_account_name)

    log.info(f"File uploaded to {path}")

    logged_parameters = vars(args).copy()

    # Do not log the actual Hugging Face token for security reasons
    logged_parameters["hf_token"] = ""

    mlflow.log_params(logged_parameters)
    mlflow.log_param("run_id", run_id)
    mlflow.log_metrics(json_quality_metrics)
    if has_gpt_evidence:
        error_counts = {
            f"{prefix}_rag_error_count": int(predictions[f"{prefix}_rag_error"].ne("").sum())
            for prefix in reported_modes
        }
        mlflow.log_metrics(error_counts)
        if any(error_counts.values()):
            raise RuntimeError(
                f"Incomplete RAG evaluation: {error_counts}. "
                f"Responses, scores, reasons, and errors were saved to {output_file}."
            )
        if args.compare_with_gpt and llm_baseline_output:
            baseline_file, baseline_manifest = write_llm_baseline(
                predictions,
                llm_baseline_output,
                data_path / "test.jsonl",
                gpt_deployment=args.gpt_deployment,
                judge_deployment=args.judge_deployment or args.gpt_deployment,
                gpt_api_version=args.gpt_api_version,
                max_new_tokens=args.max_new_tokens,
            )
            log.info(
                "Saved reusable GPT baseline to %s with manifest %s",
                baseline_file,
                baseline_manifest,
            )
        mlflow.log_metrics({
            f"{prefix}_{metric}_avg": float(predictions[f"{prefix}_{metric}_score"].mean())
            for prefix in reported_modes for metric in RAG_METRICS
        })
    for latency_column in [
        "oob_slm_latency_ms",
        "fine_tuned_slm_latency_ms",
        "gpt_slm_latency_ms",
    ]:
        if latency_column not in predictions:
            continue
        latency_values = predictions[latency_column].dropna()
        if latency_values.empty:
            continue
        metric_prefix = latency_column.removesuffix("_slm_latency_ms")
        mlflow.log_metric(
            f"{metric_prefix}_latency_avg_ms",
            float(latency_values.mean()),
        )
        mlflow.log_metric(
            f"{metric_prefix}_latency_p95_ms",
            float(latency_values.quantile(0.95)),
        )

if __name__ == "__main__":
    main(parse_args())
