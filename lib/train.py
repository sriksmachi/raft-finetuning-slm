"""Azure ML command-job entry point for SLM Studio fine-tuning."""

from __future__ import annotations

import argparse
import logging
import math
import statistics
from importlib.metadata import version
from pathlib import Path

import mlflow
import torch
import unsloth  # noqa: F401 - must preload before importing Transformers
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from lib.data import validate_records
from lib.lora_artifacts import (
    DEFAULT_ADAPTER_NAME,
    save_lora_adapter,
    validate_adapter_name,
)
from lib.prompts import SYSTEM_PROMPT, user_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
for logger_name in ("httpx", "azure"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# bf16 is preferred when the GPU supports it (e.g. A100/H100); fp16 otherwise.
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def training_profile(args: argparse.Namespace, bf16_supported: bool) -> dict[str, int | str]:
    tokens_per_microbatch = args.batch_size * args.max_seq_length
    return {
        "precision": "bf16" if bf16_supported else "fp16",
        "sequences_per_microbatch": args.batch_size,
        "tokens_per_microbatch_upper_bound": tokens_per_microbatch,
        "microbatches_per_optimizer_step": args.gradient_accumulation,
        "sequences_per_optimizer_step": args.batch_size * args.gradient_accumulation,
        "tokens_per_optimizer_step_upper_bound": (
            tokens_per_microbatch * args.gradient_accumulation
        ),
    }


def mlflow_parameters(
    args: argparse.Namespace, profile: dict[str, int | str]
) -> dict[str, object]:
    parameters = vars(args).copy()
    for name in ("base_model", "data", "hf_token_secret_name", "model_output"):
        parameters.pop(name, None)
    parameters.update({f"profile_{key}": value for key, value in profile.items()})
    return parameters


def log_mlflow_parameters(parameters: dict[str, object], run_id: str) -> None:
    existing = mlflow.get_run(run_id).data.params
    pending = {key: value for key, value in parameters.items() if key not in existing}
    conflicts = {
        key: (existing[key], str(value))
        for key, value in parameters.items()
        if key in existing and existing[key] != str(value)
    }
    if conflicts:
        log.warning(
            "Preserving %d existing MLflow parameter value(s) during job retry: %s",
            len(conflicts),
            sorted(conflicts),
        )
    if pending:
        mlflow.log_params(pending)


class ResponseOnlyDataCollator:
    """Pad causal-LM inputs and preserve response-only labels.

    The stock ``DataCollatorForSeq2Seq`` is unsafe to use blindly with
    hand-rolled ``-100``-masked labels: depending on how it is configured it
    can fail to batch variable-length label sequences correctly. This
    collator pads ``input_ids``/``attention_mask`` via the tokenizer as
    normal, then pads ``labels`` separately with ``-100`` so padding never
    contributes to the loss, matching the tokenizer's configured
    ``padding_side``.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = [list(feature["labels"]) for feature in features]
        model_input_names = set(self.tokenizer.model_input_names)
        model_features = [
            {key: value for key, value in feature.items() if key in model_input_names}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        padded_length = batch["input_ids"].shape[1]

        padded_labels = []
        for label in labels:
            padding_length = padded_length - len(label)
            if padding_length < 0:
                raise ValueError(
                    "A label sequence is longer than its padded input sequence"
                )
            if self.tokenizer.padding_side == "left":
                padded_label = [-100] * padding_length + label
            else:
                padded_label = label + [-100] * padding_length
            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--chat-template", default="llama-3.2")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model in 4-bit precision (preserves prior BitsAndBytesConfig behavior).",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--lora-adapter-name", type=validate_adapter_name, default=DEFAULT_ADAPTER_NAME,
        help="Name of the final adapter directory inside --model-output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--key-vault-name")
    parser.add_argument("--hf-token-secret-name")
    parser.add_argument("--managed-identity-client-id")
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    log.info(
        "Dependency versions: torch=%s, transformers=%s, tokenizers=%s, "
        "unsloth=%s, unsloth_zoo=%s",
        version("torch"),
        version("transformers"),
        version("tokenizers"),
        version("unsloth"),
        version("unsloth_zoo"),
    )
    log.info(
        "Starting SLM Studio fine-tuning job: base_model=%s, data=%s, model_output=%s, "
        "epochs=%s, max_seq_length=%s, lora_rank=%s",
        args.base_model,
        args.data,
        args.model_output,
        args.epochs,
        args.max_seq_length,
        args.lora_rank,
    )
    profile = training_profile(args, torch.cuda.is_bf16_supported())
    log.info("Hardware-aware training profile: %s", profile)
    if profile["precision"] == "fp16":
        log.warning(
            "bf16 is unavailable on this GPU; using fp16. This is expected for T4-class GPUs."
        )
    set_seed(args.seed)
    if bool(args.key_vault_name) != bool(args.hf_token_secret_name):
        raise ValueError("Provide both Key Vault name and Hugging Face secret name")
    if args.key_vault_name:
        from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
        from azure.identity import ManagedIdentityCredential
        from azure.keyvault.secrets import SecretClient
        from huggingface_hub import login

        secret_client = SecretClient(
            vault_url=f"https://{args.key_vault_name}.vault.azure.net",
            credential=ManagedIdentityCredential(
                client_id=args.managed_identity_client_id
            ),
        )
        try:
            hf_token = secret_client.get_secret(args.hf_token_secret_name).value
            if len(hf_token) > 0:
                log.info("Successfully retrieved Hugging Face token from Key Vault")
        except ClientAuthenticationError as exc:
            log.exception(
                "Managed identity authentication to Key Vault %s failed",
                args.key_vault_name,
            )
            raise RuntimeError(
                "Azure ML compute managed identity is unavailable. Enable a managed "
                "identity on the compute target before submitting this job."
            ) from exc
        except HttpResponseError as exc:
            log.exception(
                "Key Vault secret retrieval for %s failed", args.hf_token_secret_name
            )
            raise RuntimeError(
                "Azure ML compute managed identity could not read the Hugging Face "
                "token. Grant it Key Vault Secrets User access to the configured vault."
            ) from exc
        login(token=hf_token)

    data_path = Path(args.data)
    files = {
        "train": str(data_path / "train.jsonl"),
        "validation": str(data_path / "validation.jsonl"),
    }
    dataset = load_dataset("json", data_files=files)
    log.info(
        "Loaded dataset from %s: train=%d, validation=%d",
        data_path,
        len(dataset["train"]),
        len(dataset["validation"]),
    )

    for split in ("train", "validation"):
        summary = validate_records(dataset[split], split)
        log.info("%s structural validation: %s", split, summary)

    missing_answer_tag = {
        split: [
            index
            for index, record in enumerate(dataset[split])
            if "<ANSWER>" not in record["cot_answer"]
        ]
        for split in ("train", "validation")
    }

    if len(missing_answer_tag["train"]) < 5 or len(missing_answer_tag["validation"]) < 5:
        log.warning(
            "Less than 5 records are missing the <ANSWER> tag in either the train or validation split"
        )
    else:
        for split, indices in missing_answer_tag.items():
            raise ValueError(
                f"{split} has {len(indices)} record(s) whose cot_answer is missing "
                f"the required <ANSWER> tag, e.g. indices {indices[:10]}. Fix the "
                "source data before training on it."
            )

    base_model_path = resolve_base_model(args.base_model)
    log.info("Loading base model: %s", base_model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=args.max_seq_length,
        dtype=DTYPE,
        load_in_4bit=args.load_in_4bit,
    )
    log.info(
        "Model max_seq_length=%s, requested=%s",
        getattr(model, "max_seq_length", None),
        args.max_seq_length,
    )

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=True,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    log.info("Attached LoRA adapters: rank=%d, alpha=%d", args.lora_rank, 2 * args.lora_rank)

    def tokenize_record(record: dict) -> dict:
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(record["instruction"])},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": record["cot_answer"]}
        ]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        encoded = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=args.max_seq_length,
        )

        prompt_length = min(len(prompt_ids), len(encoded))
        # Defensive check: the prompt/response boundary below assumes the
        # chat template renders an identical prefix whether a real assistant
        # message follows the generation header or not. That assumption is
        # verified here, rather than trusted silently -- a mismatch would
        # otherwise mask the wrong tokens with no visible error.
        if encoded[:prompt_length] != prompt_ids[:prompt_length]:
            raise ValueError(
                "Prompt/response tokenization boundary mismatch: the first "
                f"{prompt_length} tokens of the full conversation do not match "
                "the prompt-only tokenization for this record."
            )

        labels = list(encoded)
        labels[:prompt_length] = [-100] * prompt_length
        supervised_token_count = sum(1 for label in labels if label != -100)
        return {
            "input_ids": encoded,
            "attention_mask": [1] * len(encoded),
            "labels": labels,
            "supervised_token_count": supervised_token_count,
            "prompt_token_count": len(prompt_ids),
        }

    tokenized = dataset.map(
        tokenize_record,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing training records",
    )

    audit_summary = {}
    for split in ("train", "validation"):
        split_data = tokenized[split]
        supervised_counts = split_data["supervised_token_count"]
        prompt_counts = split_data["prompt_token_count"]
        zero_supervised = [i for i, count in enumerate(supervised_counts) if count == 0]
        truncated_prompts = [
            i for i, count in enumerate(prompt_counts) if count >= args.max_seq_length
        ]
        audit_summary[split] = {
            "records": len(split_data),
            "zero_supervised_tokens": len(zero_supervised),
            "prompt_at_or_above_max_seq_length": len(truncated_prompts),
            "supervised_tokens_min": min(supervised_counts),
            "supervised_tokens_median": statistics.median(supervised_counts),
            "supervised_tokens_max": max(supervised_counts),
        }
        log.info("%s tokenization audit: %s", split, audit_summary[split])
        if zero_supervised:
            log.error(
                "%s has %d record(s) with zero supervised tokens after truncation "
                "to max_seq_length=%d, e.g. indices %s",
                split,
                len(zero_supervised),
                args.max_seq_length,
                zero_supervised[:10],
            )
            raise ValueError(
                f"{split} has {len(zero_supervised)} record(s) with zero supervised "
                f"tokens after truncation to max_seq_length={args.max_seq_length}, "
                f"e.g. indices {zero_supervised[:10]}. Increase --max-seq-length or "
                "shorten the retrieved-document context for these records."
            )

    tokenized = tokenized.remove_columns(["supervised_token_count", "prompt_token_count"])

    output_dir = Path(args.model_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Model output directory: %s", output_dir)

    training_args = TrainingArguments(
        output_dir="training_checkpoints",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        logging_steps=5,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        optim="paged_adamw_8bit",
        report_to=[],
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )

    class MlflowMetricsCallback(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_kwargs):
            metrics = {}
            for name, value in (logs or {}).items():
                try:
                    metrics[name] = float(value)
                except (TypeError, ValueError):
                    continue
            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)

    with mlflow.start_run() as run:
        log.info("Started MLflow run: %s", run.info.run_id)
        log_mlflow_parameters(mlflow_parameters(args, profile), run.info.run_id)
        mlflow.set_tags(
            {
                "task": "context-grounded-generation",
                "framework": "SLM Studio",
                "base_model": args.base_model,
            }
        )
        mlflow.log_metric("train_samples_count", len(dataset["train"]))
        mlflow.log_metric("validation_samples_count", len(dataset["validation"]))
        for split, summary in audit_summary.items():
            mlflow.log_dict(summary, f"{split}_tokenization_audit.json")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=ResponseOnlyDataCollator(tokenizer=tokenizer),
            callbacks=[MlflowMetricsCallback()],
        )
        log.info("Starting training")
        train_result = trainer.train()
        log.info("Training complete: %s", train_result.metrics)

        metrics = trainer.evaluate()
        log.info("Evaluation metrics: %s", metrics)
        if "eval_loss" in metrics:
            validation_perplexity = math.exp(min(metrics["eval_loss"], 20))
            mlflow.log_metric("validation_perplexity", validation_perplexity)
            log.info("Validation perplexity: %s", validation_perplexity)

        adapter_dir = save_lora_adapter(
            trainer.model, tokenizer, output_dir, args.lora_adapter_name
        )
        log.info("Saved LoRA adapter '%s' and tokenizer to %s", args.lora_adapter_name, adapter_dir)
        (output_dir / "run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        log.info(
            "SLM Studio fine-tuning job complete: run_id=%s, output_dir=%s",
            run.info.run_id,
            output_dir,
        )


if __name__ == "__main__":
    main(parse_args())
