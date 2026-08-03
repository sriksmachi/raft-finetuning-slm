"""Azure ML command-job entry point for RAFT QLoRA fine-tuning."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from lib.prompts import SYSTEM_PROMPT, user_prompt


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
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--key-vault-name")
    parser.add_argument("--hf-token-secret-name")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if bool(args.key_vault_name) != bool(args.hf_token_secret_name):
        raise ValueError("Provide both Key Vault name and Hugging Face secret name")
    if args.key_vault_name:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
        from huggingface_hub import login

        secret_client = SecretClient(
            vault_url=f"https://{args.key_vault_name}.vault.azure.net",
            credential=DefaultAzureCredential(),
        )
        login(token=secret_client.get_secret(args.hf_token_secret_name).value)

    data_path = Path(args.data)
    files = {
        "train": str(data_path / "train.jsonl"),
        "validation": str(data_path / "validation.jsonl"),
    }
    dataset = load_dataset("json", data_files=files)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
    )

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
        labels = list(encoded)
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {"input_ids": encoded, "attention_mask": [1] * len(encoded), "labels": labels}

    tokenized = dataset.map(
        tokenize_record,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing RAFT records",
    )
    output_dir = Path(args.model_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
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
        report_to=["mlflow"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )
    mlflow.transformers.autolog(log_models=False)
    with mlflow.start_run() as run:
        logged_parameters = vars(args).copy()
        logged_parameters.pop("hf_token_secret_name", None)
        mlflow.log_params(logged_parameters)
        mlflow.set_tags(
            {
                "task": "context-grounded-generation",
                "framework": "RAFT",
                "base_model": args.base_model,
            }
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            ),
        )
        trainer.train()
        metrics = trainer.evaluate()
        if "eval_loss" in metrics:
            mlflow.log_metric("validation_perplexity", math.exp(min(metrics["eval_loss"], 20)))

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "run_id.txt").write_text(run.info.run_id, encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
