"""Submit an Azure ML fine-tuning job and record its immutable lineage.

Registers a pinned CUDA training environment, submits a managed-identity SLM
fine-tuning command job (``lib/train.py``, Unsloth-based LoRA adaptation with
prompt-token loss masking), optionally waits for completion, and records the
training run for Stage 03 evaluation. Registration and deployment intentionally
belong to ``scripts/04_release_candidate.py`` after candidate-bound evaluation
evidence and explicit approval.

This is the control plane; ``lib/train.py`` is the remote execution unit that
actually runs on the GPU compute cluster.

**Mandatory production controls**
- Use a dedicated GPU compute cluster with managed identity and minimum nodes
  set to zero.
- Pin the curated environment and data/model asset versions.
- Keep the test split sealed until Stage 03.
- Do not register artifacts from this training stage. Stage 04 owns the release
  policy, human approval, registration, and zero-traffic deployment.

Examples:
    # Submit a new fine-tuning job and exit immediately (job keeps running).
    python scripts/02_azureml_fine_tuning.py

    # Submit and block until the job finishes.
    python scripts/02_azureml_fine_tuning.py --wait
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from azure.identity import CredentialUnavailableError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.azureml_ops import (
    create_training_environment,
    get_training_environment_version,
    submit_finetuning_job,
)
from lib.config import AzureMLConfig
from lib.run_registry import record_run

log = logging.getLogger("azureml_fine_tuning")


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def get_job_after_stream(ml_client, job_name: str, attempts: int = 3, delay: float = 5.0):
    for attempt in range(1, attempts + 1):
        try:
            return ml_client.jobs.get(job_name)
        except CredentialUnavailableError:
            if attempt == attempts:
                raise
            log.warning(
                "Azure credential refresh failed after log streaming; retrying status lookup (%d/%d)",
                attempt,
                attempts,
            )
            time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Run configuration -- same defaults as the former notebook constants.
    parser.add_argument("--data-asset-name", default="slm-studio-instance-security")
    parser.add_argument("--data-asset-version", default="20260805.042350")
    parser.add_argument(
        "--base-model-source",
        choices=["foundry", "huggingface"],
        default="foundry",
        help="'foundry' needs no HF token; 'huggingface' needs a Key Vault token.",
    )
    parser.add_argument("--base-model-name", default="Llama-3.2-3B-Instruct")
    parser.add_argument("--base-model-registry", default="azureml-meta")
    parser.add_argument("--base-model-label", default="latest")
    parser.add_argument("--hf-base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--compute-name",
        default=os.environ.get("AZURE_GPU_CLUSTER_NAME", "gpu-cluster"),
    )
    parser.add_argument(
        "--managed-identity-client-id",
        default="39793817-16f4-45f9-8181-b8e198144c49",
    )
    parser.add_argument("--train-environment-name", default="slm-studio-finetuning")
    parser.add_argument(
        "--train-environment-version", default=get_training_environment_version(),
        help="Overrides AZURE_TRAIN_ENVIRONMENT_VERSION in .env; successful version bumps are persisted there.",
    )
    parser.add_argument("--experiment-name", default="slm-studio-finetuning")
    # Optional train.py overrides (forwarded only if explicitly set).
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--chat-template", default=None)
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-adapter-name", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Phase control.
    parser.add_argument(
        "--skip-environment", action="store_true", help="Reuse an already-registered environment"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block and stream logs until the submitted job reaches a terminal state.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def register_environment(ml_client, args: argparse.Namespace):
    log.info(
        "Registering training environment %s:%s",
        args.train_environment_name,
        args.train_environment_version,
    )
    training_environment = create_training_environment(
        ml_client, args.train_environment_name, args.train_environment_version
    )
    args.train_environment_version = str(training_environment.version)
    log.info("Environment: %s", training_environment.id)
    return training_environment


def submit_training(ml_client, args: argparse.Namespace, environment_id: str):
    from azure.identity import DefaultAzureCredential

    if args.base_model_source == "foundry":
        base_model_arg = args.base_model_name
        key_vault_name = None
        hf_token_secret_name = None
    else:
        base_model_arg = args.hf_base_model
        key_vault_name = os.environ.get("AZURE_KEY_VAULT_NAME")
        hf_token_secret_name = os.environ.get("AZURE_KEY_VAULT_HF_TOKEN_SECRET_NAME")
        if not (key_vault_name and hf_token_secret_name):
            raise ValueError(
                "AZURE_KEY_VAULT_NAME and AZURE_KEY_VAULT_HF_TOKEN_SECRET_NAME must be set "
                "in the environment when --base-model-source=huggingface"
            )

    # Reuse a pre-warmed credential to avoid interactive-browser fallback stalls inside the helper.
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    log.info("Resolving base model...")

    submitted_job = submit_finetuning_job(
        ml_client=ml_client,
        data_asset=f"azureml:{args.data_asset_name}:{args.data_asset_version}",
        environment=environment_id,
        compute=args.compute_name,
        experiment_name=args.experiment_name,
        base_model=base_model_arg,
        display_name=f"slm-studio-sft-{datetime.now(timezone.utc).strftime('%Y%m%d.%H%M%S')}",
        base_model_source=args.base_model_source,
        base_model_registry=args.base_model_registry,
        base_model_label=args.base_model_label,
        key_vault_name=key_vault_name,
        hf_token_secret_name=hf_token_secret_name,
        managed_identity_client_id=args.managed_identity_client_id,
        credential=credential,
        max_seq_length=args.max_seq_length,
        chat_template=args.chat_template,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_adapter_name=args.lora_adapter_name,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        seed=args.seed,
    )
    log.info("Job name: %s", submitted_job.name)

    run_path = record_run(
        submitted_job.name,
        experiment_name=args.experiment_name,
        display_name=submitted_job.display_name,
        base_model=base_model_arg,
        base_model_source=args.base_model_source,
        compute_name=args.compute_name,
        data_asset=f"{args.data_asset_name}:{args.data_asset_version}",
        status="submitted",
    )
    log.info("Recorded run %s", run_path)
    return submitted_job


def main() -> int:
    configure_console_encoding()
    # Load .env before parse_args() so argparse defaults sourced from the
    # environment (e.g. --compute-name) see .env's values, not just whatever
    # the shell already exported.
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("azure").setLevel(logging.WARNING)

    config = AzureMLConfig.from_env()
    ml_client = config.create_ml_client()
    data_asset = ml_client.data.get(args.data_asset_name, version=args.data_asset_version)
    log.info("Data input: %s", data_asset.id)
    log.info("Compute: %s", ml_client.compute.get(args.compute_name).name)

    training_environment = None
    if not args.skip_environment:
        training_environment = register_environment(ml_client, args)
    else:
        log.info(
            "Reusing registered training environment %s:%s (--skip-environment)",
            args.train_environment_name, args.train_environment_version,
        )
        training_environment = ml_client.environments.get(
            args.train_environment_name, version=args.train_environment_version,
        )

    submitted_job = submit_training(ml_client, args, training_environment.id)
    if args.wait:
        log.info("Waiting for job %s to complete...", submitted_job.name)
        ml_client.jobs.stream(submitted_job.name)
        completed_job = get_job_after_stream(ml_client, submitted_job.name)
        if completed_job.status != "Completed":
            raise RuntimeError(f"Training did not complete successfully: {completed_job.status}")
        record_run(submitted_job.name, status="completed")
        log.info("Training completed. Run Stage 03 evaluation for job %s.", submitted_job.name)
    else:
        log.info(
            "Training submitted asynchronously. Stage 03 resolves this recorded run by default."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
