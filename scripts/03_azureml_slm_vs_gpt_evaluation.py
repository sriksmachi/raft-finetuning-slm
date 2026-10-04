"""Submit an SLM-vs-GPT inference/evaluation Azure ML command job.

Runs ``lib/inference.py`` as a managed-identity command job: loads
the fine-tuned model from a ``scripts/02_azureml_fine_tuning.py`` training job's
raw ``outputs/model`` artifact (mounted directly, *not* a registered model --
registration happens in Stage 04 only after this evaluation's gate passes, so
requiring registration here would be circular), generates responses for
the sealed test split, additionally generates responses from an Azure
OpenAI / Azure AI Foundry GPT deployment for the same records, computes
DeepEval faithfulness/relevancy/correctness scores and reasons for OOB, fine-tuned,
and GPT responses against cot_answer, saves the final CSV, and logs means to
the job's MLflow run -- which is how they become visible as Azure ML job metrics.
The output also contains a hash-bound evidence manifest consumed by
``scripts/04_release_candidate.py``.

``--finetuning-job-name`` is optional: every training job submitted by
``scripts/02_azureml_fine_tuning.py`` is recorded as a YAML file under
``runs/`` (see ``lib/run_registry.py``), and if the flag is omitted this
script evaluates the most recently submitted run for
``--training-experiment-name`` instead.

Typical sequence:
    1. python scripts/02_azureml_fine_tuning.py --wait   # trains, records the run under runs/
    2. python scripts/03_azureml_slm_vs_gpt_evaluation.py --gpt-endpoint <endpoint> --wait  # evaluates the latest run
    3. python scripts/04_release_candidate.py ...  # attest, approve, register, deploy at zero traffic

Examples:
    # Evaluate the most recently submitted training run.
    python scripts/03_azureml_slm_vs_gpt_evaluation.py \\
        --gpt-endpoint https://<resource>.openai.azure.com

    # Evaluate a specific training job and block until the job finishes.
    python scripts/03_azureml_slm_vs_gpt_evaluation.py \\
        --finetuning-job-name <job name> --gpt-endpoint <endpoint> --wait
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.azureml_ops import (
    create_inference_environment,
    submit_inference_evaluation_job,
)
from lib.config import AzureMLConfig
from lib.run_registry import RUNS_DIR, latest_run, record_run

log = logging.getLogger("azureml_slm_vs_gpt_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--finetuning-job-name",
        default=None,
        help="Job name of the scripts/02_azureml_fine_tuning.py training run "
        "(lib/train.py) whose outputs/model artifact will be mounted and scored. "
        f"Defaults to the most recently submitted run recorded under {RUNS_DIR.name}/ "
        "(see --training-experiment-name and lib/run_registry.py).",
    )
    parser.add_argument(
        "--training-experiment-name",
        default="slm-studio-finetuning",
        help="Experiment name used to pick the latest recorded run when "
        "--finetuning-job-name is not given; matches "
        "scripts/02_azureml_fine_tuning.py's --experiment-name.",
    )
    parser.add_argument("--test-data-asset-name", default="slm-studio-instance-security")
    parser.add_argument("--test-data-asset-version", default="20260805.042350")

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
    parser.add_argument("--storage-account-name", required=True)

    parser.add_argument("--inference-environment-name", default="slm-studio-inference")
    parser.add_argument("--inference-environment-version", default="2")
    parser.add_argument("--experiment-name", default="slm-studio-slm-vs-gpt-eval")
    parser.add_argument(
        "--display-name",
        default=f"slm-vs-gpt-eval-{datetime.now(timezone.utc).strftime('%Y%m%d.%H%M%S')}",
    )

    # GPT comparison.
    parser.add_argument(
        "--gpt-endpoint",
        required=True,
        help="Azure OpenAI / Azure AI Foundry endpoint to compare the fine-tuned SLM against.",
    )
    parser.add_argument("--gpt-deployment", default="gpt-4.1")
    parser.add_argument(
        "--judge-deployment", default=None,
        help="Separate RAG judge deployment on --gpt-endpoint; defaults to --gpt-deployment.",
    )
    parser.add_argument("--gpt-api-version", default="2024-10-21")
    parser.add_argument("--gpt-max-workers", type=int, default=16)
    parser.add_argument("--gpt-max-retries", type=int, default=4)

    # Optional inference.py overrides (forwarded only if explicitly set).
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--sample-records", type=int, default=None)

    parser.add_argument(
        "--skip-environment", action="store_true", help="Reuse an already-registered environment"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block and stream logs until the job reaches a terminal state, then "
        "print the metrics logged to its MLflow run.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_hf_token(args: argparse.Namespace) -> str:
    if args.base_model_source != "huggingface":
        return ""
    key_vault_name = os.environ.get("AZURE_KEY_VAULT_NAME")
    hf_token_secret_name = os.environ.get("AZURE_KEY_VAULT_HF_TOKEN_SECRET_NAME")
    if not (key_vault_name and hf_token_secret_name):
        raise ValueError(
            "AZURE_KEY_VAULT_NAME and AZURE_KEY_VAULT_HF_TOKEN_SECRET_NAME must be set "
            "in the environment when --base-model-source=huggingface"
        )
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    secret_client = SecretClient(
        vault_url=f"https://{key_vault_name}.vault.azure.net", credential=credential
    )
    return secret_client.get_secret(hf_token_secret_name).value


def print_job_metrics(config: AzureMLConfig, ml_client, job_name: str) -> None:
    import mlflow

    workspace = ml_client.workspaces.get(config.workspace_name)
    mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)
    run = mlflow.get_run(job_name)
    if not run.data.metrics:
        log.warning("No MLflow run found yet for job %s", job_name)
        return
    metrics = "\n".join(
        f"{name}: {value}" for name, value in sorted(run.data.metrics.items())
    )
    log.info(
        "Metrics logged for job %s (OOB, fine-tuned SLM, GPT: RAG scores and latency):\n%s",
        job_name,
        metrics,
    )


def main() -> int:
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
    credential = config.create_credential()
    ml_client = config.create_ml_client(credential=credential)
    test_data_asset = ml_client.data.get(
        args.test_data_asset_name, version=args.test_data_asset_version
    )
    log.info("Test data input: %s", test_data_asset.id)
    log.info("Compute: %s", ml_client.compute.get(args.compute_name).name)

    if not args.skip_environment:
        inference_environment = create_inference_environment(
            ml_client, args.inference_environment_name, args.inference_environment_version
        )
        log.info("Environment: %s", inference_environment.id)
        environment_id = inference_environment.id
    else:
        log.info("Skipping environment registration (--skip-environment)")
        environment_id = (
            f"azureml:{args.inference_environment_name}:{args.inference_environment_version}"
        )

    base_model_arg = (
        args.base_model_name if args.base_model_source == "foundry" else args.hf_base_model
    )
    hf_token = resolve_hf_token(args)

    finetuning_job_name = args.finetuning_job_name
    if not finetuning_job_name:
        record = latest_run(experiment_name=args.training_experiment_name)
        if not record:
            raise RuntimeError(
                "No --finetuning-job-name given and no matching run recorded under "
                f"{RUNS_DIR}. Submit a training job via scripts/02_azureml_fine_tuning.py "
                "first, or pass --finetuning-job-name explicitly."
            )
        finetuning_job_name = record["job_name"]
        log.info(
            "Using latest recorded training job %s (submitted_at=%s) from %s",
            finetuning_job_name,
            record.get("submitted_at"),
            RUNS_DIR,
        )

    submitted_job = submit_inference_evaluation_job(
        ml_client=ml_client,
        test_data_asset=f"azureml:{args.test_data_asset_name}:{args.test_data_asset_version}",
        environment=environment_id,
        compute=args.compute_name,
        experiment_name=args.experiment_name,
        base_model=base_model_arg,
        display_name=args.display_name,
        finetuning_job_name=finetuning_job_name,
        storage_account_name=args.storage_account_name,
        managed_identity_client_id=args.managed_identity_client_id,
        base_model_source=args.base_model_source,
        base_model_registry=args.base_model_registry,
        base_model_label=args.base_model_label,
        mode="both",
        compare_with_gpt=True,
        gpt_endpoint=args.gpt_endpoint,
        gpt_deployment=args.gpt_deployment,
        judge_deployment=args.judge_deployment,
        gpt_api_version=args.gpt_api_version,
        gpt_max_workers=args.gpt_max_workers,
        gpt_max_retries=args.gpt_max_retries,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        max_input_tokens=args.max_input_tokens,
        inference_batch_size=args.inference_batch_size,
        load_in_4bit=args.load_in_4bit,
        max_records=args.max_records,
        sample_records=args.sample_records,
        hf_token=hf_token,
        credential=credential,
    )
    log.info("Job name: %s", submitted_job.name)
    record_run(
        finetuning_job_name,
        evaluation_job_name=submitted_job.name,
        evaluation_environment=environment_id,
        evaluation_submitted_at=datetime.now(timezone.utc).isoformat(),
    )

    if args.wait:
        log.info("Waiting for job %s to complete...", submitted_job.name)
        ml_client.jobs.stream(submitted_job.name)
        print_job_metrics(config, ml_client, submitted_job.name)
    else:
        log.info(
            "Job submitted asynchronously. Per-model RAG scores/reasons will be saved "
            "in the final predictions CSV; RAG means and latency for all three models "
            "will appear in Azure ML / MLflow experiment '%s' once it completes.",
            args.experiment_name,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
