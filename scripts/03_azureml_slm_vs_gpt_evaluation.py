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

The first run for a test dataset registers GPT responses, diagnostics, latency,
and scores as a timestamped ``llm_baseline_dataset`` version. Pass that version
through ``--llm-baseline-data-asset-version`` on later runs to validate and
reuse the GPT evidence without generating or judging the baseline again.

``--finetuning-job-name`` is optional: every training job submitted by
``scripts/02_azureml_fine_tuning.py`` is recorded as a YAML file under
``runs/`` (see ``lib/run_registry.py``), and if the flag is omitted this
script evaluates the most recently submitted run for
``--training-experiment-name`` instead.

Typical sequence:
    1. python scripts/03_azureml_slm_vs_gpt_evaluation.py --mode llm_baseline --wait
    2. python scripts/02_azureml_fine_tuning.py --wait
    3. python scripts/03_azureml_slm_vs_gpt_evaluation.py --mode slm_evaluation \
           --llm-baseline-data-asset-version <version> --wait
    4. python scripts/04_release_candidate.py ...

Examples:
    # Generate the immutable GPT baseline once.
    python scripts/03_azureml_slm_vs_gpt_evaluation.py \\
        --mode llm_baseline --wait

    # Evaluate a specific training job against the cached baseline.
    python scripts/03_azureml_slm_vs_gpt_evaluation.py \\
        --mode slm_evaluation --finetuning-job-name <job name> \\
        --llm-baseline-data-asset-version <version> --wait
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
        "--mode",
        choices=["llm_baseline", "slm_evaluation"],
        required=True,
        help=(
            "llm_baseline generates and registers GPT answers/scores only; "
            "slm_evaluation runs OOB and fine-tuned SLM inference, merges an "
            "existing baseline version, and writes final evaluation evidence."
        ),
    )

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
    parser.add_argument(
        "--storage-account-name",
        default=None,
        help=(
            "Storage account for the compatibility predictions upload. Defaults "
            "to the account backing the Azure ML workspace's default datastore."
        ),
    )

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
        default=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        help="Azure OpenAI / Azure AI Foundry endpoint used for GPT generation "
        "and for judging new SLM responses. Defaults to AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--gpt-deployment",
        default=os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4.1"),
        help="Defaults to AZURE_OPENAI_GPT_DEPLOYMENT, then gpt-4.1.",
    )
    parser.add_argument(
        "--judge-deployment", default=None,
        help="Separate RAG judge deployment on --gpt-endpoint; defaults to --gpt-deployment.",
    )
    parser.add_argument(
        "--gpt-api-version",
        default=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        help="Defaults to AZURE_OPENAI_API_VERSION, then 2024-10-21.",
    )
    parser.add_argument("--gpt-max-workers", type=int, default=16)
    parser.add_argument("--gpt-max-retries", type=int, default=4)
    parser.add_argument(
        "--llm-baseline-data-asset-name",
        default="llm_baseline_dataset",
        help="Azure ML data asset used to register or retrieve cached GPT evidence.",
    )
    parser.add_argument(
        "--llm-baseline-data-asset-version",
        "--llm-baseline-version",
        dest="llm_baseline_data_asset_version",
        default=None,
        help=(
            "Immutable baseline version required by --mode slm_evaluation. "
            "Do not provide it with --mode llm_baseline."
        ),
    )

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


def resolve_storage_account_name(ml_client, configured_name: str | None) -> str:
    if configured_name:
        return configured_name
    datastore = ml_client.datastores.get_default()
    account_name = getattr(datastore, "account_name", None)
    if not isinstance(account_name, str) or not account_name.strip():
        raise ValueError(
            "The Azure ML workspace default datastore does not expose a storage "
            "account name. Pass --storage-account-name explicitly."
        )
    log.info(
        "Using storage account %s from default datastore %s",
        account_name,
        datastore.name,
    )
    return account_name


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
    if not args.gpt_endpoint:
        raise ValueError(
            "Set AZURE_OPENAI_ENDPOINT in .env or pass --gpt-endpoint."
        )
    storage_account_name = resolve_storage_account_name(
        ml_client, args.storage_account_name
    )
    test_data_asset = ml_client.data.get(
        args.test_data_asset_name, version=args.test_data_asset_version
    )
    log.info("Test data input: %s", test_data_asset.id)
    baseline_asset = None
    baseline_output_version = None
    if args.mode == "slm_evaluation":
        if not args.llm_baseline_data_asset_version:
            raise ValueError(
                "--llm-baseline-data-asset-version is required with "
                "--mode slm_evaluation"
            )
        baseline_asset = ml_client.data.get(
            args.llm_baseline_data_asset_name,
            version=args.llm_baseline_data_asset_version,
        )
        log.info("Reusing GPT baseline input: %s", baseline_asset.id)
    else:
        if args.llm_baseline_data_asset_version:
            raise ValueError(
                "Do not pass --llm-baseline-data-asset-version with "
                "--mode llm_baseline; this mode creates a new version"
            )
        baseline_output_version = datetime.now(timezone.utc).strftime(
            "%Y%m%d.%H%M%S"
        )
        log.info(
            "Generating GPT baseline output %s:%s",
            args.llm_baseline_data_asset_name,
            baseline_output_version,
        )
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
    hf_token = resolve_hf_token(args) if args.mode == "slm_evaluation" else ""

    finetuning_job_name = args.finetuning_job_name
    if args.mode == "slm_evaluation" and not finetuning_job_name:
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
        storage_account_name=storage_account_name,
        managed_identity_client_id=args.managed_identity_client_id,
        base_model_source=args.base_model_source,
        base_model_registry=args.base_model_registry,
        base_model_label=args.base_model_label,
        mode="both",
        workflow_mode=args.mode,
        compare_with_gpt=args.mode == "llm_baseline",
        gpt_endpoint=args.gpt_endpoint,
        gpt_deployment=args.gpt_deployment,
        judge_deployment=args.judge_deployment,
        gpt_api_version=args.gpt_api_version,
        gpt_max_workers=args.gpt_max_workers,
        gpt_max_retries=args.gpt_max_retries,
        gpt_baseline_asset=baseline_asset.id if baseline_asset else None,
        llm_baseline_output_name=(
            args.llm_baseline_data_asset_name if baseline_asset is None else None
        ),
        llm_baseline_output_version=baseline_output_version,
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
    if baseline_asset is None:
        log.info(
            "GPT baseline will be registered as %s:%s when the job succeeds",
            args.llm_baseline_data_asset_name,
            baseline_output_version,
        )
    if finetuning_job_name:
        record_run(
            finetuning_job_name,
            evaluation_job_name=submitted_job.name,
            evaluation_environment=environment_id,
            evaluation_submitted_at=datetime.now(timezone.utc).isoformat(),
            llm_baseline_data_asset=(
                f"{args.llm_baseline_data_asset_name}:"
                f"{args.llm_baseline_data_asset_version}"
            ),
            llm_baseline_reused=True,
        )

    if args.wait:
        log.info("Waiting for job %s to complete...", submitted_job.name)
        ml_client.jobs.stream(submitted_job.name)
        print_job_metrics(config, ml_client, submitted_job.name)
    else:
        baseline_action = (
            f"reuse {args.llm_baseline_data_asset_name}:"
            f"{args.llm_baseline_data_asset_version}"
            if args.mode == "slm_evaluation"
            else f"register {args.llm_baseline_data_asset_name}:{baseline_output_version}"
        )
        log.info(
            "Job submitted asynchronously and will %s. Per-model RAG evidence "
            "will be saved in the final predictions CSV and logged to Azure ML / "
            "MLflow experiment '%s'.",
            baseline_action,
            args.experiment_name,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
