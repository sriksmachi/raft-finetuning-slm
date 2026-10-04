"""Azure ML asset, job, model, and managed endpoint operations."""

from __future__ import annotations

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)
TRAIN_ENVIRONMENT_VERSION_KEY = "AZURE_TRAIN_ENVIRONMENT_VERSION"
AZUREML_GPU_BASE_IMAGE = (
    "mcr.microsoft.com/azureml/"
    "openmpi5.0-cuda12.8-ubuntu24.04"
    "@sha256:02ba2dc7f0aca7b5d296ece25005db50e5fab837ea531972793fee906029efcc"
)


def _register_environment(ml_client, environment):
    from azure.ai.ml.constants import ListViewType
    from azure.core.exceptions import ResourceExistsError

    for attempt in range(4):
        try:
            registered = ml_client.environments.create_or_update(environment)
            log.info("Using registered environment %s:%s", registered.name, registered.version)
            return registered
        except ResourceExistsError as exc:
            if "already registered and cannot be changed" not in str(exc).lower():
                raise
            if attempt == 3:
                log.exception("Repeated immutable-version conflicts registering %s", environment.name)
                raise
            versions = ml_client.environments.list(
                name=environment.name, list_view_type=ListViewType.ALL,
            )
            numeric_versions = [
                int(item.version) for item in versions if str(item.version).isdigit()
            ]
            if str(environment.version).isdigit():
                numeric_versions.append(int(environment.version))
            next_version = str(max(numeric_versions, default=0) + 1)
            log.warning(
                "Environment %s:%s is immutable and has a different definition. "
                "Registering the unchanged local definition as new version %s.",
                environment.name, environment.version, next_version,
            )
            environment.version = next_version


def get_training_environment_version() -> str:
    from dotenv import dotenv_values

    values = dotenv_values(PROJECT_ROOT / ".env")
    value = values.get(
        TRAIN_ENVIRONMENT_VERSION_KEY,
        os.environ.get(TRAIN_ENVIRONMENT_VERSION_KEY, "1"),
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{TRAIN_ENVIRONMENT_VERSION_KEY} must be non-empty")
    return value.strip()


def _persist_training_environment_version(version: str) -> None:
    from dotenv import dotenv_values, set_key

    env_path = PROJECT_ROOT / ".env"
    value = str(version)
    existing = dotenv_values(env_path).get(TRAIN_ENVIRONMENT_VERSION_KEY)
    if existing and existing.isdigit() and value.isdigit():
        value = str(max(int(existing), int(value)))
    try:
        updated, _, _ = set_key(
            str(env_path), TRAIN_ENVIRONMENT_VERSION_KEY, value, quote_mode="never",
        )
    except OSError:
        log.exception("Could not persist registered training environment version %s to %s", value, env_path)
        raise
    if not updated:
        raise RuntimeError(f"Could not persist training environment version {value} to {env_path}")
    os.environ[TRAIN_ENVIRONMENT_VERSION_KEY] = value
    log.info("Saved %s=%s to %s", TRAIN_ENVIRONMENT_VERSION_KEY, value, env_path)


def create_training_environment(ml_client, name: str, version: str | None = None):
    from azure.ai.ml.entities import Environment

    environment = Environment(
        name=name,
        version=version if version is not None else get_training_environment_version(),
        description="Reproducible CUDA environment for SLM Studio fine-tuning",
        image=AZUREML_GPU_BASE_IMAGE,
        conda_file=str(PROJECT_ROOT / "environments" / "train-conda.yml"),
    )
    registered = _register_environment(ml_client, environment)
    _persist_training_environment_version(str(registered.version))
    return registered


def create_inference_environment(ml_client, name: str, version: str = "2"):
    from azure.ai.ml.entities import Environment

    environment = Environment(
        name=name,
        version=version,
        description="Reproducible CUDA environment for SLM Studio inference",
        image=AZUREML_GPU_BASE_IMAGE,
        conda_file=str(PROJECT_ROOT / "environments" / "inference-conda.yml"),
    )
    return _register_environment(ml_client, environment)


def submit_finetuning_job(
    ml_client,
    data_asset: str,
    environment: str,
    compute: str,
    experiment_name: str,
    base_model: str,
    display_name: str,
    base_model_source: str = "foundry",
    base_model_registry: str = "azureml-meta",
    base_model_label: str = "latest",
    base_model_version: str | None = None,
    key_vault_name: str | None = None,
    hf_token_secret_name: str | None = None,
    managed_identity_client_id: str | None = None,
    credential=None,
    max_seq_length: int | None = None,
    chat_template: str | None = None,
    load_in_4bit: bool | None = None,
    lora_rank: int | None = None,
    learning_rate: float | None = None,
    epochs: float | None = None,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    seed: int | None = None,
    lora_adapter_name: str | None = None,
):
    from azure.ai.ml import Input, MLClient, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.entities import ManagedIdentityConfiguration
    from azure.identity import DefaultAzureCredential

    # Resolve base model. "foundry" mounts the registry asset at ${{inputs.base_model}};
    # "huggingface" passes the repo id through as a literal CLI argument.
    if base_model_source == "foundry":
        if credential is None:
            credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        
        print(f"[foundry] connecting to registry {base_model_registry!r}...", flush=True)
        
        registry_client = MLClient(
            credential=credential, registry_name=base_model_registry
        )
        
        if base_model_version:
            print(
                f"[foundry] fetching {base_model}:{base_model_version}...", flush=True
            )
            registry_model = registry_client.models.get(
                name=base_model, version=base_model_version
            )
        else:
            print(
                f"[foundry] fetching {base_model} label={base_model_label!r}...",
                flush=True,
            )
            registry_model = registry_client.models.get(
                name=base_model, label=base_model_label
            )
        print(f"[foundry] resolved {registry_model.id}", flush=True)
        base_model_argument = Input(
            type=AssetTypes.CUSTOM_MODEL,
            path=registry_model.id,
            mode=InputOutputModes.DOWNLOAD,
        )
    
    elif base_model_source == "huggingface":
        base_model_argument = base_model
    
    else:
        raise ValueError(
            f"Unsupported base_model_source={base_model_source!r}; use 'foundry' or 'huggingface'"
        )

    secret_arguments = ""
    
    if base_model_source == "huggingface" and key_vault_name and hf_token_secret_name:
        secret_arguments = (
            " --key-vault-name ${{inputs.key_vault_name}}"
            " --hf-token-secret-name ${{inputs.hf_token_secret_name}}"
        )
        if managed_identity_client_id:
            secret_arguments += (
                " --managed-identity-client-id "
                "${{inputs.managed_identity_client_id}}"
            )
    
    elif base_model_source == "huggingface" and (key_vault_name or hf_token_secret_name):
        raise ValueError("Provide both Key Vault name and Hugging Face secret name")

    # Optional train.py overrides (Unsloth migration: see lib/train.py). Each is
    # only appended when explicitly provided, so callers that omit them keep
    # relying on train.py's own defaults rather than this function silently
    # pinning a value.
    optional_arguments = ""
    if max_seq_length is not None:
        optional_arguments += f" --max-seq-length {max_seq_length}"
    if chat_template is not None:
        optional_arguments += f" --chat-template {chat_template}"
    if load_in_4bit is not None:
        optional_arguments += " --load-in-4bit" if load_in_4bit else " --no-load-in-4bit"
    if lora_rank is not None:
        optional_arguments += f" --lora-rank {lora_rank}"
    if learning_rate is not None:
        optional_arguments += f" --learning-rate {learning_rate}"
    if epochs is not None:
        optional_arguments += f" --epochs {epochs}"
    if batch_size is not None:
        optional_arguments += f" --batch-size {batch_size}"
    if gradient_accumulation is not None:
        optional_arguments += f" --gradient-accumulation {gradient_accumulation}"
    if seed is not None:
        optional_arguments += f" --seed {seed}"
    if lora_adapter_name is not None:
        from lib.lora_artifacts import validate_adapter_name

        optional_arguments += f" --lora-adapter-name {validate_adapter_name(lora_adapter_name)}"

    inputs = {
        "training_data": Input(
            type=AssetTypes.URI_FOLDER,
            path=data_asset,
            mode=InputOutputModes.DOWNLOAD,
        ),
        "base_model": base_model_argument,
    }
    
    if secret_arguments:
        inputs.update(
            {
                "key_vault_name": key_vault_name,
                "hf_token_secret_name": hf_token_secret_name,
            }
        )
        if managed_identity_client_id:
            inputs["managed_identity_client_id"] = managed_identity_client_id

    job = command(
        code=str(PROJECT_ROOT),
        command=(
            "python -m lib.train "
            "--data ${{inputs.training_data}} "
            "--model-output ${{outputs.model}} "
            "--base-model ${{inputs.base_model}}"
            + secret_arguments
            + optional_arguments
        ),
        inputs=inputs,
        outputs={
            "model": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT)
        },
        environment=environment,
        compute=compute,
        experiment_name=experiment_name,
        display_name=display_name,
        identity=ManagedIdentityConfiguration(client_id=managed_identity_client_id),
        tags={"framework": "SLM Studio", "stage": "fine-tuning", "model_family": "llama-3.2"},
    )
    
    return ml_client.jobs.create_or_update(job)


def submit_inference_evaluation_job(
    ml_client,
    test_data_asset: str,
    environment: str,
    compute: str,
    experiment_name: str,
    base_model: str,
    display_name: str,
    finetuning_job_name: str,
    storage_account_name: str,
    managed_identity_client_id: str,
    base_model_source: str = "foundry",
    base_model_registry: str = "azureml-meta",
    base_model_label: str = "latest",
    base_model_version: str | None = None,
    key_vault_name: str | None = None,
    hf_token_secret_name: str | None = None,
    credential=None,
    mode: str = "both",
    compare_with_gpt: bool = False,
    gpt_endpoint: str | None = None,
    gpt_deployment: str = "gpt-4.1",
    gpt_api_version: str = "2024-10-21",
    gpt_max_workers: int = 16,
    gpt_max_retries: int = 4,
    max_new_tokens: int | None = None,
    max_seq_length: int | None = None,
    max_input_tokens: int | None = None,
    inference_batch_size: int | None = None,
    load_in_4bit: bool | None = None,
    max_records: int | None = None,
    sample_records: int | None = None,
    hf_token: str = "",
    judge_deployment: str | None = None,
):
    """Submit a ``lib/inference.py`` command job.

    Runs fine-tuned-SLM-vs-GPT inference/evaluation: generates responses
    with the model produced by ``scripts/02_azureml_fine_tuning.py``'s
    training job (``finetuning_job_name``, mounted directly from that job's
    ``outputs/model`` artifact -- the same path ``register_job_model`` uses),
    optionally also against an Azure OpenAI/Azure AI Foundry GPT deployment
    for the same test questions/context, evaluates faithfulness, relevancy,
    and correctness against cot_answer, saves per-model scores/reasons to CSV,
    and logs RAG means plus JSON-validity/latency diagnostics to MLflow.

    Deliberately mounts the raw job output rather than a *registered* model:
    ``scripts/04_release_candidate.py`` only registers a candidate after this
    evaluation's gate passes, so requiring registration first would be
    circular.

    ``hf_token`` must be the literal token value (resolved from Key Vault by
    the caller, e.g. ``02_azureml_fine_tuning.py``'s Key Vault helper) since
    ``lib/inference.py`` reads it as a plain CLI argument rather than
    performing its own in-job Key Vault lookup. Pass an empty string when
    ``base_model_source="foundry"``, since no HF token is needed in that case.
    """
    from azure.ai.ml import Input, MLClient, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.entities import ManagedIdentityConfiguration
    from azure.identity import DefaultAzureCredential

    if base_model_source == "foundry":
        if credential is None:
            credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        registry_client = MLClient(credential=credential, registry_name=base_model_registry)
        if base_model_version:
            registry_model = registry_client.models.get(name=base_model, version=base_model_version)
        else:
            registry_model = registry_client.models.get(name=base_model, label=base_model_label)
        print(f"[foundry] resolved {registry_model.id}", flush=True)
        base_model_argument = Input(
            type=AssetTypes.CUSTOM_MODEL,
            path=registry_model.id,
            mode=InputOutputModes.DOWNLOAD,
        )
    elif base_model_source == "huggingface":
        base_model_argument = base_model
        if not hf_token and key_vault_name and hf_token_secret_name:
            from azure.keyvault.secrets import SecretClient

            if credential is None:
                credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
            secret_client = SecretClient(
                vault_url=f"https://{key_vault_name}.vault.azure.net",
                credential=credential,
            )
            hf_token = secret_client.get_secret(hf_token_secret_name).value
        if not hf_token:
            raise ValueError(
                "base_model_source='huggingface' requires either hf_token or both "
                "key_vault_name and hf_token_secret_name"
            )
    else:
        raise ValueError(
            f"Unsupported base_model_source={base_model_source!r}; use 'foundry' or 'huggingface'"
        )

    completed_training_job = ml_client.jobs.get(finetuning_job_name)
    if completed_training_job.status != "Completed":
        raise RuntimeError(
            f"Training job {finetuning_job_name!r} is {completed_training_job.status!r}, "
            "not 'Completed'; cannot mount its model output yet."
        )
    output_datastore = ml_client.datastores.get_default().name
    fine_tuned_model_argument = Input(
        type=AssetTypes.URI_FOLDER,
        path=(
            f"azureml://datastores/{output_datastore}/paths/"
            f"azureml/{finetuning_job_name}/model/"
        ),
        mode=InputOutputModes.DOWNLOAD,
    )

    if compare_with_gpt and not gpt_endpoint:
        raise ValueError("gpt_endpoint is required when compare_with_gpt=True")

    optional_arguments = ""
    for flag, value in (
        ("--max-new-tokens", max_new_tokens),
        ("--max-seq-length", max_seq_length),
        ("--max-input-tokens", max_input_tokens),
        ("--inference-batch-size", inference_batch_size),
        ("--max-records", max_records),
        ("--sample-records", sample_records),
    ):
        if value is not None:
            optional_arguments += f" {flag} {value}"
    if load_in_4bit:
        optional_arguments += " --load-in-4bit"
    if compare_with_gpt:
        optional_arguments += (
            " --compare-with-gpt"
            f" --gpt-endpoint {gpt_endpoint}"
            f" --gpt-deployment {gpt_deployment}"
            f" --gpt-api-version {gpt_api_version}"
            f" --gpt-max-workers {gpt_max_workers}"
            f" --gpt-max-retries {gpt_max_retries}"
        )
        if judge_deployment:
            optional_arguments += f" --judge-deployment {judge_deployment}"

    inputs = {
        "test_data": Input(
            type=AssetTypes.URI_FOLDER,
            path=test_data_asset,
            mode=InputOutputModes.DOWNLOAD,
        ),
        "base_model": base_model_argument,
        "fine_tuned_model": fine_tuned_model_argument,
    }
    hf_token_argument = ""
    if hf_token:
        # Plain string input: lib/inference.py reads --hf-token as a literal
        # value, not via its own Key Vault lookup (unlike train.py).
        inputs["hf_token"] = hf_token
        hf_token_argument = " --hf-token ${{inputs.hf_token}}"

    job = command(
        # lib/inference.py uses `from data import read_jsonl` (a bare
        # import), so the job's code snapshot must be the lib/ folder itself
        # -- not the repo root -- matching how that script expects to be invoked.
        code=str(PROJECT_ROOT / "lib"),
        command=(
            "python inference.py "
            "--model ${{inputs.base_model}} "
            "--fine-tuned-model ${{inputs.fine_tuned_model}} "
            f"--finetuning-job-name {finetuning_job_name} "
            "--data ${{inputs.test_data}} "
            "--output ${{outputs.predictions}} "
            f"--storage-account-name {storage_account_name} "
            f"--azure-client-id {managed_identity_client_id} "
            f"--mode {mode}"
            + hf_token_argument
            + optional_arguments
        ),
        inputs=inputs,
        outputs={
            "predictions": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT)
        },
        environment=environment,
        compute=compute,
        experiment_name=experiment_name,
        display_name=display_name,
        identity=ManagedIdentityConfiguration(client_id=managed_identity_client_id),
        tags={
            "framework": "SLM Studio",
            "stage": "inference-evaluation",
            "finetuning_job_name": finetuning_job_name,
            "compare_with_gpt": str(compare_with_gpt),
        },
    )

    return ml_client.jobs.create_or_update(job)


def register_job_model(
    ml_client,
    job_name: str,
    model_name: str,
    version: str,
    validation_metric: float | None = None,
):
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    tags = {"framework": "SLM Studio", "source_job": job_name, "stage": "candidate"}
    if validation_metric is not None:
        tags["validation_answer_token_f1"] = str(validation_metric)
    model = Model(
        name=model_name,
        version=version,
        path=f"azureml://jobs/{job_name}/outputs/model",
        type=AssetTypes.CUSTOM_MODEL,
        description="Named SLM Studio LoRA adapter and tokenizer (base weights remain separate)",
        tags=tags,
    )
    return ml_client.models.create_or_update(model)


def deploy_managed_endpoint(
    ml_client,
    endpoint_name: str,
    deployment_name: str,
    model: str,
    environment: str,
    instance_type: str,
    instance_count: int = 1,
    traffic_percent: int = 0,
    base_model: str | None = None,
    key_vault_name: str | None = None,
    hf_token_secret_name: str | None = None,
):
    from azure.ai.ml.entities import (
        CodeConfiguration,
        DataCollector,
        DeploymentCollection,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        OnlineRequestSettings,
        ProbeSettings,
    )
    from azure.core.exceptions import ResourceNotFoundError

    if traffic_percent != 0:
        raise ValueError("Candidate deployment must have zero traffic; use the separate promotion stage")
    if bool(key_vault_name) != bool(hf_token_secret_name):
        raise ValueError("Provide both Key Vault name and Hugging Face secret name")
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="SLM Studio context-grounded generation endpoint",
            auth_mode="aml_token",
            tags={"framework": "SLM Studio", "data_collection": "enabled"},
        )
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    previous_traffic = dict(endpoint.traffic or {})
    try:
        ml_client.online_deployments.get(endpoint_name=endpoint_name, name=deployment_name)
    except ResourceNotFoundError:
        pass
    else:
        raise ValueError("Choose a new deployment name; existing deployments are never overwritten")
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=str(PROJECT_ROOT),
            scoring_script="lib/score.py",
        ),
        instance_type=instance_type,
        instance_count=instance_count,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=120_000,
            max_concurrent_requests_per_instance=1,
        ),
        liveness_probe=ProbeSettings(initial_delay=600),
        readiness_probe=ProbeSettings(initial_delay=600),
        environment_variables={
            "TOKENIZERS_PARALLELISM": "false",
            "SLM_ENABLE_DATA_COLLECTION": "true",
            **({"SLM_BASE_MODEL": base_model} if base_model else {}),
            **({"SLM_KEY_VAULT_NAME": key_vault_name,
                "SLM_HF_TOKEN_SECRET_NAME": hf_token_secret_name} if key_vault_name else {}),
        },
        data_collector=DataCollector(
            collections={
                "model_inputs": DeploymentCollection(enabled="true"),
                "model_outputs": DeploymentCollection(enabled="true"),
            }
        ),
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    if dict(endpoint.traffic or {}) != previous_traffic:
        raise RuntimeError("Endpoint traffic changed during candidate deployment; review before any promotion")
    return endpoint


def promote_deployment(
    ml_client,
    endpoint_name: str,
    deployment_name: str,
    traffic_percent: int = 100,
):
    if (
        isinstance(traffic_percent, bool)
        or not isinstance(traffic_percent, int)
        or not 1 <= traffic_percent <= 100
    ):
        raise ValueError("traffic_percent must be an integer between 1 and 100")
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    current = {
        name: percent
        for name, percent in dict(endpoint.traffic or {}).items()
        if name != deployment_name and percent > 0
    }
    if traffic_percent == 100:
        endpoint.traffic = {deployment_name: 100}
    elif len(current) == 1:
        incumbent = next(iter(current))
        endpoint.traffic = {
            incumbent: 100 - traffic_percent,
            deployment_name: traffic_percent,
        }
    elif not current:
        raise ValueError(
            "Partial traffic requires one incumbent deployment; use 100 for the first release"
        )
    else:
        raise ValueError(
            "Partial traffic requires exactly one incumbent deployment to avoid "
            "silently changing an existing multi-deployment routing policy"
        )
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()
