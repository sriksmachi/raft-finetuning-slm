"""Azure ML asset, job, model, and managed endpoint operations."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_training_environment(ml_client, name: str, version: str = "1"):
    from azure.ai.ml.entities import Environment

    environment = Environment(
        name=name,
        version=version,
        description="Reproducible CUDA environment for RAFT QLoRA fine-tuning",
        image="nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        conda_file=str(PROJECT_ROOT / "environments" / "train-conda.yml"),
    )
    return ml_client.environments.create_or_update(environment)


def create_inference_environment(ml_client, name: str, version: str = "1"):
    from azure.ai.ml.entities import Environment

    environment = Environment(
        name=name,
        version=version,
        description="Reproducible CUDA environment for RAFT online inference",
        image="nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        conda_file=str(PROJECT_ROOT / "environments" / "inference-conda.yml"),
    )
    return ml_client.environments.create_or_update(environment)


def submit_finetuning_job(
    ml_client,
    data_asset: str,
    environment: str,
    compute: str,
    experiment_name: str,
    base_model: str,
    display_name: str,
    key_vault_name: str | None = None,
    hf_token_secret_name: str | None = None,
):
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.entities import ManagedIdentityConfiguration

    secret_arguments = ""
    if key_vault_name and hf_token_secret_name:
        secret_arguments = (
            " --key-vault-name ${{inputs.key_vault_name}}"
            " --hf-token-secret-name ${{inputs.hf_token_secret_name}}"
        )
    elif key_vault_name or hf_token_secret_name:
        raise ValueError("Provide both Key Vault name and Hugging Face secret name")

    inputs = {
        "raft_data": Input(
            type=AssetTypes.URI_FOLDER,
            path=data_asset,
            mode=InputOutputModes.DOWNLOAD,
        ),
        "base_model": base_model,
    }
    if secret_arguments:
        inputs.update(
            {
                "key_vault_name": key_vault_name,
                "hf_token_secret_name": hf_token_secret_name,
            }
        )

    job = command(
        code=str(PROJECT_ROOT),
        command=(
            "python -m lib.train "
            "--data ${{inputs.raft_data}} "
            "--model-output ${{outputs.model}} "
            "--base-model ${{inputs.base_model}}"
            + secret_arguments
        ),
        inputs=inputs,
        outputs={
            "model": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT)
        },
        environment=environment,
        compute=compute,
        experiment_name=experiment_name,
        display_name=display_name,
        identity=ManagedIdentityConfiguration(),
        tags={"framework": "RAFT", "stage": "fine-tuning", "model_family": "llama-3.2"},
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

    tags = {"framework": "RAFT", "source_job": job_name, "stage": "candidate"}
    if validation_metric is not None:
        tags["validation_answer_token_f1"] = str(validation_metric)
    model = Model(
        name=model_name,
        version=version,
        path=f"azureml://jobs/{job_name}/outputs/model",
        type=AssetTypes.CUSTOM_MODEL,
        description="Merged RAFT fine-tuned causal language model",
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
):
    from azure.ai.ml.entities import (
        CodeConfiguration,
        DataCollector,
        DeploymentCollection,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        OnlineRequestSettings,
    )
    from azure.core.exceptions import ResourceNotFoundError

    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="RAFT context-grounded generation endpoint",
            auth_mode="aml_token",
            tags={"framework": "RAFT", "data_collection": "enabled"},
        )
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

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
        environment_variables={"TOKENIZERS_PARALLELISM": "false"},
        data_collector=DataCollector(
            collections={
                "model_inputs": DeploymentCollection(enabled="true"),
                "model_outputs": DeploymentCollection(enabled="true"),
            }
        ),
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    if traffic_percent:
        endpoint.traffic = {deployment_name: traffic_percent}
        return ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    return ml_client.online_endpoints.get(endpoint_name)


def promote_deployment(ml_client, endpoint_name: str, deployment_name: str):
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()
