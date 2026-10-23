import copy
import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from azure.core.exceptions import ResourceNotFoundError

from lib import candidate_release, deployment_planning


def test_retail_price_uses_only_matching_linux_payg_hourly_meter(monkeypatch):
    valid = {"armSkuName": "Standard_NC4as_T4_v3", "armRegionName": "eastus", "type": "Consumption",
             "unitOfMeasure": "1 Hour", "currencyCode": "USD", "retailPrice": .52,
             "effectiveStartDate": "2026-01-01T00:00:00Z", "meterId": "meter-id",
             "skuName": "NC4as T4 v3", "productName": "Virtual Machines NCasT4 v3 Series"}
    windows = {**valid, "productName": "Virtual Machines Windows", "retailPrice": .9}
    spot = {**valid, "skuName": "NC4as T4 v3 Spot", "retailPrice": .1}
    monkeypatch.setattr(deployment_planning, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps({"Items": [valid, windows, spot], "NextPageLink": None}).encode()))
    result = deployment_planning.retail_price(valid["armSkuName"], "eastus")
    assert result["hourly_usd"] == .52
    assert result["source"].startswith("https://prices.azure.com/")
    assert result["meter_id"] == "meter-id"


def test_missing_or_malformed_retail_price_is_not_invented(monkeypatch):
    monkeypatch.setattr(deployment_planning, "urlopen", lambda *args, **kwargs: io.BytesIO(b'{"Items": [], "NextPageLink": null}'))
    with pytest.raises(ValueError, match="No Linux"):
        deployment_planning.retail_price("Standard_NC4as_T4_v3", "eastus")


@pytest.mark.parametrize("sku,success", [("Standard_NC4as_T4_v3", True), ("invented-GPU", False)])
def test_gpt_can_only_choose_available_memory_safe_priced_sku(monkeypatch, sku, success):
    import openai

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://unit.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "unit-version")
    monkeypatch.setenv("AZURE_OPENAI_GPT_DEPLOYMENT", "unit-gpt")
    monkeypatch.setattr(deployment_planning, "retail_price", lambda *args: {"hourly_usd": .52})
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"sku": sku, "rationale": "Fits fp16 base weights with headroom."})))])
    monkeypatch.setattr(openai, "AzureOpenAI", Mock(return_value=client))
    if success:
        result = deployment_planning.recommend_gpu("Llama-3.2-3B-Instruct", {"load_in_4bit": True}, {"Standard_NC4as_T4_v3"}, "eastus", Mock())
        assert result["serving_dtype"] == "float16"
        assert result["estimated_required_gpu_memory_gb"] == 13
        assert result["instance_count"] == 1
    else:
        with pytest.raises(ValueError, match="eligible SKU"):
            deployment_planning.recommend_gpu("Llama-3.2-3B-Instruct", {}, {"Standard_NC4as_T4_v3"}, "eastus", Mock())


def test_unknown_size_or_insufficient_gpu_memory_blocks_without_gpt(monkeypatch):
    prices = Mock()
    monkeypatch.setattr(deployment_planning, "retail_price", prices)
    with pytest.raises(ValueError, match="parameter size"):
        deployment_planning.recommend_gpu("unknown-model", {}, set(), "eastus", Mock())
    with pytest.raises(ValueError, match="No supported"):
        deployment_planning.recommend_gpu("Llama-70B-Instruct", {}, {"Standard_NC4as_T4_v3"}, "eastus", Mock())
    prices.assert_not_called()


def test_exact_plan_hash_and_expiration_are_required():
    plan = {"expires_at": "2000-01-01T00:00:00+00:00", "readiness": {"passed": True}}
    with pytest.raises(ValueError, match="expired"):
        candidate_release.validate_plan(plan, candidate_release.plan_hash(plan))
    plan["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError, match="changed"):
        candidate_release.validate_plan(plan, "0" * 64)


def cloud_fixture(monkeypatch, tmp_path, action):
    from lib import azureml_ops

    client = Mock()
    client.models.get.side_effect = ResourceNotFoundError("not found")
    base = "azureml://registries/azureml-meta/models/Llama-3.2-3B-Instruct/versions/1"
    training = SimpleNamespace(inputs={"base_model": base})
    evidence = {"manifest_sha256": "a" * 64, "predictions_sha256": "b" * 64}
    gate = {"passed": True, "policy": {"version": "paired-quality-v1"}}
    test_path = tmp_path / "test.jsonl"
    record = {"id": "test-1", "instruction": "Answer from the document.", "question": "What is HBM?",
              "context": "HBM means high-bandwidth memory.", "cot_answer": "High-bandwidth memory."}
    test_path.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(candidate_release, "cloud_context", Mock(return_value=(client, Mock(), training, evidence, gate, test_path, {})))
    model = SimpleNamespace(name="candidate", version="1", id="/models/candidate/versions/1")
    register = Mock(return_value=model)
    monkeypatch.setattr(azureml_ops, "register_job_model", register)
    environment = Mock(return_value=SimpleNamespace(id="azureml:env:1"))
    monkeypatch.setattr(azureml_ops, "create_inference_environment", environment)
    endpoint = SimpleNamespace(name="candidate-endpoint", scoring_uri="https://unit.inference.ml.azure.com/score")
    deploy = Mock(return_value=endpoint)
    monkeypatch.setattr(azureml_ops, "deploy_managed_endpoint", deploy)
    client.online_deployments.get.return_value = SimpleNamespace(model=model.id, provisioning_state="Succeeded")
    plan = {
        "action": action, "model_name": "candidate", "model_version": "1",
        "training_job_name": "training-job", "evaluation_job_name": "evaluation-job",
        "base_model_reference": base, "evidence": evidence, "readiness": gate,
        "endpoint_name": endpoint.name, "deployment_name": "candidate",
        "gpu": {"sku": "Standard_NC4as_T4_v3", "price": {"hourly_usd": .52}, "max_input_tokens": 2048, "max_output_tokens": 512},
        "production_data_uri": "azureml://datastores/unit/paths/production", "monitor_compute": "cpu-cluster",
    }
    return client, plan, register, environment, deploy


def test_dedicated_deployment_rejects_existing_endpoint_without_changes():
    from lib.azureml_ops import deploy_managed_endpoint

    client = Mock()
    client.online_endpoints.get.return_value = SimpleNamespace(traffic={"existing": 100})
    with pytest.raises(ValueError, match="existing routing is never modified"):
        deploy_managed_endpoint(client, "existing", "candidate", "model:1", "env:1", "gpu")
    client.online_endpoints.begin_create_or_update.assert_not_called()
    client.online_deployments.begin_create_or_update.assert_not_called()


@pytest.mark.parametrize("routing_verified", [True, False])
def test_dedicated_deployment_routes_only_new_endpoint_and_keeps_scoring_limits(monkeypatch, routing_verified):
    from azure.ai.ml import entities

    from lib.azureml_ops import deploy_managed_endpoint

    for name in ("CodeConfiguration", "DataCollector", "DeploymentCollection",
                 "ManagedOnlineDeployment", "ManagedOnlineEndpoint",
                 "OnlineRequestSettings", "ProbeSettings"):
        monkeypatch.setattr(entities, name, lambda **kwargs: SimpleNamespace(**kwargs))
    client = Mock()
    endpoint = SimpleNamespace(name="new-endpoint", traffic={})
    client.online_endpoints.get.side_effect = [ResourceNotFoundError("not found"), endpoint]
    client.online_deployments.get.side_effect = ResourceNotFoundError("not found")

    def update_endpoint(value):
        if hasattr(value, "traffic") and not routing_verified:
            value.traffic = {}
        return SimpleNamespace(result=lambda: value)

    client.online_endpoints.begin_create_or_update.side_effect = update_endpoint
    if routing_verified:
        assert deploy_managed_endpoint(client, "new-endpoint", "candidate", "model:1", "env:1", "gpu") is endpoint
        assert endpoint.traffic == {"candidate": 100}
    else:
        with pytest.raises(RuntimeError, match="did not route requests"):
            deploy_managed_endpoint(client, "new-endpoint", "candidate", "model:1", "env:1", "gpu")
    deployment = client.online_deployments.begin_create_or_update.call_args.args[0]
    assert deployment.environment_variables["SLM_MAX_INPUT_TOKENS"] == "2048"
    assert deployment.environment_variables["SLM_MAX_OUTPUT_TOKENS"] == "512"
    assert deployment.code_configuration.scoring_script == "lib/score.py"
    assert deployment.data_collector.collections["model_inputs"].enabled == "true"


def test_register_only_never_creates_environment_endpoint_or_schedule(monkeypatch, tmp_path):
    client, plan, register, environment, deploy = cloud_fixture(monkeypatch, tmp_path, "register")
    persisted = []
    result = candidate_release.execute_plan({}, plan, "reviewer", "Evidence reviewed.", tmp_path / "release" / "execution",
                                            Mock(), lambda data: persisted.append(copy.deepcopy(data)))
    assert result["status"] == "registered"
    register.assert_called_once()
    environment.assert_not_called()
    deploy.assert_not_called()
    client.schedules.begin_create_or_update.assert_not_called()
    assert persisted[0]["registered_model"]["version"] == "1"


def test_deploy_records_identity_then_validates_default_scoring_and_monitoring(monkeypatch, tmp_path):
    from lib import azureml_ops, monitoring

    client, plan, _, environment, deploy = cloud_fixture(monkeypatch, tmp_path, "register_and_deploy")
    schedule = Mock(return_value=SimpleNamespace(name="request-drift"))
    monkeypatch.setattr(monitoring, "create_drift_schedule", schedule)
    client.schedules.get.return_value = SimpleNamespace(is_enabled=True)
    invoke = Mock(return_value=[{"prediction": "High-bandwidth memory.", "latency_ms": 100}])
    monkeypatch.setattr(azureml_ops, "evaluate_endpoint", invoke)
    persisted = []
    result = candidate_release.execute_plan({}, plan, "reviewer", "Evidence and cost reviewed.",
                                            tmp_path / "release" / "execution", Mock(),
                                            lambda data: persisted.append(copy.deepcopy(data)))
    assert persisted[0]["status"] == "registered"
    assert result["endpoint"]["validated"]
    assert result["monitoring"]["status"] == "enabled"
    assert "dedicated" not in deploy.call_args.kwargs
    assert "deployment_name" not in invoke.call_args.kwargs
    environment.assert_called_once()
    schedule.assert_called_once()


def test_cloud_evidence_change_blocks_registration_and_gpu_allocation(monkeypatch, tmp_path):
    _, plan, register, environment, deploy = cloud_fixture(monkeypatch, tmp_path, "register_and_deploy")
    plan["evidence"] = {**plan["evidence"], "manifest_sha256": "c" * 64}
    with pytest.raises(ValueError, match="changed after approval"):
        candidate_release.execute_plan({}, plan, "reviewer", "Reviewed.", tmp_path / "execution", Mock(), Mock())
    register.assert_not_called()
    environment.assert_not_called()
    deploy.assert_not_called()
