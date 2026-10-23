import copy
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.candidate_api import candidate_details
from app.main import create_app
from app.workflow import new_record


def evaluation_record():
    record = new_record("prod-evaluation", "data/", "context-grounded")
    record.update(mode="production", simulated=False, run_type="evaluation")
    record["stages"]["evaluation"]["status"] = "succeeded"
    record["training"] = {"dataset_id": "azureml:demo-data:1", "dataset_run_id": "prod-data",
                          "config": {"base_model": "Llama-3.2-3B-Instruct", "epochs": 2, "lora_adapter_name": "adapter"}}
    record["evaluation"] = {
        "job_name": "evaluated-job", "job_status": "Completed", "candidate_id": "trained-job",
        "dataset_id": "azureml:demo-data:1", "data_fingerprint": "a" * 64,
        "judge_deployment": "configured-judge", "baseline_id": "baseline:1",
        "training_job": {"training_run_id": "prod-training", "base_model": "Llama-3.2-3B-Instruct", "sample_count": 240},
        "records": [{"answers": {model: {"scores": {metric: score for metric in ("faithfulness", "relevancy", "correctness")}}
                                for model, score in (("fine_tuned", .8), ("oob", .7), ("gpt", .82))}}
                    for _ in range(30)],
    }
    return record


def wait(client, run_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.get(f"/api/production/runs/{run_id}").json()
        if result["status"] != "running":
            return result
        time.sleep(.01)
    pytest.fail("Candidate operation did not finish")


@pytest.fixture
def studio(tmp_path):
    app = create_app(production_runs_dir=tmp_path / "production")
    source = evaluation_record()
    app.state.production.store.save(source)
    calls = []

    def planner(selected, options, directory, report):
        calls.append(("plan", options["action"]))
        report("Validating mock candidate lineage.")
        plan = {
            "action": options["action"], "readiness": candidate_details(selected)["readiness"],
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
            "model_name": options["model_name"], "model_version": "version1",
            "base_model_reference": "azureml://registries/azureml-meta/models/Llama-3.2-3B-Instruct/versions/1",
        }
        if options["action"] == "register_and_deploy":
            plan["gpu"] = {"sku": "Standard_NC4as_T4_v3", "price": {"hourly_usd": .52}}
        return plan

    def executor(selected, plan, actor, reason, directory, report, persist):
        calls.append(("execute", plan["action"]))
        result = {"status": "registered", "registered_model": {"name": plan["model_name"], "version": "version1", "id": "/models/candidate/versions/1"}}
        persist(result)
        if plan["action"] == "register_and_deploy":
            result.update(status="deployed", endpoint={"uri": "https://unit.inference.ml.azure.com/score", "validated": True})
        return result

    app.state.production.candidates.planner = planner
    app.state.production.candidates.executor = executor
    with TestClient(app) as client:
        yield client, app.state.production, calls


def request_plan(client, action="register"):
    response = client.post("/api/production/release-plans", json={
        "evaluation_run_id": "prod-evaluation", "action": action, "confirmed": True,
        **({"endpoint_name": "unit-endpoint", "production_data_uri": "azureml://datastores/unit/paths/production"} if action != "register" else {}),
    })
    assert response.status_code == 202, response.text
    return wait(client, response.json()["run_id"])


def approval(plan, **changes):
    return {"confirmed": True, "plan_hash": plan["release"]["plan_hash"],
            "actor": "reviewer", "reason": "Reviewed scientific evidence.", **changes}


def test_catalogue_exposes_transparent_fixed_policy_and_full_lineage(studio):
    client, _, _ = studio
    response = client.get("/api/production/release-evaluations")
    assert response.status_code == 200
    entry = response.json()["evaluations"][0]
    assert entry["readiness"]["passed"]
    assert entry["lineage"]["training_run"] == "prod-training"
    assert entry["lineage"]["preparation_run"] == "prod-data"
    assert entry["lineage"]["evaluation_dataset"] == entry["lineage"]["training_dataset"]
    assert entry["lineage"]["adapter_name"] == "adapter"
    assert response.json()["policy"]["editable"] is False


def test_perfect_baseline_exception_is_recomputed_for_catalogue_and_planning(studio):
    client, production, calls = studio
    source = production.store.read("prod-evaluation")
    for record in source["evaluation"]["records"]:
        for model in ("oob", "fine_tuned"):
            record["answers"][model]["scores"]["faithfulness"] = 1
    production.store.save(source)
    entry = client.get("/api/production/release-evaluations").json()["evaluations"][0]
    assert entry["readiness"]["policy"]["version"] == "paired-quality-v2"
    assert entry["readiness"]["metrics"][0]["ceiling_exception_applied"]
    assert entry["readiness"]["passed"]
    plan = request_plan(client)
    assert plan["release"]["plan"]["readiness"]["metrics"][0]["ceiling_exception_applied"]
    assert calls == [("plan", "register")]
    source["evaluation"]["records"][0]["answers"]["fine_tuned"]["scores"]["faithfulness"] = .99
    production.store.save(source)
    response = client.post("/api/production/release-plans", json={
        "evaluation_run_id": "prod-evaluation", "action": "register", "confirmed": True,
    })
    assert response.status_code == 409
    assert not response.json()["detail"]["readiness"]["passed"]
    assert calls == [("plan", "register")]


def test_register_only_is_two_phase_and_persistent_without_deployment(studio):
    client, production, calls = studio
    plan = request_plan(client)
    assert calls == [("plan", "register")]
    assert plan["release"]["status"] == "awaiting_approval"
    response = client.post(f"/api/production/release-plans/{plan['run_id']}/approve", json=approval(plan))
    assert response.status_code == 202
    final = wait(client, plan["run_id"])
    assert final["release"]["status"] == "registered"
    assert "endpoint" not in final["release"]
    assert final["stages"]["deployment"]["status"] == "not_started"
    assert production.store.read(plan["run_id"])["decisions"]["registration"]["actor"] == "reviewer"
    assert client.post(f"/api/production/release-plans/{plan['run_id']}/approve", json=approval(plan)).status_code == 409
    assert len(calls) == 2


def test_deployment_requires_cost_approval_and_reports_validated_endpoint(studio):
    client, _, calls = studio
    plan = request_plan(client, "register_and_deploy")
    url = f"/api/production/release-plans/{plan['run_id']}/approve"
    assert client.post(url, json=approval(plan)).status_code == 409
    assert calls == [("plan", "register_and_deploy")]
    assert client.post(url, json=approval(plan, accept_gpu_cost=True)).status_code == 202
    final = wait(client, plan["run_id"])
    assert final["stages"]["deployment"]["status"] == "succeeded"
    assert final["release"]["endpoint"]["validated"]
    log = client.get(f"/api/production/runs/{plan['run_id']}/log").json()
    assert log["stage"] == "deployment"
    assert "Validating mock candidate" in log["log"]


def test_blocked_gate_cannot_be_overridden(studio):
    client, production, calls = studio
    source = production.store.read("prod-evaluation")
    source["evaluation"]["records"] = source["evaluation"]["records"][:29]
    production.store.save(source)
    response = client.post("/api/production/release-plans", json={"evaluation_run_id": "prod-evaluation", "confirmed": True})
    assert response.status_code == 409
    assert not calls
    assert client.post("/api/production/release-plans", json={"evaluation_run_id": "prod-evaluation", "confirmed": True, "override": True}).status_code == 422


@pytest.mark.parametrize("mutation", ["expired", "tampered"])
def test_stale_or_tampered_plans_cannot_be_approved(studio, mutation):
    client, production, calls = studio
    plan = request_plan(client)
    changed = copy.deepcopy(plan)
    if mutation == "expired":
        changed["release"]["plan"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    else:
        changed["release"]["plan"]["model_name"] = "different-model"
    production.store.save(changed)
    response = client.post(f"/api/production/release-plans/{plan['run_id']}/approve", json=approval(plan))
    assert response.status_code == 409
    assert len(calls) == 1


def test_deployment_failure_retains_registered_model_and_warns_about_billing(studio):
    client, production, _ = studio
    plan = request_plan(client, "register_and_deploy")

    def fail(source, plan, actor, reason, directory, report, persist):
        persist({"registered_model": {"name": "candidate", "version": "1", "id": "/models/candidate/1"}})
        raise RuntimeError("GPU quota exhausted")

    production.candidates.executor = fail
    client.post(f"/api/production/release-plans/{plan['run_id']}/approve", json=approval(plan, accept_gpu_cost=True))
    final = wait(client, plan["run_id"])
    assert final["status"] == "failed"
    assert final["release"]["registered_model"]["id"] == "/models/candidate/1"
    assert final["stages"]["decision"]["status"] == "succeeded"
    assert "allocated" in final["stages"]["deployment"]["error"]
