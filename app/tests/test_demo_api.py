"""API coverage using isolated demo stores; repository fixture files are read-only."""

import ast
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.fixtures import METRICS, STAGES
from app.main import ROOT, create_app
from app.models import Parameters


@pytest.fixture
def studio(tmp_path):
    application = create_app(tmp_path / "demo", stage_delay=0.005)
    with TestClient(application) as client:
        yield client, application.state.store


def wait(client, run_id, stage):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        record = response.json()
        if record["stages"][stage]["status"] != "running":
            return record
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for {stage} of {run_id}")


def prepare(client, scenario="passing"):
    response = client.post("/api/runs", json={
        "source_folder": r"C:\not-real\never-read.pdf", "recipe": "context-grounded",
        "scenario": scenario, "confirmed": True,
    })
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["stages"]["data_preparation"]["status"] == "running"
    return record


def trained(client, scenario="passing"):
    record = prepare(client, scenario)
    run_id = record["run_id"]
    wait(client, run_id, "data_preparation")
    response = client.post(f"/api/runs/{run_id}/training",
                           json={"confirmed": True, **Parameters().model_dump()})
    assert response.status_code == 200, response.text
    return wait(client, run_id, "training")


def evaluated(client, scenario="passing"):
    record = trained(client, scenario)
    response = client.post(f"/api/runs/{record['run_id']}/evaluation", json={"confirmed": True})
    assert response.status_code == 200, response.text
    return wait(client, record["run_id"], "evaluation")


def assert_shape(record):
    assert record["schema_version"] == 1
    assert record["mode"] == "demo" and record["simulated"] is True
    assert record["run_id"].startswith("demo-")
    assert record["created_at"].endswith("Z") and record["updated_at"].endswith("Z")
    assert set(record["stages"]) == set(STAGES)
    assert set(record["decisions"]) == {"registration", "deployment"}
    assert record["current_stage"] in STAGES
    if record["data_card"]:
        card = record["data_card"]
        assert card["split_counts"] == card["splits"]
        assert card["fingerprint"] == card["data_fingerprint"]
        assert card["source_references"] == card["source_refs"]
        assert card["asset_reference"].startswith("demo://")
    for stage in record["stages"].values():
        assert stage["status"] in {"not_started", "running", "succeeded", "failed"}
        assert 0 <= stage["progress"] <= 100
        assert set(stage) >= {"error", "started_at", "completed_at"}
    for event in record["audit"]:
        assert event["simulated"] is True
        assert {"action", "actor", "timestamp"} <= event.keys()
    if record["evaluation"]:
        evidence = record["evaluation"]
        assert evidence["simulated"] is True and evidence["sealed"] is True
        assert evidence["candidate_id"] == record["training"]["candidate_id"]
        assert {row["role"] for row in evidence["models"]} == {"candidate", "base_slm", "gpt"}
        for row in evidence["models"]:
            assert row["simulated"] is True and row["sample_count"] == 36
            assert row["test_split"] == evidence["test_split"]
            assert row["data_fingerprint"] == evidence["data_fingerprint"]
            assert set(row["metrics"]) == {metric["key"] for metric in METRICS}
            for key in ("faithfulness", "accuracy", "relevancy"):
                assert 0 <= row["metrics"][key] <= 1
    if record.get("release"):
        assert record["release"]["simulated"] is True
        assert record["release"]["status"] == "candidate_deployed_zero_traffic"
        assert record["release"]["endpoint"]["traffic_percent"] == 0
        assert record["release"]["drift_reference"]["uri"].startswith("demo://")


def test_config_and_production_disabled(studio):
    client, _ = studio
    config = client.get("/api/config").json()
    assert config["mode"] == "demo" and config["production_enabled"] is False
    assert config["simulated"] is True
    assert config["stage_order"] == STAGES
    assert config["wizard"] == {
        "sequential": True, "allow_completed_review": True, "production_actions": False
    }
    assert len(config["metrics"]) == 5
    units = {metric["key"]: metric["unit"] for metric in config["metrics"]}
    assert units["latency"] == "ms"
    assert units["token_cost"] == "USD"
    models = {model["id"] for model in config["models"]}
    assert models == {
        "demo-phi-3-mini",
        "demo-phi-4-mini",
        "demo-llama-3.2-1b",
        "demo-llama-3.2-3b",
        "demo-qwen-2.5-1.5b",
        "demo-qwen-2.5-3b",
        "demo-ministral-3b",
        "demo-mistral-7b-instruct",
    }
    for model in models:
        assert Parameters(base_model=model).base_model == model
    for record in client.get("/api/runs").json()["runs"]:
        assert record["training"]["config"]["base_model"] in models
    assert client.get("/api/config?mode=production").status_code == 400
    assert client.post("/api/mode", json={"mode": "production"}).status_code == 400
    assert client.post("/api/runs", json={
        "mode": "production", "source_folder": "demo", "recipe": "context-grounded",
        "confirmed": True,
    }).status_code == 422


def test_header_presents_product_tagline():
    html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert (
        "Cut token costs with task-aligned open models—fine-tune, compare, "
        "and deploy in minutes."
    ) in html


def test_isolation_and_no_source_reads(studio, tmp_path, monkeypatch):
    client, store = studio
    sentinel = tmp_path / "production.yaml"
    sentinel.write_text("private: never open", encoding="utf-8")
    original = Path.read_text
    reads = []

    def guarded_read(path, *args, **kwargs):
        reads.append(path)
        assert path.parent == store.directory
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    record = prepare(client)
    result = wait(client, record["run_id"], "data_preparation")
    assert result["data_card"]["source_is_label_only"] is True
    assert len(client.get("/api/runs").json()["runs"]) == 4
    assert reads and sentinel not in reads


def test_no_pipeline_or_cloud_imports():
    for filename in ("main.py", "store.py", "models.py", "fixtures.py"):
        tree = ast.parse((ROOT / "app" / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert all(name.split(".")[0] not in {
                "lib", "scripts", "azure", "openai", "huggingface", "torch", "transformers"
            } for name in names)


def test_seeds_and_latest(studio):
    client, _ = studio
    records = client.get("/api/runs").json()["runs"]
    assert [record["run_id"] for record in records] == [
        "demo-training-failure", "demo-blocked", "demo-completed"
    ]
    for record in records:
        assert_shape(record)
    assert client.get("/api/runs/latest").json()["run_id"] == "demo-training-failure"
    assert records[0]["evaluation"] is None
    assert records[1]["evaluation"]["gate"]["status"] == "blocked"
    assert records[2]["deployment"]["monitoring"]["simulated"] is True
    record = prepare(client)
    assert client.get("/api/runs/latest").json()["run_id"] == record["run_id"]


def test_committed_seeds_are_rich_and_readonly():
    for run_id in ("demo-training-failure", "demo-blocked", "demo-completed"):
        path = ROOT / "runs" / "demo" / f"{run_id}.yaml"
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert_shape(record)
        assert record["data_card"]["record_count"] == sum(record["data_card"]["splits"].values())
        assert record["training"]["job_uri"].startswith("demo://")
        assert record["training"]["model_artifact"].startswith("demo://")


def test_empty_latest_and_no_reseed_until_new_factory(studio):
    client, store = studio
    for path in store.directory.glob("*.yaml"):
        path.unlink()
    assert client.get("/api/runs").json() == {"runs": []}
    assert client.get("/api/runs/latest").status_code == 404


def test_seeding_preserves_user_edits(tmp_path):
    directory = tmp_path / "demo"
    first = create_app(directory)
    record = first.state.store.read("demo-blocked")
    record["source_folder"] = "User-edited label"
    first.state.store.save(record)
    second = create_app(directory)
    assert second.state.store.read("demo-blocked")["source_folder"] == "User-edited label"
    assert len(second.state.store.list()) == 3


@pytest.mark.parametrize("path,body", [
    ("/api/runs", {"source_folder": "x", "recipe": "context-grounded"}),
    ("/api/runs/demo-blocked/training", Parameters().model_dump()),
    ("/api/runs/demo-blocked/evaluation", {}),
    ("/api/runs/demo-blocked/register", {}),
    ("/api/runs/demo-completed/deployment", {}),
])
def test_confirmation_required(studio, path, body):
    client, _ = studio
    response = client.post(path, json=body)
    assert response.status_code == 409
    assert "confirmation" in response.json()["detail"].lower()


@pytest.mark.parametrize("changes", [
    {"confirmed": "yes"}, {"recipe": "unknown"}, {"scenario": "unknown"}, {"source_folder": ""},
])
def test_preparation_validation(studio, changes):
    client, _ = studio
    body = {"confirmed": True, "recipe": "context-grounded", "scenario": "passing", "source_folder": "x"}
    body.update(changes)
    assert client.post("/api/runs", json=body).status_code == 422


@pytest.mark.parametrize("changes", [
    {"batch_size": 0}, {"learning_rate": -1}, {"epochs": 100}, {"base_model": "real-cloud-model"},
    {"max_seq_length": 9000}, {"lora_rank": 0}, {"gradient_accumulation": 0},
])
def test_training_validation(studio, changes):
    client, _ = studio
    body = {"confirmed": True, **Parameters().model_dump(), **changes}
    assert client.post("/api/runs/demo-blocked/training", json=body).status_code == 422


def test_prerequisites(studio):
    client, _ = studio
    run_id = prepare(client)["run_id"]
    assert client.post(f"/api/runs/{run_id}/training",
                       json={"confirmed": True, **Parameters().model_dump()}).status_code == 409
    assert client.post(f"/api/runs/{run_id}/evaluation", json={"confirmed": True}).status_code == 409
    assert client.post(f"/api/runs/{run_id}/register", json={
        "confirmed": True, "override": True, "confirmation": "OVERRIDE", "reason": "Demo"
    }).status_code == 409
    assert client.post("/api/runs/demo-blocked/deployment", json={"confirmed": True}).status_code == 409


def test_suggestions_are_simulated(studio):
    client, _ = studio
    response = client.post("/api/runs/demo-blocked/training/suggestions", json={})
    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "demo_fixture" and result["simulated"] is True
    assert result["parameters"] == Parameters().model_dump()
    assert "no GPT call" in result["rationale"]


def test_full_workflow(studio):
    client, store = studio
    record = evaluated(client)
    run_id = record["run_id"]
    assert record["evaluation"]["gate"]["status"] == "passed"
    assert_shape(record)
    result = client.post(f"/api/runs/{run_id}/register", json={"confirmed": True})
    assert result.status_code == 200, result.text
    registered = result.json()
    assert registered["deployment"] is None
    assert registered["current_stage"] == "deployment"
    assert registered["release"]["status"] == "candidate_deployed_zero_traffic"
    assert registered["release"]["endpoint"]["traffic_percent"] == 0
    assert registered["release"]["drift_reference"]["status"] == "created"
    registration = registered["decisions"]["registration"]
    assert registration["evidence_candidate_id"] == record["training"]["candidate_id"]
    assert registration["actor"] == "local-demo-user"
    response = client.post(f"/api/runs/{run_id}/deployment", json={
        "confirmed": True, "actor": "release-operator", "traffic_percent": 25,
    })
    assert response.status_code == 200
    final = wait(client, run_id, "deployment")
    assert final["status"] == "completed"
    assert final["deployment"]["endpoint_uri"].startswith("demo://")
    assert final["deployment"]["deployment_gate"]["status"] == "passed"
    assert final["deployment"]["deployment_gate"]["actual_requests_sent"] == 0
    assert final["deployment"]["traffic"]["candidate_percent"] == 25
    assert final["deployment"]["traffic"]["after"] == {"incumbent": 75, "candidate-v1": 25}
    assert final["deployment"]["monitoring"]["reference_uri"] == \
        final["release"]["drift_reference"]["uri"]
    assert final["decisions"]["deployment"]["actor"] == "release-operator"
    assert all(stage["status"] == "succeeded" for stage in final["stages"].values())
    assert_shape(final)
    assert store.read(run_id) == final
    assert not list(store.directory.glob("*.pending"))


@pytest.mark.parametrize("body", [
    {"override": False}, {"override": True, "confirmation": "wrong", "reason": "demo"},
    {"override": True, "confirmation": "OVERRIDE", "reason": ""},
    {"override": True, "confirmation": "OVERRIDE", "reason": "   "},
])
def test_blocked_gate_requires_deliberate_override(studio, body):
    client, _ = studio
    assert client.post("/api/runs/demo-blocked/register", json={
        "confirmed": True, **body
    }).status_code == 409


def test_override_audit_and_idempotent_registration(studio):
    client, _ = studio
    body = {"confirmed": True, "override": True, "confirmation": "OVERRIDE",
            "reason": "Presenting blocked-gate demo only", "actor": "demo-presenter"}
    first = client.post("/api/runs/demo-blocked/register", json=body)
    assert first.status_code == 200, first.text
    record = first.json()
    registration = record["decisions"]["registration"]
    assert registration["override"] is True and registration["actor"] == "demo-presenter"
    assert registration["reason"] == body["reason"]
    assert registration["gate_status"] == "blocked"
    second = client.post("/api/runs/demo-blocked/register", json={"confirmed": True})
    assert second.json() == record
    assert sum(event["action"] == "registration_overridden" for event in record["audit"]) == 1
    assert client.post("/api/runs/demo-blocked/deployment", json={"confirmed": True}).status_code == 200
    assert wait(client, "demo-blocked", "deployment")["status"] == "completed"


@pytest.mark.parametrize("body", [
    {"confirmed": True, "traffic_percent": 0},
    {"confirmed": True, "traffic_percent": 101},
    {"confirmed": True, "traffic_percent": 10, "actor": "   "},
])
def test_deployment_gate_input_validation(studio, body):
    client, _ = studio
    # Input validation runs before completed-action idempotency.
    assert client.post("/api/runs/demo-completed/deployment", json=body).status_code == 422


def test_deployment_requires_zero_traffic_release(studio):
    client, store = studio
    record = store.read("demo-blocked")
    record["decisions"]["registration"] = {
        "simulated": True, "model_name": "legacy-model", "version": "1",
        "override": True, "reason": "legacy", "actor": "legacy-user",
        "decided_at": record["updated_at"],
        "evidence_candidate_id": record["training"]["candidate_id"],
        "evidence_data_fingerprint": record["evaluation"]["data_fingerprint"],
        "evidence_test_split": record["evaluation"]["test_split"],
        "gate_status": "blocked",
    }
    record["stages"]["decision"].update(status="succeeded", progress=100)
    record["release"] = {"status": "candidate_deployed_zero_traffic",
                         "endpoint": {"traffic_percent": 10}}
    store.save(record)
    response = client.post("/api/runs/demo-blocked/deployment", json={"confirmed": True})
    assert response.status_code == 409
    assert "zero-traffic" in response.json()["detail"]


def test_legacy_registration_derives_release_without_rewriting(studio):
    _, store = studio
    path = store.path("demo-blocked")
    record = store.read("demo-blocked")
    record["decisions"]["registration"] = {
        "simulated": True, "model_name": "legacy-model", "version": "1",
        "override": True, "reason": "legacy approval", "actor": "legacy-user",
        "decided_at": record["updated_at"],
        "evidence_candidate_id": record["training"]["candidate_id"],
        "evidence_data_fingerprint": record["evaluation"]["data_fingerprint"],
        "evidence_test_split": record["evaluation"]["test_split"],
        "gate_status": "blocked",
    }
    record["release"] = None
    store.save(record)
    before = path.read_text(encoding="utf-8")
    migrated = store.read("demo-blocked")
    assert migrated["release"]["endpoint"]["traffic_percent"] == 0
    assert migrated["release"]["approval"]["actor"] == "legacy-user"
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("damage", [
    "missing", "candidate", "artifact", "test_split", "fingerprint", "sample_count", "metric",
    "failed", "pending", "false_pass", "policy", "values",
])
def test_invalid_evidence_cannot_be_overridden(studio, damage):
    client, store = studio
    record = store.read("demo-blocked")
    evidence = record["evaluation"]
    if damage == "missing":
        record["evaluation"] = None
    elif damage == "candidate":
        evidence["candidate_id"] = "demo-other-candidate"
    elif damage == "artifact":
        evidence["candidate_artifact"] = "demo://wrong/artifact"
    elif damage == "test_split":
        evidence["test_split"] = "demo://wrong/test"
    elif damage == "fingerprint":
        evidence["data_fingerprint"] = "demo-sha256:wrong"
    elif damage == "sample_count":
        evidence["models"][0]["sample_count"] = 10
    elif damage == "metric":
        evidence["models"][0]["metrics"]["accuracy"] = None
    elif damage == "failed":
        record["stages"]["evaluation"]["status"] = "failed"
    elif damage == "pending":
        evidence["gate"]["status"] = "pending"
    elif damage == "false_pass":
        evidence["gate"]["status"] = "passed"
    elif damage == "policy":
        evidence["gate"]["policy"]["id"] = "untrusted-policy"
    elif damage == "values":
        evidence["gate"]["values"]["candidate_json_validity"] = 1.0
    store.save(record)
    result = client.post("/api/runs/demo-blocked/register", json={
        "confirmed": True, "override": True, "confirmation": "OVERRIDE", "reason": "Demo",
    })
    assert result.status_code == 409, result.text
    assert store.read("demo-blocked")["decisions"]["registration"] is None


def test_duplicate_submissions_do_not_restart_or_downgrade(tmp_path):
    application = create_app(tmp_path / "demo", stage_delay=0.02)
    with TestClient(application) as client:
        record = prepare(client)
        run_id = record["run_id"]
        wait(client, run_id, "data_preparation")
        for stage, body in [
            ("training", {"confirmed": True, **Parameters().model_dump()}),
            ("evaluation", {"confirmed": True}),
            ("deployment", {"confirmed": True}),
        ]:
            if stage == "deployment":
                assert client.post(f"/api/runs/{run_id}/register", json={"confirmed": True}).status_code == 200
            first = client.post(f"/api/runs/{run_id}/{stage}", json=body).json()
            second = client.post(f"/api/runs/{run_id}/{stage}", json=body).json()
            assert first["stages"][stage]["started_at"] == second["stages"][stage]["started_at"]
            assert len(application.state.tasks) == 1
            complete = wait(client, run_id, stage)
            repeated = client.post(f"/api/runs/{run_id}/{stage}", json=body).json()
            assert repeated["stages"][stage] == complete["stages"][stage]
            assert sum(event["action"] == f"{stage}_started" for event in repeated["audit"]) == 1
        final = client.get(f"/api/runs/{run_id}").json()
        assert client.post(f"/api/runs/{run_id}/training", json={
            "confirmed": True, **Parameters().model_dump(), "epochs": 5
        }).json() == final
        assert final["status"] == "completed"


def test_progress_is_persisted(tmp_path):
    application = create_app(tmp_path / "demo", stage_delay=0.04)
    with TestClient(application) as client:
        record = prepare(client)
        run_id = record["run_id"]
        time.sleep(0.06)
        result = client.get(f"/api/runs/{run_id}").json()
        assert 0 < result["stages"]["data_preparation"]["progress"] < 100
        assert result["status"] == "running"
        assert application.state.store.read(run_id)["stages"] == result["stages"]
        assert wait(client, run_id, "data_preparation")["stages"]["data_preparation"]["progress"] == 100


def test_training_failure_is_visible_and_logged(studio, caplog):
    client, store = studio
    record = trained(client, "failure")
    assert record["status"] == "failed"
    assert "Simulated training failure" in record["stages"]["training"]["error"]
    assert record["stages"]["training"]["progress"] == 65
    assert record["evaluation"] is None
    assert "Simulated training failure" in caplog.text
    assert store.read(record["run_id"]) == record
    assert client.post(f"/api/runs/{record['run_id']}/evaluation", json={"confirmed": True}).status_code == 409


@pytest.mark.parametrize("run_id", [
    "production-run", "demo-..", "demo-evil.yaml", "demo-C%3Asecret", "demo-evil%5Coutside",
])
def test_traversal_and_invalid_ids_are_rejected(studio, run_id):
    client, _ = studio
    assert client.get(f"/api/runs/{run_id}").status_code == 400
    assert client.post(f"/api/runs/{run_id}/evaluation", json={"confirmed": True}).status_code == 400
    assert client.get("/api/runs/demo-no-such-run").status_code == 404


def test_slash_traversal_never_reads_outside_store(studio):
    client, store = studio
    for run_id in ("../production", "demo-%2E%2E%2Foutside", "demo-%2Fetc"):
        assert client.get(f"/api/runs/{run_id}").status_code in {400, 404}
    assert len(store.list()) == 3


@pytest.mark.parametrize("contents", ["not: [valid", "null", "run_id: demo-corrupted\nmode: production\n"])
def test_corruption_is_an_explicit_error(studio, contents, caplog):
    client, store = studio
    (store.directory / "demo-corrupted.yaml").write_text(contents, encoding="utf-8")
    for url in ("/api/runs/demo-corrupted", "/api/runs", "/api/runs/latest"):
        response = client.get(url)
        assert response.status_code == 500, response.text
        assert "Corrupted" in response.json()["detail"]
    assert "Unable to read demo run demo-corrupted" in caplog.text


def test_corrupted_record_does_not_hide_startup(tmp_path, caplog):
    directory = tmp_path / "demo"
    directory.mkdir()
    (directory / "demo-corrupted.yaml").write_text("[", encoding="utf-8")
    application = create_app(directory)
    with TestClient(application) as client:
        assert client.get("/api/config").status_code == 200
        assert client.get("/api/runs").status_code == 500
    assert "Unable to recover demo record" in caplog.text


def test_restart_recovers_unfinished_stage(tmp_path, caplog):
    application = create_app(tmp_path / "demo")
    store = application.state.store
    record = store.read("demo-blocked")
    record.update(status="running", current_stage="deployment")
    record["stages"]["deployment"].update(status="running", progress=45)
    store.save(record)
    restarted = create_app(store.directory)
    with TestClient(restarted) as client:
        recovered = client.get("/api/runs/demo-blocked").json()
        assert recovered["status"] == "failed"
        assert recovered["stages"]["deployment"]["status"] == "failed"
        assert recovered["stages"]["deployment"]["progress"] == 45
        assert "Interrupted" in recovered["stages"]["deployment"]["error"]
        assert recovered["deployment"] is None
        assert any(event["action"] == "stage_interrupted" for event in recovered["audit"])
    assert "Recovered interrupted" in caplog.text


def test_shutdown_cancels_and_persists_interruption(tmp_path):
    application = create_app(tmp_path / "demo", stage_delay=5)
    with TestClient(application) as client:
        run_id = prepare(client)["run_id"]
        assert application.state.tasks
    assert not application.state.tasks
    record = application.state.store.read(run_id)
    assert record["status"] == "failed"
    assert record["stages"]["data_preparation"]["status"] == "failed"
    assert "Interrupted" in record["stages"]["data_preparation"]["error"]


def test_write_failure_is_visible(studio, monkeypatch, caplog):
    client, _ = studio

    def failure(*args):
        raise OSError("Simulated disk failure")

    monkeypatch.setattr("app.store.os.replace", failure)
    response = client.post("/api/runs", json={
        "confirmed": True, "source_folder": "Demo", "recipe": "context-grounded",
    })
    assert response.status_code == 500
    assert "Unable to persist" in response.json()["detail"]
    assert "Simulated disk failure" in caplog.text


def test_static_mount_contract(studio):
    client, _ = studio
    root = client.get("/")
    static_index = client.get("/static/index.html")
    assert root.status_code == 200
    assert static_index.status_code == 200
    assert root.text == static_index.text
    assert root.headers["content-type"].startswith("text/html")
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert any(getattr(route, "path", None) == "/static" for route in client.app.routes)


def test_evaluation_showcase_exposes_real_read_only_evidence(studio):
    client, _ = studio
    response = client.get("/api/evaluation-showcase")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["sample_count"] == len(payload["records"])
    assert payload["summary"]["sample_count"] > 0
    assert set(payload["summary"]["models"]) == {"oob", "fine_tuned", "gpt"}
    assert payload["summary"]["evaluation_run_id"]
    record = payload["records"][0]
    assert {"id", "type", "query", "cot_answer", "answers"} <= set(record)
    assert set(record["answers"]) == {"oob", "fine_tuned", "gpt"}
    for answer in record["answers"].values():
        assert {"text", "latency_ms", "scores"} <= set(answer)
        assert set(answer["scores"]) == {"faithfulness", "relevancy", "correctness"}


def test_ui_uses_one_banner_and_production_style_actions(studio):
    client, _ = studio
    html = client.get("/").text
    assert html.count("Demo only. No Azure services are called.") == 1
    for label in ("Prepare data", "Suggest parameters", "Start training",
                  "Run evaluation", "Approve &amp; release candidate",
                  "Run gate &amp; route traffic"):
        assert label in html
    for marker in ("wizard-status", "wizard-back", "wizard-next",
                   "Candidate traffic", "zero-traffic candidate",
                   "Response-level model comparison", "evidence-search",
                   "CoT answer", "Fine-tuned SLM answer",
                   "AI-guided dataset creation, model recommendations"):
        assert marker in html
    script = client.get("/static/app.js").text
    assert "Quality scores and measured inference latencies are shown once" in script
    assert "function renderMetrics" not in script
    for label in ("SIMULATED", "Simulate training", "demo fixtures",
                  "Create demo run", "Register demo candidate"):
        assert label not in html
    assert client.get("/api/config").json()["production_enabled"] is False


def test_recipe_labels_explain_raft_and_rag_without_changing_api_values(studio):
    client, _ = studio
    html = client.get("/").text
    assert 'value="context-grounded">Context-grounded (RAFT)' in html
    assert 'value="context-assisted">Context-assisted (RAG)' in html
    assert 'aria-describedby="recipe-explanation"' in html
    assert "not a fine-tuning method by itself" in html
    assert client.get("/api/config").json()["recipes"] == [
        "context-grounded", "context-assisted"
    ]
