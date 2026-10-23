"""Production composition and persistence checks without cloud execution."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.store import RunStore, StoreError, validate_id
from app.workflow import new_record


def test_production_app_starts_empty_and_serves_uncached_ui(tmp_path):
    application = create_app(production_runs_dir=tmp_path / "production")
    with TestClient(application) as client:
        assert client.get("/api/production/runs").json() == {"runs": []}
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert 'id="mode"' not in response.text
        assert 'id="demo-options"' not in response.text
        for asset in ("app.js", "api.js", "styles.css"):
            response = client.get(f"/static/{asset}")
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
        config = client.get("/api/production/config").json()
        assert config["mode"] == "production" and config["simulated"] is False
        assert all(not model["id"].startswith("demo-") for model in config["models"])
        schema = client.get("/openapi.json").json()
        assert all(path.startswith("/api/production/") for path in schema["paths"])
        assert not {"Preparation", "Parameters", "Training", "Registration", "Deployment"} & schema["components"]["schemas"].keys()
    assert not (tmp_path / "demo").exists()
    assert list((tmp_path / "production").iterdir()) == []


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/config"), ("GET", "/api/runs"), ("POST", "/api/runs"),
    ("GET", "/api/evaluation-showcase"), ("POST", "/api/mode"),
    ("POST", "/api/runs/demo-completed/training"),
    ("POST", "/api/runs/demo-completed/evaluation"),
    ("POST", "/api/runs/demo-completed/register"),
    ("POST", "/api/runs/demo-completed/deployment"),
    ("POST", "/api/runs/demo-completed/training/suggestions"),
])
def test_removed_routes_are_unavailable(tmp_path, method, path):
    with TestClient(create_app(production_runs_dir=tmp_path)) as client:
        assert client.request(method, path).status_code == 404


def test_store_preserves_old_production_records_without_creating_release_evidence(tmp_path):
    store = RunStore(tmp_path)
    record = new_record("prod-existing", "data/", "context-grounded")
    record["scenario"] = "passing"  # Historical production records carried this unused field.
    record["decisions"]["registration"] = {"model_name": "existing-candidate"}
    record["evaluation"] = {"candidate_id": "existing-training-job"}
    store.save(record)
    loaded = store.read(record["run_id"])
    assert loaded == record
    assert loaded["release"] is None
    fresh = new_record("prod-new", "data/", "context-grounded")
    assert "scenario" not in fresh
    assert fresh["audit"][0]["actor"] == "local-production-user"
    assert fresh["audit"][0]["simulated"] is False


@pytest.mark.parametrize("run_id", ["demo-completed", "../prod-run", "prod-../run", "", None])
def test_store_rejects_unsafe_or_nonproduction_ids(run_id):
    with pytest.raises(ValueError, match="production run ID"):
        validate_id(run_id)


def test_store_rejects_simulated_writes(tmp_path):
    store = RunStore(tmp_path)
    record = new_record("prod-invalid", "data/", "context-grounded")
    record["simulated"] = True
    with pytest.raises(StoreError, match="Only production records"):
        store.save(record)
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_failure_keeps_saved_record(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    record = new_record("prod-atomic", "data/", "context-grounded")
    store.save(record)
    original = store.path(record["run_id"]).read_bytes()

    def fail_replace(*args):
        raise OSError("Disk unavailable")

    monkeypatch.setattr("app.store.os.replace", fail_replace)
    record["source_folder"] = "changed"
    with pytest.raises(StoreError, match="Disk unavailable"):
        store.save(record)
    assert store.path(record["run_id"]).read_bytes() == original
    assert not list(tmp_path.glob("*.pending"))


@pytest.mark.parametrize("stage", ["training", "evaluation", "decision", "deployment"])
def test_restart_retains_cloud_identity_and_warns_about_interrupted_work(tmp_path, stage):
    store = RunStore(tmp_path)
    record = new_record("prod-interrupted", "data/", "context-grounded")
    record.update(status="running", current_stage=stage)
    record["stages"][stage]["status"] = "running"
    record["training"] = {"job_name": "existing-azure-job"}
    store.save(record)
    store.recover()
    loaded = store.read(record["run_id"])
    assert loaded["status"] == "failed"
    assert loaded["stages"][stage]["status"] == "failed"
    assert any(word in loaded["stages"][stage]["error"].lower()
               for word in ("azure", "billable", "billing", "remote"))
    assert loaded["training"]["job_name"] == "existing-azure-job"


def test_corruption_is_an_explicit_api_error(tmp_path):
    application = create_app(production_runs_dir=tmp_path)
    with TestClient(application) as client:
        (tmp_path / "prod-broken.yaml").write_text("[invalid yaml", encoding="utf-8")
        response = client.get("/api/production/runs")
        assert response.status_code == 500
        assert "Corrupted or unreadable production record" in response.json()["detail"]
        assert response.json()["simulated"] is False
