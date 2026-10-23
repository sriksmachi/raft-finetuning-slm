"""Production fine-tuning contracts without billable Azure operations."""

import asyncio
import json
import runpy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.main import ROOT, create_app
from app.models import ProductionTraining
from app.training import train_model, training_status
from app.workflow import new_record

DATASET = "azureml:gpu-demo:data-211026150828"


def receipt(dataset_id=DATASET, status="Completed"):
    return {
        "job_name": "azure-training-job", "job_id": "/jobs/azure-training-job",
        "job_uri": "https://ml.azure.com/runs/azure-training-job",
        "dataset_id": dataset_id, "status": status,
        "model_artifact": "azureml://jobs/azure-training-job/outputs/model/paths/",
    }


def wait(client, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/production/runs/{run_id}")
        assert result.status_code == 200, result.text
        record = result.json()
        if record["status"] != "running":
            return record
        time.sleep(.01)
    pytest.fail("Training did not reach a terminal state")


@pytest.fixture
def studio(tmp_path):
    app = create_app(production_runs_dir=tmp_path / "production")
    source = new_record("prod-published-dataset", "data", "context-grounded")
    source.update(mode="production", simulated=False, audit=[])
    source["stages"]["data_preparation"].update(status="succeeded", progress=100)
    source["data_card"] = {
        "simulated": False, "asset_reference": DATASET,
        "recipe": "context-grounded", "record_count": 7,
        "split_counts": {"train": 5, "validation": 1, "test": 1},
        "fingerprint": "test-fingerprint", "data_fingerprint": "test-fingerprint",
    }
    app.state.production.store.save(source)

    async def train(body, dataset_id, directory, on_submitted):
        on_submitted(receipt(dataset_id, "Submitted"))
        await asyncio.sleep(.02)
        return receipt(dataset_id)

    app.state.production.training_runner = AsyncMock(side_effect=train)
    app.state.production.training_status_reader = Mock(side_effect=lambda job_name: next(
        record["training"]["job_status"]
        for record in app.state.production.store.list()
        if (record.get("training") or {}).get("job_name") == job_name
    ))
    with TestClient(app) as client:
        yield client, app


@pytest.mark.parametrize("azure_status,run_status,stage_status", [
    ("Running", "running", "running"), ("Queued", "running", "running"),
    ("Completed", "ready", "succeeded"), ("Failed", "failed", "failed"),
    ("Canceled", "cancelled", "cancelled"), ("Cancelled", "cancelled", "cancelled"),
])
def test_select_running_training_refreshes_and_persists_azure_status(
    studio, azure_status, run_status, stage_status,
):
    client, app = studio
    api = app.state.production
    record = new_record("prod-away-training", DATASET, "context-grounded")
    record.update(run_type="training", current_stage="training", status="running")
    record["stages"]["training"]["status"] = "running"
    record["training"] = {
        "job_name": "azure-training-job", "job_status": "Submitted",
        "dataset_run_id": None, "config": {},
    }
    api.store.save(record)
    api.training_status_reader = Mock(return_value=azure_status)
    response = client.get("/api/production/runs/prod-away-training")
    assert response.status_code == 200, response.text
    saved = api.store.read(record["run_id"])
    assert response.json() == saved
    assert saved["training"]["job_status"] == azure_status
    assert saved["status"] == run_status
    assert saved["stages"]["training"]["status"] == stage_status
    api.training_status_reader.assert_called_once_with("azure-training-job")
    assert azure_status in client.get(
        "/api/production/runs/prod-away-training/log",
    ).json()["current_status"]
    api.training_status_reader.assert_called_once()
    if run_status != "running":
        assert saved["stages"]["training"]["completed_at"]
        assert client.get("/api/production/runs/prod-away-training").json() == saved
        api.training_status_reader.assert_called_once()
    if azure_status == "Completed":
        assert saved["stages"]["training"]["progress"] == 100


@pytest.mark.parametrize("run_type,status,job_name", [
    ("training", "ready", "azure-training-job"),
    ("training", "failed", "azure-training-job"),
    ("training", "cancelled", "azure-training-job"),
    ("training", "running", None),
    ("evaluation", "running", "azure-training-job"),
    ("release", "running", "azure-training-job"),
    (None, "running", "azure-training-job"),
])
def test_only_running_submitted_training_queries_azure(studio, run_type, status, job_name):
    client, app = studio
    api = app.state.production
    record = new_record("prod-no-refresh", DATASET, "context-grounded")
    record.update(run_type=run_type, status=status)
    record["training"] = {"job_name": job_name, "job_status": "Failed" if status == "failed" else "Running"}
    record["stages"]["training"]["status"] = "running" if status == "running" else "failed"
    api.store.save(record)
    api.training_status_reader = Mock()
    assert client.get("/api/production/runs/prod-no-refresh").status_code == 200
    api.training_status_reader.assert_not_called()


@pytest.mark.parametrize("azure_status,expected_status", [
    ("Completed", "ready"), ("Running", "running"),
    ("Failed", "failed"), ("Canceled", "cancelled"),
])
@pytest.mark.parametrize("error", [
    "Training command exited with code 1. See the recent training job logs.",
    "Monitoring interrupted. The Azure ML job may still be running; check its job link before resubmitting.",
])
def test_failed_local_monitor_reconciles_unresolved_cloud_job(studio, azure_status, expected_status, error):
    client, app = studio
    api = app.state.production
    record = new_record("prod-monitor-failed", DATASET, "context-grounded")
    record.update(run_type="training", current_stage="training", status="failed")
    record["stages"]["training"].update(status="failed", error=error, completed_at=record["updated_at"])
    record["training"] = {
        "job_name": "cool_basil_nt640qk5hf", "job_status": "Running", "config": {},
        "candidate_id": "cool_basil_nt640qk5hf",
        "model_artifact": "azureml://jobs/cool_basil_nt640qk5hf/outputs/model/paths/",
    }
    api.store.save(record)
    api.training_status_reader = Mock(return_value=azure_status)
    response = client.get("/api/production/runs/prod-monitor-failed")
    assert response.status_code == 200, response.text
    saved = api.store.read(record["run_id"])
    assert saved["status"] == response.json()["status"] == expected_status
    assert saved["training"]["job_status"] == azure_status
    assert saved["training"]["candidate_id"] == "cool_basil_nt640qk5hf"
    api.training_status_reader.assert_called_once_with("cool_basil_nt640qk5hf")
    if azure_status in {"Completed", "Running"}:
        assert saved["stages"]["training"]["error"] is None
    if azure_status == "Running":
        assert saved["stages"]["training"]["completed_at"] is None
        assert saved["stages"]["training"]["status"] == "running"
        assert client.post("/api/production/runs/prod-monitor-failed/cancel").status_code == 200
        assert api.store.read(record["run_id"])["status"] == "cancelled"
        assert client.get("/api/production/runs/prod-monitor-failed").json()["status"] == "cancelled"
        api.training_status_reader.assert_called_once()
    else:
        client.get("/api/production/runs/prod-monitor-failed")
        api.training_status_reader.assert_called_once()


def test_azure_refresh_failure_is_visible_and_preserves_running_status(studio):
    client, app = studio
    api = app.state.production
    record = new_record("prod-refresh-failure", DATASET, "context-grounded")
    record.update(run_type="training", status="running", current_stage="training")
    record["training"] = {"job_name": "azure-training-job", "job_status": "Running"}
    record["stages"]["training"]["status"] = "running"
    api.store.save(record)
    api.training_status_reader = Mock(side_effect=RuntimeError("Azure unavailable"))
    response = client.get("/api/production/runs/prod-refresh-failure")
    assert response.status_code == 502
    assert "Azure ML" in response.json()["detail"]
    assert api.store.read(record["run_id"]) == record
    api.training_status_reader = Mock(return_value="Completed")
    assert client.get("/api/production/runs/prod-refresh-failure").json()["status"] == "ready"


def test_azure_response_does_not_overwrite_a_concurrent_cancellation(studio):
    client, app = studio
    api = app.state.production
    record = new_record("prod-refresh-race", DATASET, "context-grounded")
    record.update(run_type="training", status="running", current_stage="training")
    record["training"] = {"job_name": "azure-training-job", "job_status": "Running"}
    record["stages"]["training"]["status"] = "running"
    api.store.save(record)

    def cancel_during_lookup(job_name):
        api.mark_cancelled(record)
        return "Completed"

    api.training_status_reader = Mock(side_effect=cancel_during_lookup)
    assert client.get("/api/production/runs/prod-refresh-race").json()["status"] == "cancelled"
    assert api.store.read(record["run_id"])["status"] == "cancelled"


def test_reconciliation_updates_the_live_monitor_without_stale_receipt_overwrites(studio):
    client, app = studio
    api = app.state.production
    advance = asyncio.Queue()

    async def train(body, dataset_id, directory, on_submitted):
        on_submitted(receipt(dataset_id, "Running"))
        await advance.get()
        on_submitted(receipt(dataset_id, "Running"))
        await advance.get()
        return receipt(dataset_id)

    api.training_runner = train
    run_id = client.post("/api/production/training", json={
        "confirmed": True, "dataset_id": DATASET,
    }).json()["run_id"]
    api.training_status_reader = Mock(return_value="Completed")
    assert client.get(f"/api/production/runs/{run_id}").json()["status"] == "ready"
    client.portal.call(advance.put_nowait, True)
    saved = api.store.read(run_id)
    assert saved["status"] == "ready"
    assert saved["training"]["job_status"] == "Completed"
    client.portal.call(advance.put_nowait, True)
    client.portal.call(asyncio.sleep, .02)
    assert api.store.read(run_id)["stages"]["training"]["status"] == "succeeded"


@pytest.mark.parametrize("status", ["Completed", "", None])
def test_status_reader_uses_the_configured_workspace_and_closes_credential(monkeypatch, status):
    from lib.config import AzureMLConfig

    client = Mock()
    client.jobs.get.return_value.status = status
    credential = MagicMock()
    credential.__enter__.return_value = credential
    config = Mock()
    config.create_credential.return_value = credential
    config.create_ml_client.return_value = client
    monkeypatch.setattr(AzureMLConfig, "from_env", Mock(return_value=config))
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    if status:
        assert training_status("azure-training-job") == status
    else:
        with pytest.raises(ValueError, match="empty training job status"):
            training_status("azure-training-job")
    client.jobs.get.assert_called_once_with("azure-training-job")
    config.create_ml_client.assert_called_once_with(credential=credential)
    credential.__exit__.assert_called_once()


def test_training_from_published_run_persists_job_and_dataset_lineage(studio, tmp_path):
    client, app = studio
    before = client.get("/api/production/runs/prod-published-dataset").json()
    response = client.post("/api/production/training", json={
        "confirmed": True, "dataset_run_id": before["run_id"],
        "seed": 7, "load_in_4bit": True, "epochs": .5,
    })
    assert response.status_code == 202, response.text
    result = wait(client, response.json()["run_id"])
    assert result["stages"]["training"]["status"] == "succeeded"
    assert result["training"]["dataset_id"] == DATASET
    assert result["training"]["dataset_run_id"] == before["run_id"]
    assert result["training"]["config"]["seed"] == 7
    assert result["training"]["config"]["load_in_4bit"] is True
    assert result["training"]["job_status"] == "Completed"
    assert result["training"]["job_uri"].startswith("https://ml.azure.com/")
    assert result["data_card"] == before["data_card"]
    assert all(event["simulated"] is False for event in result["audit"])
    assert client.get("/api/production/runs/prod-published-dataset").json() == before
    app.state.production.training_runner.assert_awaited_once()
    assert client.get(f"/api/runs/{result['run_id']}").status_code == 404
    log = client.get(f"/api/production/runs/{result['run_id']}/log").json()
    assert log["stage"] == "training" and log["stage_status"] == "succeeded"
    reopened = create_app(production_runs_dir=tmp_path / "production")
    with TestClient(reopened) as other:
        assert other.get(f"/api/production/runs/{result['run_id']}").json()["training"] == result["training"]


def test_manual_dataset_can_train_without_a_preparation_run(studio):
    client, _ = studio
    response = client.post("/api/production/training", json={
        "confirmed": True, "dataset_id": "azureml:already-published:1",
    })
    assert response.status_code == 202, response.text
    result = wait(client, response.json()["run_id"])
    assert result["training"]["dataset_id"] == "azureml:already-published:1"
    assert result["training"]["dataset_run_id"] is None
    assert result["data_card"] is None
    assert result["stages"]["training"]["status"] == "succeeded"


def test_training_run_log_updates_while_job_is_running(studio):
    client, app = studio
    advance = asyncio.Queue()

    async def train(body, dataset_id, directory, on_submitted):
        for status in ("Submitted", "Running"):
            on_submitted(receipt(dataset_id, status))
            await advance.get()
        return receipt(dataset_id)

    app.state.production.training_runner = train
    response = client.post("/api/production/training", json={"confirmed": True, "dataset_id": DATASET})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    for expected in ("Submitted", "Running"):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get(f"/api/production/runs/{run_id}/log")
            assert response.status_code == 200
            log = response.json()
            if log["current_status"].endswith(f": {expected}"):
                break
            time.sleep(.01)
        else:
            pytest.fail(f"Training log did not publish {expected}")
        assert response.headers["cache-control"] == "no-store"
        assert log["stage"] == "training" and log["stage_status"] == "running"
        assert client.get(f"/api/production/runs/{run_id}").json()["training"]["job_status"] == expected
        assert "Submitting billable Azure ML fine-tuning job." in log["log"]
        client.portal.call(advance.put_nowait, True)
    wait(client, run_id)
    log = client.get(f"/api/production/runs/{run_id}/log").json()
    assert log["stage_status"] == "succeeded"
    assert "Fine-tuning completed" in log["current_status"]
    assert all(f": {status}" in log["log"] for status in ("Submitted", "Running", "Completed"))


@pytest.mark.parametrize("changes,status", [
    ({"confirmed": False}, 409),
    ({"dataset_id": None}, 422),
    ({"dataset_id": "azureml:name:latest/unsafe"}, 422),
    ({"dataset_id": "azureml:name:" + "v" * 31}, 422),
    ({"dataset_id": "https://external/data.jsonl"}, 422),
    ({"dataset_run_id": "prod-published-dataset"}, 422),
    ({"dataset_recipe": "context-assisted"}, 422),
    ({"base_model": "model; echo unsafe"}, 422),
    ({"chat_template": "template --override"}, 422),
    ({"lora_adapter_name": "../adapter"}, 422),
    ({"batch_size": 0}, 422), ({"gradient_accumulation": True}, 422),
    ({"seed": -1}, 422), ({"epochs": 0}, 422),
    ({"learning_rate": .5}, 422), ({"load_in_4bit": "yes"}, 422),
    ({"compute_name": "browser-must-not-configure-infrastructure"}, 422),
])
def test_invalid_training_never_submits_a_job(studio, changes, status):
    client, app = studio
    response = client.post("/api/production/training", json={
        "confirmed": True, "dataset_id": DATASET, **changes,
    })
    assert response.status_code == status, response.text
    app.state.production.training_runner.assert_not_awaited()


def test_unpublished_and_missing_runs_are_rejected(studio):
    client, app = studio
    record = app.state.production.store.read("prod-published-dataset")
    record["stages"]["data_preparation"]["status"] = "failed"
    app.state.production.store.save(record)
    assert client.post("/api/production/training", json={
        "confirmed": True, "dataset_run_id": record["run_id"],
    }).status_code == 409
    assert client.post("/api/production/training", json={
        "confirmed": True, "dataset_run_id": "prod-missing",
    }).status_code == 404
    app.state.production.training_runner.assert_not_awaited()


@pytest.mark.parametrize("run_type", ["training", "evaluation"])
def test_derived_runs_do_not_replace_the_preparation_dataset_source(studio, run_type):
    client, app = studio
    record = app.state.production.store.read("prod-published-dataset")
    record["run_type"] = run_type
    app.state.production.store.save(record)
    response = client.post("/api/production/training", json={
        "confirmed": True, "dataset_run_id": record["run_id"],
    })
    assert response.status_code == 409
    app.state.production.training_runner.assert_not_awaited()


def test_submission_identity_survives_monitoring_interruption(studio):
    client, app = studio

    async def blocked(body, dataset_id, directory, on_submitted):
        on_submitted(receipt(dataset_id, "Submitted"))
        await asyncio.Event().wait()

    app.state.production.training_runner = blocked
    response = client.post("/api/production/training", json={"confirmed": True, "dataset_id": DATASET})
    run_id = response.json()["run_id"]
    active = client.get(f"/api/production/runs/{run_id}").json()
    assert active["training"]["job_name"] == "azure-training-job"
    assert client.post("/api/production/training", json={
        "confirmed": True, "dataset_id": DATASET,
    }).status_code == 409
    client.portal.call(app.state.production.close)
    failed = app.state.production.store.read(run_id)
    assert "may still be running" in failed["stages"]["training"]["error"]
    assert failed["training"]["job_name"] == "azure-training-job"
    app.state.production.training_status_reader = Mock(return_value="Completed")
    completed = client.get(f"/api/production/runs/{run_id}").json()
    assert completed["status"] == "ready"
    assert completed["training"]["job_name"] == "azure-training-job"
    assert completed["stages"]["training"]["error"] is None


def test_training_failure_is_explicit_and_retains_job_identity(studio):
    client, app = studio

    async def failed(body, dataset_id, directory, on_submitted):
        on_submitted(receipt(dataset_id, "Submitted"))
        raise RuntimeError("GPU job failed: out of memory")

    app.state.production.training_runner = failed
    app.state.production.training_status_reader = Mock(return_value="Failed")
    run_id = client.post("/api/production/training", json={
        "confirmed": True, "dataset_id": DATASET,
    }).json()["run_id"]
    result = wait(client, run_id)
    assert result["status"] == "failed"
    assert "out of memory" in result["stages"]["training"]["error"]
    assert result["training"]["job_uri"].startswith("https://ml.azure.com/")


def test_adapter_forwards_every_supported_trainer_setting(tmp_path, monkeypatch):
    body = ProductionTraining(
        confirmed=True, dataset_id=DATASET, base_model="Llama-3.1-8B-Instruct",
        max_seq_length=4096, learning_rate=.0001, epochs=.5, batch_size=3,
        gradient_accumulation=8, lora_rank=32, lora_adapter_name="test-adapter",
        chat_template="llama-3.1", seed=17, load_in_4bit=True,
    )

    async def finished():
        (tmp_path / "training_receipt.json").write_text(json.dumps(receipt()), encoding="utf-8")
        return 0

    process = SimpleNamespace(wait=AsyncMock(side_effect=finished), returncode=0, terminate=Mock())
    launch = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    submitted = Mock()
    result = asyncio.run(train_model(body, DATASET, tmp_path, submitted))
    args = launch.call_args.args
    for flag, expected in {
        "--base-model-name": body.base_model, "--max-seq-length": "4096",
        "--learning-rate": "0.0001", "--epochs": "0.5", "--batch-size": "3",
        "--gradient-accumulation-steps": "8", "--lora-rank": "32",
        "--lora-adapter-name": "test-adapter", "--chat-template": "llama-3.1",
        "--seed": "17", "--data-asset-name": "gpu-demo", "--data-asset-version": "data-211026150828",
    }.items():
        assert args[args.index(flag) + 1] == expected
    assert "--wait" in args and "--load-in-4bit" in args
    assert args[1] == "-u"
    assert result["status"] == "Completed"
    submitted.assert_called_once()


@pytest.mark.parametrize("code,payload", [
    (1, receipt(status="Failed")), (0, None), (0, receipt(status="Failed")),
    (0, {"status": "Completed"}), (0, receipt(dataset_id="azureml:different:1")),
])
def test_adapter_rejects_failed_or_incomplete_receipts(tmp_path, monkeypatch, code, payload):
    async def finished():
        if payload is not None:
            (tmp_path / "training_receipt.json").write_text(json.dumps(payload), encoding="utf-8")
        return code

    process = SimpleNamespace(wait=AsyncMock(side_effect=finished), returncode=code, terminate=Mock())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(train_model(ProductionTraining(dataset_id=DATASET), DATASET, tmp_path, Mock()))


def test_stage_02_writes_atomic_job_receipt(tmp_path):
    write = runpy.run_path(str(ROOT / "scripts" / "02_azureml_fine_tuning.py"))["write_job_receipt"]
    path = tmp_path / "receipt.json"
    job = SimpleNamespace(name="job-1", id="/jobs/job-1", studio_url="https://ml.azure.com/runs/job-1")
    args = SimpleNamespace(receipt_path=str(path), data_asset_name="dataset", data_asset_version="1")
    write(job, args, "Submitted")
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "Submitted"
    write(job, args, "Completed")
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "Completed"
    assert not path.with_suffix(".pending").exists()


@pytest.mark.parametrize("terminal", ["Completed", "Failed", "Canceled"])
def test_stage_02_publishes_status_heartbeats_every_ten_seconds(tmp_path, monkeypatch, caplog, terminal):
    namespace = runpy.run_path(str(ROOT / "scripts" / "02_azureml_fine_tuning.py"))
    monitor = namespace["monitor_training_job"]
    statuses = ["Queued", "Preparing", "Running", "Running", terminal]
    jobs = [SimpleNamespace(
        name="job-1", id="/jobs/job-1", studio_url="https://ml.azure.com/runs/job-1", status=status,
    ) for status in statuses]
    lookup = Mock(side_effect=jobs)
    sleep = Mock()
    writer = Mock(wraps=namespace["write_job_receipt"])
    monkeypatch.setitem(monitor.__globals__, "get_job_after_stream", lookup)
    monkeypatch.setitem(monitor.__globals__, "write_job_receipt", writer)
    monkeypatch.setattr("lib.azureml_ops.time", SimpleNamespace(sleep=sleep))
    args = SimpleNamespace(receipt_path=str(tmp_path / "receipt.json"),
                           data_asset_name="dataset", data_asset_version="1")
    client = Mock()
    with caplog.at_level("INFO", logger="azureml_fine_tuning"):
        result = monitor(client, "job-1", args)
    client.jobs.stream.assert_called_once_with("job-1")
    assert result.status == terminal
    assert [call.args[2] for call in writer.call_args_list] == statuses
    assert [call.args for call in sleep.call_args_list] == [(10,)] * 4
    assert [record.getMessage() for record in caplog.records] == [
        f"Azure ML fine-tuning job job-1: {status}" for status in statuses
    ]
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))["status"] == terminal


def test_stage_02_monitoring_errors_are_not_hidden(tmp_path, monkeypatch):
    namespace = runpy.run_path(str(ROOT / "scripts" / "02_azureml_fine_tuning.py"))
    monitor = namespace["monitor_training_job"]
    monkeypatch.setitem(
        monitor.__globals__, "get_job_after_stream", Mock(side_effect=RuntimeError("Azure unavailable")),
    )
    with pytest.raises(RuntimeError, match="Azure unavailable"):
        monitor(Mock(), "job-1", SimpleNamespace(receipt_path=str(tmp_path / "receipt.json")))
    assert not (tmp_path / "receipt.json").exists()


def test_stage_02_streams_remote_logs_and_surfaces_stream_errors(tmp_path, monkeypatch, capsys):
    namespace = runpy.run_path(str(ROOT / "scripts" / "02_azureml_fine_tuning.py"))
    monitor = namespace["monitor_training_job"]
    monkeypatch.setitem(monitor.__globals__, "get_job_after_stream", Mock(return_value=SimpleNamespace(
        name="job-1", id="/jobs/job-1", status="Completed",
    )))
    args = SimpleNamespace(receipt_path=str(tmp_path / "receipt.json"),
                           data_asset_name="dataset", data_asset_version="1")
    client = Mock()
    client.jobs.stream.side_effect = lambda name: print("Training epoch 1: loss 0.42", flush=True)
    monitor(client, "job-1", args)
    assert "Training epoch 1: loss 0.42" in capsys.readouterr().out
    client.jobs.stream.side_effect = RuntimeError("Azure log streaming disconnected")
    with pytest.raises(RuntimeError, match="Azure log streaming disconnected"):
        monitor(client, "job-1", args)


def test_frontend_assets_are_not_cached(studio):
    client, _ = studio
    for path in ("/", "/static/app.js", "/static/styles.css"):
        assert client.get(path).headers["cache-control"] == "no-store"


def test_training_ledger_records_resolved_adapter_and_template(monkeypatch):
    namespace = runpy.run_path(str(ROOT / "scripts" / "02_azureml_fine_tuning.py"))
    monkeypatch.setattr("sys.argv", ["stage02"])
    args = namespace["parse_args"]()
    submit = namespace["submit_training"]
    job = SimpleNamespace(name="trained-job", id="/jobs/trained-job", display_name="test")
    remote = Mock(return_value=job)
    ledger = Mock()
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", Mock())
    monkeypatch.setitem(submit.__globals__, "submit_finetuning_job", remote)
    monkeypatch.setitem(submit.__globals__, "record_run", ledger)
    submit(Mock(), args, "azureml:environment:1")
    parameters = ledger.call_args.kwargs["training_parameters"]
    assert parameters["lora_adapter_name"] == remote.call_args.kwargs["lora_adapter_name"]
    assert parameters["chat_template"] == remote.call_args.kwargs["chat_template"]
    assert parameters["lora_adapter_name"] and parameters["chat_template"]
