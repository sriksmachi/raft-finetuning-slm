"""Candidate-bound production evaluation without Azure or judge calls."""

import asyncio
import csv
import hashlib
import json
import runpy
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import yaml
from fastapi.testclient import TestClient

from app.evaluation import evaluate_model, evaluation_result, refresh_evaluation
from app.main import ROOT, create_app
from app.workflow import new_record


def evaluation_receipt(job, mode="slm_evaluation", baseline="baseline:1", status="Completed"):
    return {
        "job_name": f"{mode}-job", "job_id": f"/jobs/{mode}-job",
        "job_uri": f"https://ml.azure.com/runs/{mode}-job",
        "status": status, "mode": mode, "dataset_id": job["dataset_id"],
        "training_job_name": job["job_name"] if mode == "slm_evaluation" else None,
        "baseline_id": baseline, "judge_deployment": "unit-judge",
    }


def write_evidence(directory, job, row_changes=None, manifest_changes=None):
    directory.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "test-1", "type": "oracle", "question": "What is HBM?", "cot_answer": "High-bandwidth memory.",
        "source_index": 0, "evaluation_framework": "deepeval", "rag_judge_deployment": "unit-judge",
    }
    for model in ("oob", "fine_tuned", "gpt"):
        row.update({
            f"{model}_slm_response": "High-bandwidth memory.", f"{model}_slm_latency_ms": 123,
            f"{model}_inference_error": "", f"{model}_rag_error": "",
            **{f"{model}_{score}_score": .8 for score in ("faithfulness", "relevancy", "correctness")},
        })
    row.update(row_changes or {})
    path = directory / "predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    manifest = {
        "schema_version": 1, "stage": 3, "framework": "deepeval",
        "training_job_name": job["job_name"], "evaluation_run_id": "slm_evaluation-job",
        "created_at": "2026-10-21T17:00:00Z", "test_sha256": "a" * 64,
        "predictions_file": path.name, "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **(manifest_changes or {}),
    }
    (directory / "evaluation_evidence.json").write_text(json.dumps(manifest), encoding="utf-8")


def wait(client, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = client.get(f"/api/production/runs/{run_id}").json()
        if record["status"] != "running":
            return record
        time.sleep(.01)
    pytest.fail("Evaluation did not finish")


@pytest.fixture
def studio(tmp_path):
    ledger = tmp_path / "runs"
    app = create_app(production_runs_dir=ledger / "production")
    original = {
        "job_name": "trained-job", "status": "completed", "submitted_at": "2026-10-21T12:00:00Z",
        "base_model": "Llama-3.2-3B-Instruct", "base_model_source": "foundry",
        "data_asset": "gpu-demo:1", "sample_count": 40,
        "training_parameters": {"epochs": 2, "lora_adapter_name": "test-adapter"},
        "llm_baseline_data_asset": "baseline:1",
    }
    path = ledger / "trained-job.yaml"
    path.write_text(yaml.safe_dump(original), encoding="utf-8")

    async def evaluate(job, directory, on_submitted):
        on_submitted(evaluation_receipt(job, status="Running"))
        await asyncio.sleep(.02)
        write_evidence(directory / "outputs", job)
        return evaluation_result(directory / "outputs", job, evaluation_receipt(job))

    app.state.production.evaluation_runner = AsyncMock(side_effect=evaluate)
    app.state.production.evaluation_refresher = Mock(side_effect=AssertionError("Unexpected Azure recovery"))
    with TestClient(app) as client:
        yield client, app, path


def test_catalogue_returns_selected_training_metadata_and_no_demo_jobs(studio):
    client, _, path = studio
    response = client.get("/api/production/training-jobs")
    assert response.status_code == 200
    jobs = response.json()["jobs"]
    assert len(jobs) == 1
    job = jobs[0]
    assert job["job_name"] == path.stem and job["eligible"] is True
    assert job["training_date"] == "2026-10-21T12:00:00Z"
    assert job["base_model"] == "Llama-3.2-3B-Instruct"
    assert job["dataset_version"] == "1"
    assert job["sample_count"] == 40 and job["epochs"] == 2
    assert job["adapter_name"] == "test-adapter" and job["baseline_id"] == "baseline:1"


def test_evaluation_persists_bound_metrics_logs_and_source_is_unchanged(studio, tmp_path):
    client, app, path = studio
    before = path.read_bytes()
    response = client.post("/api/production/evaluation", json={"job_name": path.stem, "confirmed": True})
    assert response.status_code == 202, response.text
    record = wait(client, response.json()["run_id"])
    assert record["stages"]["evaluation"]["status"] == "succeeded"
    assert record["stages"]["decision"]["status"] == "not_started"
    evidence = record["evaluation"]
    assert evidence["candidate_id"] == path.stem and evidence["dataset_id"] == "azureml:gpu-demo:1"
    assert evidence["judge_deployment"] == "unit-judge"
    assert evidence["summary"]["sample_count"] == 1
    assert evidence["metrics"][0]["fine_tuned"] == .8
    assert evidence["metrics"][-1]["fine_tuned"] is None
    assert evidence["records"][0]["answers"]["gpt"]["text"] == "High-bandwidth memory."
    log = client.get(f"/api/production/runs/{record['run_id']}/log").json()
    assert log["stage"] == "evaluation" and "completed" in log["current_status"]
    assert path.read_bytes() == before
    app.state.production.evaluation_runner.assert_awaited_once()
    assert client.get(f"/api/runs/{record['run_id']}").status_code == 404
    assert all(event["simulated"] is False for event in record["audit"])
    reopened = create_app(production_runs_dir=path.parent / "production")
    with TestClient(reopened) as other:
        assert other.get(f"/api/production/runs/{record['run_id']}").json()["evaluation"] == evidence


def test_catalogue_merges_studio_training_metadata_without_duplicate_job_ids(studio):
    client, app, _ = studio
    record = new_record("prod-trained", "azureml:gpu-demo:1", "context-grounded")
    record.update(mode="production", simulated=False, run_type="training")
    record["stages"]["training"].update(status="succeeded", progress=100)
    record["training"] = {
        "job_name": "trained-job", "job_status": "Completed", "dataset_id": "azureml:gpu-demo:1",
        "config": {"base_model": "Llama-3.2-3B-Instruct", "epochs": 3, "lora_adapter_name": "studio-adapter"},
    }
    record["data_card"] = {"record_count": 64, "asset_reference": "azureml:gpu-demo:1"}
    app.state.production.store.save(record)
    jobs = client.get("/api/production/training-jobs").json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["training_run_id"] == record["run_id"]
    assert jobs[0]["epochs"] == 3 and jobs[0]["adapter_name"] == "studio-adapter"
    assert jobs[0]["sample_count"] == 64 and jobs[0]["baseline_id"] == "baseline:1"


@pytest.mark.parametrize("body,status", [
    ({"job_name": "trained-job"}, 409),
    ({"job_name": "missing", "confirmed": True}, 404),
    ({"job_name": "../other", "confirmed": True}, 422),
    ({"job_name": "trained-job", "confirmed": True, "judge_deployment": "browser-override"}, 422),
])
def test_invalid_evaluation_is_not_submitted(studio, body, status):
    client, app, _ = studio
    assert client.post("/api/production/evaluation", json=body).status_code == status
    app.state.production.evaluation_runner.assert_not_awaited()


def test_failed_and_incomplete_training_is_not_selectable(studio):
    client, app, path = studio
    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    record["status"] = "failed"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    assert client.get("/api/production/training-jobs").json()["jobs"] == []
    assert client.post("/api/production/evaluation", json={"job_name": path.stem, "confirmed": True}).status_code == 404
    record.update(status="completed", data_asset=None)
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    assert client.get("/api/production/training-jobs").json()["jobs"][0]["eligible"] is False
    assert client.post("/api/production/evaluation", json={"job_name": path.stem, "confirmed": True}).status_code == 409
    app.state.production.evaluation_runner.assert_not_awaited()


@pytest.mark.parametrize("status", ["candidate_deployed_zero_traffic", "traffic_routed"])
def test_previously_released_training_jobs_remain_selectable(studio, status):
    client, _, path = studio
    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    record["status"] = status
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    assert client.get("/api/production/training-jobs").json()["jobs"][0]["eligible"] is True


def test_monitoring_interruption_keeps_remote_job_link_and_blocks_duplicate(studio):
    client, app, path = studio

    async def blocked(job, directory, on_submitted):
        on_submitted(evaluation_receipt(job, status="Running"))
        await asyncio.Event().wait()

    app.state.production.evaluation_runner = blocked
    body = {"job_name": path.stem, "confirmed": True}
    record = client.post("/api/production/evaluation", json=body).json()
    assert client.post("/api/production/evaluation", json=body).status_code == 409
    log = client.get(f"/api/production/runs/{record['run_id']}/log").json()
    assert log["stage_status"] == "running" and "Running" in log["current_status"]
    client.portal.call(app.state.production.close)
    failed = app.state.production.store.read(record["run_id"])
    assert "may still be running" in failed["stages"]["evaluation"]["error"]
    assert failed["evaluation"]["job_uri"].startswith("https://ml.azure.com/")


def test_failure_surfaces_without_success_metrics(studio):
    client, app, path = studio
    app.state.production.evaluation_runner = AsyncMock(side_effect=RuntimeError("Judge quota exceeded"))
    record = client.post("/api/production/evaluation", json={"job_name": path.stem, "confirmed": True}).json()
    result = wait(client, record["run_id"])
    assert result["status"] == "failed" and "Judge quota" in result["stages"]["evaluation"]["error"]
    assert "metrics" not in result["evaluation"]


@pytest.mark.parametrize("row_changes,manifest_changes", [
    ({}, {"training_job_name": "different"}),
    ({}, {"evaluation_run_id": "different"}),
    ({}, {"predictions_sha256": "0" * 64}),
    ({}, {"predictions_file": "../outside.csv"}),
    ({"fine_tuned_correctness_score": "NaN"}, {}),
    ({"fine_tuned_faithfulness_score": 1.2}, {}),
    ({"fine_tuned_slm_latency_ms": 0}, {}),
    ({"gpt_rag_error": "judge failed"}, {}),
    ({"rag_judge_deployment": "unrequested-judge"}, {}),
    ({"evaluation_framework": "unverified"}, {}),
    ({"source_index": 2}, {}),
])
def test_metrics_require_complete_candidate_bound_judge_evidence(tmp_path, row_changes, manifest_changes):
    job = {"job_name": "trained-job", "dataset_id": "azureml:gpu-demo:1"}
    write_evidence(tmp_path, job, row_changes, manifest_changes)
    with pytest.raises(ValueError):
        evaluation_result(tmp_path, job, evaluation_receipt(job))


def test_p95_latency_matches_pipeline_interpolated_quantile(tmp_path):
    job = {"job_name": "trained-job", "dataset_id": "azureml:gpu-demo:1"}
    write_evidence(tmp_path, job)
    path = tmp_path / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        for index, latency in enumerate((100, 200, 300)):
            writer.writerow({
                **row, "id": f"test-{index}", "source_index": index,
                **{f"{model}_slm_latency_ms": latency for model in ("oob", "fine_tuned", "gpt")},
            })
    manifest_path = tmp_path / "evaluation_evidence.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predictions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = evaluation_result(tmp_path, job, evaluation_receipt(job))
    for model in ("oob", "fine_tuned", "gpt"):
        assert result["summary"]["models"][model]["latency_avg_ms"] == 200
        assert result["summary"]["models"][model]["latency_p95_ms"] == pytest.approx(290)


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("locked", [True, False])
def test_adapter_reuses_baseline_or_generates_one_then_evaluates(studio, tmp_path, monkeypatch, cached, locked):
    client, _, _ = studio
    job = client.get("/api/production/training-jobs").json()["jobs"][0]
    if not cached:
        job["baseline_id"] = None
    commands = []
    read_text = Path.read_text
    locks = []

    def receipt_read(path, *args, **kwargs):
        if locked and path.name == "slm_evaluation_receipt.json" and len(locks) < 2:
            locks.append(True)
            raise PermissionError("Windows receipt sharing violation")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", receipt_read)
    monkeypatch.setattr("lib.job_receipts.time.sleep", Mock())

    async def launch(*command, **kwargs):
        commands.append(command)
        mode = command[command.index("--mode") + 1]
        path = Path(command[command.index("--receipt-path") + 1])

        async def finish():
            path.write_text(json.dumps(evaluation_receipt(job, mode)), encoding="utf-8")
            if mode == "slm_evaluation":
                write_evidence(tmp_path / "outputs", job)
            return 0

        return SimpleNamespace(wait=AsyncMock(side_effect=finish), returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    submitted = Mock()
    result = asyncio.run(evaluate_model(job, tmp_path, submitted))
    assert [command[command.index("--mode") + 1] for command in commands] == (
        ["slm_evaluation"] if cached else ["llm_baseline", "slm_evaluation"])
    command = commands[-1]
    assert command[1] == "-u"
    assert command[command.index("--finetuning-job-name") + 1] == "trained-job"
    assert command[command.index("--test-data-asset-version") + 1] == "1"
    assert command[command.index("--llm-baseline-data-asset-version") + 1] == "1"
    assert "--judge-deployment" not in command and "--gpt-deployment" not in command
    assert result["job_status"] == "Completed" and result["judge_deployment"] == "unit-judge"
    assert submitted.call_count == (1 if cached else 2)
    assert len(locks) == (2 if locked else 0)


@pytest.mark.parametrize("changes", [
    {"training_job_name": "another-candidate"}, {"dataset_id": "azureml:another-dataset:1"},
])
def test_adapter_does_not_publish_receipts_for_a_different_candidate(studio, tmp_path, monkeypatch, changes):
    client, _, _ = studio
    job = client.get("/api/production/training-jobs").json()["jobs"][0]

    async def launch(*command, **kwargs):
        path = Path(command[command.index("--receipt-path") + 1])

        async def finish():
            path.write_text(json.dumps({**evaluation_receipt(job), **changes}), encoding="utf-8")
            return 0

        return SimpleNamespace(wait=AsyncMock(side_effect=finish), returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    submitted = Mock()
    with pytest.raises(ValueError, match="selected job and dataset"):
        asyncio.run(evaluate_model(job, tmp_path, submitted))
    submitted.assert_not_called()


def test_stage_03_judge_defaults_and_receipt_use_backend_environment(tmp_path, monkeypatch):
    namespace = runpy.run_path(str(ROOT / "scripts" / "03_azureml_slm_vs_gpt_evaluation.py"))
    monkeypatch.setenv("AZURE_OPENAI_GPT_DEPLOYMENT", "configured-gpt")
    monkeypatch.setenv("AZURE_OPENAI_JUDGE_DEPLOYMENT", "configured-judge")
    monkeypatch.setattr("sys.argv", ["stage03", "--mode", "slm_evaluation"])
    args = namespace["parse_args"]()
    assert args.gpt_deployment == "configured-gpt" and args.judge_deployment == "configured-judge"
    args.receipt_path = tmp_path / "receipt.json"
    args.finetuning_job_name = "trained-job"
    job = SimpleNamespace(name="eval-job", id="/jobs/eval-job", studio_url="https://ml.azure.com/runs/eval-job")
    namespace["write_evaluation_receipt"](job, args, "1", "Running")
    receipt = json.loads(args.receipt_path.read_text(encoding="utf-8"))
    assert receipt["judge_deployment"] == "configured-judge"
    assert receipt["training_job_name"] == "trained-job" and receipt["status"] == "Running"


@pytest.mark.parametrize("status", ["Completed", "Failed"])
def test_stage_03_downloads_only_completed_candidate_outputs(tmp_path, monkeypatch, status):
    namespace = runpy.run_path(str(ROOT / "scripts" / "03_azureml_slm_vs_gpt_evaluation.py"))
    main = namespace["main"]
    monkeypatch.setenv("AZURE_OPENAI_GPT_DEPLOYMENT", "configured-gpt")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://unit-test.openai.azure.com/")
    monkeypatch.setattr("sys.argv", [
        "stage03", "--mode", "slm_evaluation", "--finetuning-job-name", "trained-job",
        "--test-data-asset-name", "gpu-demo", "--test-data-asset-version", "1",
        "--llm-baseline-data-asset-version", "1", "--skip-environment", "--wait",
        "--receipt-path", str(tmp_path / "receipt.json"), "--result-dir", str(tmp_path / "outputs"),
    ])
    client = Mock()
    config = Mock()
    config.create_ml_client.return_value = client
    submitted = SimpleNamespace(name="eval-job", id="/jobs/eval-job", status=status)
    submission = Mock(return_value=submitted)
    metrics = Mock()

    def monitor(ml_client, name, callback):
        callback(submitted)
        return submitted

    for key, value in {
        "load_dotenv": Mock(), "AzureMLConfig": SimpleNamespace(from_env=lambda: config),
        "resolve_storage_account_name": Mock(return_value="unit-storage"), "resolve_hf_token": Mock(return_value=""),
        "submit_inference_evaluation_job": submission, "record_run": Mock(),
        "monitor_job": monitor, "print_job_metrics": metrics,
    }.items():
        monkeypatch.setitem(main.__globals__, key, value)
    if status == "Completed":
        assert main() == 0
        client.jobs.download.assert_called_once_with(
            "eval-job", download_path=str(tmp_path / "outputs"), output_name="predictions",
        )
        metrics.assert_called_once()
        assert submission.call_args.kwargs["test_data_asset"] == "azureml:gpu-demo:1"
        assert submission.call_args.kwargs["finetuning_job_name"] == "trained-job"
    else:
        with pytest.raises(RuntimeError, match="Failed"):
            main()
        client.jobs.download.assert_not_called()
        metrics.assert_not_called()


@pytest.mark.parametrize("status,cached", [
    ("Running", False), ("Failed", False), ("Canceled", False),
    ("Completed", False), ("Completed", True),
])
def test_recovery_checks_existing_job_and_downloads_only_completed_evidence(tmp_path, monkeypatch, status, cached):
    from lib.config import AzureMLConfig

    job = {"job_name": "trained-job", "dataset_id": "azureml:gpu-demo:1"}
    receipt = evaluation_receipt(job, status="Queued")
    client = Mock()
    client.jobs.get.return_value.status = status
    credential = MagicMock()
    credential.__enter__.return_value = credential
    config = Mock()
    config.create_credential.return_value = credential
    config.create_ml_client.return_value = client
    monkeypatch.setattr(AzureMLConfig, "from_env", Mock(return_value=config))
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    client.jobs.download.side_effect = lambda *args, **kwargs: write_evidence(Path(kwargs["download_path"]), job)
    if cached:
        write_evidence(tmp_path / "outputs", job)
    updated, result = refresh_evaluation(job, tmp_path, receipt)
    assert updated["status"] == status
    client.jobs.get.assert_called_once_with("slm_evaluation-job")
    client.jobs.create_or_update.assert_not_called()
    credential.__exit__.assert_called_once()
    if status == "Completed":
        assert result["candidate_id"] == "trained-job"
        assert result["job_status"] == "Completed" and result["metrics"]
        if not cached:
            client.jobs.download.assert_called_once_with(
                "slm_evaluation-job", download_path=str(tmp_path / "outputs"), output_name="predictions",
            )
        else:
            client.jobs.download.assert_not_called()
    else:
        assert result is None
        client.jobs.download.assert_not_called()


def interrupted_evaluation(api, job):
    record = new_record("prod-recover-evaluation", job["dataset_id"], "context-grounded")
    record.update(run_type="evaluation", current_stage="evaluation", status="failed")
    record["stages"]["evaluation"].update(
        status="failed", error="Permission denied: slm_evaluation_receipt.json",
        completed_at=record["updated_at"],
    )
    record["evaluation"] = {
        "training_job": job, "jobs": {"slm_evaluation": evaluation_receipt(job, status="Queued")},
        "job_status": "Queued", "candidate_id": job["job_name"], "dataset_id": job["dataset_id"],
    }
    api.store.save(record)
    return record


def test_selecting_failed_receipt_monitor_restores_existing_evaluation_then_results(studio):
    client, app, path = studio
    api = app.state.production
    job = client.get("/api/production/training-jobs").json()["jobs"][0]
    record = interrupted_evaluation(api, job)
    directory = api.store.directory / record["run_id"]
    write_evidence(directory / "outputs", job)
    result = evaluation_result(directory / "outputs", job, evaluation_receipt(job))
    api.evaluation_refresher = Mock(side_effect=[
        (evaluation_receipt(job, status="Running"), None), (evaluation_receipt(job), result),
    ])
    original = path.read_bytes()
    url = f"/api/production/runs/{record['run_id']}"
    running = client.get(url).json()
    assert running["status"] == running["stages"]["evaluation"]["status"] == "running"
    assert running["stages"]["evaluation"]["error"] is None
    assert running["stages"]["evaluation"]["completed_at"] is None
    completed = client.get(url).json()
    assert completed["status"] == "ready"
    assert completed["stages"]["evaluation"]["status"] == "succeeded"
    assert completed["stages"]["evaluation"]["progress"] == 100
    assert completed["evaluation"]["job_status"] == "Completed"
    assert completed["evaluation"]["metrics"] == result["metrics"]
    assert api.store.read(record["run_id"]) == completed
    assert client.get(url).json() == completed
    assert api.evaluation_refresher.call_count == 2
    api.evaluation_runner.assert_not_awaited()
    assert path.read_bytes() == original


@pytest.mark.parametrize("status", ["Failed", "Canceled"])
def test_recovered_evaluation_preserves_confirmed_cloud_failure(studio, status):
    client, app, _ = studio
    api = app.state.production
    job = client.get("/api/production/training-jobs").json()["jobs"][0]
    record = interrupted_evaluation(api, job)
    api.evaluation_refresher = Mock(return_value=(evaluation_receipt(job, status=status), None))
    url = f"/api/production/runs/{record['run_id']}"
    result = client.get(url).json()
    assert result["status"] == ("failed" if status == "Failed" else "cancelled")
    assert status in result["stages"]["evaluation"]["error"]
    assert "metrics" not in result["evaluation"]
    client.get(url)
    api.evaluation_refresher.assert_called_once()


def test_evaluation_recovery_failure_is_visible_without_changing_saved_state(studio):
    client, app, _ = studio
    api = app.state.production
    job = client.get("/api/production/training-jobs").json()["jobs"][0]
    record = interrupted_evaluation(api, job)
    api.evaluation_refresher = Mock(side_effect=ValueError("Evidence does not match the selected candidate"))
    response = client.get(f"/api/production/runs/{record['run_id']}")
    assert response.status_code == 502
    assert "Evidence does not match" in response.json()["detail"]
    assert api.store.read(record["run_id"]) == record


def test_restored_evaluation_monitor_can_be_cancelled_without_touching_cloud_job(studio):
    client, app, _ = studio
    api = app.state.production
    job = client.get("/api/production/training-jobs").json()["jobs"][0]
    record = interrupted_evaluation(api, job)
    api.evaluation_refresher = Mock(return_value=(evaluation_receipt(job, status="Running"), None))
    url = f"/api/production/runs/{record['run_id']}"
    assert client.get(url).json()["status"] == "running"
    assert client.post(f"{url}/cancel").status_code == 200
    assert client.get(url).json()["status"] == "cancelled"
    api.evaluation_refresher.assert_called_once()
    api.evaluation_runner.assert_not_awaited()
