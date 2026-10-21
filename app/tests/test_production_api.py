"""Fast API integration tests: real persistence, fake billable execution."""

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import ROOT, create_app
from app.models import Preparation, ProductionPreparation
from app.preparation import build_data_card, prepare_data
from app.production_api import MAX_LOG_BYTES, MAX_LOG_LINES


def write_dataset(directory, count=7):
    dataset = directory / "dataset"
    dataset.mkdir(parents=True)
    for split, size in (("train", count - 2), ("validation", 1), ("test", 1)):
        rows = [{
            "id": f"{split}-{index}", "type": "oracle", "question": f"Question {index}?",
            "oracle_context": f"Distinct source for {split}", "context": {"sentences": [["source"]]},
            "cot_answer": "<ANSWER>Answer</ANSWER>", "instruction": "Source and question",
        } for index in range(size)]
        (dataset / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )


@pytest.fixture
def production(tmp_path):
    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "example.pdf").write_bytes(b"test PDF placeholder")
    application = create_app(tmp_path / "demo", production_runs_dir=tmp_path / "production")

    async def runner(body, directory, version):
        write_dataset(directory, body.num_samples)
        return build_data_card(body, directory, version)

    application.state.production.runner = runner
    with TestClient(application) as client:
        yield client, application, {
            "source_folder": str(source), "num_samples": 7, "recipe": "context-grounded",
            "confirmed": True,
        }


def wait(client, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/production/runs/{run_id}")
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] != "running":
            return record
        time.sleep(.01)
    pytest.fail("Production preparation did not complete")


def test_prepare_preview_isolation_and_persistence(production, tmp_path):
    client, _application, body = production
    assert client.get("/api/production/config").json()["enabled_stages"] == ["data_preparation", "training"]
    assert client.get("/api/production/runs").json() == {"runs": []}
    assert client.get("/api/production/runs/latest").status_code == 404
    response = client.post("/api/production/runs", json=body)
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    result = wait(client, run_id)
    assert result["mode"] == "production" and result["simulated"] is False
    assert result["stages"]["data_preparation"]["status"] == "succeeded"
    card = result["data_card"]
    assert card["record_count"] == body["num_samples"] == sum(card["split_counts"].values())
    assert card["validation"]["status"] == "passed"
    assert {sample["split"] for sample in card["samples"]} == {"train", "validation", "test"}
    assert all(sample["question"] and sample["cot_answer"] for sample in card["samples"])
    assert all(set(sample) == {
        "split", "id", "type", "question", "context", "oracle_context", "cot_answer", "instruction",
    } for sample in card["samples"])
    assert card["samples"][0]["context"] == {"sentences": [["source"]]}
    assert card["samples"][0]["oracle_context"] == "Distinct source for train"
    assert 1 <= len(result["data_asset_version"]) <= 30
    assert result["data_asset_version"] != run_id
    assert card["asset_reference"] == f"azureml:slm-studio-data:{result['data_asset_version']}"
    assert len(card["fingerprint"]) == 64
    assert all(event["simulated"] is False for event in result["audit"])
    assert client.get(f"/api/runs/{run_id}").status_code == 400
    assert client.get("/api/production/runs/demo-completed").status_code == 400
    assert all(run["mode"] == "demo" for run in client.get("/api/runs").json()["runs"])
    assert client.post(f"/api/production/runs/{run_id}/training", json={"confirmed": True}).status_code == 404
    assert client.get("/api/production/runs/latest").json()["run_id"] == run_id
    reopened = create_app(tmp_path / "other-demo", production_runs_dir=tmp_path / "production")
    with TestClient(reopened) as other:
        assert other.get(f"/api/production/runs/{run_id}").json()["data_card"] == card


def test_old_saved_previews_are_refreshed_from_jsonl(production):
    client, application, body = production
    result = wait(client, client.post("/api/production/runs", json=body).json()["run_id"])
    original = result["data_card"]["samples"]
    result["data_card"]["samples"] = [
        {"split": sample["split"], "question": sample["question"], "cot_answer": sample["cot_answer"]}
        for sample in original
    ]
    application.state.production.store.save(result)
    run_id = result["run_id"]
    responses = [
        client.get(f"/api/production/runs/{run_id}").json(),
        client.get("/api/production/runs/latest").json(),
        client.get("/api/production/runs").json()["runs"][0],
    ]
    assert all(record["data_card"]["samples"] == original for record in responses)
    assert len(original) == 4


def test_corrupt_local_samples_surface_an_error(production):
    client, application, body = production
    result = wait(client, client.post("/api/production/runs", json=body).json()["run_id"])
    path = application.state.production.store.directory / result["run_id"] / "dataset" / "train.jsonl"
    path.write_text('["not a dataset record"]\n', encoding="utf-8")
    response = client.get(f"/api/production/runs/{result['run_id']}")
    assert response.status_code == 500
    assert "Unable to load dataset samples" in response.json()["detail"]


@pytest.mark.parametrize("changes,status", [
    ({"confirmed": False}, 409), ({"num_samples": 0}, 422),
    ({"num_samples": 2}, 422), ({"num_samples": 3.5}, 422),
    ({"num_samples": True}, 422), ({"num_samples": 100001}, 422),
    ({"recipe": "context-assisted"}, 422), ({"source_folder": "missing-folder"}, 422),
    ({"num_questions": 0}, 422), ({"oracle_probability": 2}, 422),
    ({"data_asset_name": "../invalid"}, 422), ({"scenario": "passing"}, 422),
])
def test_validation_does_not_execute(production, changes, status):
    client, application, body = production
    runner = AsyncMock()
    application.state.production.runner = runner
    assert client.post("/api/production/runs", json={**body, **changes}).status_code == status
    runner.assert_not_called()
    assert client.get("/api/production/runs").json()["runs"] == []


def test_failure_is_visible_and_persisted(production):
    client, application, body = production
    application.state.production.runner = AsyncMock(side_effect=RuntimeError("Generation failed"))
    result = client.post("/api/production/runs", json=body).json()
    record = wait(client, result["run_id"])
    assert record["status"] == "failed"
    assert record["data_card"] is None
    assert record["stages"]["data_preparation"]["error"] == "Generation failed"


def test_shutdown_and_concurrent_submission(production):
    client, application, body = production

    async def blocked(*args):
        await asyncio.Event().wait()

    application.state.production.runner = blocked
    result = client.post("/api/production/runs", json=body).json()
    assert client.post("/api/production/runs", json=body).status_code == 409
    client.portal.call(application.state.production.close)
    assert wait(client, result["run_id"])["status"] == "failed"


def test_recover_interrupted_run(production):
    client, application, body = production
    application.state.production.runner = AsyncMock(side_effect=RuntimeError("failure"))
    result = wait(client, client.post("/api/production/runs", json=body).json()["run_id"])
    result["status"] = "running"
    result["stages"]["data_preparation"]["status"] = "running"
    application.state.production.store.save(result)
    application.state.production.store.recover()
    recovered = client.get(f"/api/production/runs/{result['run_id']}").json()
    assert recovered["status"] == "failed"
    assert "Interrupted" in recovered["stages"]["data_preparation"]["error"]


def test_adapter_cli_arguments_and_exact_count(tmp_path, monkeypatch):
    body = ProductionPreparation(source_folder=str(tmp_path), num_samples=7)
    directory = tmp_path / "run"
    write_dataset(directory)
    process = AsyncMock()
    process.wait.return_value = 0
    launch = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    card = asyncio.run(prepare_data(body, directory, "prod-test"))
    args = launch.call_args.args
    assert args[args.index("--num-samples") + 1] == "7"
    assert args[args.index("--source-folder") + 1] == str(tmp_path)
    assert args[args.index("--data-asset-version") + 1] == "prod-test"
    assert "--skip-publish" not in args
    assert args[1] == "-u"
    assert launch.call_args.kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
    assert card["record_count"] == 7
    with pytest.raises(ValueError, match="sample count"):
        build_data_card(body.model_copy(update={"num_samples": 8}), directory, "prod-test")
    process.wait.return_value = 1
    with pytest.raises(RuntimeError, match="exited with code 1"):
        asyncio.run(prepare_data(body, directory, "prod-test"))


def test_log_updates_while_running_and_retains_terminal_status(production):
    client, application, body = production
    config = client.get("/api/production/config").json()
    assert config["poll_interval_ms"] == 10000

    async def blocked(body, directory, version):
        with (directory / "preparation.log").open("a", encoding="utf-8") as log:
            log.write("Extracting page 1/45\n")
            log.flush()
        await asyncio.Event().wait()

    application.state.production.runner = blocked
    run_id = client.post("/api/production/runs", json=body).json()["run_id"]
    path = application.state.production.store.directory / run_id / "preparation.log"
    response = client.get(f"/api/production/runs/{run_id}/log")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["stage_status"] == "running"
    assert result["current_status"] == "Extracting page 1/45"
    assert "queued" in result["log"] and "started" in result["log"]
    with path.open("a", encoding="utf-8") as log:
        log.write("Extracting page 2/45\nGenerated samples: 1/7\n")
    updated = client.get(f"/api/production/runs/{run_id}/log").json()
    assert updated["current_status"] == "Generated samples: 1/7"
    assert "Extracting page 2/45" in updated["log"]
    client.portal.call(application.state.production.close)
    interrupted = client.get(f"/api/production/runs/{run_id}/log").json()
    assert interrupted["stage_status"] == "failed"
    assert "Interrupted" in interrupted["current_status"]
    assert "Interrupted" in interrupted["log"]


def test_completed_log_is_available_after_restart(production, tmp_path):
    client, _application, body = production
    run_id = client.post("/api/production/runs", json=body).json()["run_id"]
    wait(client, run_id)
    result = client.get(f"/api/production/runs/{run_id}/log").json()
    assert "succeeded" in result["current_status"]
    assert result["stage_status"] == "succeeded"
    reopened = create_app(tmp_path / "other-demo", production_runs_dir=tmp_path / "production")
    with TestClient(reopened) as other:
        assert other.get(f"/api/production/runs/{run_id}/log").json() == result


def test_run_log_tail_is_bounded_and_handles_carriage_returns(production):
    client, application, body = production
    run_id = client.post("/api/production/runs", json=body).json()["run_id"]
    wait(client, run_id)
    path = application.state.production.store.directory / run_id / "preparation.log"
    path.write_bytes(
        b"x" * (MAX_LOG_BYTES + 100) + b"\n"
        + b"old log line\n" * (MAX_LOG_LINES + 10)
        + "Pages: 1/45\rPages: 2/45\r\nLatest status: caf\u00e9\n".encode("utf-8")
    )
    response = client.get(f"/api/production/runs/{run_id}/log")
    result = response.json()
    assert result["truncated"] is True
    assert len(result["log"].splitlines()) <= MAX_LOG_LINES
    assert len(result["log"].encode("utf-8")) <= MAX_LOG_BYTES
    assert result["current_status"] == "Latest status: caf\u00e9"
    assert "Pages: 1/45\nPages: 2/45" in result["log"]


def test_run_log_rejects_missing_invalid_and_unreadable_logs(production, monkeypatch):
    client, application, body = production
    assert client.get("/api/production/runs/demo-completed/log").status_code == 400
    assert client.get("/api/production/runs/prod-missing/log").status_code == 404
    run_id = client.post("/api/production/runs", json=body).json()["run_id"]
    wait(client, run_id)
    path = application.state.production.store.directory / run_id / "preparation.log"
    path.unlink()
    assert client.get(f"/api/production/runs/{run_id}/log").status_code == 404
    original = type(path).open

    def unreadable(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("Denied")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", unreadable)
    assert client.get(f"/api/production/runs/{run_id}/log").status_code == 500


def test_failed_run_log_contains_the_failure(production):
    client, application, body = production
    application.state.production.runner = AsyncMock(side_effect=RuntimeError("Azure publication failed"))
    run_id = client.post("/api/production/runs", json=body).json()["run_id"]
    wait(client, run_id)
    result = client.get(f"/api/production/runs/{run_id}/log").json()
    assert result["current_status"] == "Azure publication failed"
    assert "Data preparation failed: Azure publication failed" in result["log"]


def test_worker_flushes_log_before_process_completion(tmp_path, monkeypatch):
    from app import preparation

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "01_prepare_data.py").write_text(
        "import time\n"
        "from pathlib import Path\n"
        "print('First live status')\n"
        "while not Path('continue').exists():\n"
        "    time.sleep(0.01)\n"
        "print('Final live status')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(preparation, "build_data_card", lambda *args: {"record_count": 7})
    directory = tmp_path / "run"
    preparation.append_job_status(directory, "Queued")
    body = ProductionPreparation(source_folder=str(tmp_path), num_samples=7)

    async def check_live_output():
        task = asyncio.create_task(prepare_data(body, directory, "prod-test"))
        try:
            deadline = asyncio.get_running_loop().time() + 10
            while "First live status" not in (directory / "preparation.log").read_text(encoding="utf-8"):
                if task.done():
                    await task
                    pytest.fail("Worker exited without logging progress")
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("Worker output was buffered until process exit")
                await asyncio.sleep(.01)
            assert not task.done()
            (tmp_path / "continue").touch()
            assert await asyncio.wait_for(task, 10) == {"record_count": 7}
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check_live_output())
    content = (directory / "preparation.log").read_text(encoding="utf-8")
    assert "Queued" in content
    assert "First live status" in content and "Final live status" in content


def test_production_uses_distinct_persisted_versions_within_azure_limit(production):
    client, application, body = production
    runner = AsyncMock(side_effect=application.state.production.runner)
    application.state.production.runner = runner
    versions = []
    for timestamp in ("211026201403", "211026201404"):
        response = client.post("/api/production/runs", json={
            **body, "data_asset_version": f"data-{timestamp}",
        })
        assert response.status_code == 202
        submitted = response.json()
        record = wait(client, submitted["run_id"])
        version = record["data_asset_version"]
        assert version == submitted["data_asset_version"]
        assert len(submitted["run_id"]) == 37
        assert version == f"data-{timestamp}"
        assert len(version) == 17
        assert runner.call_args.args[2] == version
        assert record["data_card"]["asset_reference"].endswith(f":{version}")
        versions.append(version)
    assert versions[0] != versions[1]


@pytest.mark.parametrize("version", [
    "prod-" + "a" * 32, "data-2110262014", "data-211026201403-extra",
    "data-321026201403", "data-211026251403",
])
def test_request_rejects_invalid_dataset_version(production, version):
    client, application, body = production
    runner = AsyncMock()
    application.state.production.runner = runner
    assert client.post("/api/production/runs", json={
        **body, "data_asset_version": version,
    }).status_code == 422
    runner.assert_not_called()


def test_generated_version_uses_utc_date_suffix():
    before = datetime.now(timezone.utc).strftime("data-%d%m%y%H%M%S")
    body = ProductionPreparation(source_folder="pdfs")
    after = datetime.now(timezone.utc).strftime("data-%d%m%y%H%M%S")
    assert body.data_asset_version in {before, after}
    assert body.data_asset_name == "slm-studio-data"


def test_duplicate_dataset_version_is_rejected_before_generation(production):
    client, application, body = production
    body = {**body, "data_asset_version": "data-211026201403"}
    run_id = client.post("/api/production/runs", json=body).json()["run_id"]
    wait(client, run_id)
    runner = AsyncMock()
    application.state.production.runner = runner
    assert client.post("/api/production/runs", json=body).status_code == 409
    runner.assert_not_called()


def test_default_data_folder_in_ui_and_request_models():
    assert Preparation(recipe="context-grounded").source_folder == "data/"
    assert ProductionPreparation().source_folder == "data/"
    html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'name="source_folder" value="data/"' in html


def test_production_prepares_default_data_folder(production, tmp_path, monkeypatch):
    client, _application, body = production
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "example.pdf").write_bytes(b"test PDF placeholder")
    monkeypatch.chdir(tmp_path)
    body = {key: value for key, value in body.items() if key != "source_folder"}
    response = client.post("/api/production/runs", json=body)
    assert response.status_code == 202, response.text
    record = wait(client, response.json()["run_id"])
    assert record["source_folder"] == str(directory)
    assert record["data_card"]["source_folder"] == str(directory)
    assert record["stages"]["data_preparation"]["status"] == "succeeded"
