"""Real preparation, fine-tuning, and evaluation endpoints."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.candidate_api import CandidateAPI
from app.evaluation import (
    evaluate_model,
    evaluation_needs_refresh,
    refresh_evaluation,
    training_jobs,
)
from app.models import (
    ProductionEvaluation,
    ProductionPreparation,
    ProductionTraining,
    TrainerParameters,
)
from app.preparation import append_job_status, load_data_samples, prepare_data
from app.store import RunStore, StoreError, validate_id
from app.training import train_model, training_needs_refresh, training_status
from app.workflow import STAGES, audit, new_record, now

logger = logging.getLogger(__name__)
MAX_LOG_BYTES = 64 * 1024
MAX_LOG_LINES = 200


class ProductionAPI:
    def __init__(self, directory: Path):
        self.store = RunStore(directory)
        self.tasks: set[asyncio.Task] = set()
        self.cancelling: set[str] = set()
        self.runner = prepare_data
        self.training_runner = train_model
        self.training_status_reader = training_status
        self.training_records: dict[str, dict] = {}
        self.evaluation_runner = evaluate_model
        self.evaluation_refresher = refresh_evaluation
        self.ledger_dir = self.store.directory.parent
        self.router = APIRouter(prefix="/api/production")
        self.router.add_api_route("/config", self.config, methods=["GET"])
        self.router.add_api_route("/runs", self.runs, methods=["GET"])
        self.router.add_api_route("/runs/latest", self.latest, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}", self.read_latest, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}/log", self.run_log, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}/cancel", self.cancel, methods=["POST"])
        self.router.add_api_route("/runs", self.submit, methods=["POST"], status_code=202)
        self.router.add_api_route("/training", self.submit_training, methods=["POST"], status_code=202)
        self.router.add_api_route("/training-jobs", self.training_jobs, methods=["GET"])
        self.router.add_api_route("/evaluation", self.submit_evaluation, methods=["POST"], status_code=202)
        self.candidates = CandidateAPI(self)
        self.router.include_router(self.candidates.router)

    async def config(self):
        return {
            "mode": "production", "production_enabled": True, "simulated": False,
            "models": [{"id": name, "label": name} for name in (
                "Llama-3.2-3B-Instruct", "Llama-3.2-1B-Instruct", "Llama-3.1-8B-Instruct",
            )],
            "recipes": ["context-grounded"], "metrics": [],
            "stage_order": STAGES, "enabled_stages": STAGES,
            "training_defaults": TrainerParameters().model_dump(),
            "poll_interval_ms": 10000,
            "wizard": {"sequential": True, "allow_completed_review": True,
                       "production_actions": True},
        }

    async def runs(self):
        return {"runs": [self.with_samples(record) for record in self.store.list()]}

    def with_samples(self, record: dict) -> dict:
        if not record.get("data_card"):
            return record
        source_id = (record.get("training") or {}).get("dataset_run_id") or record["run_id"]
        try:
            validate_id(source_id)
        except (ValueError, TypeError) as exc:
            logger.exception("Invalid sample source for %s", record["run_id"])
            raise StoreError(f"Invalid sample source for {record['run_id']}.") from exc
        dataset_dir = self.store.directory / source_id / "dataset"
        if dataset_dir.is_dir():
            if not dataset_dir.resolve().is_relative_to(self.store.directory):
                raise StoreError("Sample dataset path must remain inside the production store.")
            try:
                record["data_card"]["samples"] = load_data_samples(dataset_dir)
            except (OSError, ValueError, TypeError) as exc:
                logger.exception("Unable to load dataset samples for %s", record["run_id"])
                raise StoreError(f"Unable to load dataset samples for {record['run_id']}: {exc}") from exc
        return record

    async def latest(self):
        records = self.store.list()
        if not records:
            raise HTTPException(404, "No production runs exist.")
        return self.with_samples(records[0])

    async def read(self, run_id: str):
        try:
            return self.with_samples(self.store.read(run_id))
        except ValueError as exc:
            raise HTTPException(400, "Invalid production run ID.") from exc
        except KeyError as exc:
            raise HTTPException(404, "Production run not found.") from exc

    async def read_latest(self, run_id: str):
        record = await self.read(run_id)
        if record.get("run_type") == "evaluation":
            return await self.read_latest_evaluation(record)
        training = record.get("training") or {}
        if not training_needs_refresh(record):
            return record
        job_name = training["job_name"]
        try:
            status = await asyncio.to_thread(self.training_status_reader, job_name)
        except Exception as exc:
            logger.exception("Unable to refresh Azure ML training status for %s", run_id)
            raise HTTPException(502, "Unable to fetch the latest training status from Azure ML. Retry refreshing the run.") from exc
        record = self.training_records.get(run_id) or self.store.read(run_id)
        if not training_needs_refresh(record) or record["training"]["job_name"] != job_name:
            return self.with_samples(record)
        training = record["training"]
        changed = training["job_status"] != status
        interrupted = record["status"] == "failed" or record["stages"]["training"]["status"] == "failed"
        training["job_status"] = status
        if status in {"Completed", "Failed", "Canceled", "Cancelled"}:
            succeeded = status == "Completed"
            cancelled = status in {"Canceled", "Cancelled"}
            message = None if succeeded else f"Azure ML training ended with status: {status}."
            if not succeeded and record["stages"]["training"]["error"]:
                message += f" {record['stages']['training']['error']}"
            record["stages"]["training"].update(
                status="succeeded" if succeeded else "cancelled" if cancelled else "failed",
                error=message, completed_at=now(),
            )
            if succeeded:
                record["stages"]["training"]["progress"] = 100
            record["status"] = "ready" if succeeded else "cancelled" if cancelled else "failed"
            audit(record, "training_status_reconciled", job_name=job_name, job_status=status)
        elif interrupted:
            record["status"] = "running"
            record["stages"]["training"].update(status="running", error=None, completed_at=None)
            audit(record, "training_monitoring_resumed", job_name=job_name, job_status=status)
        if changed or interrupted or record["status"] != "running":
            record["updated_at"] = now()
            append_job_status(self.store.directory / run_id,
                              f"Azure ML fine-tuning job {job_name}: {status}", "training.log")
            self.store.save(record)
        return self.with_samples(record)

    async def read_latest_evaluation(self, record: dict):
        run_id = record["run_id"]
        if not evaluation_needs_refresh(record) or any(
            task.get_name().startswith(f"{run_id}:") and not task.done() for task in self.tasks
        ):
            return record
        receipt = record["evaluation"]["jobs"]["slm_evaluation"]
        job = record["evaluation"]["training_job"]
        try:
            updated, result = await asyncio.to_thread(
                self.evaluation_refresher, job, self.store.directory / run_id, receipt,
            )
        except Exception as exc:
            logger.exception("Unable to restore Azure ML evaluation for %s", run_id)
            raise HTTPException(502, f"Unable to restore evaluation status or results from Azure ML: {exc}") from exc
        record = self.store.read(run_id)
        if (not evaluation_needs_refresh(record)
                or record["evaluation"]["jobs"]["slm_evaluation"]["job_name"] != receipt["job_name"]):
            return self.with_samples(record)
        evaluation = record["evaluation"]
        status = updated["status"]
        changed = updated != evaluation["jobs"]["slm_evaluation"] or record["status"] == "failed"
        evaluation["jobs"]["slm_evaluation"] = updated
        evaluation.update(job_status=status, phase="slm_evaluation")
        stage = record["stages"]["evaluation"]
        if status == "Completed":
            self.finish_evaluation(record, result, job)
        elif status in {"Failed", "Canceled", "Cancelled"}:
            cancelled = status in {"Canceled", "Cancelled"}
            record["status"] = "cancelled" if cancelled else "failed"
            stage.update(status=record["status"], completed_at=now(),
                         error=f"Azure ML evaluation ended with status: {status}.")
            audit(record, "evaluation_status_reconciled", job_name=updated["job_name"], job_status=status)
        else:
            record["status"] = "running"
            stage.update(status="running", error=None, completed_at=None)
        if changed or status == "Completed":
            record["updated_at"] = now()
            append_job_status(self.store.directory / run_id,
                              f"Restored Azure ML evaluation job {updated['job_name']}: {status}", "evaluation.log")
            self.store.save(record)
        return self.with_samples(record)

    async def run_log(self, run_id: str):
        record = await self.read(run_id)
        directory = self.store.directory / run_id
        stage = record.get("run_type") if record.get("run_type") in {"training", "evaluation"} else "data_preparation"
        label = {"training": "Training", "evaluation": "Evaluation", "data_preparation": "Preparation"}[stage]
        path = directory / ("preparation.log" if stage == "data_preparation" else f"{stage}.log")
        if record.get("run_type") == "release":
            stage = record["current_stage"]
            label = "Candidate release"
            path = directory / "release.log"
        if (directory.is_symlink() or path.is_symlink()
                or not path.resolve().is_relative_to(self.store.directory)):
            raise HTTPException(400, "Symlink run logs are not supported.")
        try:
            with path.open("rb") as log:
                size = log.seek(0, 2)
                offset = max(0, size - MAX_LOG_BYTES)
                log.seek(offset)
                tail = log.read(MAX_LOG_BYTES)
                stamp = os.fstat(log.fileno()).st_mtime
        except FileNotFoundError as exc:
            raise HTTPException(404, f"{label} log is not available for this run.") from exc
        except OSError as exc:
            logger.exception("Unable to read %s log for %s", stage, run_id)
            raise HTTPException(500, f"Unable to read the {label.lower()} log.") from exc
        text = tail.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        lines = text.splitlines()
        if offset and lines:
            lines = lines[1:]
        lines = [line for line in lines if line.strip()]
        truncated = bool(offset) or len(lines) > MAX_LOG_LINES
        lines = lines[-MAX_LOG_LINES:]
        return JSONResponse(content={
            "run_id": run_id, "status": record["status"],
            "stage": stage, "stage_status": record["stages"][stage]["status"],
            "current_status": record["stages"][stage]["error"]
                or (lines[-1] if lines else f"Waiting for {label.lower()} output."),
            "log": "\n".join(lines), "truncated": truncated,
            "updated_at": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
        }, headers={"Cache-Control": "no-store"})

    def mark_cancelled(self, record: dict) -> None:
        message = "Task cancelled by user."
        if record.get("run_type") in {"training", "evaluation"}:
            message += " The Azure ML job may still be running; check its job link and cancel it in Azure ML before resubmitting."
        elif record.get("run_type") == "release":
            message += " Azure resources may remain allocated and billable; inspect the saved model and endpoint before retrying."
        record["stages"][record["current_stage"]].update(
            status="cancelled", error=message, completed_at=now(),
        )
        record.update(status="cancelled", updated_at=now())
        if record.get("run_type") == "release":
            record["release"]["status"] = "cancelled"
        audit(record, "task_cancelled", stage=record["current_stage"])
        filename = {
            "training": "training.log", "evaluation": "evaluation.log", "release": "release.log",
        }.get(record.get("run_type"), "preparation.log")
        append_job_status(self.store.directory / record["run_id"], message, filename)
        self.store.save(record)

    async def cancel(self, run_id: str):
        record = await self.read(run_id)
        task = next((task for task in self.tasks
                     if task.get_name().split(":")[0] == run_id and not task.done()), None)
        detached_job = (
            record.get("run_type") == "training" and (record.get("training") or {}).get("job_name")
        ) or evaluation_needs_refresh(record)
        if (record["status"] != "running" or (task is None and not detached_job)
                or run_id in self.cancelling):
            raise HTTPException(409, "This task is not running or cancellation is already in progress.")
        if task is None:
            self.mark_cancelled(record)
            return self.with_samples(record)
        self.cancelling.add(run_id)
        try:
            task.cancel()
            result, = await asyncio.gather(task, return_exceptions=True)
            self.tasks.discard(task)
            if isinstance(result, Exception):
                logger.error("Unable to stop production task %s: %s", run_id, result)
                raise HTTPException(500, "Unable to stop the task. Inspect its saved status and logs.")
            record = self.store.read(run_id)
            if record["status"] != "cancelled":
                self.mark_cancelled(record)
            return self.with_samples(record)
        finally:
            self.cancelling.discard(run_id)

    async def submit(self, body: ProductionPreparation):
        if not body.confirmed:
            raise HTTPException(409, "Confirm billable GPT generation and Azure ML publication.")
        source = Path(body.source_folder).expanduser().resolve()
        if not source.is_dir() or not any(source.rglob("*.pdf")):
            raise HTTPException(422, "source_folder must be a server-accessible folder containing PDFs.")
        if self.tasks:
            raise HTTPException(409, "Data preparation is already running; wait before submitting another run.")
        if any(
            record.get("data_asset_version") == body.data_asset_version
            and record.get("preparation", {}).get("data_asset_name") == body.data_asset_name
            for record in self.store.list()
        ):
            raise HTTPException(409, "This dataset name and timestamp version already exist; retry with a new timestamp.")
        body = body.model_copy(update={"source_folder": str(source)})
        run_key = uuid4().hex
        run_id = f"prod-{run_key}"
        record = new_record(run_id, str(source), body.recipe)
        record.update(mode="production", simulated=False, audit=[], status="running",
                      num_samples=body.num_samples, data_asset_version=body.data_asset_version,
                      preparation=body.model_dump(exclude={"confirmed"}))
        record["stages"]["data_preparation"].update(status="running", started_at=now())
        audit(record, "data_preparation_confirmed", actor="local-production-user")
        append_job_status(self.store.directory / run_id, "Data preparation queued.")
        self.store.save(record)
        task = asyncio.create_task(self.execute(record, body), name=run_id)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return record

    async def submit_training(self, body: ProductionTraining):
        if not body.confirmed:
            raise HTTPException(409, "Confirm the selected dataset, trainer settings, and billable GPU job.")
        if self.tasks:
            raise HTTPException(409, "A production operation is already running. Wait before submitting training.")
        source = None
        if body.dataset_run_id:
            source = await self.read(body.dataset_run_id)
            if (source.get("run_type") in {"training", "evaluation", "release"}
                    or source["stages"]["data_preparation"]["status"] != "succeeded"
                    or not source.get("data_card")
                    or source["data_card"].get("simulated") is not False
                    or source.get("recipe") != "context-grounded"):
                raise HTTPException(409, "Select a successfully published production RAFT dataset.")
            dataset_id = source["data_card"].get("asset_reference")
            # Apply the same pinned-ID validation to persisted dataset selections.
            try:
                body = ProductionTraining.model_validate({
                    **body.model_dump(exclude={"dataset_run_id", "dataset_id"}),
                    "dataset_id": dataset_id,
                })
            except ValidationError as exc:
                raise HTTPException(409, "The published run does not contain a valid pinned Azure dataset reference.") from exc
        else:
            dataset_id = body.dataset_id
        if not dataset_id:
            raise HTTPException(422, "A pinned Azure dataset ID is required.")
        run_id = f"prod-{uuid4().hex}"
        record = new_record(run_id, dataset_id, "context-grounded")
        record.update(mode="production", simulated=False, audit=[], run_type="training",
                      status="running", current_stage="training")
        if source:
            record["data_card"] = source["data_card"]
            record["stages"]["data_preparation"] = source["stages"]["data_preparation"]
        record["training"] = {
            "simulated": False, "config": TrainerParameters.model_validate(
                body.model_dump(include=set(TrainerParameters.model_fields)),
            ).model_dump(),
            "dataset_id": dataset_id, "dataset_run_id": source["run_id"] if source else None,
            "job_name": None, "job_uri": None, "model_artifact": None, "job_status": "Submitting",
        }
        record["stages"]["training"].update(status="running", started_at=now())
        audit(record, "training_confirmed", actor="local-production-user", dataset_id=dataset_id)
        append_job_status(self.store.directory / run_id, "Submitting billable Azure ML fine-tuning job.", "training.log")
        self.store.save(record)
        task = asyncio.create_task(self.execute_training(record, body, dataset_id), name=f"{run_id}:training")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return record

    async def execute_training(self, record: dict, body: ProductionTraining, dataset_id: str):
        directory = self.store.directory / record["run_id"]
        stage = record["stages"]["training"]
        self.training_records[record["run_id"]] = record

        def submitted(receipt: dict):
            if stage["status"] != "running":
                return
            changed = (record["training"]["job_name"], record["training"]["job_status"]) != (
                receipt["job_name"], receipt["status"],
            )
            record["training"].update(
                job_name=receipt["job_name"], job_uri=receipt.get("job_uri"),
                job_id=receipt["job_id"], job_status=receipt["status"],
                model_artifact=receipt["model_artifact"], candidate_id=receipt["job_name"],
            )
            record["updated_at"] = now()
            if changed:
                append_job_status(
                    directory, f"Azure ML fine-tuning job {receipt['job_name']}: {receipt['status']}",
                    "training.log",
                )
            self.store.save(record)

        try:
            receipt = await self.training_runner(body, dataset_id, directory, submitted)
            submitted(receipt)
            if receipt["status"] != "Completed":
                raise RuntimeError(f"Azure ML training did not complete: {receipt['status']}")
            stage.update(status="succeeded", progress=100, completed_at=now())
            record["status"] = "ready"
            audit(record, "training_succeeded", actor="local-production-user")
            append_job_status(directory, "Fine-tuning completed. Review the saved adapter in Azure ML.", "training.log")
        except asyncio.CancelledError:
            if record["run_id"] in self.cancelling:
                self.mark_cancelled(record)
                raise
            message = "Monitoring interrupted. The Azure ML job may still be running; check its job link before resubmitting."
            stage.update(status="failed", error=message, completed_at=now())
            record["status"] = "failed"
            audit(record, "training_monitoring_interrupted", actor="local-production-user")
            append_job_status(directory, message, "training.log")
            raise
        except Exception as exc:
            logger.exception("Production training failed for %s", record["run_id"])
            stage.update(status="failed", error=str(exc), completed_at=now())
            record["status"] = "failed"
            audit(record, "training_failed", actor="local-production-user", error=str(exc))
            append_job_status(directory, f"Fine-tuning failed: {exc}", "training.log")
        finally:
            record["updated_at"] = now()
            self.store.save(record)
            self.training_records.pop(record["run_id"], None)

    async def training_jobs(self):
        return {"jobs": training_jobs(self.ledger_dir, self.store)}

    async def submit_evaluation(self, body: ProductionEvaluation):
        if not body.confirmed:
            raise HTTPException(409, "Confirm the training job, sealed test split, and billable GPU and judge calls.")
        if self.tasks:
            raise HTTPException(409, "A production operation is already running. Wait before submitting evaluation.")
        job = next((entry for entry in training_jobs(self.ledger_dir, self.store)
                    if entry["job_name"] == body.job_name), None)
        if not job:
            raise HTTPException(404, "Completed training job not found in the saved runs.")
        if not job["eligible"]:
            raise HTTPException(409, job["unavailable_reason"])
        run_id = f"prod-{uuid4().hex}"
        record = new_record(run_id, job["dataset_id"], "context-grounded")
        record.update(mode="production", simulated=False, audit=[], run_type="evaluation",
                      status="running", current_stage="evaluation")
        if job["training_run_id"]:
            source = self.store.read(job["training_run_id"])
            for key in ("training", "data_card"):
                record[key] = source[key]
            for stage in ("data_preparation", "training"):
                record["stages"][stage] = source["stages"][stage]
        else:
            record["training"] = {
                "simulated": False, "config": {"base_model": job["base_model"]},
                "job_name": job["job_name"], "candidate_id": job["job_name"], "job_status": "Completed",
                "dataset_id": job["dataset_id"], "dataset_run_id": None,
                "model_artifact": f"azureml://jobs/{job['job_name']}/outputs/model/paths/",
            }
            record["stages"]["training"].update(status="succeeded", progress=100,
                                                started_at=job["training_date"])
        record["evaluation"] = {
            "simulated": False, "candidate_id": job["job_name"], "training_job": job,
            "dataset_id": job["dataset_id"], "job_status": "Submitting", "jobs": {},
        }
        record["stages"]["evaluation"].update(status="running", started_at=now())
        audit(record, "evaluation_confirmed", actor="local-production-user", job_name=body.job_name)
        append_job_status(self.store.directory / run_id, "Evaluation queued. Preparing GPT baseline and judge evaluation.", "evaluation.log")
        self.store.save(record)
        task = asyncio.create_task(self.execute_evaluation(record, job), name=f"{run_id}:evaluation")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return record

    async def execute_evaluation(self, record: dict, job: dict):
        directory = self.store.directory / record["run_id"]
        stage = record["stages"]["evaluation"]

        def submitted(receipt):
            evaluation = record["evaluation"]
            previous = evaluation["jobs"].get(receipt["mode"], {})
            evaluation["jobs"][receipt["mode"]] = receipt
            evaluation.update({key: receipt.get(key) for key in (
                "job_name", "job_uri", "baseline_id", "judge_deployment",
            )})
            evaluation.update(job_status=receipt["status"], phase=receipt["mode"])
            if previous.get("status") != receipt["status"]:
                append_job_status(directory, f"{receipt['mode']}: {receipt['status']}", "evaluation.log")
            record["updated_at"] = now()
            self.store.save(record)

        try:
            result = await self.evaluation_runner(job, directory, submitted)
            self.finish_evaluation(record, result, job)
        except asyncio.CancelledError:
            if record["run_id"] in self.cancelling:
                self.mark_cancelled(record)
                raise
            message = "Evaluation monitoring interrupted. The Azure ML job may still be running; check its job link before resubmitting."
            stage.update(status="failed", error=message, completed_at=now())
            record["status"] = "failed"
            audit(record, "evaluation_monitoring_interrupted", actor="local-production-user")
            append_job_status(directory, message, "evaluation.log")
            raise
        except Exception as exc:
            logger.exception("Production evaluation failed for %s", record["run_id"])
            stage.update(status="failed", error=str(exc), completed_at=now())
            record["status"] = "failed"
            audit(record, "evaluation_failed", actor="local-production-user", error=str(exc))
            append_job_status(directory, f"Evaluation failed: {exc}", "evaluation.log")
        finally:
            record["updated_at"] = now()
            self.store.save(record)

    def finish_evaluation(self, record: dict, result: dict | None, job: dict) -> None:
        if (not result or result.get("candidate_id") != job["job_name"]
                or result.get("dataset_id") != job["dataset_id"]
                or result.get("job_status") != "Completed" or not result.get("metrics")):
            raise ValueError("Evaluation results do not match the selected completed job and dataset.")
        record["evaluation"].update(result)
        record["stages"]["evaluation"].update(status="succeeded", progress=100, error=None, completed_at=now())
        record["status"] = "ready"
        audit(record, "evaluation_succeeded", actor="local-production-user", job_name=job["job_name"])
        append_job_status(self.store.directory / record["run_id"],
                          "Evaluation completed. Metrics and response evidence are ready.", "evaluation.log")

    async def execute(self, record: dict, body: ProductionPreparation):
        stage = record["stages"]["data_preparation"]
        directory = self.store.directory / record["run_id"]
        try:
            append_job_status(directory, "Data preparation started: extracting PDFs and generating samples.")
            card = await self.runner(body, directory, record["data_asset_version"])
            record["data_card"] = card
            stage.update(status="succeeded", progress=100, completed_at=now())
            record["status"] = "ready"
            audit(record, "data_preparation_succeeded", actor="local-production-user")
            append_job_status(directory, "Data preparation succeeded: dataset validated and published.")
        except asyncio.CancelledError:
            if record["run_id"] in self.cancelling:
                self.mark_cancelled(record)
                raise
            stage.update(status="failed", error="Interrupted: server stopped during preparation.", completed_at=now())
            record["status"] = "failed"
            audit(record, "data_preparation_interrupted", actor="local-production-user")
            append_job_status(directory, stage["error"])
            raise
        except Exception as exc:
            logger.exception("Production preparation failed for %s", record["run_id"])
            stage.update(status="failed", error=str(exc), completed_at=now())
            record["status"] = "failed"
            audit(record, "data_preparation_failed", actor="local-production-user", error=str(exc))
            append_job_status(directory, f"Data preparation failed: {exc}")
        finally:
            record["updated_at"] = now()
            self.store.save(record)

    async def close(self):
        active = list(self.tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
