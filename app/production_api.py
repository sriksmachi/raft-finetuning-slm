"""Real preparation and fine-tuning endpoints; later production stages disabled."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.fixtures import STAGES, audit, new_record, now
from app.models import ProductionPreparation, ProductionTraining, TrainerParameters
from app.preparation import append_job_status, load_data_samples, prepare_data
from app.store import DemoStore, StoreError, validate_id
from app.training import train_model

logger = logging.getLogger(__name__)
MAX_LOG_BYTES = 64 * 1024
MAX_LOG_LINES = 200


class ProductionAPI:
    def __init__(self, directory: Path):
        self.store = DemoStore(directory, mode="production")
        self.tasks: set[asyncio.Task] = set()
        self.runner = prepare_data
        self.training_runner = train_model
        self.router = APIRouter(prefix="/api/production")
        self.router.add_api_route("/config", self.config, methods=["GET"])
        self.router.add_api_route("/runs", self.runs, methods=["GET"])
        self.router.add_api_route("/runs/latest", self.latest, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}", self.read, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}/log", self.run_log, methods=["GET"])
        self.router.add_api_route("/runs", self.submit, methods=["POST"], status_code=202)
        self.router.add_api_route("/training", self.submit_training, methods=["POST"], status_code=202)

    async def config(self):
        return {
            "mode": "production", "production_enabled": True, "simulated": False,
            "models": [{"id": name, "label": name} for name in (
                "Llama-3.2-3B-Instruct", "Llama-3.2-1B-Instruct", "Llama-3.1-8B-Instruct",
            )],
            "recipes": ["context-grounded"], "metrics": [],
            "stage_order": STAGES, "enabled_stages": ["data_preparation", "training"],
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
            validate_id(source_id, "production")
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

    async def run_log(self, run_id: str):
        record = await self.read(run_id)
        directory = self.store.directory / run_id
        stage = "training" if record.get("run_type") == "training" else "data_preparation"
        path = directory / ("training.log" if stage == "training" else "preparation.log")
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
            raise HTTPException(404, f"{'Training' if stage == 'training' else 'Preparation'} log is not available for this run.") from exc
        except OSError as exc:
            logger.exception("Unable to read %s log for %s", stage, run_id)
            raise HTTPException(500, f"Unable to read the {'training' if stage == 'training' else 'preparation'} log.") from exc
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
                or (lines[-1] if lines else f"Waiting for {'training' if stage == 'training' else 'preparation'} output."),
            "log": "\n".join(lines), "truncated": truncated,
            "updated_at": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
        }, headers={"Cache-Control": "no-store"})

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
        record = new_record(run_id, str(source), body.recipe, "passing")
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
            if (source.get("run_type") == "training"
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
        record = new_record(run_id, dataset_id, "context-grounded", "passing")
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

        def submitted(receipt: dict):
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
