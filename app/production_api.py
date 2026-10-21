"""Real data-preparation endpoints; no simulated later-stage actions."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.fixtures import STAGES, audit, new_record, now
from app.models import ProductionPreparation
from app.preparation import append_job_status, prepare_data
from app.store import DemoStore

logger = logging.getLogger(__name__)
MAX_LOG_BYTES = 64 * 1024
MAX_LOG_LINES = 200


class ProductionAPI:
    def __init__(self, directory: Path):
        self.store = DemoStore(directory, mode="production")
        self.tasks: set[asyncio.Task] = set()
        self.runner = prepare_data
        self.router = APIRouter(prefix="/api/production")
        self.router.add_api_route("/config", self.config, methods=["GET"])
        self.router.add_api_route("/runs", self.runs, methods=["GET"])
        self.router.add_api_route("/runs/latest", self.latest, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}", self.read, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}/log", self.run_log, methods=["GET"])
        self.router.add_api_route("/runs", self.submit, methods=["POST"], status_code=202)

    async def config(self):
        return {
            "mode": "production", "production_enabled": True, "simulated": False,
            "models": [], "recipes": ["context-grounded"], "metrics": [],
            "stage_order": STAGES, "enabled_stages": ["data_preparation"],
            "poll_interval_ms": 10000,
            "wizard": {"sequential": True, "allow_completed_review": True,
                       "production_actions": True},
        }

    async def runs(self):
        return {"runs": self.store.list()}

    async def latest(self):
        records = self.store.list()
        if not records:
            raise HTTPException(404, "No production runs exist.")
        return records[0]

    async def read(self, run_id: str):
        try:
            return self.store.read(run_id)
        except ValueError as exc:
            raise HTTPException(400, "Invalid production run ID.") from exc
        except KeyError as exc:
            raise HTTPException(404, "Production run not found.") from exc

    async def run_log(self, run_id: str):
        record = await self.read(run_id)
        directory = self.store.directory / run_id
        path = directory / "preparation.log"
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
            raise HTTPException(404, "Preparation log is not available for this run.") from exc
        except OSError as exc:
            logger.exception("Unable to read preparation log for %s", run_id)
            raise HTTPException(500, "Unable to read the preparation log.") from exc
        text = tail.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        lines = text.splitlines()
        if offset and lines:
            lines = lines[1:]
        lines = [line for line in lines if line.strip()]
        truncated = bool(offset) or len(lines) > MAX_LOG_LINES
        lines = lines[-MAX_LOG_LINES:]
        return JSONResponse(content={
            "run_id": run_id, "status": record["status"],
            "stage_status": record["stages"]["data_preparation"]["status"],
            "current_status": record["stages"]["data_preparation"]["error"]
                or (lines[-1] if lines else "Waiting for preparation output."),
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
        body = body.model_copy(update={"source_folder": str(source)})
        run_key = uuid4().hex
        run_id = f"prod-{run_key}"
        record = new_record(run_id, str(source), body.recipe, "passing")
        record.update(mode="production", simulated=False, audit=[], status="running",
                      num_samples=body.num_samples, data_asset_version=run_key[:30],
                      preparation=body.model_dump(exclude={"confirmed"}))
        record["stages"]["data_preparation"].update(status="running", started_at=now())
        audit(record, "data_preparation_confirmed", actor="local-production-user")
        append_job_status(self.store.directory / run_id, "Data preparation queued.")
        self.store.save(record)
        task = asyncio.create_task(self.execute(record, body), name=run_id)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return record

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
