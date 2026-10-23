"""Production readiness and approved release endpoints, separate from execution."""

import asyncio
import copy
import logging
import threading
from concurrent.futures import Future
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.models import ReleaseApproval, ReleasePlan
from app.preparation import append_job_status
from app.workflow import audit, new_record, now
from lib.candidate_release import execute_plan, plan_hash, prepare_plan, validate_plan
from lib.release import (
    MODELS,
    assess_candidate_rows,
    policy_description,
)
from lib.release import (
    RAG_METRICS as METRICS,
)

logger = logging.getLogger(__name__)


def candidate_details(source: dict) -> dict:
    evaluation = source["evaluation"]
    training = evaluation.get("training_job") or {}
    configuration = (source.get("training") or {}).get("config") or {}
    rows = [
        {f"{model}_{metric}_score": record.get("answers", {}).get(model, {}).get("scores", {}).get(metric)
         for model in MODELS for metric in METRICS}
        for record in evaluation.get("records", [])
    ]
    try:
        readiness = assess_candidate_rows(rows)
    except (ValueError, TypeError) as exc:
        readiness = {"passed": False, "status": "blocked", "policy": policy_description(),
                     "metrics": [], "sample_count": len(rows), "reasons": [str(exc)]}
    return {
        "run_id": source["run_id"], "date": source["updated_at"],
        "evaluation_job_name": evaluation["job_name"], "readiness": readiness,
        "lineage": {
            "evaluation_run": source["run_id"], "evaluation_job": evaluation["job_name"],
            "training_run": training.get("training_run_id"), "training_job": evaluation["candidate_id"],
            "preparation_run": (source.get("training") or {}).get("dataset_run_id"),
            "training_dataset": (source.get("training") or {}).get("dataset_id"),
            "evaluation_dataset": evaluation["dataset_id"], "evaluation_split": "test.jsonl",
            "sealed_test_sha256": evaluation["data_fingerprint"],
            "model": training.get("base_model") or configuration.get("base_model"),
            "training_date": training.get("training_date"), "epochs": training.get("epochs") or configuration.get("epochs"),
            "adapter_name": training.get("adapter_name") or configuration.get("lora_adapter_name"),
            "training_samples": training.get("sample_count"), "evaluation_samples": len(rows),
            "judge": evaluation.get("judge_deployment"), "gpt_baseline": evaluation.get("baseline_id"),
        },
    }


class CandidateAPI:
    def __init__(self, parent):
        self.parent = parent
        self.store = parent.store
        self.planner = prepare_plan
        self.executor = execute_plan
        self.router = APIRouter()
        self.router.add_api_route("/release-evaluations", self.evaluations, methods=["GET"])
        self.router.add_api_route("/release-plans", self.submit, methods=["POST"], status_code=202)
        self.router.add_api_route("/release-plans/{run_id}/approve", self.approve, methods=["POST"], status_code=202)

    def source(self, run_id: str) -> dict:
        try:
            source = self.store.read(run_id)
        except ValueError as exc:
            raise HTTPException(400, "Invalid evaluation run ID.") from exc
        except KeyError as exc:
            raise HTTPException(404, "Evaluation run not found.") from exc
        if (source.get("run_type") != "evaluation"
                or source["stages"]["evaluation"]["status"] != "succeeded"
                or (source.get("evaluation") or {}).get("job_status") != "Completed"):
            raise HTTPException(409, "Select a successfully completed evaluation run.")
        return source

    async def evaluations(self):
        return {
            "policy": policy_description(),
            "evaluations": [candidate_details(record) for record in self.store.list()
                            if record.get("run_type") == "evaluation"
                            and record["stages"]["evaluation"]["status"] == "succeeded"][:50],
        }

    def available(self):
        if self.parent.tasks:
            raise HTTPException(409, "A production operation is running; wait before starting release work.")

    def start(self, record, source, approval=None):
        task = asyncio.create_task(self.work(record, source, approval), name=f"{record['run_id']}:release")
        self.parent.tasks.add(task)
        task.add_done_callback(self.parent.tasks.discard)

    async def submit(self, body: ReleasePlan):
        if not body.confirmed:
            raise HTTPException(409, "Confirm evidence verification and GPT sizing calls; planning allocates no GPU.")
        self.available()
        source = self.source(body.evaluation_run_id)
        details = candidate_details(source)
        if not details["readiness"]["passed"]:
            raise HTTPException(409, {"message": "Scientific readiness policy blocked.", "readiness": details["readiness"]})
        record = new_record(f"prod-{uuid4().hex}", source["source_folder"], source["recipe"])
        for key in ("training", "evaluation", "data_card"):
            record[key] = copy.deepcopy(source[key])
        for stage in ("data_preparation", "training", "evaluation"):
            record["stages"][stage] = copy.deepcopy(source["stages"][stage])
        record.update(mode="production", simulated=False, audit=[], run_type="release",
                      current_stage="decision", status="running")
        record["stages"]["decision"].update(status="running", started_at=now())
        record["release"] = {"status": "planning", "options": body.model_dump(exclude={"confirmed"}),
                             "source_run_id": source["run_id"], "lineage": details["lineage"]}
        audit(record, "release_plan_requested", actor="local-production-user", action_type=body.action)
        self.store.save(record)
        append_job_status(self.store.directory / record["run_id"], "Verifying candidate evidence; no model registration or GPU allocation during planning.", "release.log")
        self.start(record, source)
        return record

    async def approve(self, run_id: str, body: ReleaseApproval):
        self.available()
        try:
            record = self.store.read(run_id)
        except ValueError as exc:
            raise HTTPException(400, "Invalid release ID.") from exc
        except KeyError as exc:
            raise HTTPException(404, "Release plan not found.") from exc
        release = record.get("release") or {}
        if record.get("run_type") != "release" or release.get("status") != "awaiting_approval":
            raise HTTPException(409, "Release must have a fresh plan awaiting approval; failed operations are never automatically retried.")
        plan = release["plan"]
        if not body.confirmed or (plan["action"] == "register_and_deploy" and not body.accept_gpu_cost):
            raise HTTPException(409, "Explicit approval is required, including the shown hourly GPU cost when deploying.")
        try:
            validate_plan(plan, body.plan_hash)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        source = self.source(release["source_run_id"])
        record["stages"]["decision"].update(status="running", started_at=now())
        record["status"] = "running"
        release["status"] = "registering"
        release["approval_request"] = body.model_dump()
        audit(record, "release_plan_approved", actor=body.actor, plan_hash=body.plan_hash,
              gpu_cost_accepted=body.accept_gpu_cost, reason=body.reason)
        self.store.save(record)
        self.start(record, source, body)
        return record

    async def work(self, record, source, approval):
        directory = self.store.directory / record["run_id"]
        cancelled = threading.Event()
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        last_activity = "Waiting for release execution."

        def report(message):
            nonlocal last_activity
            if cancelled.is_set():
                raise InterruptedError("Release monitoring stopped. Inspect saved cloud resources before retrying.")
            append_job_status(directory, message, "release.log")
            last_activity = message

        async def heartbeat():
            while True:
                await asyncio.sleep(10)
                append_job_status(directory, f"Still running: {last_activity}", "release.log")

        heartbeat_task = asyncio.create_task(heartbeat())

        def persist_on_loop(result):
            record["release"].update(copy.deepcopy(result))
            if result.get("registered_model"):
                record["decisions"]["registration"] = {
                    **result["registered_model"], "actor": approval.actor, "reason": approval.reason,
                    "decided_at": now(), "override": False,
                }
                if not cancelled.is_set():
                    record["stages"]["decision"].update(status="succeeded", progress=100, completed_at=now())
                    if record["release"]["plan"]["action"] == "register_and_deploy":
                        record["current_stage"] = "deployment"
                        record["stages"]["deployment"].update(status="running", started_at=now())
            record["updated_at"] = now()
            self.store.save(record)
            if cancelled.is_set():
                raise InterruptedError("Registered identity saved after interruption; no further deployment steps will start.")

        def persist(result):
            if threading.get_ident() == loop_thread:
                persist_on_loop(result)
                return
            completed = Future()

            def save():
                try:
                    persist_on_loop(result)
                except Exception as exc:
                    logger.exception("Unable to persist released resource identity for %s", record["run_id"])
                    completed.set_exception(exc)
                else:
                    completed.set_result(None)

            loop.call_soon_threadsafe(save)
            completed.result()

        try:
            if approval is None:
                plan = await asyncio.to_thread(self.planner, source, record["release"]["options"], directory / "plan", report)
                record["release"].update(plan=plan, plan_hash=plan_hash(plan), status="awaiting_approval")
                record["stages"]["decision"].update(status="not_started")
                record["status"] = "ready"
                report("Release plan ready for approval. No GPU allocated.")
            else:
                result = await asyncio.to_thread(
                    self.executor, source, record["release"]["plan"], approval.actor, approval.reason,
                    directory / "execution", report, persist,
                )
                persist(result)
                stage = record["stages"][record["current_stage"]]
                stage.update(status="succeeded", progress=100, completed_at=now())
                record["status"] = "completed"
                audit(record, "release_completed", actor=approval.actor, action_type=record["release"]["plan"]["action"])
        except asyncio.CancelledError:
            cancelled.set()
            if record["run_id"] in self.parent.cancelling:
                self.parent.mark_cancelled(record)
            else:
                self.failed(record, "Release monitoring interrupted. Azure resources may still be allocated and billable. Inspect the saved registered model and endpoint before retrying.")
            raise
        except Exception as exc:
            logger.exception("Candidate release failed for %s", record["run_id"])
            self.failed(record, f"{exc} Azure resources may remain allocated; inspect saved model/endpoint before retrying.")
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            record["updated_at"] = now()
            self.store.save(record)

    def failed(self, record, message):
        record["stages"][record["current_stage"]].update(status="failed", error=message, completed_at=now())
        record["status"] = "failed"
        record["release"]["status"] = "failed"
        audit(record, "release_failed", actor="local-production-user", error=message)
        append_job_status(self.store.directory / record["run_id"], message, "release.log")
