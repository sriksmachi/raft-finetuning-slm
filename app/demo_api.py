"""Standalone simulated backend. This module never imports the real pipeline."""

import asyncio
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.fixtures import (
    METRICS, MODELS, STAGES, audit, candidate_release_result, data_card,
    deployment_result, evaluation_result, new_record, now, training_result,
)
from app.evaluation_showcase import load_evaluation_showcase
from app.models import Action, Deployment, Parameters, Preparation, Registration, Training
from app.store import DemoStore, StoreError, validate_id


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SHOWCASE_PATH = ROOT / "app" / "data" / "evaluation_showcase.json"
MAX_ACTIVE_TASKS = 32


def create_demo_app(
    runs_dir: Path | None = None,
    stage_delay: float = 0.35,
    evaluation_showcase_path: Path = EVALUATION_SHOWCASE_PATH,
) -> FastAPI:
    if not math.isfinite(stage_delay) or stage_delay < 0:
        raise ValueError("stage_delay must be finite and nonnegative")
    store = DemoStore(runs_dir if runs_dir is not None else ROOT / "runs" / "demo")
    evaluation_showcase = load_evaluation_showcase(evaluation_showcase_path)
    locks: dict[str, asyncio.Lock] = {}
    tasks: set[asyncio.Task] = set()

    def lock_for(run_id: str) -> asyncio.Lock:
        validate_id(run_id)
        return locks.setdefault(run_id, asyncio.Lock())

    def read(run_id: str) -> dict:
        try:
            return store.read(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Demo run not found.") from exc

    def confirmed(action: Action) -> None:
        if not action.confirmed:
            raise HTTPException(409, "Explicit confirmation is required for this simulated action.")

    def prerequisite(record: dict, stage: str) -> None:
        if record["stages"][stage]["status"] != "succeeded":
            raise HTTPException(409, f"{stage} must succeed before this action.")

    def capacity() -> None:
        if len(tasks) >= MAX_ACTIVE_TASKS:
            raise HTTPException(503, "Demo simulator is busy; retry after an active stage completes.")

    def unfinished(record: dict) -> None:
        if record["status"] == "failed":
            raise HTTPException(409, "This run failed or was interrupted; create a new demo run.")
        if any(state["status"] == "running" for state in record["stages"].values()):
            raise HTTPException(409, "Another stage is already running for this run.")

    async def simulate(run_id: str, stage: str) -> None:
        try:
            for progress in (20, 45, 65, 85, 100):
                await asyncio.sleep(stage_delay)
                async with lock_for(run_id):
                    record = read(run_id)
                    record["stages"][stage]["progress"] = progress
                    if stage == "training" and record["scenario"] == "failure" and progress == 65:
                        record["updated_at"] = now()
                        store.save(record)
                        raise RuntimeError("Simulated training failure: GPU allocation interrupted.")
                    if progress == 100:
                        if stage == "data_preparation":
                            record["data_card"] = data_card(record)
                        elif stage == "evaluation":
                            record["evaluation"] = evaluation_result(record)
                        elif stage == "deployment":
                            record["deployment"] = deployment_result(record)
                        record["stages"][stage].update(status="succeeded", completed_at=now())
                        record["status"] = "completed" if stage == "deployment" else "ready"
                        next_stage = {
                            "data_preparation": "training",
                            "training": "evaluation",
                            "evaluation": "decision",
                            "deployment": "deployment",
                        }
                        record["current_stage"] = next_stage[stage]
                        audit(record, f"{stage}_succeeded")
                    record["updated_at"] = now()
                    store.save(record)
        except asyncio.CancelledError:
            await mark_failed(run_id, stage, "Interrupted: demo server shut down during this stage.")
            raise
        except Exception as exc:
            logger.exception("Demo run %s stage %s failed", run_id, stage)
            await mark_failed(run_id, stage, str(exc))

    async def mark_failed(run_id: str, stage: str, message: str) -> None:
        logger.error("Demo run %s stage %s: %s", run_id, stage, message)
        try:
            async with lock_for(run_id):
                record = read(run_id)
                record["stages"][stage].update(status="failed", error=message, completed_at=now())
                record.update(status="failed", updated_at=now())
                audit(record, f"{stage}_failed", error=message)
                store.save(record)
        except Exception:
            logger.exception("Failed to persist stage failure for demo run %s", run_id)

    def start(record: dict, stage: str) -> dict:
        record["stages"][stage].update(status="running", progress=0, error=None,
                                       started_at=now(), completed_at=None)
        record.update(status="running", current_stage=stage, updated_at=now())
        audit(record, f"{stage}_started")
        store.save(record)
        task = asyncio.create_task(simulate(record["run_id"], stage), name=f"{record['run_id']}:{stage}")
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return record

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store.recover()
        yield
        active = list(tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        store.recover()

    application = FastAPI(title="SLM Studio (demo only)", lifespan=lifespan)
    application.state.store = store
    application.state.tasks = tasks

    @application.exception_handler(StoreError)
    async def store_error(request: Request, exc: StoreError):
        logger.error("Demo storage error at %s: %s", request.url.path, exc)
        return JSONResponse(status_code=500, content={
            "detail": str(exc), "simulated": not request.url.path.startswith("/api/production/"),
        })

    @application.exception_handler(ValueError)
    async def invalid_id(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @application.get("/api/config")
    async def config(mode: str = "demo"):
        if mode != "demo":
            raise HTTPException(400, "Production mode is unsupported. This standalone server only simulates demo actions.")
        return {"mode": "demo", "production_enabled": False, "simulated": True,
                "models": MODELS, "recipes": ["context-grounded", "context-assisted"],
                "metrics": METRICS, "stage_order": STAGES,
                "wizard": {"sequential": True, "allow_completed_review": True,
                           "production_actions": False}}

    @application.get("/api/evaluation-showcase")
    async def get_evaluation_showcase():
        return evaluation_showcase

    @application.post("/api/mode")
    async def unsupported_mode():
        raise HTTPException(400, "Mode switching is unsupported. This server is always demo; no production calls are available.")

    @application.get("/api/runs")
    async def runs():
        return {"runs": store.list()}

    @application.get("/api/runs/latest")
    async def latest():
        records = store.list()
        if not records:
            raise HTTPException(404, "No demo runs exist.")
        return records[0]

    @application.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        return read(run_id)

    @application.post("/api/runs")
    async def prepare(body: Preparation):
        confirmed(body)
        if not body.source_folder.strip():
            raise HTTPException(422, "source_folder must be a nonempty demo label.")
        capacity()
        record = new_record(f"demo-{uuid4().hex}", body.source_folder, body.recipe, body.scenario)
        record["num_samples"] = body.num_samples
        async with lock_for(record["run_id"]):
            return start(record, "data_preparation")

    @application.post("/api/runs/{run_id}/training/suggestions")
    async def suggestions(run_id: str):
        record = read(run_id)
        prerequisite(record, "data_preparation")
        return {"simulated": True, "source": "demo_fixture", "parameters": Parameters().model_dump(),
                "rationale": f"Editable fixture suggestion for {record['data_card']['split_counts']['train']} training examples and short handbook passages; no GPT call or production validation."}

    @application.post("/api/runs/{run_id}/training")
    async def train(run_id: str, body: Training):
        confirmed(body)
        async with lock_for(run_id):
            record = read(run_id)
            if record["stages"]["training"]["status"] in {"running", "succeeded"}:
                return record
            prerequisite(record, "data_preparation")
            unfinished(record)
            capacity()
            record["training"] = training_result(record, body.model_dump(exclude={"confirmed"}))
            return start(record, "training")

    @application.post("/api/runs/{run_id}/evaluation")
    async def evaluate(run_id: str, body: Action):
        confirmed(body)
        async with lock_for(run_id):
            record = read(run_id)
            if record["stages"]["evaluation"]["status"] in {"running", "succeeded"}:
                return record
            prerequisite(record, "training")
            unfinished(record)
            capacity()
            return start(record, "evaluation")

    def evidence(record: dict) -> dict:
        prerequisite(record, "evaluation")
        evaluation = record["evaluation"]
        training = record["training"]
        card = record["data_card"]
        if not evaluation or not training or not card:
            raise HTTPException(409, "Complete candidate-bound evidence is required; missing evidence cannot be overridden.")
        if (not isinstance(training.get("config"), dict)
                or not isinstance(card.get("assets"), dict)
                or not training.get("candidate_id") or not training.get("model_artifact")):
            raise HTTPException(409, "Complete training and sealed test lineage is required.")
        if (evaluation.get("simulated") is not True or evaluation.get("sealed") is not True
                or evaluation.get("candidate_id") != training.get("candidate_id")
                or evaluation.get("candidate_artifact") != training.get("model_artifact")
                or evaluation.get("test_split") != card.get("assets", {}).get("test")
                or evaluation.get("data_fingerprint") != card.get("data_fingerprint")
                or training.get("data_fingerprint") != card.get("data_fingerprint")):
            raise HTTPException(409, "Evaluation lineage does not match the exact candidate and sealed test fixture.")
        rows = evaluation.get("models", [])
        if (not isinstance(rows, list) or len(rows) != 3
                or not all(isinstance(row, dict) for row in rows)
                or {row.get("role") for row in rows} != {"candidate", "base_slm", "gpt"}):
            raise HTTPException(409, "Complete three-model evaluation evidence is required.")
        for row in rows:
            if (row.get("simulated") is not True or row.get("test_split") != evaluation["test_split"]
                    or row.get("data_fingerprint") != evaluation["data_fingerprint"]
                    or row.get("sample_count") != card["split_counts"]["test"]):
                raise HTTPException(409, "All model evidence must use the same sealed demo test records.")
            if not isinstance(row.get("metrics"), dict):
                raise HTTPException(409, "Complete five-metric evaluation evidence is required.")
            for metric in METRICS:
                value = row.get("metrics", {}).get(metric["key"])
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value < 0
                        or (metric["unit"] == "score" and value > 1)):
                    raise HTTPException(409, "Missing or invalid evaluation metrics cannot be overridden.")
            validity = row.get("json_validity")
            if not isinstance(validity, (int, float)) or not math.isfinite(validity) or not 0 <= validity <= 1:
                raise HTTPException(409, "Missing JSON-validity evidence cannot be overridden.")
        candidate = next(row for row in rows if row["role"] == "candidate")
        baseline = next(row for row in rows if row["role"] == "base_slm")
        gpt = next(row for row in rows if row["role"] == "gpt")
        if (candidate.get("id") != training["candidate_id"]
                or candidate.get("artifact") != training["model_artifact"]
                or baseline.get("id") != training["config"]["base_model"]
                or gpt.get("id") != "demo-gpt-baseline"):
            raise HTTPException(409, "Evaluation model identities do not match the candidate and baselines.")
        expected = "passed" if (
            candidate["json_validity"] >= gpt["json_validity"]
            and candidate["metrics"]["latency"] <= gpt["metrics"]["latency"]
        ) else "blocked"
        gate = evaluation.get("gate", {})
        if not isinstance(gate, dict):
            raise HTTPException(409, "Missing gate evidence cannot be overridden.")
        if gate.get("status") != expected or gate.get("simulated") is not True:
            raise HTTPException(409, "Pending or inconsistent gate evidence cannot be overridden.")
        policy = gate.get("policy", {})
        if (not isinstance(policy, dict) or policy.get("simulated") is not True
                or policy.get("id") != "demo-json-validity-latency-v1"
                or gate.get("values") != {
                    "candidate_json_validity": candidate["json_validity"],
                    "gpt_json_validity": gpt["json_validity"],
                    "candidate_latency_ms": candidate["metrics"]["latency"],
                    "gpt_latency_ms": gpt["metrics"]["latency"],
                }):
            raise HTTPException(409, "Gate policy or comparison values do not match the sealed evidence.")
        return evaluation

    @application.post("/api/runs/{run_id}/register")
    async def register(run_id: str, body: Registration):
        confirmed(body)
        async with lock_for(run_id):
            record = read(run_id)
            if record["decisions"]["registration"]:
                return record
            evaluation = evidence(record)
            unfinished(record)
            if not body.actor.strip():
                raise HTTPException(422, "actor must not be blank.")
            blocked = evaluation["gate"]["status"] == "blocked"
            if blocked and not (body.override and body.confirmation == "OVERRIDE" and body.reason.strip()):
                raise HTTPException(409, "Blocked gate requires override=true, typed confirmation 'OVERRIDE', and a nonempty reason.")
            registration = {
                "simulated": True, "model_name": f"{run_id}-registered-model", "version": "1",
                "override": blocked, "reason": body.reason.strip() or "Simulated gate passed.",
                "actor": body.actor, "decided_at": now(),
                "evidence_candidate_id": evaluation["candidate_id"],
                "evidence_data_fingerprint": evaluation["data_fingerprint"],
                "evidence_test_split": evaluation["test_split"], "gate_status": evaluation["gate"]["status"],
            }
            record["decisions"]["registration"] = registration
            record["release"] = candidate_release_result(record, registration)
            record["stages"]["decision"].update(status="succeeded", progress=100,
                                                 started_at=now(), completed_at=now())
            record.update(status="ready", current_stage="deployment", updated_at=now())
            audit(record, "registration_overridden" if blocked else "registration_approved",
                  actor=body.actor, reason=registration["reason"],
                  override=blocked, evidence_candidate_id=evaluation["candidate_id"])
            store.save(record)
            return record

    @application.post("/api/runs/{run_id}/deployment")
    async def deploy(run_id: str, body: Deployment):
        confirmed(body)
        if not body.actor.strip():
            raise HTTPException(422, "actor must not be blank.")
        async with lock_for(run_id):
            record = read(run_id)
            if record["stages"]["deployment"]["status"] in {"running", "succeeded"}:
                return record
            prerequisite(record, "decision")
            registration = record["decisions"]["registration"]
            if not registration or registration.get("evidence_candidate_id") != record["training"]["candidate_id"]:
                raise HTTPException(409, "Candidate-bound registration is required before deployment.")
            release = record.get("release")
            if (not isinstance(release, dict)
                    or release.get("status") != "candidate_deployed_zero_traffic"
                    or release.get("endpoint", {}).get("traffic_percent") != 0):
                raise HTTPException(409, "An approved zero-traffic candidate release is required.")
            evidence(record)
            unfinished(record)
            capacity()
            record["decisions"]["deployment"] = {
                "simulated": True, "actor": body.actor, "decided_at": now(),
                "confirmed": True, "candidate_id": record["training"]["candidate_id"],
                "traffic_percent": body.traffic_percent,
            }
            audit(record, "deployment_gate_confirmed", actor=body.actor,
                  traffic_percent=body.traffic_percent)
            return start(record, "deployment")

    static = Path(__file__).resolve().parent / "static"
    application.mount("/static", StaticFiles(directory=static, check_dir=False), name="static")

    @application.get("/")
    async def index():
        return FileResponse(static / "index.html")

    return application
