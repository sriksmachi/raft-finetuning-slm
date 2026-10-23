"""Production workflow state and audit records."""

from datetime import datetime, timezone

STAGES = ["data_preparation", "training", "evaluation", "decision", "deployment"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def audit(record: dict, action: str, actor: str = "local-production-user", **details) -> None:
    record["audit"].append({
        "action": action, "actor": actor, "timestamp": now(), "simulated": False, **details,
    })


def new_record(run_id: str, source: str, recipe: str) -> dict:
    timestamp = now()
    record = {
        "schema_version": 1, "mode": "production", "simulated": False, "run_id": run_id,
        "created_at": timestamp, "updated_at": timestamp,
        "current_stage": "data_preparation", "status": "ready",
        "stages": {
            stage: {"status": "not_started", "progress": 0, "error": None,
                    "started_at": None, "completed_at": None}
            for stage in STAGES
        },
        "source_folder": source, "recipe": recipe,
        "data_card": None, "training": None, "evaluation": None, "release": None,
        "decisions": {"registration": None, "deployment": None},
        "deployment": None, "audit": [],
    }
    audit(record, "run_created", source_folder=source, recipe=recipe)
    return record
