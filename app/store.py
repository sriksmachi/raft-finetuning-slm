"""Production YAML persistence with atomic replacement and visible corruption."""

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import yaml

from app.workflow import STAGES, audit, now

logger = logging.getLogger(__name__)
ID_PATTERN = re.compile(r"prod-[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


class StoreError(Exception):
    pass


def validate_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not ID_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid production run ID; expected prod- followed by safe letters, numbers, hyphens or underscores.")


class RunStore:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        validate_id(run_id)
        path = self.directory / f"{run_id}.yaml"
        if path.is_symlink():
            raise StoreError("Symlink run records are not supported.")
        return path

    def read(self, run_id: str) -> dict:
        path = self.path(run_id)
        if not path.exists():
            raise KeyError(run_id)
        try:
            record = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("run_id") != run_id:
                raise ValueError("Run ID does not match filename.")
            if record.get("mode") != "production" or record.get("simulated") is not False or record.get("schema_version") != 1:
                raise ValueError("Only schema version 1 production records are supported.")
            if record.get("status") not in {"running", "ready", "failed", "completed", "cancelled"}:
                raise ValueError("Invalid run status.")
            if record.get("current_stage") not in STAGES:
                raise ValueError("Invalid current stage.")
            for field in ("created_at", "updated_at"):
                stamp = datetime.fromisoformat(record[field].replace("Z", "+00:00"))
                if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
                    raise ValueError("Record timestamps must be UTC.")
            if not isinstance(record["audit"], list) or not isinstance(record["decisions"], dict):
                raise ValueError("Missing audit or decisions.")
            for field in ("registration", "deployment"):
                value = record["decisions"][field]
                if value is not None and not isinstance(value, dict):
                    raise ValueError(f"Invalid {field} decision.")
            for stage in STAGES:
                state = record["stages"][stage]
                if state["status"] not in {"not_started", "running", "succeeded", "failed", "cancelled"}:
                    raise ValueError("Invalid stage status.")
                if not isinstance(state["progress"], (int, float)) or not 0 <= state["progress"] <= 100:
                    raise ValueError("Invalid stage progress.")
                for field in ("error", "started_at", "completed_at"):
                    if field not in state:
                        raise ValueError("Incomplete stage record.")
            record.setdefault("release", None)
            for field in ("data_card", "training", "evaluation", "release", "deployment"):
                if field not in record:
                    raise ValueError(f"Missing {field}.")
                if record[field] is not None and not isinstance(record[field], dict):
                    raise ValueError(f"Invalid {field} object.")
            return record
        except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.exception("Unable to read production run %s", run_id)
            raise StoreError(f"Corrupted or unreadable production record {run_id}: {exc}") from exc

    def list(self) -> list[dict]:
        records = []
        for path in self.directory.glob("*.yaml"):
            try:
                validate_id(path.stem)
                records.append(self.read(path.stem))
            except ValueError as exc:
                logger.exception("Invalid production filename %s", path.name)
                raise StoreError(f"Invalid production record filename: {path.name}") from exc
        return sorted(records, key=lambda record: datetime.fromisoformat(
            record["created_at"].replace("Z", "+00:00")), reverse=True)

    def save(self, record: dict) -> None:
        if record.get("mode") != "production" or record.get("simulated") is not False:
            raise StoreError("Only production records may be saved.")
        path = self.path(record["run_id"])
        temporary = path.with_suffix(".yaml.pending")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(record, handle, sort_keys=False, allow_unicode=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            logger.exception("Unable to persist production run %s", record["run_id"])
            raise StoreError(f"Unable to persist production record {record['run_id']}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def recover(self) -> None:
        # One damaged record must remain visible as an API error, not prevent all startup.
        for path in self.directory.glob("*.yaml"):
            try:
                record = self.read(path.stem)
                interrupted = [name for name in STAGES if record["stages"][name]["status"] == "running"]
                if not interrupted:
                    continue
                for name in interrupted:
                    message = (
                        "Monitoring interrupted. The Azure ML job may still be running; check its job link before resubmitting."
                        if name in {"training", "evaluation"}
                        else "Release monitoring interrupted. Azure resources may still be allocated and billable; inspect the saved model and endpoint before retrying."
                        if name in {"decision", "deployment"}
                        else "Interrupted: production server stopped before this stage completed."
                    )
                    record["stages"][name].update(
                        status="failed", error=message,
                        completed_at=now(),
                    )
                    audit(record, "stage_interrupted", stage=name)
                    logger.error("Recovered interrupted production run %s stage %s", record["run_id"], name)
                record.update(status="failed", updated_at=now())
                self.save(record)
            except (StoreError, ValueError):
                logger.exception("Unable to recover production record %s", path.name)
