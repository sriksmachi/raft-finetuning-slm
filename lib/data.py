"""Dataset validation, lineage, and Azure ML publication helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = {
    "id",
    "type",
    "question",
    "context",
    "oracle_context",
    "cot_answer",
    "instruction",
}
SPLITS = ("train", "validation", "test")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def _content_key(record: dict[str, Any]) -> str:
    payload = "\n".join(
        str(record.get(field, "")).strip().lower()
        for field in ("question", "oracle_context")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_key(record: dict[str, Any]) -> str:
    source = str(record.get("oracle_context", "")).strip().lower()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def validate_records(records: Iterable[dict[str, Any]], split: str) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError(f"The {split} split is empty")

    type_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            raise ValueError(f"{split}[{index}] is missing: {sorted(missing)}")
        for field in ("question", "oracle_context", "cot_answer", "instruction"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{split}[{index}].{field} must be non-empty text")
        sample_type = str(row["type"])
        if sample_type not in {"oracle", "distractor"}:
            raise ValueError(f"{split}[{index}].type must be oracle or distractor")
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

    duplicate_count = len(rows) - len({_content_key(row) for row in rows})
    if duplicate_count:
        raise ValueError(f"{split} contains {duplicate_count} duplicate question/context pairs")

    return {"rows": len(rows), "types": type_counts, "duplicates": duplicate_count}


def validate_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    records_by_split = {
        split: read_jsonl(root / f"{split}.jsonl") for split in SPLITS
    }
    summary = {
        split: validate_records(records, split)
        for split, records in records_by_split.items()
    }

    keys_by_split = {
        split: {_content_key(row) for row in records}
        for split, records in records_by_split.items()
    }
    sources_by_split = {
        split: {_source_key(row) for row in records}
        for split, records in records_by_split.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = keys_by_split[left].intersection(keys_by_split[right])
            if overlap:
                raise ValueError(
                    f"Data leakage: {len(overlap)} samples overlap between {left} and {right}"
                )
            source_overlap = sources_by_split[left].intersection(sources_by_split[right])
            if source_overlap:
                raise ValueError(
                    f"Source leakage: {len(source_overlap)} oracle contexts overlap "
                    f"between {left} and {right}"
                )
    return summary


def resplit_dataset_by_source(dataset_dir: str | Path, seed: int = 42) -> dict[str, int]:
    """Rewrite splits so examples from one oracle context stay together."""
    root = Path(dataset_dir)
    records = [
        row
        for split in SPLITS
        for row in read_jsonl(root / f"{split}.jsonl")
    ]
    source_keys = sorted(
        {_source_key(row) for row in records},
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    train_end = int(0.8 * len(source_keys))
    validation_end = int(0.9 * len(source_keys))
    assignments = {
        key: split
        for split, keys in (
            ("train", source_keys[:train_end]),
            ("validation", source_keys[train_end:validation_end]),
            ("test", source_keys[validation_end:]),
        )
        for key in keys
    }
    grouped = {split: [] for split in SPLITS}
    for record in records:
        grouped[assignments[_source_key(record)]].append(record)
    for split, rows in grouped.items():
        path = root / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    return {split: len(rows) for split, rows in grouped.items()}


def dataset_fingerprint(dataset_dir: str | Path) -> str:
    digest = hashlib.sha256()
    root = Path(dataset_dir)
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        digest.update(split.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_manifest(dataset_dir: str | Path, summary: dict[str, Any]) -> Path:
    root = Path(dataset_dir)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint_sha256": dataset_fingerprint(root),
        "splits": summary,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def publish_data_asset(
    ml_client,
    dataset_dir: str | Path,
    name: str,
    version: str,
    description: str,
):
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data

    summary = validate_dataset(dataset_dir)
    manifest_path = write_manifest(dataset_dir, summary)
    asset = Data(
        name=name,
        version=version,
        path=str(Path(dataset_dir).resolve()),
        type=AssetTypes.URI_FOLDER,
        description=description,
        tags={
            "framework": "RAFT",
            "task": "context-grounded-generation",
            "fingerprint": dataset_fingerprint(dataset_dir),
            "manifest": manifest_path.name,
        },
    )
    return ml_client.data.create_or_update(asset)
