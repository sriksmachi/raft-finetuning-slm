import json
from pathlib import Path

import pytest

from lib.data import dataset_fingerprint, validate_dataset, write_manifest


def _record(question: str, sample_type: str = "oracle") -> dict:
    return {
        "id": question,
        "type": sample_type,
        "question": question,
        "context": {"title": [["title"]], "sentences": [["context"]]},
        "oracle_context": f"evidence for {question}",
        "cot_answer": "<ANSWER>answer</ANSWER>",
        "instruction": f"<DOCUMENT>evidence</DOCUMENT>\n### Question:\n{question}",
    }


def _write_split(root: Path, split: str, records: list[dict]) -> None:
    content = "".join(json.dumps(record) + "\n" for record in records)
    (root / f"{split}.jsonl").write_text(content, encoding="utf-8")


def test_validate_dataset_and_manifest(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [_record("train")])
    _write_split(tmp_path, "validation", [_record("validation", "distractor")])
    _write_split(tmp_path, "test", [_record("test")])

    summary = validate_dataset(tmp_path)
    manifest = write_manifest(tmp_path, summary)

    assert summary["train"]["rows"] == 1
    assert json.loads(manifest.read_text())["fingerprint_sha256"] == dataset_fingerprint(tmp_path)


def test_validate_dataset_rejects_split_leakage(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [_record("same")])
    _write_split(tmp_path, "validation", [_record("same")])
    _write_split(tmp_path, "test", [_record("test")])

    with pytest.raises(ValueError, match="Data leakage"):
        validate_dataset(tmp_path)


def test_validate_dataset_rejects_source_leakage(tmp_path: Path) -> None:
    train = _record("train")
    validation = _record("validation")
    validation["oracle_context"] = train["oracle_context"]
    _write_split(tmp_path, "train", [train])
    _write_split(tmp_path, "validation", [validation])
    _write_split(tmp_path, "test", [_record("test")])

    with pytest.raises(ValueError, match="Source leakage"):
        validate_dataset(tmp_path)
