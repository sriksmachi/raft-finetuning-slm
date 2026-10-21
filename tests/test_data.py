import json
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from lib.data import (
    dataset_fingerprint,
    publish_data_asset,
    validate_data_asset_version,
    validate_dataset,
    write_manifest,
)


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


def test_validate_dataset_warns_and_continues_for_duplicates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    duplicate = _record("duplicate")
    _write_split(tmp_path, "train", [duplicate, duplicate.copy()])
    _write_split(tmp_path, "validation", [_record("validation")])
    _write_split(tmp_path, "test", [_record("test")])

    summary = validate_dataset(tmp_path)

    assert summary["train"]["duplicates"] == 1
    assert "train contains 1 duplicate question/context pairs" in caplog.text


def test_publish_data_asset_uses_selected_datastore(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [_record("train")])
    _write_split(tmp_path, "validation", [_record("validation")])
    _write_split(tmp_path, "test", [_record("test")])
    ml_client = Mock()
    ml_client.data.create_or_update.side_effect = lambda asset: asset

    asset = publish_data_asset(
        ml_client=ml_client,
        dataset_dir=tmp_path,
        name="dataset",
        version="1",
        description="description",
        datastore="uploads",
    )

    assert asset.datastore == "uploads"
    ml_client.data.create_or_update.assert_called_once_with(asset)


@pytest.mark.parametrize("version", ["1", "20261021.201403", "a" * 30])
def test_data_asset_version_accepts_azure_length_limit(version):
    validate_data_asset_version(version)


@pytest.mark.parametrize("version", ["", "a" * 31, "prod-" + "a" * 32])
def test_publication_rejects_invalid_version_before_cloud_calls(tmp_path, version):
    ml_client = Mock()
    with pytest.raises(ValueError, match="1 to 30 characters"):
        publish_data_asset(
            ml_client, tmp_path, name="dataset", version=version, description="test",
        )
    ml_client.data.create_or_update.assert_not_called()


@pytest.mark.parametrize("length", [30, 31, 37])
def test_preparation_cli_validates_version_before_generation(monkeypatch, length, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "01_prepare_data.py"
    parse_args = runpy.run_path(str(script))["parse_args"]
    monkeypatch.setattr(sys, "argv", [str(script), "--data-asset-version", "a" * length])
    if length <= 30:
        assert parse_args().data_asset_version == "a" * length
    else:
        with pytest.raises(SystemExit) as exc:
            parse_args()
        assert exc.value.code == 2
        assert "1 to 30 characters" in capsys.readouterr().err
