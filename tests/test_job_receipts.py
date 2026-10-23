"""Cross-process job receipts tolerate temporary Windows sharing violations."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from lib.job_receipts import RECEIPT_ACCESS_ATTEMPTS, read_receipt, write_receipt


def test_receipt_round_trip_and_missing_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    assert read_receipt(path) is None
    write_receipt(path, {"status": "Running"})
    assert read_receipt(path) == {"status": "Running"}
    write_receipt(path, {"status": "Completed"})
    assert read_receipt(path) == {"status": "Completed"}
    assert not path.with_suffix(".pending").exists()


def test_temporary_read_lock_is_retried_and_logged(tmp_path, monkeypatch, caplog):
    path = tmp_path / "slm_evaluation_receipt.json"
    reader = Mock(side_effect=[PermissionError("locked"), PermissionError("locked"), '{"status":"Completed"}'])
    monkeypatch.setattr(Path, "read_text", reader)
    sleep = Mock()
    monkeypatch.setattr("lib.job_receipts.time.sleep", sleep)
    assert read_receipt(path) == {"status": "Completed"}
    assert reader.call_count == 3
    assert sleep.call_count == 2
    assert "retrying read" in caplog.text


def test_persistent_read_lock_remains_an_explicit_error(tmp_path, monkeypatch):
    reader = Mock(side_effect=PermissionError("persistently locked"))
    monkeypatch.setattr(Path, "read_text", reader)
    monkeypatch.setattr("lib.job_receipts.time.sleep", Mock())
    with pytest.raises(PermissionError, match="persistently locked"):
        read_receipt(tmp_path / "receipt.json")
    assert reader.call_count == RECEIPT_ACCESS_ATTEMPTS


def test_temporary_replacement_lock_preserves_atomic_receipt(tmp_path, monkeypatch, caplog):
    import os

    path = tmp_path / "receipt.json"
    write_receipt(path, {"status": "Running"})
    replace = os.replace
    attempts = []

    def locked(source, destination):
        attempts.append(True)
        if len(attempts) < 3:
            assert read_receipt(path) == {"status": "Running"}
            raise PermissionError("reader sharing lock")
        replace(source, destination)

    monkeypatch.setattr("lib.job_receipts.os.replace", locked)
    monkeypatch.setattr("lib.job_receipts.time.sleep", Mock())
    write_receipt(path, {"status": "Completed"})
    assert read_receipt(path) == {"status": "Completed"}
    assert len(attempts) == 3
    assert "retrying replacement" in caplog.text


def test_persistent_replacement_lock_does_not_report_success(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    write_receipt(path, {"status": "Running"})
    replace = Mock(side_effect=PermissionError("persistently locked"))
    monkeypatch.setattr("lib.job_receipts.os.replace", replace)
    monkeypatch.setattr("lib.job_receipts.time.sleep", Mock())
    with pytest.raises(PermissionError, match="persistently locked"):
        write_receipt(path, {"status": "Completed"})
    assert read_receipt(path) == {"status": "Running"}
    assert replace.call_count == RECEIPT_ACCESS_ATTEMPTS


@pytest.mark.parametrize("content,error", [
    ("not JSON", json.JSONDecodeError), ("[]", TypeError),
])
def test_invalid_receipts_are_not_silently_retried(tmp_path, monkeypatch, content, error):
    path = tmp_path / "receipt.json"
    path.write_text(content, encoding="utf-8")
    sleep = Mock()
    monkeypatch.setattr("lib.job_receipts.time.sleep", sleep)
    with pytest.raises(error):
        read_receipt(path)
    sleep.assert_not_called()
