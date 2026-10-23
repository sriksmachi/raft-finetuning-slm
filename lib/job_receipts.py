"""Atomic job receipts with bounded retries for Windows file sharing."""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)
RECEIPT_ACCESS_ATTEMPTS = 5
RECEIPT_RETRY_DELAY = 0.05


def read_receipt(path: Path) -> dict | None:
    for attempt in range(RECEIPT_ACCESS_ATTEMPTS):
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except PermissionError:
            if attempt == RECEIPT_ACCESS_ATTEMPTS - 1:
                raise
            logger.warning("Job receipt temporarily locked; retrying read: %s", path)
            time.sleep(RECEIPT_RETRY_DELAY * (attempt + 1))
        else:
            receipt = json.loads(text)
            if not isinstance(receipt, dict):
                raise TypeError(f"Job receipt must contain a mapping: {path}")
            return receipt


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for attempt in range(RECEIPT_ACCESS_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == RECEIPT_ACCESS_ATTEMPTS - 1:
                raise
            logger.warning("Job receipt temporarily locked; retrying replacement: %s", path)
            time.sleep(RECEIPT_RETRY_DELAY * (attempt + 1))
