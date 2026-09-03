from lib.monitoring import create_reference_profile, detect_drift


def _record(size: int, documents: int = 1) -> dict:
    return {
        "instruction": "<DOCUMENT>" * documents + "x" * size,
        "question": "q" * max(1, size // 10),
    }


def test_matching_distribution_has_no_drift() -> None:
    records = [_record(size) for size in range(100, 200, 10)]
    profile = create_reference_profile(records, bins=5)
    report = detect_drift(profile, records)
    assert report["drift_detected"] is False


def test_shifted_distribution_detects_drift() -> None:
    reference = [_record(size, 1) for size in range(100, 200, 10)]
    production = [_record(size, 8) for size in range(1000, 2000, 100)]
    report = detect_drift(create_reference_profile(reference, bins=5), production)
    assert report["drift_detected"] is True
