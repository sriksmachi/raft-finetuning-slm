import argparse
import copy
import json
import math
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from lib.monitoring import create_reference_profile, detect_drift, population_stability_index


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


@pytest.mark.parametrize("bins", [0, -1, 1.5, True])
def test_invalid_bin_count_fails(bins) -> None:
    with pytest.raises(ValueError, match="bins"):
        create_reference_profile([_record(100)], bins=bins)


@pytest.mark.parametrize("record", [None, [], {}, {"instruction": "text"},
                                          {"instruction": None, "question": "q"},
                                          {"instruction": "text", "question": 1}])
def test_malformed_request_records_fail(record) -> None:
    with pytest.raises(ValueError, match="instruction and question"):
        create_reference_profile([record])
    with pytest.raises(ValueError, match="instruction and question"):
        detect_drift(create_reference_profile([_record(100)]), [record])


@pytest.mark.parametrize("threshold", [0, -1, math.nan, math.inf, -math.inf, True, "0.2"])
def test_invalid_threshold_fails(threshold) -> None:
    profile = create_reference_profile([_record(100)])
    with pytest.raises(ValueError, match="threshold"):
        detect_drift(profile, [_record(100)], threshold)


@pytest.mark.parametrize(
    "reference,observed",
    [
        ([], []),
        ([1.0], [0.5, 0.5]),
        ([0.0, 1.0], [0.5, 0.5]),
        ([-0.1, 1.1], [0.5, 0.5]),
        ([math.nan, 0.5], [0.5, 0.5]),
        ([0.5, 0.5], [math.inf, 0.5]),
        ([0.2, 0.2], [0.5, 0.5]),
        ([[0.5, 0.5]], [0.5, 0.5]),
        (None, [1.0]),
        (["0.5", "0.5"], [0.5, 0.5]),
        ([True], [1.0]),
    ],
)
def test_invalid_psi_vectors_fail(reference, observed) -> None:
    with pytest.raises(ValueError):
        population_stability_index(reference, observed)


def test_psi_known_value_and_extreme_positive_probabilities() -> None:
    assert population_stability_index([0.5, 0.5], [0.25, 0.75]) == pytest.approx(
        0.25 * math.log(3)
    )
    assert math.isfinite(population_stability_index([1e-320, 1.0], [1.0, 1e-320]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", math.nan), ("std", math.inf), ("std", -1),
        ("edges", [0, math.inf]), ("edges", [1, 1]), ("edges", [2, 1]),
        ("edges", [0]), ("edges", [[0, 1]]), ("edges", "bad"),
        ("edges", ["0", "1"]),
        ("probabilities", [0.5, 0.5]), ("probabilities", [0]), ("probabilities", None),
    ],
)
def test_invalid_baseline_features_fail(field, value) -> None:
    profile = create_reference_profile([_record(100)])
    profile["features"]["instruction_chars"][field] = value
    with pytest.raises(ValueError):
        detect_drift(profile, [_record(100)])


@pytest.mark.parametrize("profile", [None, {}, {"row_count": 1, "features": {}}])
def test_invalid_profile_structure_fails(profile) -> None:
    with pytest.raises(ValueError):
        detect_drift(profile, [_record(100)])


def test_empty_populations_and_invalid_profile_count_fail() -> None:
    with pytest.raises(ValueError, match="Reference records"):
        create_reference_profile([])
    profile = create_reference_profile([_record(100)])
    with pytest.raises(ValueError, match="Production records"):
        detect_drift(profile, [])
    for count in [0, -1, True, 1.5, math.inf]:
        invalid = copy.deepcopy(profile)
        invalid["row_count"] = count
        with pytest.raises(ValueError, match="row_count"):
            detect_drift(invalid, [_record(100)])


def test_out_of_range_observations_are_folded_into_tail_bins() -> None:
    reference = [_record(size) for size in [100, 200, 300, 400]]
    profile = create_reference_profile(reference, bins=2)
    report = detect_drift(profile, [_record(0), _record(10000)])
    assert report["row_count"] == 2
    assert report["features"]["instruction_chars"]["psi"] == pytest.approx(0)
    assert all(math.isfinite(feature["psi"]) for feature in report["features"].values())


@pytest.fixture
def monitoring_files():
    directory = Path(__file__).parent / f".monitoring-test-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def test_command_reads_normalized_production_and_logs_metrics(monkeypatch, monitoring_files) -> None:
    from lib import monitor_drift

    reference = [_record(size) for size in range(100, 200, 10)]
    production = [_record(size, 8) for size in range(1000, 2000, 100)]
    profile_path = monitoring_files / "reference_profile.json"
    profile_path.write_text(json.dumps(create_reference_profile(reference)), encoding="utf-8")
    folder = monitoring_files / "production"
    folder.mkdir()
    (folder / "requests.jsonl").write_text(
        "\n".join(json.dumps(record) for record in production), encoding="utf-8"
    )
    mlflow = MagicMock()
    monkeypatch.setattr(monitor_drift, "mlflow", mlflow)
    output = monitoring_files / "report"
    monitor_drift.main(argparse.Namespace(
        reference_profile=str(profile_path), production_data=str(folder),
        output=str(output), psi_threshold=0.2,
    ))
    report = json.loads((output / "drift_report.json").read_text(encoding="utf-8"))
    assert report["drift_detected"]
    mlflow.log_metric.assert_any_call("drift_detected", 1.0)
    for feature, values in report["features"].items():
        mlflow.log_metric.assert_any_call(f"psi_{feature}", values["psi"])
    mlflow.log_artifact.assert_called_once_with(
        str(output / "drift_report.json"), artifact_path="monitoring"
    )


@pytest.mark.parametrize("content", ["not json", '{"instruction":"text"}', "null", ""])
def test_command_malformed_production_fails_without_logging(
    monkeypatch, monitoring_files, content
) -> None:
    from lib import monitor_drift

    profile = monitoring_files / "reference_profile.json"
    profile.write_text(json.dumps(create_reference_profile([_record(100)])), encoding="utf-8")
    production = monitoring_files / "requests.jsonl"
    production.write_text(content, encoding="utf-8")
    mlflow = MagicMock()
    monkeypatch.setattr(monitor_drift, "mlflow", mlflow)
    with pytest.raises(ValueError):
        monitor_drift.main(argparse.Namespace(
            reference_profile=str(profile), production_data=str(production),
            output=str(monitoring_files / "report"), psi_threshold=0.2,
        ))
    mlflow.start_run.assert_not_called()
    assert not (monitoring_files / "report").exists()
