import hashlib
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lib.monitoring import create_drift_schedule


@pytest.fixture
def sdk(monkeypatch):
    ml = ModuleType("azure.ai.ml")
    dsl = ModuleType("azure.ai.ml.dsl")
    entities = ModuleType("azure.ai.ml.entities")
    constants = ModuleType("azure.ai.ml.constants")
    constructors = {}
    for module, names in [
        (ml, ["Input", "Output", "command"]),
        (entities, ["JobSchedule", "ManagedIdentityConfiguration",
                    "RecurrencePattern", "RecurrenceTrigger"]),
    ]:
        for name in names:
            constructor = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
            setattr(module, name, constructor)
            constructors[name] = constructor
    class CommandNode(SimpleNamespace):
        def __call__(self):
            self.outputs = SimpleNamespace(**self.outputs)
            return self

    constructors["command"].side_effect = lambda **kwargs: CommandNode(**kwargs)

    def pipeline_factory(**kwargs):
        def decorate(function):
            def build():
                outputs = function()
                job = constructors["command"].call_args
                return SimpleNamespace(
                    jobs={"drift": constructors["command"].side_effect(**job.kwargs)},
                    outputs=outputs,
                    settings=SimpleNamespace(),
                    **kwargs,
                )
            return build
        return decorate

    dsl.pipeline = MagicMock(side_effect=pipeline_factory)
    constructors["pipeline"] = dsl.pipeline
    constants.AssetTypes = SimpleNamespace(URI_FILE="uri_file", URI_FOLDER="uri_folder")
    for name, module in [
        ("azure.ai.ml", ml), ("azure.ai.ml.entities", entities),
        ("azure.ai.ml.dsl", dsl),
        ("azure.ai.ml.constants", constants),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    monkeypatch.setattr(Path, "read_bytes", lambda _: b'{"baseline": "immutable"}')
    return constructors


def _options():
    return {
        "schedule_name": "endpoint-request-drift",
        "reference_profile": Path("baseline.json"),
        "production_data_uri": "azureml://datastores/governed/paths/production/requests/",
        "environment": "azureml:inference:2",
        "compute": "cpu-monitoring",
    }


def test_schedule_uses_package_snapshot_governed_inputs_and_completed_result(sdk) -> None:
    client = MagicMock()
    created = create_drift_schedule(client, **_options())
    assert created is client.schedules.begin_create_or_update.return_value.result.return_value
    client.schedules.begin_create_or_update.return_value.result.assert_called_once_with()
    schedule = client.schedules.begin_create_or_update.call_args.args[0]
    assert schedule.name == "endpoint-request-drift"
    job = schedule.create_job.jobs["drift"]
    assert Path(job.code) == Path(__file__).resolve().parents[1]
    assert job.command.startswith("python -m lib.monitor_drift ")
    for argument in ["reference-profile", "production-data", "output", "psi-threshold"]:
        assert f"--{argument}" in job.command
    assert job.inputs["reference_profile"].type == "uri_file"
    assert job.inputs["reference_profile"].path == str(Path("baseline.json").resolve())
    assert job.inputs["production_data"].type == "uri_folder"
    assert job.inputs["production_data"].path == _options()["production_data_uri"]
    assert job.inputs["psi_threshold"] == 0.2
    assert job.outputs["report"].type == "uri_folder"
    assert job.environment == "azureml:inference:2"
    assert job.compute == "cpu-monitoring"
    assert job.is_deterministic is False
    assert schedule.create_job.settings.force_rerun is True
    assert not hasattr(job, "identity")
    assert schedule.tags == job.tags
    assert schedule.tags["baseline_sha256"] == hashlib.sha256(
        b'{"baseline": "immutable"}'
    ).hexdigest()
    assert schedule.tags["monitoring_schedule"] == schedule.name
    assert schedule.trigger.frequency == "week"
    assert schedule.trigger.interval == 1
    assert schedule.trigger.time_zone == "UTC"
    assert schedule.trigger.schedule.hours == [2]
    assert schedule.trigger.schedule.minutes == [0]


def test_custom_schedule_and_managed_identity(sdk) -> None:
    client = MagicMock()
    create_drift_schedule(client, **_options(), managed_identity_client_id="managed-client",
                          frequency="day", interval=3, hour=0, psi_threshold=0.4)
    schedule = client.schedules.begin_create_or_update.call_args.args[0]
    assert schedule.create_job.jobs["drift"].identity.client_id == "managed-client"
    assert schedule.trigger.frequency == "day"
    assert schedule.trigger.interval == 3
    assert schedule.trigger.schedule.hours == [0]
    assert schedule.create_job.jobs["drift"].inputs["psi_threshold"] == 0.4


@pytest.mark.parametrize(
    "uri", ["azureml:production-requests:7",
            "azureml:/subscriptions/123/resourceGroups/governed/providers/"
            "Microsoft.MachineLearningServices/workspaces/studio/data/requests/versions/7",
            "https://governed.blob.core.windows.net/production/requests/"]
)
def test_supported_governed_uri_is_preserved(sdk, uri) -> None:
    options = _options()
    options["production_data_uri"] = uri
    client = MagicMock()
    create_drift_schedule(client, **options)
    job = client.schedules.begin_create_or_update.call_args.args[0].create_job.jobs["drift"]
    assert job.inputs["production_data"].path == uri


@pytest.mark.parametrize(
    "field,value",
    [
        ("schedule_name", ""), ("environment", " "), ("compute", None),
        ("production_data_uri", ""), ("production_data_uri", "data/test"),
        ("production_data_uri", "http://governed.blob.core.windows.net/requests/"),
        ("production_data_uri", "https://example.com/requests/"),
        ("production_data_uri", "azureml://datastores/<store>/paths/production/"),
        ("production_data_uri", "azureml://datastores/your_store/paths/production/"),
        ("production_data_uri", "azureml:placeholder:1"),
        ("production_data_uri", "azureml://"),
        ("production_data_uri", "https://governed.blob.core.windows.net"),
        ("production_data_uri", "https://user:password@host/requests/"),
        ("production_data_uri", "azureml:requests"),
        ("psi_threshold", 0), ("psi_threshold", -1), ("psi_threshold", math.nan),
        ("psi_threshold", math.inf), ("psi_threshold", True),
        ("frequency", ""), ("frequency", "year"),
        ("interval", 0), ("interval", -1), ("interval", True), ("interval", 1.5),
        ("hour", -1), ("hour", 24), ("hour", True), ("hour", 2.5),
        ("managed_identity_client_id", " "),
    ],
)
def test_invalid_configuration_fails_before_sdk_or_azure(sdk, field, value) -> None:
    options = _options()
    options[field] = value
    client = MagicMock()
    with pytest.raises(ValueError):
        create_drift_schedule(client, **options)
    client.schedules.begin_create_or_update.assert_not_called()
    for constructor in sdk.values():
        constructor.assert_not_called()


def test_missing_reference_fails_before_sdk_or_azure(sdk, monkeypatch) -> None:
    monkeypatch.setattr(Path, "is_file", lambda _: False)
    client = MagicMock()
    with pytest.raises(ValueError, match="existing file"):
        create_drift_schedule(client, **_options())
    client.schedules.begin_create_or_update.assert_not_called()
    sdk["command"].assert_not_called()


def test_schedule_creation_failure_propagates(sdk) -> None:
    client = MagicMock()
    client.schedules.begin_create_or_update.return_value.result.side_effect = RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"):
        create_drift_schedule(client, **_options())
