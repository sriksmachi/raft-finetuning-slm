import logging
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dotenv import dotenv_values

import lib.azureml_ops as operations
from test_lora_artifacts import ROOT, _functions
from lib.azureml_ops import (
    TRAIN_ENVIRONMENT_VERSION_KEY,
    _persist_training_environment_version,
    _register_environment,
    create_inference_environment,
    create_training_environment,
    get_training_environment_version,
)


class ResourceExistsError(Exception):
    pass


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    monkeypatch.setattr(operations, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(TRAIN_ENVIRONMENT_VERSION_KEY, "1")
    monkeypatch.setitem(sys.modules, "azure.core.exceptions",
                        SimpleNamespace(ResourceExistsError=ResourceExistsError))
    monkeypatch.setitem(sys.modules, "azure.ai.ml.constants",
                        SimpleNamespace(ListViewType=SimpleNamespace(ALL="all")))
    monkeypatch.setitem(sys.modules, "azure.ai.ml.entities",
                        SimpleNamespace(Environment=lambda **kwargs: SimpleNamespace(**kwargs)))
    return tmp_path


def _conflict():
    return ResourceExistsError("Environment version is already registered and cannot be changed.")


def _client(create, versions=("1", "2")):
    return SimpleNamespace(environments=SimpleNamespace(
        create_or_update=Mock(side_effect=create),
        list=Mock(return_value=[SimpleNamespace(version=version) for version in versions]),
    ))


def test_matching_environment_is_reused_without_new_version(sdk):
    environment = SimpleNamespace(name="training", version="1")
    client = _client(lambda value: value)
    assert _register_environment(client, environment) is environment
    client.environments.list.assert_not_called()
    assert environment.version == "1"


@pytest.mark.parametrize("create_environment", [create_training_environment, create_inference_environment])
def test_immutable_definition_gets_new_version_without_changing_yaml(sdk, caplog, create_environment):
    attempted = []
    definitions = []

    def create(environment):
        attempted.append(environment.version)
        definitions.append((environment.image, environment.conda_file))
        if environment.version == "1":
            raise _conflict()
        return SimpleNamespace(
            name=environment.name, version=environment.version,
            id=f"azureml:{environment.name}:{environment.version}",
        )

    root = Path(__file__).resolve().parents[1]
    train_yaml = root / "environments" / "train-conda.yml"
    original = train_yaml.read_bytes()
    client = _client(create, versions=("1", "2", "release-tag"))
    with caplog.at_level(logging.WARNING):
        registered = create_environment(client, "slm-studio-environment", "1")
    assert registered.id == "azureml:slm-studio-environment:3"
    assert attempted == ["1", "3"]
    assert definitions[0] == definitions[1]
    assert train_yaml.read_bytes() == original
    client.environments.list.assert_called_once_with(name="slm-studio-environment", list_view_type="all")
    assert "new version 3" in caplog.text


def test_registration_handles_concurrent_version_conflicts(sdk):
    attempted = []

    def create(environment):
        attempted.append(environment.version)
        if environment.version in {"1", "3"}:
            raise _conflict()
        return environment

    client = _client(create)
    client.environments.list.side_effect = [
        [SimpleNamespace(version="1"), SimpleNamespace(version="2")],
        [SimpleNamespace(version="1"), SimpleNamespace(version="2"), SimpleNamespace(version="3")],
    ]
    assert _register_environment(client, SimpleNamespace(name="training", version="1")).version == "4"
    assert attempted == ["1", "3", "4"]


def test_repeated_conflicts_are_bounded_and_propagated(sdk):
    client = _client(lambda environment: (_ for _ in ()).throw(_conflict()))
    with pytest.raises(ResourceExistsError, match="cannot be changed"):
        _register_environment(client, SimpleNamespace(name="training", version="1"))
    assert client.environments.create_or_update.call_count == 4


def test_other_conflicts_are_not_silently_retried(sdk):
    client = _client(lambda environment: (_ for _ in ()).throw(ResourceExistsError("Other conflict")))
    with pytest.raises(ResourceExistsError, match="Other conflict"):
        _register_environment(client, SimpleNamespace(name="training", version="1"))
    client.environments.list.assert_not_called()


def test_saved_training_version_is_authoritative_over_stale_process_value(sdk):
    (sdk / ".env").write_text(f"{TRAIN_ENVIRONMENT_VERSION_KEY}=7\n", encoding="utf-8")
    assert get_training_environment_version() == "7"


def test_absent_training_version_has_explicit_initial_default(sdk):
    assert get_training_environment_version() == "1"


def test_empty_training_version_is_rejected(sdk):
    (sdk / ".env").write_text(f"{TRAIN_ENVIRONMENT_VERSION_KEY}=\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be non-empty"):
        get_training_environment_version()


def test_training_reads_saved_version_and_persists_bump_for_next_run(sdk):
    path = sdk / ".env"
    path.write_text(
        f"# Keep this comment\nOTHER_SETTING=keep-me\n{TRAIN_ENVIRONMENT_VERSION_KEY}=2\n",
        encoding="utf-8",
    )
    attempted = []

    def create(environment):
        attempted.append(environment.version)
        if environment.version == "2":
            raise _conflict()
        return environment

    client = _client(create, versions=("1", "2"))
    assert create_training_environment(client, "training").version == "3"
    assert get_training_environment_version() == "3"
    assert dotenv_values(path)["OTHER_SETTING"] == "keep-me"
    assert "# Keep this comment" in path.read_text(encoding="utf-8")
    assert create_training_environment(client, "training").version == "3"
    assert attempted == ["2", "3", "3"]
    assert path.read_text(encoding="utf-8").count(TRAIN_ENVIRONMENT_VERSION_KEY) == 1


def test_failed_training_registration_does_not_update_saved_version(sdk):
    path = sdk / ".env"
    path.write_text(f"{TRAIN_ENVIRONMENT_VERSION_KEY}=2\n", encoding="utf-8")
    original = path.read_bytes()
    client = _client(lambda environment: (_ for _ in ()).throw(RuntimeError("Access denied")))
    with pytest.raises(RuntimeError, match="Access denied"):
        create_training_environment(client, "training")
    assert path.read_bytes() == original


def test_explicit_older_version_does_not_downgrade_saved_latest(sdk):
    path = sdk / ".env"
    path.write_text(f"{TRAIN_ENVIRONMENT_VERSION_KEY}=8\n", encoding="utf-8")
    client = _client(lambda value: value)
    assert create_training_environment(client, "training", "2").version == "2"
    assert get_training_environment_version() == "8"


def test_inference_registration_does_not_replace_training_version(sdk):
    path = sdk / ".env"
    path.write_text(f"{TRAIN_ENVIRONMENT_VERSION_KEY}=8\n", encoding="utf-8")
    client = _client(lambda value: value)
    create_inference_environment(client, "inference", "2")
    assert get_training_environment_version() == "8"


def test_unsloth_environments_pin_torchao_for_torch_29() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("train-conda.yml", "inference-conda.yml"):
        environment = (root / "environments" / name).read_text(encoding="utf-8")
        assert "torch>=2.9,<2.10" in environment
        assert "torchao==0.17.0" in environment


def test_cli_configures_utf8_console_streams() -> None:
    stdout = Mock()
    stderr = Mock()
    functions = _functions(
        ROOT / "scripts" / "02_azureml_fine_tuning.py",
        {"configure_console_encoding"},
        {"sys": SimpleNamespace(stdout=stdout, stderr=stderr)},
    )

    functions["configure_console_encoding"]()

    stdout.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")
    stderr.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")


def test_job_status_lookup_retries_transient_credential_failure() -> None:
    class CredentialUnavailableError(Exception):
        pass

    completed = SimpleNamespace(status="Completed")
    client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=Mock(side_effect=[CredentialUnavailableError(), completed])
        )
    )
    sleep = Mock()
    functions = _functions(
        ROOT / "scripts" / "02_azureml_fine_tuning.py",
        {"get_job_after_stream"},
        {
            "CredentialUnavailableError": CredentialUnavailableError,
            "log": Mock(),
            "time": SimpleNamespace(sleep=sleep),
        },
    )

    assert functions["get_job_after_stream"](client, "job-name", delay=0.25) is completed
    assert client.jobs.get.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_persistence_failure_is_explicit(sdk, monkeypatch):
    import dotenv

    monkeypatch.setattr(dotenv, "set_key", Mock(side_effect=PermissionError("Read-only .env")))
    with pytest.raises(PermissionError, match="Read-only .env"):
        _persist_training_environment_version("3")


def test_cli_can_reuse_saved_environment_without_registration(sdk):
    (sdk / ".env").write_text(f"{TRAIN_ENVIRONMENT_VERSION_KEY}=3\n", encoding="utf-8")
    client = SimpleNamespace(
        data=SimpleNamespace(get=Mock(return_value=SimpleNamespace(id="azureml:data:1"))),
        compute=SimpleNamespace(get=Mock(return_value=SimpleNamespace(name="gpu"))),
        environments=SimpleNamespace(get=Mock(return_value=SimpleNamespace(id="azureml:training:3"))),
    )
    args = argparse.Namespace(
        log_level="INFO", data_asset_name="data", data_asset_version="1", compute_name="gpu",
        skip_environment=True, train_environment_name="training",
        train_environment_version=get_training_environment_version(),
        skip_submit=False, wait=False,
    )
    submit = Mock(return_value=SimpleNamespace(name="job"))
    register = Mock()
    config = SimpleNamespace(create_ml_client=lambda: client)
    functions = _functions(ROOT / "scripts" / "02_azureml_fine_tuning.py", {"main"}, {
        "PROJECT_ROOT": sdk, "load_dotenv": Mock(), "parse_args": lambda: args,
        "logging": logging, "log": logging.getLogger("test"),
        "AzureMLConfig": SimpleNamespace(from_env=lambda: config),
        "register_environment": register, "submit_training": submit,
        "configure_console_encoding": Mock(),
    })
    assert functions["main"]() == 0
    register.assert_not_called()
    client.environments.get.assert_called_once_with("training", version="3")
    submit.assert_called_once_with(client, args, "azureml:training:3")
