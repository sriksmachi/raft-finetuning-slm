from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lib.azureml_ops import promote_deployment


def _client(traffic):
    endpoint = SimpleNamespace(traffic=traffic)
    client = MagicMock()
    client.online_endpoints.get.return_value = endpoint
    client.online_endpoints.begin_create_or_update.return_value.result.return_value = endpoint
    return client, endpoint


def test_full_promotion_routes_all_traffic_to_candidate() -> None:
    client, endpoint = _client({"blue": 100})
    result = promote_deployment(client, "endpoint", "green")
    assert result is endpoint
    assert endpoint.traffic == {"green": 100}


def test_canary_promotion_preserves_single_incumbent() -> None:
    client, endpoint = _client({"blue": 100})
    promote_deployment(client, "endpoint", "green", traffic_percent=10)
    assert endpoint.traffic == {"blue": 90, "green": 10}


@pytest.mark.parametrize("traffic", [{}, {"blue": 50, "yellow": 50}])
def test_canary_requires_exactly_one_incumbent(traffic) -> None:
    client, _ = _client(traffic)
    with pytest.raises(ValueError, match="incumbent deployment"):
        promote_deployment(client, "endpoint", "green", traffic_percent=10)
    client.online_endpoints.begin_create_or_update.assert_not_called()


@pytest.mark.parametrize("percent", [0, 101, 10.5, True])
def test_invalid_traffic_percent_fails_before_azure(percent) -> None:
    client, _ = _client({"blue": 100})
    with pytest.raises(ValueError, match="integer between 1 and 100"):
        promote_deployment(client, "endpoint", "green", traffic_percent=percent)
    client.online_endpoints.get.assert_not_called()
