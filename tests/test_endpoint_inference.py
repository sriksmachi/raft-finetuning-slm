import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lib.endpoint_inference import evaluate_endpoint, invoke_endpoint
from lib.prompts import question_context_instruction


@pytest.mark.parametrize("double_encoded", [False, True])
def test_invoke_endpoint_preserves_payload_and_cleans_request(double_encoded):
    requests = []
    predictions = [{"prediction": "answer", "latency_ms": 12}]

    def invoke(**kwargs):
        path = Path(kwargs["request_file"])
        requests.append(path)
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "input_data": [{"instruction": "question"}]
        }
        assert kwargs["endpoint_name"] == "endpoint"
        assert kwargs["deployment_name"] == "candidate"
        response = json.dumps({"predictions": predictions})
        return json.dumps(response) if double_encoded else response

    client = SimpleNamespace(online_endpoints=SimpleNamespace(invoke=Mock(side_effect=invoke)))
    assert invoke_endpoint(
        client, "endpoint", [{"instruction": "question"}], deployment_name="candidate"
    ) == predictions
    assert not requests[0].exists()


def test_evaluate_endpoint_preserves_batch_order_and_record_fields():
    records = [
        {"id": str(index), "type": "answer", "question": f"question-{index}",
         "instruction": f"input-{index}", "context": "context", "cot_answer": "reference"}
        for index in range(3)
    ]
    batches = []

    def invoke(**kwargs):
        items = json.loads(Path(kwargs["request_file"]).read_text(encoding="utf-8"))["input_data"]
        batches.append(items)
        return json.dumps({"predictions": [
            {"prediction": item["instruction"], "latency_ms": 10} for item in items
        ]})

    client = SimpleNamespace(online_endpoints=SimpleNamespace(invoke=Mock(side_effect=invoke)))
    results = evaluate_endpoint(client, "endpoint", records, batch_size=2)
    assert [len(batch) for batch in batches] == [2, 1]
    assert [result["id"] for result in results] == ["0", "1", "2"]
    assert [result["prediction"] for result in results] == [
        question_context_instruction(record["question"], record["context"])
        for record in records
    ]
    assert all(result["reference"] == "reference" and result["latency_ms"] == 10 for result in results)


def test_invoke_endpoint_cleans_request_on_failure():
    requests = []

    def invoke(**kwargs):
        requests.append(Path(kwargs["request_file"]))
        raise RuntimeError("Endpoint unavailable")

    client = SimpleNamespace(online_endpoints=SimpleNamespace(invoke=Mock(side_effect=invoke)))
    with pytest.raises(RuntimeError, match="Endpoint unavailable"):
        invoke_endpoint(client, "endpoint", [{"instruction": "question"}])
    assert not requests[0].exists()
