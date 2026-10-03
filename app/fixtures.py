"""Deterministic demo evidence. No source documents or remote services are read."""

from datetime import datetime, timezone

from app.models import Parameters


STAGES = ["data_preparation", "training", "evaluation", "decision", "deployment"]
MODELS = [
    {"id": "demo-phi-3-mini", "label": "Phi-3 Mini (simulated)"},
    {"id": "demo-llama-3.2-3b", "label": "Llama 3.2 3B (simulated)"},
]
METRICS = [
    {"key": "faithfulness", "label": "Faithfulness", "unit": "score", "direction": "higher"},
    {"key": "accuracy", "label": "Accuracy", "unit": "score", "direction": "higher"},
    {"key": "relevancy", "label": "Relevancy", "unit": "score", "direction": "higher"},
    {"key": "latency", "label": "Latency", "unit": "ms", "direction": "lower"},
    {"key": "token_cost", "label": "Token cost", "unit": "USD", "direction": "lower"},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def audit(record: dict, action: str, actor: str = "local-demo-user", **details) -> None:
    record["audit"].append(
        {"action": action, "actor": actor, "timestamp": now(), "simulated": True, **details}
    )


def new_record(run_id: str, source: str, recipe: str, scenario: str) -> dict:
    timestamp = now()
    record = {
        "schema_version": 1, "mode": "demo", "simulated": True, "run_id": run_id,
        "created_at": timestamp, "updated_at": timestamp,
        "current_stage": "data_preparation", "status": "ready", "scenario": scenario,
        "stages": {
            stage: {"status": "not_started", "progress": 0, "error": None,
                    "started_at": None, "completed_at": None}
            for stage in STAGES
        },
        "source_folder": source, "recipe": recipe,
        "data_card": None, "training": None, "evaluation": None,
        "decisions": {"registration": None, "deployment": None},
        "deployment": None, "audit": [],
    }
    audit(record, "run_created", source_folder=source, recipe=recipe)
    return record


def data_card(record: dict) -> dict:
    run_id = record["run_id"]
    return {
        "simulated": True, "source": "demo_fixture",
        "source_folder": record["source_folder"], "source_is_label_only": True,
        "source_refs": ["demo://documents/employee-handbook.pdf",
                        "demo://documents/benefits-faq.pdf"],
        "source_references": ["demo://documents/employee-handbook.pdf",
                              "demo://documents/benefits-faq.pdf"],
        "recipe": record["recipe"],
        "recipe_description": (
            "Simulated answers restricted to supplied passages."
            if record["recipe"] == "context-grounded"
            else "Demo-only illustrative recipe; not a defined production generation recipe."
        ),
        "document_count": 2, "chunk_count": 64, "record_count": 240,
        "splits": {"train": 168, "validation": 36, "test": 36},
        "split_counts": {"train": 168, "validation": 36, "test": 36},
        "split_policy": "Sealed document-group split; illustrative fixture, no live generation.",
        "validation": {"simulated": True, "status": "passed", "checked_records": 240,
                       "invalid_records": 0, "checks": ["schema", "nonempty_context", "split_disjointness"]},
        "data_fingerprint": "demo-sha256:handbook-v1-sealed-36",
        "fingerprint": "demo-sha256:handbook-v1-sealed-36",
        "asset_reference": f"demo://{run_id}/data",
        "assets": {split: f"demo://{run_id}/data/{split}.jsonl"
                   for split in ["train", "validation", "test"]},
    }


def training_result(record: dict, parameters: dict) -> dict:
    run_id = record["run_id"]
    return {
        "simulated": True, "config": parameters, "candidate_id": f"{run_id}-candidate-v1",
        "job_name": f"{run_id}-training", "job_uri": f"demo://{run_id}/jobs/training",
        "model_artifact": f"demo://{run_id}/artifacts/candidate-v1",
        "data_fingerprint": record["data_card"]["data_fingerprint"],
        "train_asset": record["data_card"]["assets"]["train"],
        "validation_asset": record["data_card"]["assets"]["validation"],
        "estimated_cost_usd": 0.42, "cost_is_simulated": True,
        "compute": "demo-gpu-simulator", "seed": 42,
    }


def evaluation_result(record: dict) -> dict:
    training = record["training"]
    blocked = record["scenario"] == "blocked"
    candidate_values = [0.78, 0.76, 0.81, 180.0, 0.00004] if blocked else [
        0.94, 0.92, 0.95, 120.0, 0.00004
    ]
    rows = []
    for role, model_id, values, validity in [
        ("candidate", training["candidate_id"], candidate_values, 0.86 if blocked else 0.97),
        ("base_slm", training["config"]["base_model"], [0.72, 0.69, 0.75, 105.0, 0.00003], 0.81),
        ("gpt", "demo-gpt-baseline", [0.91, 0.90, 0.93, 420.0, 0.0012], 0.95),
    ]:
        rows.append({
            "simulated": True, "role": role, "id": model_id, "sample_count": 36,
            "metrics": dict(zip([metric["key"] for metric in METRICS], values)),
            "json_validity": validity, "test_split": record["data_card"]["assets"]["test"],
            "data_fingerprint": record["data_card"]["data_fingerprint"],
            "artifact": training["model_artifact"] if role == "candidate" else f"demo://models/{model_id}",
        })
    return {
        "simulated": True, "source": "demo_fixture",
        "candidate_id": training["candidate_id"],
        "candidate_artifact": training["model_artifact"],
        "test_split": record["data_card"]["assets"]["test"],
        "data_fingerprint": record["data_card"]["data_fingerprint"],
        "sample_count": 36, "sealed": True, "models": rows,
        "scoring_policy": {
            "simulated": True,
            "id": "demo-handbook-rubric-v1",
            "faithfulness": "Fixture support score in [0,1]; no judge call.",
            "accuracy": "Fixture reference-match score in [0,1].",
            "relevancy": "Fixture question relevance score in [0,1].",
            "latency": "Illustrative median response milliseconds, not measured.",
            "token_cost": "Illustrative USD allocation per response, not actual billing.",
        },
        "cost_assumptions": {"simulated": True, "input_tokens": 800, "output_tokens": 120,
                             "allocation": "Fixed fixture USD per response; SLM amortized GPU, GPT illustrative tokens.",
                             "actual_charges_usd": 0},
        "gate": {
            "simulated": True, "status": "blocked" if blocked else "passed",
            "policy": {
                "id": "demo-json-validity-latency-v1", "simulated": True,
                "description": "Illustrative original JSON-validity/latency comparison, not a production gate.",
                "rules": ["candidate.json_validity >= gpt.json_validity",
                          "candidate.latency <= gpt.latency"],
            },
            "values": {"candidate_json_validity": rows[0]["json_validity"],
                       "gpt_json_validity": rows[2]["json_validity"],
                       "candidate_latency_ms": candidate_values[3], "gpt_latency_ms": 420.0},
            "message": ("Candidate JSON validity falls below GPT fixture; registration blocked."
                        if blocked else "Candidate meets simulated JSON-validity and latency comparisons."),
        },
        "recommendation": {
            "simulated": True, "source": "demo_fixture", "ready": not blocked,
            "summary": ("Do not deploy without an explicit, justified override."
                        if blocked else "Ready for an illustrative demo deployment after human confirmation."),
            "advisory_only": True, "limitations": ["36 synthetic sealed examples", "No real quality or billing measurements"],
        },
        "completed_at": now(),
    }


def deployment_result(record: dict) -> dict:
    run_id = record["run_id"]
    return {
        "simulated": True, "endpoint_name": f"{run_id}-endpoint",
        "endpoint_uri": f"demo://{run_id}/endpoints/online",
        "candidate_id": record["training"]["candidate_id"],
        "registration": record["decisions"]["registration"]["model_name"],
        "job_uri": f"demo://{run_id}/jobs/deployment",
        "evaluation": {"simulated": True, "status": "passed", "smoke_test_count": 12,
                       "successful_responses": 12, "actual_requests_sent": 0},
        "monitoring": {"simulated": True, "status": "healthy", "window": "illustrative 24h",
                       "request_count": 1200, "error_rate": 0.002, "p95_latency_ms": 165,
                       "drift_score": 0.03, "alerts": [],
                       "uri": f"demo://{run_id}/monitoring"},
        "completed_at": now(),
    }


def seed_records() -> list[dict]:
    records = []
    for run_id, scenario, timestamp in [
        ("demo-completed", "passing", "2026-09-29T09:00:00Z"),
        ("demo-blocked", "blocked", "2026-09-30T10:00:00Z"),
        ("demo-training-failure", "failure", "2026-10-01T11:00:00Z"),
    ]:
        record = new_record(run_id, "Demo employee handbook (label only)", "context-grounded", scenario)
        record["created_at"] = timestamp
        record["data_card"] = data_card(record)
        record["training"] = training_result(record, Parameters().model_dump())
        for stage in ["data_preparation", "training", "evaluation"]:
            record["stages"][stage].update(status="succeeded", progress=100,
                                           started_at=timestamp, completed_at=timestamp)
        record["current_stage"] = "decision"
        if scenario == "failure":
            record["status"] = "failed"
            record["current_stage"] = "training"
            record["stages"]["training"].update(status="failed", progress=65,
                                                 error="Simulated training failure: GPU allocation interrupted.")
            record["stages"]["evaluation"].update(status="not_started", progress=0,
                                                   started_at=None, completed_at=None)
            audit(record, "training_failed", error=record["stages"]["training"]["error"])
        else:
            record["evaluation"] = evaluation_result(record)
            audit(record, "evaluation_succeeded", evidence_candidate_id=record["training"]["candidate_id"])
            if scenario == "passing":
                record["decisions"]["registration"] = {
                    "simulated": True, "model_name": f"{run_id}-registered-model", "version": "1",
                    "override": False, "reason": "Simulated gate passed.", "actor": "local-demo-user",
                    "decided_at": timestamp, "evidence_candidate_id": record["training"]["candidate_id"],
                    "evidence_data_fingerprint": record["data_card"]["data_fingerprint"],
                    "evidence_test_split": record["evaluation"]["test_split"], "gate_status": "passed",
                }
                record["decisions"]["deployment"] = {
                    "simulated": True, "actor": "local-demo-user", "decided_at": timestamp,
                    "candidate_id": record["training"]["candidate_id"], "confirmed": True,
                }
                for stage in ["decision", "deployment"]:
                    record["stages"][stage].update(status="succeeded", progress=100,
                                                   started_at=timestamp, completed_at=timestamp)
                record["deployment"] = deployment_result(record)
                record["current_stage"] = "deployment"
                record["status"] = "completed"
                audit(record, "registration_approved", override=False, reason="Simulated gate passed.")
                audit(record, "deployment_confirmed")
                audit(record, "deployment_succeeded", endpoint_uri=record["deployment"]["endpoint_uri"])
        record["updated_at"] = timestamp
        for event in record["audit"]:
            event["timestamp"] = timestamp
        if record["evaluation"]:
            record["evaluation"]["completed_at"] = timestamp
        if record["deployment"]:
            record["deployment"]["completed_at"] = timestamp
        records.append(record)
    return records
