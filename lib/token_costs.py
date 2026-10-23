"""Measured evaluation token usage with explicit, snapshotted price estimates."""

import hashlib
import json
import math
import re
from pathlib import Path

MODELS = ("gpt", "oob", "fine_tuned", "judge")
DEFAULT_PRICES = Path(__file__).resolve().parents[1] / "token_prices.json"


def load_prices(path: Path = DEFAULT_PRICES) -> dict:
    raw = path.read_bytes()
    prices = json.loads(raw)
    if (not isinstance(prices, dict) or prices.get("schema_version") != 1 or prices.get("currency") != "USD"
            or prices.get("unit") != "per_million_tokens"
            or not isinstance(prices.get("models"), dict) or not prices["models"]):
        raise ValueError("Invalid token price schema; use USD per_million_tokens keyed by explicit frontier model IDs.")
    for model, rates in prices["models"].items():
        if not re.fullmatch(r"(?:gpt-[a-z0-9][a-z0-9.-]*|o[134](?:-[a-z0-9.-]+)?|claude-[a-z0-9][a-z0-9.-]*)", model):
            raise ValueError(f"Use an explicit frontier model ID, not a role or deployment alias: {model}")
        if not isinstance(rates, dict) or set(rates) != {"input", "output"}:
            raise ValueError(f"Invalid {model} token price fields.")
        for rate in rates.values():
            if rate is not None and (
                isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate < 0
            ):
                raise ValueError("Token prices must be null or finite nonnegative numbers.")
    return {"configuration": prices, "sha256": hashlib.sha256(raw).hexdigest()}


def _count(value) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise TypeError("Token counts cannot be booleans.")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Token counts must be nonnegative integers.")
    return int(number)


def _price_model(model_id, prices: dict) -> str | None:
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    models = prices["configuration"]["models"]
    if model_id in models:
        return model_id
    # Azure returns dated snapshots; never confuse GPT-4.1 with GPT-4.1-mini.
    family = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_id)
    return family if family in models else None


def summarize_usage(rows: list[dict], prices: dict, *, baseline_reused: bool) -> dict:
    if not rows:
        raise ValueError("Token usage requires evaluation records.")
    result = {}
    for model in MODELS:
        counts = {}
        for direction in ("input", "output"):
            if model == "judge":
                fields = [f"{prefix}_judge_{direction}_token_count" for prefix in MODELS[:3]]
                measured = [_count(row.get(field)) for row in rows for field in fields]
            else:
                field = f"{model}_{'prompt' if direction == 'input' else 'generated'}_token_count"
                measured = [_count(row.get(field)) for row in rows]
            counts[f"{direction}_tokens"] = sum(measured) if all(n is not None for n in measured) else None
        complete = all(counts[f"{d}_tokens"] is not None for d in ("input", "output"))
        identities = []
        price_models = []
        costs = []
        configured = True
        if model in {"gpt", "judge"}:
            for row in rows:
                prefixes = MODELS[:3] if model == "judge" else ("gpt",)
                for prefix in prefixes:
                    identity = row.get(f"{prefix}_judge_model_id" if model == "judge" else "gpt_model_id")
                    if isinstance(identity, str) and identity.strip():
                        identities.append(identity)
                    price_model = _price_model(identity, prices)
                    if price_model is not None:
                        price_models.append(price_model)
                    rates = prices["configuration"]["models"].get(price_model)
                    if rates is None or any(rates[d] is None for d in ("input", "output")):
                        configured = False
                        continue
                    for direction, suffix in (("input", "prompt"), ("output", "generated")):
                        key = f"{prefix}_judge_{direction}_token_count" if model == "judge" else f"gpt_{suffix}_token_count"
                        count = _count(row.get(key))
                        if count is not None:
                            costs.append(count * rates[direction] / 1_000_000)
        cost = sum(costs) if configured and complete and model in {"gpt", "judge"} else None
        self_hosted = model in {"oob", "fine_tuned"}
        result[model] = {
            **counts, "total_tokens": sum(counts.values()) if complete else None, "estimated_cost_usd": cost,
            "model_ids": sorted(set(identities)), "price_model_ids": sorted(set(price_models)),
            "status": "self_hosted_no_token_price" if self_hosted else "estimated" if cost is not None else "usage_not_recorded" if not complete else "model_or_price_not_configured",
            "basis": "Self-hosted: no per-token tariff. GPU hourly costs are separate." if self_hosted else "Configured model-specific standard-input/output API prices; cached-input discounts are not applied.",
        }
    return {
        "models": result, "price_snapshot": prices, "baseline_reused": baseline_reused,
        "note": "GPT and its judge usage describe baseline evidence, not new charges when reused. Judge totals include all three models; provider retries with no returned usage are not measurable.",
    }
