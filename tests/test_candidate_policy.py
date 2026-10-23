import copy
import json

import pytest

from lib.release import RAG_METRICS as METRICS
from lib.release import assess_candidate_rows
from lib.token_costs import load_prices, summarize_usage


def scored_rows(count=30, *, candidate=.8, original=.7, gpt=.82):
    return [
        {f"{model}_{metric}_score": value
         for model, value in (("fine_tuned", candidate), ("oob", original), ("gpt", gpt))
         for metric in METRICS}
        for _ in range(count)
    ]


def test_fixed_policy_is_paired_deterministic_and_quality_only():
    rows = scored_rows()
    for row in rows:
        row["fine_tuned_slm_latency_ms"] = 1e9
        row["token_cost"] = 1e9
    result = assess_candidate_rows(rows)
    assert result == assess_candidate_rows(rows)
    assert result["passed"]
    assert result["policy"]["editable"] is False
    assert all(metric["uplift_ci95"] == pytest.approx([.1, .1]) for metric in result["metrics"])


@pytest.mark.parametrize("count,ready", [(29, False), (30, True)])
def test_exact_minimum_examples(count, ready):
    assert assess_candidate_rows(scored_rows(count))["passed"] is ready


@pytest.mark.parametrize("candidate,original,gpt,ready", [
    (.72, .7, .74, True),  # Exactly two percentage points is sufficient.
    (.7199, .7, .74, False),
    (.76, .7, .8, False),  # Exactly five percent shortfall is NOT sufficient.
    (.760001, .7, .8, True),
    (.85, .7, .8, True),
    (.02, 0, 0, True),
])
def test_scientific_threshold_boundaries(candidate, original, gpt, ready):
    assert assess_candidate_rows(scored_rows(candidate=candidate, original=original, gpt=gpt))["passed"] is ready


def test_positive_mean_without_positive_paired_confidence_interval_is_blocked():
    rows = scored_rows(candidate=.625, original=.6, gpt=.64)
    for index, row in enumerate(rows):
        row["fine_tuned_correctness_score"] = .9 if index < 15 else .35
    result = assess_candidate_rows(rows)
    assert result["metrics"][2]["absolute_uplift"] > .02
    assert result["metrics"][2]["checks"]["gpt_gap"]
    assert not result["metrics"][2]["checks"]["positive_confidence_interval"]
    assert not result["passed"]


def test_perfect_original_baseline_allows_only_a_perfect_tie():
    result = assess_candidate_rows(scored_rows(candidate=1, original=1, gpt=1))
    assert result["passed"]
    assert result["policy"]["version"] == "paired-quality-v2"
    for metric in result["metrics"]:
        assert metric["baseline_at_ceiling"]
        assert metric["ceiling_exception_applied"]
        assert metric["absolute_uplift"] == 0
        assert metric["uplift_ci95"] == [0, 0]
        assert all(metric["checks"].values())


@pytest.mark.parametrize("score", [.99, .9999999999999999])
def test_any_candidate_regression_from_perfect_baseline_is_blocked(score):
    rows = scored_rows(candidate=1, original=1, gpt=1)
    rows[0]["fine_tuned_correctness_score"] = score
    result = assess_candidate_rows(rows)
    assert not result["passed"]
    metric = result["metrics"][2]
    assert metric["baseline_at_ceiling"]
    assert not metric["ceiling_exception_applied"]
    assert not metric["checks"]["uplift"]
    assert not metric["checks"]["positive_confidence_interval"]
    assert "must remain exactly 100%" in result["reasons"][0]


@pytest.mark.parametrize("count", [2, 29])
def test_perfect_tie_does_not_waive_minimum_examples(count):
    result = assess_candidate_rows(scored_rows(count, candidate=1, original=1, gpt=1))
    assert not result["passed"]
    assert all(metric["ceiling_exception_applied"] for metric in result["metrics"])
    assert all(not metric["passed"] for metric in result["metrics"])


@pytest.mark.parametrize("score", [.999, .9999999999999999])
def test_nearly_perfect_original_baseline_still_requires_normal_uplift(score):
    rows = scored_rows(candidate=1, original=1, gpt=1)
    rows[0]["oob_correctness_score"] = score
    result = assess_candidate_rows(rows)
    assert not result["passed"]
    assert not result["metrics"][2]["baseline_at_ceiling"]
    assert not result["metrics"][2]["checks"]["uplift"]


def test_ceiling_exception_is_independent_for_each_metric():
    rows = scored_rows()
    for row in rows:
        row["oob_faithfulness_score"] = row["fine_tuned_faithfulness_score"] = 1
    result = assess_candidate_rows(rows)
    assert result["passed"]
    assert [metric["ceiling_exception_applied"] for metric in result["metrics"]] == [True, False, False]
    for row in rows:
        row["fine_tuned_relevancy_score"] = .7
    result = assess_candidate_rows(rows)
    assert not result["passed"]
    assert result["metrics"][0]["passed"]
    assert not result["metrics"][1]["checks"]["positive_confidence_interval"]


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1, 1.01])
def test_invalid_or_missing_individual_scores_are_rejected(value):
    rows = scored_rows()
    rows[0]["oob_relevancy_score"] = value
    with pytest.raises(ValueError):
        assess_candidate_rows(rows)


def usage_rows():
    return [{"gpt_model_id": "gpt-4.1-2025-04-14",
             **{f"{model}_judge_model_id": "gpt-5.2" for model in ("gpt", "oob", "fine_tuned")},
             **{f"{model}_{suffix}_token_count": value
                for model in ("gpt", "oob", "fine_tuned")
                for suffix, value in (("prompt", 100), ("generated", 20), ("judge_input", 10), ("judge_output", 5))}}]


def test_unset_prices_are_not_zero_and_missing_usage_is_explicit():
    prices = load_prices()
    result = summarize_usage(usage_rows(), prices, baseline_reused=True)
    assert result["models"]["gpt"]["input_tokens"] == 100
    assert result["models"]["gpt"]["estimated_cost_usd"] is None
    assert result["models"]["judge"]["input_tokens"] == 30
    assert result["baseline_reused"]
    legacy = summarize_usage([{}], prices, baseline_reused=True)
    assert legacy["models"]["fine_tuned"]["input_tokens"] is None
    assert legacy["models"]["fine_tuned"]["status"] == "self_hosted_no_token_price"


def test_costs_use_configured_per_million_rates_and_preserve_snapshot(tmp_path):
    document = copy.deepcopy(load_prices()["configuration"])
    document["models"]["gpt-4.1"] = {"input": 2, "output": 8}
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(document))
    prices = load_prices(path)
    result = summarize_usage(usage_rows(), prices, baseline_reused=False)
    assert result["models"]["gpt"]["estimated_cost_usd"] == pytest.approx(.00036)
    assert result["price_snapshot"] == prices
    document["models"]["gpt-4.1"]["input"] = -1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="nonnegative"):
        load_prices(path)
    assert result["price_snapshot"]["configuration"]["models"]["gpt-4.1"]["input"] == 2


def test_prices_only_apply_to_exact_frontier_model_families_and_judge(tmp_path):
    document = load_prices()["configuration"]
    document["models"]["gpt-4.1"] = {"input": 100, "output": 100}
    document["models"]["gpt-4.1-mini"] = {"input": 1, "output": 4}
    document["models"]["gpt-5.2"] = {"input": 2, "output": 8}
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(document))
    rows = usage_rows()
    rows[0]["gpt_model_id"] = "gpt-4.1-mini-2025-04-14"
    result = summarize_usage(rows, load_prices(path), baseline_reused=False)
    assert result["models"]["gpt"]["estimated_cost_usd"] == pytest.approx(.00018)
    assert result["models"]["judge"]["estimated_cost_usd"] == pytest.approx(.00018)
    assert result["models"]["gpt"]["price_model_ids"] == ["gpt-4.1-mini"]
    assert result["models"]["fine_tuned"]["estimated_cost_usd"] is None
    assert result["models"]["fine_tuned"]["status"] == "self_hosted_no_token_price"
    rows[0]["gpt_model_id"] = "custom-deployment-name"
    assert summarize_usage(rows, load_prices(path), baseline_reused=False)["models"]["gpt"]["estimated_cost_usd"] is None
    document["models"]["fine_tuned"] = {"input": 1, "output": 1}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="explicit frontier model ID"):
        load_prices(path)
