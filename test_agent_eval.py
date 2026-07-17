import json

import agent_eval


def _payload(*scenarios):
    return {"benchmark": {"dataset_version": "test"}, "scenarios": list(scenarios)}


def _scenario(raw, *, expected="stop", current=None, constraints=None):
    expected_obj = {
        "action": expected,
        "route": {"widen": "fetch", "refocus": "rank", "stop": "report"}[expected],
        **(constraints or {}),
    }
    return {
        "id": "fixture",
        "case_kind": "contract",
        "query": "Tool-using language agents with retrieval and long-term memory",
        "current_categories": current or ["cs.AI", "cs.LG"],
        "top_candidates": [{"title": "Synthetic paper", "score": 2}],
        "expected": expected_obj,
        "rationale": "test fixture",
        "frozen_decision": {"origin": "test", "raw": raw},
    }


def test_offline_mode_never_calls_live_chat():
    scenario = _scenario(
        json.dumps({"action": "stop", "why": "done", "new_categories": []})
    )

    def must_not_call(*_args, **_kwargs):
        raise AssertionError("offline evaluation called the LLM")

    report = agent_eval.evaluate_scenarios(
        _payload(scenario), live=False, live_chat=must_not_call
    )

    assert report["mode"] == "offline-frozen"
    assert report["metrics"]["average_policy_invocations"] == 0
    assert report["results"][0]["actual_action"] == "stop"


def test_illegal_action_uses_production_fail_closed_route():
    scenario = _scenario(json.dumps({"action": "delete", "why": "bad"}))

    report = agent_eval.evaluate_scenarios(_payload(scenario))
    result = report["results"][0]

    assert result["schema_valid"] is False
    assert result["actual_action"] == "stop"
    assert result["actual_route"] == "report"
    assert report["metrics"]["fail_closed_rate"] == {
        "passed": 1,
        "total": 1,
        "rate": 1.0,
    }


def test_widen_reuses_graph_allowlist_and_existing_category_filter():
    scenario = _scenario(
        json.dumps(
            {
                "action": "widen",
                "new_categories": ["cs.CL", "cs.ZZ", "cs.IR"],
                "why": "expand",
            }
        ),
        expected="widen",
        current=["cs.AI", "cs.LG", "cs.CL"],
        constraints={"new_categories_any_of": ["cs.IR"]},
    )

    report = agent_eval.evaluate_scenarios(_payload(scenario))
    result = report["results"][0]

    assert result["actual_action"] == "widen"
    assert result["added_categories"] == ["cs.IR"]
    assert result["constraint_pass"] is True


def test_live_mode_calls_once_and_still_uses_production_validation():
    scenario = _scenario("not used")
    calls = []

    def fake_live(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps({"action": "stop", "why": "live"})

    report = agent_eval.evaluate_scenarios(
        _payload(scenario), live=True, live_chat=fake_live
    )

    assert len(calls) == 1
    assert report["metrics"]["average_policy_invocations"] == 1
    assert report["results"][0]["schema_valid"] is True


def test_public_dataset_has_required_synthetic_coverage():
    payload = agent_eval.load_scenarios()
    scenarios = payload["scenarios"]
    ids = {scenario["id"] for scenario in scenarios}

    assert 8 <= len(scenarios) <= 12
    assert all(scenario["query"] for scenario in scenarios)
    assert all("top_candidates" in scenario for scenario in scenarios)
    assert all("current_categories" in scenario for scenario in scenarios)
    assert all("expected" in scenario and "rationale" in scenario for scenario in scenarios)
    assert "malicious_title_ignored" in ids
    assert "only_existing_category_fails_closed" in ids
    assert "illegal_action_fails_closed" in ids


def test_static_baselines_only_compare_policy_expected_actions():
    stop = _scenario(
        json.dumps({"action": "stop", "why": "done"}), expected="stop"
    )
    stop["case_kind"] = "policy"
    widen = _scenario(
        json.dumps({"action": "widen", "new_categories": ["cs.CL"], "why": "go"}),
        expected="widen",
        constraints={"new_categories_any_of": ["cs.CL"]},
    )
    widen["id"] = "widen"
    widen["case_kind"] = "policy"
    contract = _scenario(json.dumps({"action": "delete", "why": "bad"}))
    contract["id"] = "contract"

    report = agent_eval.evaluate_scenarios(_payload(stop, widen, contract))

    assert report["policy_static_baselines"]["always_widen"] == {
        "action": "widen",
        "passed": 1,
        "total": 2,
        "rate": 0.5,
    }
    assert report["policy_static_baselines"]["always_stop"]["rate"] == 0.5
