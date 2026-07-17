"""Small offline-first benchmark for the Agent's reflect decision.

The production parser, validators and route mapping are exercised through
``graph.reflect_node`` and ``graph.route_action``.  The default mode injects
frozen responses from a synthetic dataset and therefore never calls an LLM.
Use ``--live`` explicitly to replace those responses with one DeepSeek call
per scenario.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import daily
import graph


BASE = Path(__file__).parent
DEFAULT_DATA = BASE / "agent_eval_data" / "scenarios_v1.json"


def load_scenarios(path: str | Path = DEFAULT_DATA) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("评测数据必须包含非空 scenarios 数组")
    return payload


def _paper(case_id: str, index: int, row: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        entry_id=f"synthetic:{case_id}:{index}",
        title=str(row.get("title", "")),
        summary=str(row.get("summary", "synthetic abstract omitted")),
    )


def _state_for(scenario: dict[str, Any]) -> dict[str, Any]:
    current_categories = list(scenario.get("current_categories", graph.CATEGORIES))
    extra_categories = [
        category for category in current_categories if category not in graph.CATEGORIES
    ]
    top = [
        (
            _paper(scenario["id"], index, row),
            int(row["score"]),
            "synthetic fixture",
        )
        for index, row in enumerate(scenario.get("top_candidates", []), 1)
    ]
    return {
        "attempts": 1,
        "query": str(scenario["query"]),
        "base_query": str(scenario["query"]),
        "extra_cats": extra_categories,
        "new_papers": [item[0] for item in top],
        "top": top,
        "hallu": False,
        "relevant_count": sum(1 for _, score, _ in top if score >= 7),
        "action": "",
        "decision_reason": "",
        "decision_trace": [],
        "reflections": 0,
        "fetch_failed": False,
        "fetch_error": "",
        "fetch_since": "2026-07-10T00:00:00Z",
        "fetch_until": "2026-07-17T00:00:00Z",
    }


def raw_schema_valid(raw: Any) -> tuple[bool, dict[str, Any] | None]:
    """Check the response schema promised by the reflect prompt.

    Routing acceptance is still decided by ``graph.reflect_node``.  This
    helper only makes JSON/schema compliance separately visible.
    """
    if not isinstance(raw, str):
        return False, None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, None
    if not isinstance(obj, dict) or obj.get("action") not in graph.REFLECT_ACTIONS:
        return False, obj if isinstance(obj, dict) else None
    if not isinstance(obj.get("why"), str):
        return False, obj
    action = obj["action"]
    if action == "refocus" and not isinstance(obj.get("new_query"), str):
        return False, obj
    if action == "widen":
        categories = obj.get("new_categories")
        if not isinstance(categories, list) or not all(
            isinstance(category, str) for category in categories
        ):
            return False, obj
    return True, obj


def _constraint_pass(
    scenario: dict[str, Any], decision: dict[str, Any], route: str
) -> tuple[bool, list[str]]:
    expected = scenario["expected"]
    problems: list[str] = []
    action = decision.get("action")
    if action != expected["action"]:
        problems.append(f"action={action!r}, expected={expected['action']!r}")
    if route != expected["route"]:
        problems.append(f"route={route!r}, expected={expected['route']!r}")

    if action == "widen":
        current = set(scenario.get("current_categories", graph.CATEGORIES))
        added = set(decision.get("extra_cats", [])) - current
        any_of = set(expected.get("new_categories_any_of", []))
        all_of = set(expected.get("new_categories_all_of", []))
        if any_of and not (added & any_of):
            problems.append(f"added_categories={sorted(added)!r} misses any_of={sorted(any_of)!r}")
        if all_of and not all_of.issubset(added):
            problems.append(f"added_categories={sorted(added)!r} misses all_of={sorted(all_of)!r}")

    if action == "refocus":
        query = str(decision.get("query", ""))
        folded = query.casefold()
        must_all = [str(term).casefold() for term in expected.get("query_must_include_all", [])]
        must_any = [str(term).casefold() for term in expected.get("query_must_include_any", [])]
        if must_all and not all(term in folded for term in must_all):
            problems.append(f"query misses required terms: {must_all!r}")
        if must_any and not any(term in folded for term in must_any):
            problems.append(f"query misses all alternative terms: {must_any!r}")
        if query.strip().casefold() == str(scenario["query"]).strip().casefold():
            problems.append("refocus query did not change")

    return not problems, problems


def _evaluate_one(
    scenario: dict[str, Any],
    *,
    live: bool,
    live_chat: Callable[..., str],
    verbose: bool,
) -> dict[str, Any]:
    state = _state_for(scenario)
    raw_box: dict[str, Any] = {"value": None}
    call_count = 0

    def injected_chat(*args: Any, **kwargs: Any) -> str:
        nonlocal call_count
        if live:
            call_count += 1
            raw = live_chat(*args, **kwargs)
        else:
            raw = scenario["frozen_decision"]["raw"]
        raw_box["value"] = raw
        return raw

    original_chat = graph.chat
    output = io.StringIO()
    try:
        graph.chat = injected_chat
        if verbose:
            decision = graph.reflect_node(state)
        else:
            with redirect_stdout(output):
                decision = graph.reflect_node(state)
    finally:
        graph.chat = original_chat

    raw = raw_box["value"]
    schema_valid, raw_obj = raw_schema_valid(raw)
    route = graph.route_action(decision)
    constraint_pass, constraint_problems = _constraint_pass(scenario, decision, route)

    raw_action = raw_obj.get("action") if isinstance(raw_obj, dict) else None
    # An invalid/unsupported response, or an action rejected by production
    # validators, is the denominator for fail-closed behavior.
    fail_closed_applicable = (
        not schema_valid
        or (
            raw_action in {"widen", "refocus"}
            and decision.get("action") == "stop"
        )
    )
    fail_closed_pass = (
        fail_closed_applicable
        and decision.get("action") == "stop"
        and route == "report"
    )
    current = set(scenario.get("current_categories", graph.CATEGORIES))
    added_categories = sorted(set(decision.get("extra_cats", [])) - current)

    return {
        "id": scenario["id"],
        "case_kind": scenario.get("case_kind", "policy"),
        "expected_action": scenario["expected"]["action"],
        "actual_action": decision.get("action"),
        "actual_route": route,
        "action_correct": decision.get("action") == scenario["expected"]["action"],
        "schema_valid": schema_valid,
        "constraint_pass": constraint_pass,
        "constraint_problems": constraint_problems,
        "fail_closed_applicable": fail_closed_applicable,
        "fail_closed_pass": fail_closed_pass,
        "policy_invocations": call_count,
        "added_categories": added_categories,
        "new_query": decision.get("query"),
        "decision_reason": decision.get("decision_reason", ""),
        "raw": raw,
        "rationale": scenario.get("rationale", ""),
    }


def _ratio(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    passed = sum(bool(row[key]) for row in rows)
    total = len(rows)
    return {"passed": passed, "total": total, "rate": passed / total if total else None}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fail_rows = [row for row in rows if row["fail_closed_applicable"]]
    return {
        "action_accuracy": _ratio(rows, "action_correct"),
        "schema_valid_rate": _ratio(rows, "schema_valid"),
        "constraint_pass_rate": _ratio(rows, "constraint_pass"),
        "fail_closed_rate": _ratio(fail_rows, "fail_closed_pass"),
        # This counts reflect policy invocations. daily.chat may perform
        # transport retries internally, so it is not an HTTP request counter.
        "average_policy_invocations": (
            sum(row["policy_invocations"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }


def _static_action_baseline(
    rows: list[dict[str, Any]], action: str
) -> dict[str, Any]:
    total = len(rows)
    passed = sum(row["expected_action"] == action for row in rows)
    return {
        "action": action,
        "passed": passed,
        "total": total,
        "rate": passed / total if total else None,
    }


def evaluate_scenarios(
    payload: dict[str, Any],
    *,
    live: bool = False,
    live_chat: Callable[..., str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    scenarios = payload["scenarios"]
    caller = live_chat or daily.chat
    rows = [
        _evaluate_one(
            scenario,
            live=live,
            live_chat=caller,
            verbose=verbose,
        )
        for scenario in scenarios
    ]
    policy_rows = [row for row in rows if row["case_kind"] == "policy"]
    contract_rows = [row for row in rows if row["case_kind"] == "contract"]
    limitations = [
        "Synthetic scenarios test routing contracts, not real recommendation quality.",
        "Frozen outputs are reproducible fixtures, not a guarantee about the current model.",
        "The benchmark does not evaluate arxiv freshness, ranking quality, or paper usefulness.",
    ]
    if live:
        limitations.append(
            "Live mode replaces contract fault fixtures with model responses; use policy_metrics "
            "for model comparison and offline contract_metrics for deterministic fail-closed checks."
        )
    return {
        "benchmark": payload.get("benchmark", {}),
        "mode": "live" if live else "offline-frozen",
        "scenario_count": len(rows),
        "metrics": _metrics(rows),
        "policy_metrics": _metrics(policy_rows),
        "policy_static_baselines": {
            "always_widen": _static_action_baseline(policy_rows, "widen"),
            "always_stop": _static_action_baseline(policy_rows, "stop"),
        },
        "contract_metrics": _metrics(contract_rows),
        "results": rows,
        "limitations": limitations,
    }


def _format_ratio(metric: dict[str, Any]) -> str:
    if not metric["total"]:
        return "N/A (no applicable cases)"
    return f"{metric['passed']}/{metric['total']} ({metric['rate']:.1%})"


def format_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    baselines = report["policy_static_baselines"]
    lines = [
        "Agent reflect synthetic benchmark",
        f"mode: {report['mode']}",
        f"scenarios: {report['scenario_count']}",
        f"action accuracy: {_format_ratio(metrics['action_accuracy'])}",
        f"schema valid rate: {_format_ratio(metrics['schema_valid_rate'])}",
        f"constraint pass rate: {_format_ratio(metrics['constraint_pass_rate'])}",
        f"fail-closed rate: {_format_ratio(metrics['fail_closed_rate'])}",
        "average policy invocations/scenario: "
        f"{metrics['average_policy_invocations']:.2f}",
        "policy action baselines (synthetic expected actions only):",
        f"- always-widen: {_format_ratio(baselines['always_widen'])}",
        f"- always-stop: {_format_ratio(baselines['always_stop'])}",
        "",
        "case                         kind      expected  actual    schema  constraints  route",
    ]
    for row in report["results"]:
        lines.append(
            f"{row['id'][:28]:28} "
            f"{row['case_kind'][:9]:9} "
            f"{str(row['expected_action']):9} "
            f"{str(row['actual_action']):9} "
            f"{('ok' if row['schema_valid'] else 'bad'):7} "
            f"{('ok' if row['constraint_pass'] else 'bad'):12} "
            f"{row['actual_route']}"
        )
    lines.extend(
        [
            "",
            "Limitations:",
            *[f"- {item}" for item in report["limitations"]],
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="评测 reflect 决策、约束验证与 fail-closed 路由"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--live",
        action="store_true",
        help="对每个场景调用一次 DeepSeek；默认使用冻结输出且不联网",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument("--verbose", action="store_true", help="保留 graph 的 reflect 日志")
    args = parser.parse_args(argv)

    report = evaluate_scenarios(
        load_scenarios(args.data),
        live=args.live,
        verbose=args.verbose,
    )
    print(
        json.dumps(report, ensure_ascii=False, indent=2)
        if args.json
        else format_report(report)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
