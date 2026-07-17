"""PREF_WEIGHT 调参：只用 dev 选参，最后在 held-out test 评估一次。

默认按论文 id 和类别做稳定、分层的 70/30 拆分。同一 seed 下拆分
不随运行顺序变化，且 test 指标不参与权重选择。

用法：
    python tune.py
    python tune.py --dev-ratio 0.7 --seed 20260717 --weights 0,1,2,3,5,8
"""

from __future__ import annotations

import argparse
import hashlib
from types import SimpleNamespace
from typing import Mapping, Sequence

import retrieval
from eval import (
    K,
    RankingMetrics,
    filter_labels_to_papers,
    judged_metric,
    llm_candidates,
    rank_scored_candidates,
)
from label import load_labels
from memory import get_anchor_ids
from qa import load_store
from retrieval import INTEREST, hybrid_recall, llm_score, preference_bonus

DEFAULT_SEED = 20260717
DEFAULT_WEIGHTS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)


def _stable_order(ids: Sequence[str], seed: int, namespace: str) -> list[str]:
    return sorted(
        ids,
        key=lambda paper_id: hashlib.sha256(
            f"{seed}:{namespace}:{paper_id}".encode("utf-8")
        ).digest(),
    )


def deterministic_label_split(
    labels: Mapping[str, str],
    *,
    dev_ratio: float = 0.7,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, str], dict[str, str]]:
    """对 judged labels 做稳定的分层 dev/test 拆分。

    每类至少有两个样本时，保证 dev/test 都有该类。拆分只决定哪些
    judgment 可用于调参，不改变检索或 LLM 分数。
    """

    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio 必须在 0 和 1 之间")

    dev_ids: set[str] = set()
    test_ids: set[str] = set()
    for label in ("rel", "irrel"):
        ids = [paper_id for paper_id, value in labels.items() if value == label]
        ordered = _stable_order(ids, seed, label)
        if len(ordered) >= 2:
            dev_count = round(len(ordered) * dev_ratio)
            dev_count = min(max(dev_count, 1), len(ordered) - 1)
        elif ordered:
            # 极小数据集无法同时覆盖两个 split，放入份额更大的一侧。
            dev_count = 1 if dev_ratio >= 0.5 else 0
        else:
            dev_count = 0
        dev_ids.update(ordered[:dev_count])
        test_ids.update(ordered[dev_count:])

    dev = {paper_id: labels[paper_id] for paper_id in dev_ids}
    test = {paper_id: labels[paper_id] for paper_id in test_ids}
    return dev, test


def _score_key(item: tuple[float, RankingMetrics]):
    """只基于 dev 指标选权重：主目标 F1，其后 P/R/覆盖，并偏好小权重。"""

    weight, result = item
    value = lambda metric: -1.0 if metric is None else metric
    return (
        value(result.f1),
        value(result.precision),
        value(result.recall),
        result.coverage,
        -abs(weight),
        -weight,
    )


def select_best_weight(
    dev_results: Sequence[tuple[float, RankingMetrics]],
) -> tuple[float, RankingMetrics]:
    """从 dev 结果选权重；函数不接收 test 指标，防止误用。"""

    if not dev_results:
        raise ValueError("至少需要一个候选权重")
    return max(dev_results, key=_score_key)


def _parse_weights(raw: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("权重必须是逗号分隔的数字") from error
    if not weights:
        raise argparse.ArgumentTypeError("至少需要一个权重")
    return weights


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _without_other_split(ranked, other_labels: Mapping[str, str]):
    """从某个 split 的候选 pool 移除另一 split 的 judged 文档。"""

    other_ids = set(other_labels)
    return [paper for paper in ranked if paper.entry_id not in other_ids]


def _paper(record):
    return SimpleNamespace(
        title=record["title"], summary=record["summary"], entry_id=record["id"]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在确定性 dev split 调 PREF_WEIGHT，并只在最后评估一次 test。"
    )
    parser.add_argument("--dev-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--weights",
        type=_parse_weights,
        default=DEFAULT_WEIGHTS,
        help="逗号分隔的候选权重（默认: 0,1,2,3,5,8）",
    )
    parser.add_argument("-k", type=int, default=K)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.k < 1:
        raise SystemExit("k 必须 >= 1")

    anchors = set(get_anchor_ids())
    papers = [_paper(record) for record in load_store() if record["id"] not in anchors]
    labels = filter_labels_to_papers(load_labels(), papers)
    dev_labels, test_labels = deterministic_label_split(
        labels, dev_ratio=args.dev_ratio, seed=args.seed
    )
    if not dev_labels or not test_labels:
        raise SystemExit("标签太少，无法形成非空 dev/test；请先增加标注。")

    recalled = hybrid_recall(INTEREST, papers)
    candidates = llm_candidates(papers, recalled)
    scores, score_failed = llm_score(INTEREST, candidates)

    # preference_bonus 内部会乘 PREF_WEIGHT。只计算一次权重=1 的基础信号，
    # 后续扫描是纯本地排序，且不重复 embed。
    original_weight = retrieval.PREF_WEIGHT
    try:
        retrieval.PREF_WEIGHT = 1.0
        unit_bonus = preference_bonus(candidates)
    finally:
        retrieval.PREF_WEIGHT = original_weight

    print(
        f"拆分：seed={args.seed}, dev_ratio={args.dev_ratio:.2f}, "
        f"dev={len(dev_labels)}(rel={sum(v == 'rel' for v in dev_labels.values())}), "
        f"test={len(test_labels)}(rel={sum(v == 'rel' for v in test_labels.values())})"
    )
    print(
        f"漏斗：全库 {len(papers)} -> RRF -> top-{retrieval.LLM_CAND} "
        f"(本次 {len(candidates)}) -> LLM 精排。"
    )
    if score_failed:
        print("LLM 打分校验失败：按生产降级规则回退 RRF，所有权重将等价。")
    print(
        "选参目标：dev judged F1（并报告 P/R/Coverage）；"
        "test 结果不参与选权重。\n"
    )
    print(f"{'PREF_WEIGHT':>12} | {'dev P@' + str(args.k):>10} | "
          f"{'dev R@' + str(args.k):>10} | {'dev F1':>8} | {'Coverage':>8}")

    dev_results = []
    ranked_by_weight = {}
    for weight in args.weights:
        bonus = [weight * value for value in unit_bonus]
        ranked = rank_scored_candidates(
            candidates, scores, bonus, score_failed=score_failed
        )
        ranked_by_weight[weight] = ranked
        dev_ranked = _without_other_split(ranked, test_labels)
        result = judged_metric(dev_ranked, dev_labels, args.k)
        dev_results.append((weight, result))
        print(
            f"{weight:>12g} | {_fmt(result.precision):>10} | "
            f"{_fmt(result.recall):>10} | {_fmt(result.f1):>8} | "
            f"{_fmt(result.coverage):>8}"
        )

    best_weight, best_dev = select_best_weight(dev_results)

    # test 只在 dev 完成选参后计算这一次，不扫描其他权重。
    test_ranked = _without_other_split(ranked_by_weight[best_weight], dev_labels)
    test_result = judged_metric(test_ranked, test_labels, args.k)
    print(
        f"\ndev 选出 PREF_WEIGHT={best_weight:g}: "
        f"P={_fmt(best_dev.precision)}, R={_fmt(best_dev.recall)}, "
        f"F1={_fmt(best_dev.f1)}, Coverage={_fmt(best_dev.coverage)}"
    )
    print(
        f"held-out test（仅评估选中权重一次）: "
        f"P@{args.k}={_fmt(test_result.precision)}, "
        f"R@{args.k}={_fmt(test_result.recall)}, F1={_fmt(test_result.f1)}, "
        f"Coverage={_fmt(test_result.coverage)} "
        f"({test_result.judged_in_topk}/{test_result.topk_size} judged)"
    )


if __name__ == "__main__":
    main()
