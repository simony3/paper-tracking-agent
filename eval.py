"""离线排序与 LLM 打分评测。

主要口径：
1. 只在已判定（judged）的论文上计算排序 P/R，未标注样本不默认为不相关。
2. 同时报告 top-k 的 judgment coverage，避免低覆盖率的高分被误读。
3. LLM 评测严格使用生产的 RRF -> top-LLM_CAND -> LLM 精排漏斗。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

from label import load_labels
from memory import get_anchor_ids
from qa import load_store
from retrieval import (
    INTEREST,
    LLM_CAND,
    RELEVANT_THRESHOLD,
    bm25_recall,
    embed_recall,
    hybrid_recall,
    llm_score,
    preference_bonus,
)
from storage import DataFileError, read_json

K = 10
RANDOM_TRIALS = 1000
VALID_LABELS = frozenset({"rel", "irrel"})
SNAPSHOT_SCHEMA = "paper-tracker.eval-snapshot/1.0.0"
DEFAULT_SNAPSHOT = Path(__file__).parent / "eval_data" / "eval_v1.json"


@dataclass(frozen=True)
class RankingMetrics:
    """在 judged pool 上的排序指标。

    precision 的分母是 top-k 中已判定的数量，recall 的分母是
    当前可评测集中所有已判定的相关论文。它们都不是“全库完整真值”。
    """

    precision: float | None
    recall: float | None
    coverage: float
    relevant_retrieved: int
    judged_in_topk: int
    topk_size: int
    total_judged_relevant: int

    @property
    def f1(self) -> float | None:
        if self.precision is None or self.recall is None:
            return None
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass(frozen=True)
class Estimate:
    mean: float | None
    std: float | None
    valid_trials: int


@dataclass(frozen=True)
class RandomBaseline:
    precision: Estimate
    recall: Estimate
    coverage: Estimate
    trials: int


@dataclass(frozen=True)
class ClassificationMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    balanced_accuracy: float | None
    kappa: float | None
    majority_accuracy: float | None
    judged: int


@dataclass(frozen=True)
class EvaluationInput:
    records: list[dict]
    labels: dict[str, str]
    anchors: set[str]
    frozen_scores: dict[str, tuple[int | float, str]] | None
    source: str
    split: str


def to_paper(record):
    return SimpleNamespace(
        title=record["title"], summary=record["summary"], entry_id=record["id"]
    )


def load_snapshot_input(path: Path, split: str = "test") -> EvaluationInput:
    """读取公开冻结快照，不访问本地 profile/cache，也不调用 DeepSeek。

    ``test`` 排除 dev/anchor，保留 unjudged 作为真实检索干扰项；``dev`` 同理
    排除 test；``all`` 仅排除 anchors。这样 held-out 标签不会参与目标 split 的指标。
    """

    if split not in {"all", "dev", "test"}:
        raise ValueError("split 必须是 all/dev/test")
    try:
        payload = read_json(path, {})
    except DataFileError as exc:
        raise ValueError(f"快照无法读取: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError(f"不支持的评测快照 schema，期望 {SNAPSHOT_SCHEMA}")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("评测快照 records 必须是非空数组")
    actual_records_hash = hashlib.sha256(
        json.dumps(
            raw_records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if payload.get("records_sha256") != actual_records_hash:
        raise ValueError("评测快照 records_sha256 校验失败，请重新获取或运行导出校验")

    records = []
    all_labels = {}
    anchors = set()
    frozen_scores = {}
    seen_ids = set()
    for row in raw_records:
        if not isinstance(row, dict):
            raise ValueError("评测快照包含非对象 record")
        paper_id = row.get("id")
        if not isinstance(paper_id, str) or not paper_id or paper_id in seen_ids:
            raise ValueError("评测快照 paper id 缺失或重复")
        seen_ids.add(paper_id)
        if not isinstance(row.get("title"), str) or not isinstance(row.get("summary"), str):
            raise ValueError(f"评测快照论文元数据不完整: {paper_id}")
        label = row.get("label")
        if label is not None and label not in VALID_LABELS:
            raise ValueError(f"评测快照标签非法: {paper_id}")
        row_split = row.get("split")
        if row_split not in {"anchor", "dev", "test", "unjudged"}:
            raise ValueError(f"评测快照 split 非法: {paper_id}")
        if row.get("is_anchor") is True:
            anchors.add(paper_id)
        if label in VALID_LABELS:
            all_labels[paper_id] = label

        score, reason = row.get("score"), row.get("reason")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 10
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError(f"评测快照 legacy score/reason 非法: {paper_id}")
        frozen_scores[paper_id] = (score, reason.strip())

        excluded_for_split = row_split == "anchor" or (
            split == "test" and row_split == "dev"
        ) or (split == "dev" and row_split == "test")
        if not excluded_for_split:
            records.append({"id": paper_id, "title": row["title"], "summary": row["summary"]})

    papers = [to_paper(record) for record in records]
    labels = filter_labels_to_papers(all_labels, papers)
    version = payload.get("dataset_version", path.name)
    return EvaluationInput(
        records=records,
        labels=labels,
        anchors=anchors,
        frozen_scores=frozen_scores,
        source=f"冻结快照 {version}（legacy scorer）",
        split=split,
    )


def load_runtime_input() -> EvaluationInput:
    """读取私有运行态；LLM 分数按当前版本化缓存/DeepSeek 生成。"""

    all_labels = load_labels()
    records = load_store()
    anchors = set(get_anchor_ids())
    eligible = [record for record in records if record.get("id") not in anchors]
    papers = [to_paper(record) for record in eligible]
    return EvaluationInput(
        records=eligible,
        labels=filter_labels_to_papers(all_labels, papers),
        anchors=anchors,
        frozen_scores=None,
        source="本地运行态（当前 scorer/cache）",
        split="all",
    )


def frozen_llm_score(
    papers: Sequence,
    scores_by_id: Mapping[str, tuple[int | float, str]],
):
    """按候选顺序读取快照分数；缺失或非法时触发与生产相同的 RRF 回退。"""

    scores = []
    failed = False
    for paper in papers:
        value = scores_by_id.get(paper.entry_id)
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or isinstance(value[0], bool)
            or not isinstance(value[0], (int, float))
            or not math.isfinite(value[0])
            or not 0 <= value[0] <= 10
            or not isinstance(value[1], str)
            or not value[1].strip()
        ):
            scores.append((0, "冻结分数缺失"))
            failed = True
        else:
            scores.append((value[0], value[1].strip()))
    return scores, failed


def filter_labels_to_papers(labels: Mapping[str, str], papers: Iterable) -> dict[str, str]:
    """只保留当前可评测论文的合法标签。

    调用方先移除 feedback anchors，再调用此函数，便能保证
    anchors 同时从样本和 Recall 分母中移除。
    """

    paper_ids = {paper.entry_id for paper in papers}
    return {
        paper_id: label
        for paper_id, label in labels.items()
        if paper_id in paper_ids and label in VALID_LABELS
    }


def judged_metric(ranked: Sequence, labels: Mapping[str, str], k: int = K) -> RankingMetrics:
    """计算 judged P@k / judged Recall@k 和 top-k judgment coverage。"""

    topk = list(ranked[:k])
    judged = [paper for paper in topk if labels.get(paper.entry_id) in VALID_LABELS]
    relevant_retrieved = sum(labels[paper.entry_id] == "rel" for paper in judged)
    total_relevant = sum(label == "rel" for label in labels.values() if label in VALID_LABELS)

    precision = relevant_retrieved / len(judged) if judged else None
    recall = relevant_retrieved / total_relevant if total_relevant else None
    coverage = len(judged) / len(topk) if topk else 0.0
    return RankingMetrics(
        precision=precision,
        recall=recall,
        coverage=coverage,
        relevant_retrieved=relevant_retrieved,
        judged_in_topk=len(judged),
        topk_size=len(topk),
        total_judged_relevant=total_relevant,
    )


def metric(ranked, labels, k=K, *, judged_only=False):
    """保留旧调用方的三元组接口。

    新评测一律调用 :func:`judged_metric`。默认的 legacy 口径仅为
    不破坏旧脚本：它会把未标注项纳入 Precision 分母，不用于本文的报告。
    过渡期调用方可传 ``judged_only=True`` 获得新口径三元组。
    """

    result = judged_metric(ranked, labels, k)
    if judged_only:
        return result.precision, result.recall, result.judged_in_topk
    legacy_precision = (
        result.relevant_retrieved / result.topk_size if result.topk_size else 0.0
    )
    return legacy_precision, result.recall or 0.0, result.judged_in_topk


def llm_candidates(papers: Sequence, recalled_indices: Sequence[int], limit: int = LLM_CAND):
    """把混合召回排名截断成与生产一致的 LLM 候选漏斗。"""

    return [papers[index] for index in recalled_indices[:limit]]


def rank_scored_candidates(candidates, scores, bonus=None, *, score_failed=False):
    """按生产语义精排；LLM 校验失败时回退到 RRF 顺序。"""

    if score_failed:
        return list(candidates)
    bonus = bonus if bonus is not None else [0.0] * len(candidates)
    ranked = sorted(
        zip(candidates, scores, bonus),
        key=lambda item: item[1][0] + item[2],
        reverse=True,
    )
    return [paper for paper, _, _ in ranked]


def _estimate(values: Sequence[float | None]) -> Estimate:
    valid = [value for value in values if value is not None]
    if not valid:
        return Estimate(None, None, 0)
    return Estimate(
        mean=statistics.fmean(valid),
        std=statistics.pstdev(valid),
        valid_trials=len(valid),
    )


def random_baseline(
    papers: Sequence,
    labels: Mapping[str, str],
    *,
    k: int = K,
    trials: int = RANDOM_TRIALS,
    seed: int = 42,
) -> RandomBaseline:
    """多次无放回随机抽取 top-k，报告均值和标准差。"""

    if trials < 1:
        raise ValueError("trials 必须 >= 1")
    rng = random.Random(seed)
    sample_size = min(k, len(papers))
    precision_values = []
    recall_values = []
    coverage_values = []
    for _ in range(trials):
        sample = rng.sample(list(papers), sample_size)
        result = judged_metric(sample, labels, k)
        precision_values.append(result.precision)
        recall_values.append(result.recall)
        coverage_values.append(result.coverage)
    return RandomBaseline(
        precision=_estimate(precision_values),
        recall=_estimate(recall_values),
        coverage=_estimate(coverage_values),
        trials=trials,
    )


def binary_classification_metrics(
    papers: Sequence,
    scores: Sequence,
    labels: Mapping[str, str],
    *,
    threshold: float = RELEVANT_THRESHOLD,
) -> ClassificationMetrics:
    """在候选集的已判定样本上评估 LLM 分数阈值。"""

    tp = fp = fn = tn = 0
    for paper, score_item in zip(papers, scores):
        label = labels.get(paper.entry_id)
        if label not in VALID_LABELS:
            continue
        score = score_item[0] if isinstance(score_item, (tuple, list)) else score_item
        predicted_rel = score >= threshold
        if label == "rel" and predicted_rel:
            tp += 1
        elif label == "rel":
            fn += 1
        elif predicted_rel:
            fp += 1
        else:
            tn += 1

    judged = tp + fp + fn + tn
    accuracy = (tp + tn) / judged if judged else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if precision is not None and recall is not None else None)
    )
    specificity = tn / (tn + fp) if tn + fp else None
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )

    if judged:
        actual_rel = tp + fn
        actual_irrel = tn + fp
        predicted_rel = tp + fp
        predicted_irrel = tn + fn
        expected_agreement = (
            actual_rel * predicted_rel + actual_irrel * predicted_irrel
        ) / (judged * judged)
        kappa = (
            (accuracy - expected_agreement) / (1 - expected_agreement)
            if expected_agreement < 1
            else None
        )
        majority_accuracy = max(actual_rel, actual_irrel) / judged
    else:
        kappa = None
        majority_accuracy = None

    return ClassificationMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=balanced_accuracy,
        kappa=kappa,
        majority_accuracy=majority_accuracy,
        judged=judged,
    )


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _fmt_estimate(estimate: Estimate) -> str:
    if estimate.mean is None:
        return "N/A"
    return f"{estimate.mean:.3f}±{estimate.std:.3f}"


def build_parser():
    parser = argparse.ArgumentParser(
        description="评测生产一致的 RRF -> top-30 -> LLM scorer 漏斗"
    )
    parser.add_argument(
        "--snapshot",
        nargs="?",
        const=DEFAULT_SNAPSHOT,
        type=Path,
        help=(
            "使用公开冻结快照和 legacy 分数，不调用 DeepSeek；"
            f"省略路径时默认 {DEFAULT_SNAPSHOT.relative_to(Path(__file__).parent)}"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("all", "dev", "test"),
        help="快照标签划分（--snapshot 默认 test；本地运行态仅支持 all）",
    )
    parser.add_argument("-k", type=int, default=K, help=f"排名指标截断（默认 {K}）")
    parser.add_argument(
        "--random-trials",
        type=int,
        default=RANDOM_TRIALS,
        help=f"随机基线重复次数（默认 {RANDOM_TRIALS}）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机基线种子（默认 42）")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("-k 必须 >= 1")
    if args.random_trials < 1:
        parser.error("--random-trials 必须 >= 1")

    split = args.split or ("test" if args.snapshot else "all")
    if not args.snapshot and split != "all":
        parser.error("本地运行态没有冻结 split；请使用 --snapshot 或将 --split 设为 all")
    try:
        evaluation = (
            load_snapshot_input(args.snapshot, split)
            if args.snapshot
            else load_runtime_input()
        )
    except (ValueError, DataFileError) as exc:
        parser.error(str(exc))

    labels = evaluation.labels
    if sum(value == "rel" for value in labels.values()) < 3 or len(labels) < 10:
        if args.snapshot:
            raise SystemExit("冻结 split 的标签太少，无法形成有效评测。")
        raise SystemExit(
            f"当前仅有 {len(labels)} 个可评测标签"
            f"（rel={sum(value == 'rel' for value in labels.values())}）。"
            "请先用 `python label.py` 增加标注，或运行 `python eval.py --snapshot`。"
        )

    # anchors 在构造 EvaluationInput 时已从论文池和 Recall 分母同步排除。
    papers = [to_paper(record) for record in evaluation.records]
    anchors = evaluation.anchors

    b_indices, _ = bm25_recall(INTEREST, papers)
    e_indices, _ = embed_recall(INTEREST, papers)
    hybrid_indices = hybrid_recall(INTEREST, papers)
    hybrid_ranked = [papers[index] for index in hybrid_indices]

    # 生产路径的关键漏斗：只对 RRF top-LLM_CAND 打分。
    candidates = llm_candidates(papers, hybrid_indices)
    if evaluation.frozen_scores is None:
        scores, score_failed = llm_score(INTEREST, candidates)
        bonus = preference_bonus(candidates)
        llm_name = "混合+当前LLM打分(记忆关)"
    else:
        scores, score_failed = frozen_llm_score(candidates, evaluation.frozen_scores)
        # 公共快照刻意不发布私人 liked/disliked 文本，不能伪造“记忆开”结果。
        bonus = [0.0] * len(candidates)
        llm_name = "混合+冻结legacy LLM分"
    memory_off = rank_scored_candidates(
        candidates, scores, score_failed=score_failed
    )
    memory_on = rank_scored_candidates(
        candidates, scores, bonus, score_failed=score_failed
    )

    ablation = [
        ("纯 BM25", [papers[index] for index in b_indices]),
        ("纯向量", [papers[index] for index in e_indices]),
        ("混合 RRF", hybrid_ranked),
        (llm_name, memory_off),
    ]
    if evaluation.frozen_scores is None:
        ablation.append(("混合+当前LLM+记忆配置", memory_on))
    random_result = random_baseline(
        papers,
        labels,
        k=args.k,
        trials=args.random_trials,
        seed=args.seed,
    )

    print(
        f"数据源：{evaluation.source} | split={evaluation.split}"
    )
    print(
        f"候选池：论文 {len(papers)} 篇 / judged {len(labels)} 篇 "
        f"(rel={sum(value == 'rel' for value in labels.values())}, "
        f"irrel={sum(value == 'irrel' for value in labels.values())})"
    )
    print(
        f"anchors 排除 {len(anchors)} 个；已同步从论文池和 Recall 分母排除。"
    )
    print(
        f"LLM 漏斗：全库 -> RRF -> top-{LLM_CAND} "
        f"(本次 {len(candidates)} 篇) -> LLM 精排；"
        f"打分校验{'失败，已回退 RRF' if score_failed else '通过'}。"
    )
    if evaluation.frozen_scores is not None:
        print(
            "注意：快照分数来自 legacy scorer，只用于复核历史系统；"
            "它不等于当前 prompt/model 的新在线结果，也不包含私人偏好记忆。"
        )
    print(
        "口径：P@k(judged)=top-k 中 rel / top-k 中 judged；"
        "R@k(judged)=top-k 中 rel / 评测集全部 judged rel。"
    )
    print(
        "未标注论文不视为 irrel；Coverage 是 top-k 中已判定比例，"
        "低覆盖结果应谨慎解读。\n"
    )

    print(
        f"{'\u6392\u5e8f\u65b9\u6848':<25}"
        f"{'P@' + str(args.k) + '(judged)':>15}"
        f"{'R@' + str(args.k) + '(judged)':>15}"
        f"{'F1':>9}{'Coverage':>12}"
    )
    print(
        f"{'\u968f\u673a\u57fa\u7ebf(' + str(random_result.trials) + '\u6b21)':<25}"
        f"{_fmt_estimate(random_result.precision):>15}"
        f"{_fmt_estimate(random_result.recall):>15}"
        f"{'-':>9}{_fmt_estimate(random_result.coverage):>12}"
    )
    if random_result.precision.valid_trials != random_result.trials:
        print(
            "  注：随机 top-k 没有任何 judged 时 P(judged) 未定义；"
            f"Precision 均值基于 {random_result.precision.valid_trials}/"
            f"{random_result.trials} 个有效试验。"
        )
    for name, ranked in ablation:
        result = judged_metric(ranked, labels, args.k)
        print(
            f"{name:<25}{_fmt(result.precision):>15}{_fmt(result.recall):>15}"
            f"{_fmt(result.f1):>9}{_fmt(result.coverage):>12}"
            f"  ({result.judged_in_topk}/{result.topk_size} judged)"
        )

    classification = binary_classification_metrics(
        candidates, scores, labels, threshold=RELEVANT_THRESHOLD
    )
    candidate_coverage = (
        classification.judged / len(candidates) if candidates else 0.0
    )
    pool_coverage = classification.judged / len(labels) if labels else 0.0
    print(f"\nLLM >= {RELEVANT_THRESHOLD} 分的二分类评测（仅 judged 候选）")
    print(
        f"覆盖：{classification.judged}/{len(candidates)} 个候选 "
        f"({_fmt(candidate_coverage)})；覆盖整个 judged pool "
        f"{classification.judged}/{len(labels)} ({_fmt(pool_coverage)})"
    )
    print("\n                     预测 rel    预测 irrel")
    print(
        f"实际 rel             {classification.tp:>4}"
        f"          {classification.fn:>4}"
    )
    print(
        f"实际 irrel           {classification.fp:>4}"
        f"          {classification.tn:>4}"
    )
    print(
        f"相关类 Precision={_fmt(classification.precision)}  "
        f"Recall={_fmt(classification.recall)}  F1={_fmt(classification.f1)}"
    )
    print(
        f"Accuracy={_fmt(classification.accuracy)}  "
        f"Balanced Accuracy={_fmt(classification.balanced_accuracy)}  "
        f"Cohen's kappa={_fmt(classification.kappa)}"
    )
    print(
        f"多数类基线 Accuracy={_fmt(classification.majority_accuracy)}"
        "（不平衡数据不应只看 Accuracy）"
    )


if __name__ == "__main__":
    main()
