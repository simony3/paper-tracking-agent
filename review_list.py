"""生成按检索排名的人工核对清单；默认只让 LLM 检查 RRF top-30。"""

import argparse
from pathlib import Path
from types import SimpleNamespace

from label import load_labels
from qa import load_store
from retrieval import INTEREST, hybrid_recall, llm_score
from storage import atomic_write_text

BASE = Path(__file__).parent
DEFAULT_REVIEW_COUNT = 30
DEFAULT_LLM_CANDIDATES = 30


def build_parser():
    parser = argparse.ArgumentParser(
        description="生成不把未标注项当负例的人工核对清单"
    )
    parser.add_argument(
        "--review-count",
        type=int,
        default=DEFAULT_REVIEW_COUNT,
        help=f"优先核对的检索头部数量（默认 {DEFAULT_REVIEW_COUNT}）",
    )
    parser.add_argument(
        "--llm-candidates",
        type=int,
        default=DEFAULT_LLM_CANDIDATES,
        help=f"送 LLM 打分的 RRF 头部数量（默认 {DEFAULT_LLM_CANDIDATES}）",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.review_count < 1:
        parser.error("--review-count 必须 >= 1")
    if args.llm_candidates < 1:
        parser.error("--llm-candidates 必须 >= 1")

    store = load_store()
    if not store:
        parser.error("论文库为空，请先运行 graph.py")
    labels = load_labels()
    papers = [
        SimpleNamespace(title=row["title"], summary=row["summary"], entry_id=row["id"])
        for row in store
    ]
    store_idx = {row["id"]: index for index, row in enumerate(store)}

    order = hybrid_recall(INTEREST, papers)
    ranked = [papers[index] for index in order]
    score_candidates = ranked[:args.llm_candidates]
    scores, score_failed = llm_score(INTEREST, score_candidates)
    score_by_id = {
        paper.entry_id: score_item
        for paper, score_item in zip(score_candidates, scores)
    }

    lines = [
        "# 待核对清单（按检索排名）\n",
        f"全库 {len(store)} 篇 | 已标 {len(labels)} 篇"
        f"（rel {sum(value == 'rel' for value in labels.values())}） | "
        "标注命令：`python label.py <rel|irrel> <序号> [序号...]`\n",
        f"> ⭐=建议优先核对（RRF 前 {args.review_count} 且尚非 rel，或 LLM 分数在 6–7 边界）。"
        f"只对 RRF 前 {len(score_candidates)} 篇调用 LLM；其余显示 —。\n",
        "| 序号 | 标注 | LLM分 | ⭐ | 标题 |",
        "|---|---|---:|---|---|",
    ]
    if score_failed:
        lines.insert(3, "> ⚠️ LLM 打分异常，本次分数仅供参考；异常结果不会写入缓存。\n")

    suggested = []
    for rank, paper in enumerate(ranked, 1):
        number = store_idx[paper.entry_id] + 1
        current = labels.get(paper.entry_id, "—")
        score_item = score_by_id.get(paper.entry_id)
        score = score_item[0] if score_item else None
        star = "⭐" if (
            (rank <= args.review_count and current != "rel") or score in {6, 7}
        ) else ""
        if star:
            suggested.append(number)
        title = paper.title.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {number} | {current} | {score if score is not None else '—'} | {star} | {title} |"
        )

    atomic_write_text(BASE / "待核对清单.md", "\n".join(lines))

    by_id = {row["id"]: row for row in store}
    details = [
        "# 核对详情（建议优先核对）\n",
        "> 请独立阅读标题/摘要后再标注；LLM 分数和理由不是 ground truth。\n",
    ]
    suggested_set = set(suggested)
    for paper in ranked:
        number = store_idx[paper.entry_id] + 1
        if number not in suggested_set:
            continue
        current = labels.get(paper.entry_id, "未标")
        score_item = score_by_id.get(paper.entry_id)
        score_text = f"{score_item[0]}/10" if score_item else "未送 LLM"
        reason = score_item[1] if score_item else "—"
        details.extend(
            [
                f"\n## 序号 {number} | 当前：{current} | LLM：{score_text}",
                f"**{paper.title}**\n",
                f"LLM 理由：{reason}\n",
                f"摘要：{by_id[paper.entry_id]['summary']}\n",
            ]
        )
    atomic_write_text(BASE / "核对详情.md", "\n".join(details))

    print(
        f"清单和详情已生成。建议优先核对 {len(suggested)} 篇，"
        f"序号：{' '.join(map(str, sorted(suggested)))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
