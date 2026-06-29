"""生成"待核对清单":把全库按当前检索排名排序,标出每篇的标注状态/银标签/LLM分,
挑出最该人工核对的篇(排名靠前却没标 rel 的 + 分数 6~7 边界的)。
顺便给未打分的新论文打分入缓存,供阶段3 eval 覆盖全库。用法: python review_list.py"""
from types import SimpleNamespace

from qa import load_store
from retrieval import INTEREST, RELEVANT_THRESHOLD, hybrid_recall, llm_score
from label import load_labels

TOP_REVIEW = 30   # 排名前这么多篇优先核对

if __name__ == "__main__":
    store = load_store()
    labels = load_labels()
    papers = [SimpleNamespace(title=r["title"], summary=r["summary"], entry_id=r["id"]) for r in store]
    store_idx = {r["id"]: i for i, r in enumerate(store)}  # label.py 用的序号 = 此处+1

    order = hybrid_recall(INTEREST, papers)          # 检索排名(下标)
    cand = [papers[i] for i in order]
    scores, _ = llm_score(INTEREST, cand)            # 未缓存的会打分(花 token)

    lines = ["# 待核对清单(按检索排名)\n",
             f"全库 {len(store)} 篇 | 已标 {len(labels)} 篇(rel {sum(v=='rel' for v in labels.values())}) "
             f"| 标注命令:`python label.py <rel|irrel> <序号> [序号...]`\n",
             "> ⭐=建议优先核对(排名靠前没标rel / 分数卡6~7边界)。序号即 label.py 用的序号。\n",
             "| 序号 | 标注 | LLM分 | ⭐ | 标题 |", "|---|---|---|---|---|"]
    suggest = []
    for rank, (p, (s, _)) in enumerate(zip(cand, scores), 1):
        num = store_idx[p.entry_id] + 1
        cur = labels.get(p.entry_id, "—")
        star = ""
        if (rank <= TOP_REVIEW and cur != "rel") or (s in (6, 7)):
            star = "⭐"
            suggest.append(num)
        title = p.title.replace("|", "/")
        lines.append(f"| {num} | {cur} | {s} | {star} | {title} |")

    from pathlib import Path
    base = Path(__file__).parent
    (base / "待核对清单.md").write_text("\n".join(lines), encoding="utf-8")

    # 核对详情:只列建议核对的篇,摊开 摘要+LLM理由,供逐篇读着标
    by_id = {r["id"]: r for r in store}
    detail = ["# 核对详情(建议优先核对的篇)\n",
              "> 读摘要判断该篇对你研究兴趣是否相关,然后:\n"
              "> 相关→记下序号统一 `python label.py rel <序号...>`;不相关→`python label.py irrel <序号...>`\n"]
    for rank, (p, (s, reason)) in enumerate(zip(cand, scores), 1):
        num = store_idx[p.entry_id] + 1
        if num not in suggest:
            continue
        cur = labels.get(p.entry_id, "未标")
        detail.append(f"\n## 序号 {num} | 当前:{cur} | LLM {s}/10")
        detail.append(f"**{p.title}**\n")
        detail.append(f"LLM理由:{reason}\n")
        detail.append(f"摘要:{by_id[p.entry_id]['summary']}\n")
    (base / "核对详情.md").write_text("\n".join(detail), encoding="utf-8")

    print(f"清单+详情已生成。建议优先核对 {len(suggest)} 篇,序号:{' '.join(map(str, sorted(suggest)))}")
