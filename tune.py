"""常量扫描:对 PREF_WEIGHT 各取值跑一遍 Precision/Recall@10,挑最优,兑现 P2-G。
零 API —— 打分走缓存、偏好重排是本地向量计算。用法: python tune.py"""
import retrieval
from types import SimpleNamespace

from qa import load_store
from retrieval import INTEREST, hybrid_recall, llm_score, preference_bonus
from label import load_labels
from memory import get_anchor_ids
from eval import metric, K

if __name__ == "__main__":
    labels = load_labels()
    anchors = get_anchor_ids()
    papers = [SimpleNamespace(title=r["title"], summary=r["summary"], entry_id=r["id"])
              for r in load_store() if r["id"] not in anchors]
    cand = [papers[i] for i in hybrid_recall(INTEREST, papers)]
    scores, _ = llm_score(INTEREST, cand)

    print(f"测试集 {len(papers)} 篇(已排除反馈 {len(anchors)} 条)\n")
    print(f"{'PREF_WEIGHT':>12} | Precision@{K} | Recall@{K}")
    best = None
    for w in (0, 1, 2, 3, 5, 8):
        retrieval.PREF_WEIGHT = w
        bonus = preference_bonus(cand)
        on = [p for p, _, _ in sorted(zip(cand, scores, bonus), key=lambda x: x[1][0] + x[2], reverse=True)]
        p, r, _ = metric(on, labels)
        print(f"{w:>12} | {p:>11.2f} | {r:>8.2f}")
        if best is None or p > best[1] or (p == best[1] and r > best[2]):
            best = (w, p, r)
    print(f"\n本数据上最优 PREF_WEIGHT = {best[0]}(Precision@{K}={best[1]:.2f} Recall@{K}={best[2]:.2f})")
