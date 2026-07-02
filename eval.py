import random
from types import SimpleNamespace

from qa import load_store
from retrieval import INTEREST, RELEVANT_THRESHOLD, bm25_recall, embed_recall, hybrid_recall, llm_score, preference_bonus
from label import load_labels
from memory import get_anchor_ids

K = 10

def to_paper(r):
    return SimpleNamespace(title=r["title"], summary=r["summary"], entry_id=r["id"])

def metric(ranked, labels, k=K):
    topk = ranked[:k]
    # 标准口径:未标注的论文按"不相关"处理,分母固定为 top-k 实际篇数(通常=10)
    rel_in_topk = sum(1 for p in topk if labels.get(p.entry_id) == "rel")
    total_rel = sum(1 for v in labels.values() if v == "rel")
    precision = rel_in_topk / len(topk) if topk else 0.0
    recall = rel_in_topk / total_rel if total_rel else 0.0
    labeled_in_topk = sum(1 for p in topk if p.entry_id in labels)
    return precision, recall, labeled_in_topk

if __name__ == "__main__":
    labels = load_labels()
    store = load_store()

    if len([v for v in labels.values() if v == "rel"]) < 3 or len(labels) < 10:
        print(f"当前已标注 {len(labels)} 篇(相关 {sum(v=='rel' for v in labels.values())} 篇)。")
        print("评测需要先标注 ~20 篇:用 `python label.py` 看列表,`python label.py <rel|irrel> <序号> [序号...]` 逐篇标。")
        print("标够后再跑 `python eval.py` 出 Precision@10 数字。")
        raise SystemExit

    # 隔离训练/测试:把有反馈的论文从评测集排除,杜绝 anchor 给自己/近邻送分(数据泄漏)
    anchors = get_anchor_ids()
    papers = [to_paper(r) for r in store if r["id"] not in anchors]
    feedback_n = len([1 for r in store if r["id"] in anchors])

    # 消融各级排序:BM25/向量/RRF 全是本地计算零 API;LLM 打分走 scores_cache 复用
    b_ranked, _ = bm25_recall(INTEREST, papers)
    e_ranked, _ = embed_recall(INTEREST, papers)
    cand = [papers[i] for i in hybrid_recall(INTEREST, papers)]
    # 只打一次分,记忆开/关共用同一组分数;唯一差别是偏好重排 bonus → LLM 打分噪声在对比中抵消
    scores, _ = llm_score(INTEREST, cand)
    bonus = preference_bonus(cand)
    off = [p for p, _ in sorted(zip(cand, scores), key=lambda x: x[1][0], reverse=True)]
    on = [p for p, _, _ in sorted(zip(cand, scores, bonus), key=lambda x: x[1][0] + x[2], reverse=True)]

    # 随机基线:证明"我的检索"显著优于瞎排,比记忆 on/off 的微弱差异更有说服力
    rng = random.Random(42)
    rand = cand[:]
    rng.shuffle(rand)

    ablation = [
        ("随机基线", rand),
        ("纯 BM25", [papers[i] for i in b_ranked]),
        ("纯向量", [papers[i] for i in e_ranked]),
        ("混合 RRF", cand),
        ("混合+LLM打分(记忆关)", off),
        ("混合+LLM+记忆(开)", on),
    ]
    _, _, cov = metric(off, labels)

    # 一致性:LLM 打分≥阈值 与 人工 rel 标签 的吻合率,验证"7 分阈值"是否靠谱(P2-H)
    agree = total = 0
    for p, (s, _) in zip(cand, scores):
        if p.entry_id in labels:
            total += 1
            agree += (s >= RELEVANT_THRESHOLD) == (labels[p.entry_id] == "rel")
    consistency = agree / total if total else 0.0

    print(f"评测口径:标注 {len(labels)} 篇 / 反馈 {feedback_n} 条(已从测试集排除)/ "
          f"测试集 {len(papers)} 篇 / top-{K} 中已标注 {cov} 篇")
    print("注意:标注为 LLM-as-a-judge 银标签、反馈样本小,数字仅供横向对比参考\n")
    print(f"{'排序方案':<24}Precision@{K}  Recall@{K}")
    for name, ranked in ablation:
        p, r, _ = metric(ranked, labels)
        print(f"{name:<24}{p:>12.2f}{r:>10.2f}")
    print(f"\nLLM≥{RELEVANT_THRESHOLD}分 与人工 rel 标签一致率:{consistency:.2f}(共 {total} 篇有标签)")
