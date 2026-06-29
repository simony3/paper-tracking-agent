"""从未标注论文里随机抽一批,生成独立核对清单(故意不显示 LLM 分/理由,
让人工判断不被模型锚定)→ 标注后评测集不再只含"检索的favorites",洗掉 P@10 选择偏差。
用法: python random_check.py [N]  默认抽 18 篇,随机种子固定可复现。"""
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 18
SEED = 2026

if __name__ == "__main__":
    store = json.loads((BASE / "papers_store.json").read_text(encoding="utf-8"))
    labels = json.loads((BASE / "labels.json").read_text(encoding="utf-8"))
    unlabeled = [(i, r) for i, r in enumerate(store) if r["id"] not in labels]
    sample = random.Random(SEED).sample(unlabeled, min(N, len(unlabeled)))
    sample.sort(key=lambda x: x[0])  # 按序号排,读着顺

    lines = ["# 随机核对清单(独立标注,不看模型分)\n",
             f"从 {len(unlabeled)} 篇未标注里随机抽 {len(sample)} 篇。读摘要判断是否相关,然后告诉我序号。\n"]
    nums = []
    for i, r in sample:
        num = i + 1
        nums.append(num)
        lines.append(f"\n## 序号 {num}")
        lines.append(f"**{r['title']}**\n")
        lines.append(f"{r['summary']}\n")
    (BASE / "随机核对.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"随机核对.md 已生成,{len(sample)} 篇,序号:{' '.join(map(str, nums))}")
