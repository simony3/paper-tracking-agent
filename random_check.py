"""从未标注论文里随机抽一批,生成独立核对清单(故意不显示 LLM 分/理由,
让人工判断不被模型锚定)→ 标注后评测集不再只含"检索的 favorites",降低选择偏差。
用法: python random_check.py [N]  默认抽 18 篇,随机种子固定可复现。"""
import argparse
import random
from pathlib import Path

from storage import atomic_write_text, read_json

BASE = Path(__file__).parent
SEED = 2026


def main(argv=None):
    parser = argparse.ArgumentParser(description="随机抽取未标注论文，生成无模型提示的盲核清单")
    parser.add_argument("count", nargs="?", type=int, default=18, help="抽样篇数（默认 18）")
    parser.add_argument("--seed", type=int, default=SEED, help=f"随机种子（默认 {SEED}）")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("抽样篇数必须大于 0")

    store = read_json(BASE / "papers_store.json", [])
    labels = read_json(BASE / "labels.json", {})
    if not store:
        parser.error("论文库为空，请先运行 graph.py")
    unlabeled = [(i, r) for i, r in enumerate(store) if r["id"] not in labels]
    if not unlabeled:
        parser.error("当前没有未标注论文")
    sample = random.Random(args.seed).sample(unlabeled, min(args.count, len(unlabeled)))
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
    atomic_write_text(BASE / "随机核对.md", "\n".join(lines))
    print(f"随机核对.md 已生成,{len(sample)} 篇,序号:{' '.join(map(str, nums))}")


if __name__ == "__main__":
    main()
