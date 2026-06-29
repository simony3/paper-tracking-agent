import json
import sys
from pathlib import Path

from qa import load_store

BASE = Path(__file__).parent
LABELS_FILE = BASE / "labels.json"   # 评测用标注:id -> rel/irrel(和偏好画像分开)

def load_labels():
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    return {}

def save_labels(labels):
    LABELS_FILE.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    store = load_store()
    labels = load_labels()

    if len(sys.argv) == 1:  # 不带参数:列出论文库 + 当前标注状态
        for i, r in enumerate(store, 1):
            print(f"{i}. [{labels.get(r['id'], ' ')}] {r['title']}")
        print(f"\n共 {len(store)} 篇,已标 {len(labels)} 篇。用法: python label.py <rel|irrel> <序号> [序号...]")
        sys.exit(0)

    tag = sys.argv[1]
    if tag not in ("rel", "irrel") or len(sys.argv) < 3:
        print("用法: python label.py <rel|irrel> <序号> [序号...]")
        sys.exit(1)
    nums = [int(x) - 1 for x in sys.argv[2:]]
    for idx in nums:
        if 0 <= idx < len(store):
            labels[store[idx]["id"]] = tag
    save_labels(labels)
    print(f"已批量标[{tag}] {len(nums)} 篇,当前共标 {len(labels)} 篇")
