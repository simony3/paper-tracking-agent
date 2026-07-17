import argparse
from pathlib import Path

from qa import load_store
from storage import atomic_write_json, read_json, update_json

BASE = Path(__file__).parent
LABELS_FILE = BASE / "labels.json"   # 评测用标注:id -> rel/irrel(和偏好画像分开)

def load_labels():
    return read_json(LABELS_FILE, {})

def save_labels(labels):
    atomic_write_json(LABELS_FILE, labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description="查看或批量标注论文相关性")
    parser.add_argument("tag", nargs="?", choices=("rel", "irrel"), help="相关性标签")
    parser.add_argument("indices", nargs="*", type=int, metavar="序号", help="论文序号（从 1 开始）")
    args = parser.parse_args(argv)

    store = load_store()
    labels = load_labels()

    if args.tag is None:  # 不带参数:列出论文库 + 当前标注状态
        for i, r in enumerate(store, 1):
            print(f"{i}. [{labels.get(r['id'], ' ')}] {r['title']}")
        print(f"\n共 {len(store)} 篇,已标 {len(labels)} 篇。用法: python label.py <rel|irrel> <序号> [序号...]")
        return 0

    if not args.indices:
        parser.error("至少提供一个论文序号")
    invalid = [number for number in args.indices if not 1 <= number <= len(store)]
    if invalid:
        parser.error(f"序号超范围: {', '.join(map(str, invalid))}；当前论文库共 {len(store)} 篇")

    updates = {store[number - 1]["id"]: args.tag for number in args.indices}

    def apply(current):
        if not isinstance(current, dict):
            raise ValueError("labels.json 必须是对象")
        current.update(updates)
        return current

    labels = update_json(LABELS_FILE, {}, apply)
    print(f"已批量标[{args.tag}] {len(args.indices)} 篇,当前共标 {len(labels)} 篇")
    return 0


if __name__ == "__main__":
    main()
