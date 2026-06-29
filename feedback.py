import sys

from memory import load_last_top, record_feedback

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("up", "down"):
        print("用法: python feedback.py <序号> up|down")
        sys.exit(1)
    idx = int(sys.argv[1]) - 1
    label = sys.argv[2]
    top = load_last_top()
    if not (0 <= idx < len(top)):
        print(f"序号超范围,最近速报有 {len(top)} 篇")
        sys.exit(1)
    paper = top[idx]
    record_feedback(paper["id"], paper["title"], label, paper.get("summary", ""))
    mark = "有用" if label == "up" else "没用"
    print(f"已记录[{mark}]:{paper['title']}")
