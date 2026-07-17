import argparse

from memory import load_last_top, record_feedback


def main(argv=None):
    parser = argparse.ArgumentParser(description="对最近一期速报中的论文记录偏好")
    parser.add_argument("index", type=int, help="速报中的论文序号（从 1 开始）")
    parser.add_argument("label", choices=("up", "down"), help="up=有用，down=没用")
    args = parser.parse_args(argv)

    idx = args.index - 1
    top = load_last_top()
    if not (0 <= idx < len(top)):
        parser.error(f"序号超范围，最近速报有 {len(top)} 篇")
    paper = top[idx]
    record_feedback(paper["id"], paper["title"], args.label, paper.get("summary", ""))
    mark = "有用" if args.label == "up" else "没用"
    print(f"已记录[{mark}]:{paper['title']}")


if __name__ == "__main__":
    main()
