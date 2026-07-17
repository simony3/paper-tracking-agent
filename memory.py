from datetime import date
from pathlib import Path

from retrieval import INTEREST  # 首次的基础画像种子
from storage import atomic_write_json, read_json, update_json

BASE = Path(__file__).parent
PROFILE_FILE = BASE / "profile.json"      # 画像 + 反馈历史
LAST_TOP_FILE = BASE / "last_top.json"    # 最近一期速报的精选,供反馈按序号定位

DEFAULT_PROFILE = {
    "interest": INTEREST,
    "liked": [],
    "disliked": [],
    "feedback": [],          # 每篇论文的当前标签（按 id 唯一）
    "feedback_events": [],   # 只追加的历史事件，便于审计标签反转
}


def _normalise_profile(profile):
    """兼容旧 profile.json，同时保证后续代码拿到稳定 schema。"""
    out = {key: list(value) if isinstance(value, list) else value for key, value in DEFAULT_PROFILE.items()}
    if isinstance(profile, dict):
        out.update(profile)
    for key in ("liked", "disliked", "feedback", "feedback_events"):
        if not isinstance(out.get(key), list):
            out[key] = []
    if not isinstance(out.get("interest"), str) or not out["interest"].strip():
        out["interest"] = INTEREST
    return out

def load_profile():
    return _normalise_profile(read_json(PROFILE_FILE, DEFAULT_PROFILE))

def save_profile(p):
    atomic_write_json(PROFILE_FILE, _normalise_profile(p))

def get_interest():
    """检索用的画像 = 干净的基础画像。偏好不再拼进 query 文本(嵌入不懂否定、整段标题会带偏),
    改由 retrieval.preference_bonus 在打分后做向量空间重排。"""
    return load_profile()["interest"]

def get_feedback_anchors():
    """返回 (liked, disliked) 两组锚文本(title+summary),供偏好重排算相似度。
    存 title+summary 而非纯 title:候选用的是 title+summary,两边对称信号才强。"""
    p = load_profile()
    return p["liked"], p["disliked"]

def get_anchor_ids():
    """有过反馈的论文 id,评测时从测试集排除,避免 anchor 给自己/近邻送分(数据泄漏)。"""
    return {f["id"] for f in load_profile()["feedback"] if isinstance(f, dict) and f.get("id")}

def save_last_top(top):
    data = [{"id": p.entry_id, "title": p.title, "summary": p.summary} for p, _, _ in top]
    atomic_write_json(LAST_TOP_FILE, data)

def load_last_top():
    return read_json(LAST_TOP_FILE, [])

def record_feedback(paper_id, title, label, summary=""):
    """记录当前偏好与事件历史；同一论文反转标签时不会留在两个 bucket。"""
    if label not in {"up", "down"}:
        raise ValueError("label 必须是 up 或 down")
    anchor = f"{title}. {summary}" if summary else title  # 锚文本与候选 paper_text 对称
    now = date.today().isoformat()

    def apply(profile):
        p = _normalise_profile(profile)
        previous = [f for f in p["feedback"] if isinstance(f, dict) and f.get("id") == paper_id]
        old_titles = {str(f.get("title", "")) for f in previous if f.get("title")}
        old_anchors = {str(f.get("anchor", "")) for f in previous if f.get("anchor")}

        def belongs_to_this_paper(value):
            return (
                value == anchor
                or value in old_anchors
                or value == title
                or value.startswith(f"{title}. ")
                or any(value == t or value.startswith(f"{t}. ") for t in old_titles)
            )

        # 先从两边清掉旧状态，再只写入当前 bucket。
        p["liked"] = [v for v in p["liked"] if not belongs_to_this_paper(v)]
        p["disliked"] = [v for v in p["disliked"] if not belongs_to_this_paper(v)]
        bucket = "liked" if label == "up" else "disliked"
        if anchor not in p[bucket]:
            p[bucket].append(anchor)

        event = {
            "id": paper_id,
            "title": title,
            "summary": summary,
            "anchor": anchor,
            "label": label,
            "date": now,
        }
        p["feedback"] = [
            f for f in p["feedback"]
            if not (isinstance(f, dict) and f.get("id") == paper_id)
        ]
        p["feedback"].append(event)
        p["feedback_events"].append(event)
        return p

    update_json(PROFILE_FILE, DEFAULT_PROFILE, apply)
