import json
from datetime import date
from pathlib import Path

from retrieval import INTEREST  # 首次的基础画像种子

BASE = Path(__file__).parent
PROFILE_FILE = BASE / "profile.json"      # 画像 + 反馈历史
LAST_TOP_FILE = BASE / "last_top.json"    # 最近一期速报的精选,供反馈按序号定位

def load_profile():
    if PROFILE_FILE.exists():
        return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    return {"interest": INTEREST, "liked": [], "disliked": [], "feedback": []}

def save_profile(p):
    PROFILE_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")

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
    return {f["id"] for f in load_profile()["feedback"]}

def save_last_top(top):
    data = [{"id": p.entry_id, "title": p.title, "summary": p.summary} for p, _, _ in top]
    LAST_TOP_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_last_top():
    if LAST_TOP_FILE.exists():
        return json.loads(LAST_TOP_FILE.read_text(encoding="utf-8"))
    return []

def record_feedback(paper_id, title, label, summary=""):
    """label='up'(有用)/'down'(没用)。记反馈 + 更新画像。反馈同时是评测标签。"""
    p = load_profile()
    p["feedback"].append({"id": paper_id, "title": title, "label": label, "date": date.today().isoformat()})
    anchor = f"{title}. {summary}" if summary else title  # 锚文本与候选 paper_text 对称
    bucket = "liked" if label == "up" else "disliked"
    if anchor not in p[bucket]:
        p[bucket].append(anchor)
    save_profile(p)
