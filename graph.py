import json
import re
from datetime import date
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from daily import CATEGORIES, BASE, chat, fetch_papers, load_seen, save_seen
from retrieval import RELEVANT_THRESHOLD, rank_papers, deep_read
from memory import get_interest, save_last_top
from qa import add_to_store, load_store

INITIAL_MAX = 30          # 首次抓多少篇
STEP = 30                 # 每次重抓多加多少篇
MAX_ATTEMPTS = 3          # 最多抓几轮(防止无限循环)
MIN_RELEVANT = 3          # 精选里至少要有几篇相关,否则触发反思重抓

# State = 全程传递的"公共笔记本"
class State(TypedDict):
    attempts: int          # 已抓几轮
    max_results: int       # 本轮抓多少篇
    query: str             # 当前检索用的兴趣画像(reflect 可收窄它)
    extra_cats: list       # reflect 追加的 arxiv 分类(widen)
    new_papers: list       # 去重后的新论文
    top: list              # 精选 (paper, score, reason)
    hallu: bool            # 打分一致性校验是否异常
    relevant_count: int    # 精选里相关的篇数
    action: str            # reflect 产出的下一步动作

# 节点1:抓取 + 去重
def fetch_node(state: State):
    attempts = state.get("attempts", 0)
    max_results = INITIAL_MAX + attempts * STEP
    cats = list(dict.fromkeys(CATEGORIES + state.get("extra_cats", [])))  # widen 追加分类,去重保序
    seen = load_seen()
    papers = fetch_papers(max_results, categories=cats)
    new = [p for p in papers if p.entry_id not in seen]
    print(f"[fetch] 第{attempts + 1}轮:分类{cats} 抓 {max_results} 篇,新论文 {len(new)} 篇")
    return {"new_papers": new, "attempts": attempts + 1, "max_results": max_results}

# 节点2:混合检索 + 打分 + 选 top-K
def rank_node(state: State):
    new = state["new_papers"]
    interest = state.get("query") or get_interest()  # reflect 收窄过就用新 query
    bg = [r["summary"] for r in load_store()]  # 累积论文库做 BM25 背景语料,IDF 才有统计意义
    top, hallu = rank_papers(interest, new, use_memory=True, bg_texts=bg) if new else ([], False)
    relevant = sum(1 for _, s, _ in top if s >= RELEVANT_THRESHOLD)
    print(f"[rank] 精选 {len(top)} 篇,其中相关(≥{RELEVANT_THRESHOLD}分){relevant} 篇")
    return {"top": top, "hallu": hallu, "relevant_count": relevant}

# 条件边1:相关够 / 抓够轮数 → 出报告;否则进反思
def decide(state: State):
    if state["hallu"]:
        # 打分失效时 relevant_count 基于坏分数,反思会被误导 → 直接出报告(头部已标注异常)
        print("[decide] 打分校验异常 → 跳过反思,直接出速报")
        return "report"
    if state["relevant_count"] < MIN_RELEVANT and state["attempts"] < MAX_ATTEMPTS:
        print(f"[decide] 相关仅 {state['relevant_count']} 篇 < {MIN_RELEVANT} → 反思")
        return "reflect"
    print("[decide] 相关足够(或已达上限)→ 出速报")
    return "report"

# 节点3:反思 —— 让 LLM 诊断"为什么相关太少",自主选下一步动作
def reflect_node(state: State):
    """这是本项目真正的 agent 决策点:不是固定多抓,而是让 LLM 看当前候选+兴趣,
    在 widen(分类太窄)/refocus(兴趣太宽)/stop(今天确实没好论文)三种动作里自选。"""
    interest = state.get("query") or get_interest()
    titles = "\n".join(f"- [{s}] {p.title}" for p, s, _ in state["top"]) or "(本轮无候选)"
    prompt = (
        "我在按研究兴趣筛 arxiv 论文,这是当前候选(分越低越不相关):\n"
        f"{titles}\n\n我的兴趣:{interest}\n\n"
        "相关论文太少。请判断原因并只输出一个 JSON(不要多余文字):\n"
        '{"action":"widen|refocus|stop",'
        '"new_query":"<若 refocus,给收窄后的英文兴趣描述>",'
        '"new_categories":["<若 widen,给要补充的 arxiv 分类如 cs.CL>"],'
        '"why":"<一句中文理由>"}'
    )
    text = chat(prompt, temperature=0, json_mode=True)
    try:
        obj = json.loads(text)
    except Exception:
        obj = {"action": "stop", "why": "决策解析失败,停止"}
    action = obj.get("action", "stop")
    print(f"[reflect] 决策={action} —— {obj.get('why', '')}")
    out = {"action": action}
    if action == "refocus" and obj.get("new_query"):
        out["query"] = obj["new_query"]
    if action == "widen":
        # 白名单校验:LLM 给的分类要拼进 arxiv query,只放行 cs.CL/stat.ML 这类合法格式
        cats = [c for c in obj.get("new_categories", []) if re.fullmatch(r"[a-z-]+\.[A-Z]{2}", str(c))]
        if cats:
            out["extra_cats"] = sorted(set(state.get("extra_cats", []) + cats))
        else:
            out["action"] = "stop"  # 全部不合法,widen 无从执行
    return out

# 条件边2:按反思动作路由 —— stop 去报告,widen/refocus 回去重抓
def route_action(state: State):
    return "report" if state.get("action") == "stop" else "fetch"

# 节点4:生成速报 + 记下抓过的
def report_node(state: State):
    top, hallu = state["top"], state["hallu"]
    interest = state.get("query") or get_interest()
    today = date.today().isoformat()
    lines = [
        f"# arxiv 速报 · {today}",
        f"\n方向:{', '.join(dict.fromkeys(CATEGORIES + state.get('extra_cats', [])))} | 精选 {len(top)} 篇 | "
        f"打分校验:{'异常(本期打分仅供参考)' if hallu else '通过'}\n",
    ]
    for i, (p, score, reason) in enumerate(top, 1):
        print(f"[report] 精读第 {i}/{len(top)} 篇...")
        lines.append(f"## {i}. [{score}/10] {p.title}")
        lines.append(f"**为什么相关**:{reason}\n")
        lines.append(deep_read(interest, p))
        lines.append(f"\n[原文]({p.entry_id})\n")
    report = BASE / f"速报_{today}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] 速报已生成:{report}")

    save_last_top(top)  # 供 feedback.py 按序号打标
    add_to_store(state["new_papers"])  # 论文入库,供 RAG 问答检索
    seen = load_seen()
    seen.update(p.entry_id for p in state["new_papers"])
    save_seen(seen)
    return {}

def build_graph():
    g = StateGraph(State)
    g.add_node("fetch", fetch_node)
    g.add_node("rank", rank_node)
    g.add_node("reflect", reflect_node)
    g.add_node("report", report_node)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "rank")
    g.add_conditional_edges("rank", decide, {"reflect": "reflect", "report": "report"})
    g.add_conditional_edges("reflect", route_action, {"fetch": "fetch", "report": "report"})
    g.add_edge("report", END)
    return g.compile()

if __name__ == "__main__":
    app = build_graph()
    app.invoke({"attempts": 0})
