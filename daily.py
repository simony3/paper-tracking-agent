import json
import os
import time
from pathlib import Path

import arxiv
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取 .env 里的 DEEPSEEK_API_KEY
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",  # DeepSeek 用 OpenAI 兼容接口
)

def chat(prompt, temperature=0, model="deepseek-chat", retries=3, timeout=60, json_mode=False):
    """统一的 LLM 调用入口:带超时 + 指数退避重试。
    全项目的 DeepSeek 调用都走这里,网络抖动/限流不再让整条 graph 崩。
    json_mode=True 走 DeepSeek 的 json_object 输出(要求 prompt 里出现 "json" 字样)。"""
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                timeout=timeout,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 2 ** i
            print(f"[llm] 第{i + 1}次失败({e}),{wait}s 后重试")
            time.sleep(wait)

# 抓哪些方向：cs.AI(人工智能) + cs.LG(机器学习)
CATEGORIES = ["cs.AI", "cs.LG"]
MAX_RESULTS = 30  # 抓 30 篇,再筛出最相关的
BASE = Path(__file__).parent  # 脚本所在目录,保证产物总落在项目里
SEEN_FILE = BASE / "seen.json"  # 记录抓过的论文 id,用来去重

def load_seen():
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()

def save_seen(seen):
    SEEN_FILE.write_text(json.dumps(sorted(seen)))

def fetch_papers(max_results=MAX_RESULTS, categories=None):
    # categories 可被 reflect 决策覆盖(widen 时临时补 cs.CL/cs.IR 等)
    cats = categories or CATEGORIES
    # cat:cs.AI OR cat:cs.LG —— arxiv 的查询语法,表示这些分类任一
    query = " OR ".join(f"cat:{c}" for c in cats)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 按提交时间排,最新在前
    )
    arxiv_client = arxiv.Client()
    try:
        return list(arxiv_client.results(search))
    except Exception as e:
        print(f"[fetch] arxiv 抓取失败({e}),今日跳过")
        return []
