import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from storage import DataFileError, read_json, update_json

BASE = Path(__file__).parent  # 配置和产物都绑定项目目录
load_dotenv(BASE / ".env")  # 不误读调用者当前目录中的其他 .env

# 延迟到第一次真正调用 LLM 时再创建客户端。这样无 API key 的离线测试、
# 评测脚本和库导入都不会在 import daily 时失败。保留 client 变量也方便测试注入 fake。
client = None

_RETRYABLE_LLM_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def _is_retryable_llm_error(exc):
    """只重试连接、超时、限流和服务端错误；鉴权/参数错误立即抛出。"""
    if isinstance(exc, _RETRYABLE_LLM_ERRORS):
        return True
    status = getattr(exc, "status_code", None)
    return status in _RETRYABLE_STATUS_CODES or (
        isinstance(status, int) and 500 <= status < 600
    )


def get_client():
    global client
    if client is not None:
        return client
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY；抓取和离线测试可无 key 运行，"
            "需要 LLM 打分/反思时请在 .env 中配置。"
        )
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",  # DeepSeek 用 OpenAI 兼容接口
        max_retries=0,  # 统一由 chat() 控制退避次数，避免 SDK 内外层叠加重试
    )
    return client

def chat(
    prompt,
    temperature=0,
    model="deepseek-chat",
    retries=3,
    timeout=60,
    json_mode=False,
    system=None,
):
    """统一的 LLM 调用入口:带超时 + 指数退避重试。
    全项目的 DeepSeek 调用都走这里；只对瞬时网络/限流/服务端错误重试。
    json_mode=True 走 DeepSeek 的 json_object 输出(要求 prompt 里出现 "json" 字样)。"""
    if retries < 1:
        raise ValueError("retries 必须 >= 1")
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    llm_client = get_client()  # 配置缺失不是瞬时网络错误，不做无意义的退避重试
    for i in range(retries):
        try:
            resp = llm_client.chat.completions.create(
                model=model,
                temperature=temperature,
                timeout=timeout,
                messages=messages,
                **extra,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if i == retries - 1 or not _is_retryable_llm_error(e):
                raise
            wait = 2 ** i
            detail = " ".join(str(e).split())[:200]
            print(f"[llm] 第{i + 1}次瞬时失败({type(e).__name__}: {detail}),{wait}s 后重试")
            time.sleep(wait)

# 抓哪些方向：cs.AI(人工智能) + cs.LG(机器学习)
CATEGORIES = ["cs.AI", "cs.LG"]
MAX_RESULTS = 30  # 抓 30 篇,再筛出最相关的
SEEN_FILE = BASE / "seen.json"  # 记录抓过的论文 id,用来去重
FETCH_STATE_FILE = BASE / "fetch_state.json"  # 仅在整条 graph 成功后推进水位
DEFAULT_LOOKBACK_DAYS = 7  # 首次运行覆盖一周，避免周末/节假日无投稿
CURSOR_OVERLAP_MINUTES = 15  # 水位回退一小段，用 seen/id union 消化重复


class FetchResult(list):
    """兼容原有 list API 的抓取结果，同时显式区分“成功但为空”和“失败”。"""

    def __init__(self, papers=(), *, ok=True, error="", since=None, until=None):
        super().__init__(papers)
        self.ok = bool(ok)
        self.error = str(error or "")
        self.since = since
        self.until = until


def _as_utc(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise TypeError("time value must be an ISO string or datetime")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso_utc(value):
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_fetch_cursor():
    """读取上次成功运行的 UTC 水位；文件缺失/损坏时回退到首次运行。"""
    if not FETCH_STATE_FILE.exists():
        return None
    try:
        obj = read_json(FETCH_STATE_FILE, {})
        value = obj.get("last_success_utc") if isinstance(obj, dict) else None
        return _iso_utc(value) if value else None
    except (DataFileError, ValueError, TypeError):
        return None


def save_fetch_cursor(value):
    """锁内推进成功水位；并发的较旧任务不能把水位倒退。"""
    target = _iso_utc(value)

    def advance(payload):
        payload = payload if isinstance(payload, dict) else {}
        current = payload.get("last_success_utc")
        try:
            if current and _as_utc(current) >= _as_utc(target):
                return payload
        except (ValueError, TypeError):
            pass
        return {**payload, "last_success_utc": target}

    update_json(FETCH_STATE_FILE, {}, advance)


def get_fetch_window(last_success=None, now=None):
    """固定本次 graph 的抓取窗口，多轮 widen/refocus 复用同一窗口。"""
    end = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    cursor = last_success if last_success is not None else load_fetch_cursor()
    if cursor:
        start = _as_utc(cursor) - timedelta(minutes=CURSOR_OVERLAP_MINUTES)
    else:
        start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    if start > end:
        # 系统时钟回拨时不构造反向区间；下次仍可由 seen 去重。
        start = end - timedelta(minutes=CURSOR_OVERLAP_MINUTES)
    return _iso_utc(start), _iso_utc(end)

def load_seen():
    values = read_json(SEEN_FILE, [])
    if not isinstance(values, list):
        raise DataFileError(f"seen 状态必须是数组: {SEEN_FILE}")
    return {str(value) for value in values if value}

def save_seen(seen):
    """seen 是只增集合；锁内 union，避免两个重叠运行互相覆盖。"""
    incoming = {str(value) for value in seen if value}

    def merge(values):
        if not isinstance(values, list):
            raise DataFileError(f"seen 状态必须是数组: {SEEN_FILE}")
        return sorted({str(value) for value in values if value} | incoming)

    update_json(SEEN_FILE, [], merge)

def fetch_papers(
    max_results=MAX_RESULTS,
    categories=None,
    *,
    since=None,
    until=None,
    page_size=100,
    arxiv_client=None,
):
    """
    抓 arxiv 论文。原有 ``fetch_papers(max_results, categories)`` 调用保持兼容；
    graph 传 ``since/until`` 时则以时间窗口为准，不用 top-N 截断，由 arxiv.Client
    按 ``page_size`` 自动翻页直到窗口耗尽，防止同日早些的论文永久漏抓。

    返回 FetchResult（list 子类）；旧代码可继续当 list 使用，graph 可读
    ``ok/error`` 区分空结果与网络失败。``arxiv_client`` 便于纯离线 mock。
    """
    cats = list(dict.fromkeys(categories or CATEGORIES))
    category_query = " OR ".join(f"cat:{c}" for c in cats)
    start_iso = end_iso = None
    try:
        if since is not None:
            start = _as_utc(since)
            end = _as_utc(until) if until is not None else datetime.now(timezone.utc)
            if start > end:
                raise ValueError("fetch window start must not be after end")
            start_iso, end_iso = _iso_utc(start), _iso_utc(end)
            stamp = lambda dt: dt.strftime("%Y%m%d%H%M")
            query = (
                f"({category_query}) AND "
                f"submittedDate:[{stamp(start)} TO {stamp(end)}]"
            )
            # None = 不在 Search 层截断；Client.results 会根据 page_size 翻页到末尾。
            search_limit = None
        else:
            query = category_query
            search_limit = max_results
        search = arxiv.Search(
            query=query,
            max_results=search_limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,  # 按提交时间排,最新在前
        )
        api = arxiv_client or arxiv.Client(page_size=page_size)
        return FetchResult(api.results(search), since=start_iso, until=end_iso)
    except Exception as e:
        print(f"[fetch] arxiv 抓取失败({e}),今日跳过")
        return FetchResult([], ok=False, error=str(e), since=start_iso, until=end_iso)
