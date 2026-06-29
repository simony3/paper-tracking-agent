from types import SimpleNamespace

import retrieval
from retrieval import rrf, tokenize, _score_batch
from eval import metric


def test_rrf_basic():
    # 两个榜都把 idx=2 排第一,融合后 2 必居首
    fused, _ = rrf([[2, 0, 1], [2, 1, 0]])
    assert fused[0] == 2

def test_rrf_rewards_consistency():
    # 在两榜都靠前的,应排在只在单榜靠前的之前
    fused, _ = rrf([[0, 1, 2], [0, 2, 1]])
    assert fused[0] == 0

def test_tokenize_lowercase_alnum():
    assert tokenize("RAG, Multi-Agent!") == ["rag", "multi", "agent"]

def _paper(t="t", s="s"):
    return SimpleNamespace(title=t, summary=s, entry_id="id")

def test_score_batch_detects_missing_id(monkeypatch):
    # LLM 只回 1 条,batch 有 2 篇 → 一致性校验必须判异常
    monkeypatch.setattr("daily.chat", lambda *a, **k: '{"id": 1, "score": 5, "reason": "x"}')
    parsed, bad = _score_batch("interest", [_paper(), _paper()])
    assert bad is True
    assert len(parsed) == 2          # 缺失项被补成占位,不丢位

def test_score_batch_ok_when_complete(monkeypatch):
    monkeypatch.setattr(
        "daily.chat",
        lambda *a, **k: '{"id": 1, "score": 8, "reason": "a"}\n{"id": 2, "score": 3, "reason": "b"}',
    )
    parsed, bad = _score_batch("interest", [_paper(), _paper()])
    assert bad is False
    assert parsed[0][0] == 8 and parsed[1][0] == 3

def test_score_batch_detects_out_of_range_id(monkeypatch):
    # 模型回了一个越界 id=9 → 判异常
    monkeypatch.setattr(
        "daily.chat",
        lambda *a, **k: '{"id": 1, "score": 5, "reason": "a"}\n{"id": 9, "score": 5, "reason": "b"}',
    )
    _, bad = _score_batch("interest", [_paper(), _paper()])
    assert bad is True

def test_metric_precision():
    # top-10 里 4 篇 rel → Precision@10 == 0.4
    ranked = [SimpleNamespace(entry_id=str(i)) for i in range(10)]
    labels = {"0": "rel", "1": "rel", "2": "rel", "3": "rel"}
    p, r, cov = metric(ranked, labels, k=10)
    assert p == 0.4
    assert r == 1.0          # 4 篇 rel 全在 top-10
    assert cov == 4
