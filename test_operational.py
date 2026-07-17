from types import SimpleNamespace

import numpy as np
import pytest

import daily
import feedback
import label
import memory
import qa
import review_list


def test_feedback_reversal_keeps_one_current_bucket(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "PROFILE_FILE", tmp_path / "profile.json")

    memory.record_feedback("paper-1", "Title", "up", "Summary")
    memory.record_feedback("paper-1", "Title", "down", "Summary")

    profile = memory.load_profile()
    assert profile["liked"] == []
    assert profile["disliked"] == ["Title. Summary"]
    assert [(row["id"], row["label"]) for row in profile["feedback"]] == [("paper-1", "down")]
    assert [row["label"] for row in profile["feedback_events"]] == ["up", "down"]


def test_add_to_store_deduplicates_same_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "STORE_FILE", tmp_path / "papers_store.json")
    paper = SimpleNamespace(entry_id="paper-1", title="Title", summary="Summary")

    qa.add_to_store([paper, paper])

    assert qa.load_store() == [{"id": "paper-1", "title": "Title", "summary": "Summary"}]


class _FakeEmbedder:
    def embed(self, _texts):
        return iter(
            [
                np.array([1.0, 0.0]),   # query
                np.array([1.0, 0.0]),   # relevant document
                np.array([-1.0, 0.0]),  # noisy document
            ]
        )


def test_qa_filters_each_candidate_and_requires_valid_citation(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "STORE_FILE", tmp_path / "papers_store.json")
    qa.add_to_store(
        [
            SimpleNamespace(entry_id="p1", title="Relevant", summary="Useful evidence"),
            SimpleNamespace(entry_id="p2", title="Noise", summary="Unrelated"),
        ]
    )
    monkeypatch.setattr(qa, "translate_to_en", lambda _question: "query")
    monkeypatch.setattr(qa, "hybrid_recall", lambda *_args, **_kwargs: [0, 1])
    monkeypatch.setattr(qa, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(daily, "chat", lambda *_args, **_kwargs: "结论有摘要支持[1]")

    answer = qa.ask("问题")

    assert "Relevant" in answer
    assert "Noise" not in answer

    monkeypatch.setattr(daily, "chat", lambda *_args, **_kwargs: "伪造引用[9]")
    assert qa.ask("问题") == "生成回答未通过引用校验，无法可靠作答。"


def test_label_rejects_out_of_range_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "LABELS_FILE", tmp_path / "labels.json")
    monkeypatch.setattr(label, "load_store", lambda: [{"id": "p1", "title": "Title"}])

    with pytest.raises(SystemExit):
        label.main(["rel", "99"])

    assert not label.LABELS_FILE.exists()


def test_cli_rejects_non_numeric_feedback_index():
    with pytest.raises(SystemExit):
        feedback.main(["not-a-number", "up"])


def test_qa_rejects_non_positive_top_n():
    with pytest.raises(ValueError, match="top_n"):
        qa.ask("question", top_n=0)


def test_review_list_caps_llm_candidates_and_writes_atomically(tmp_path, monkeypatch):
    rows = [
        {"id": f"p{index}", "title": f"Title {index}", "summary": "Summary"}
        for index in range(3)
    ]
    captured = []
    monkeypatch.setattr(review_list, "BASE", tmp_path)
    monkeypatch.setattr(review_list, "load_store", lambda: rows)
    monkeypatch.setattr(review_list, "load_labels", lambda: {})
    monkeypatch.setattr(review_list, "hybrid_recall", lambda *_args: [2, 1, 0])

    def fake_score(_interest, papers):
        captured.extend(paper.entry_id for paper in papers)
        return [(8, "high"), (6, "boundary")], False

    monkeypatch.setattr(review_list, "llm_score", fake_score)

    assert review_list.main(["--review-count", "1", "--llm-candidates", "2"]) == 0
    assert captured == ["p2", "p1"]
    assert (tmp_path / "待核对清单.md").exists()
    assert "未送 LLM" not in (tmp_path / "核对详情.md").read_text(encoding="utf-8")


def test_review_list_rejects_non_positive_limits():
    with pytest.raises(SystemExit):
        review_list.main(["--llm-candidates", "0"])
