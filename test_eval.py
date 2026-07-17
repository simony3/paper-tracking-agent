from types import SimpleNamespace
import hashlib
import json

import pytest

from eval import (
    ClassificationMetrics,
    RankingMetrics,
    binary_classification_metrics,
    filter_labels_to_papers,
    frozen_llm_score,
    judged_metric,
    llm_candidates,
    load_snapshot_input,
    random_baseline,
    rank_scored_candidates,
)
from retrieval import LLM_CAND
from tune import deterministic_label_split, select_best_weight


def _paper(paper_id):
    return SimpleNamespace(entry_id=paper_id, title=paper_id, summary="summary")


def test_judged_metric_does_not_treat_unlabeled_as_negative():
    papers = [_paper("rel"), _paper("unjudged"), _paper("irrel")]
    labels = {"rel": "rel", "irrel": "irrel"}

    result = judged_metric(papers, labels, k=2)

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.coverage == 0.5
    assert result.judged_in_topk == 1


def test_filtering_anchor_also_fixes_recall_denominator():
    eligible = [_paper("kept"), _paper("unjudged")]
    all_labels = {
        "kept": "rel",
        "feedback-anchor": "rel",
        "not-in-store": "rel",
    }

    labels = filter_labels_to_papers(all_labels, eligible)
    result = judged_metric([eligible[0]], labels, k=1)

    assert labels == {"kept": "rel"}
    assert result.total_judged_relevant == 1
    assert result.recall == 1.0


def test_llm_candidates_enforces_production_funnel_limit():
    papers = [_paper(str(index)) for index in range(LLM_CAND + 7)]
    recalled = list(reversed(range(len(papers))))

    candidates = llm_candidates(papers, recalled)

    assert len(candidates) == LLM_CAND
    assert [paper.entry_id for paper in candidates] == [
        str(index) for index in recalled[:LLM_CAND]
    ]


def test_scored_ranking_uses_production_fallback_on_bad_scores():
    papers = [_paper("rrf-first"), _paper("rrf-second")]
    scores = [(0, "bad"), (10, "bad")]

    ranked = rank_scored_candidates(papers, scores, score_failed=True)

    assert ranked == papers


def test_random_baseline_is_repeated_and_seeded():
    papers = [_paper(str(index)) for index in range(20)]
    labels = {
        **{str(index): "rel" for index in range(4)},
        **{str(index): "irrel" for index in range(4, 12)},
    }

    first = random_baseline(papers, labels, k=5, trials=50, seed=7)
    second = random_baseline(papers, labels, k=5, trials=50, seed=7)

    assert first == second
    assert first.trials == 50
    assert first.precision.valid_trials > 1
    assert first.precision.std is not None
    assert first.coverage.mean == pytest.approx(0.6, abs=0.15)


def test_binary_metrics_include_confusion_balance_kappa_and_majority():
    papers = [_paper(name) for name in ("a", "b", "c", "d", "unjudged")]
    labels = {"a": "rel", "b": "rel", "c": "irrel", "d": "irrel"}
    scores = [(8, ""), (3, ""), (9, ""), (1, ""), (10, "")]

    result = binary_classification_metrics(papers, scores, labels, threshold=7)

    assert isinstance(result, ClassificationMetrics)
    assert (result.tp, result.fp, result.fn, result.tn) == (1, 1, 1, 1)
    assert result.judged == 4
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5
    assert result.balanced_accuracy == 0.5
    assert result.kappa == 0.0
    assert result.majority_accuracy == 0.5


def test_label_split_is_deterministic_stratified_and_order_independent():
    labels = {
        **{f"r{index}": "rel" for index in range(5)},
        **{f"i{index}": "irrel" for index in range(5)},
    }

    dev, test = deterministic_label_split(labels, dev_ratio=0.6, seed=11)
    reverse_dev, reverse_test = deterministic_label_split(
        dict(reversed(list(labels.items()))), dev_ratio=0.6, seed=11
    )

    assert dev == reverse_dev
    assert test == reverse_test
    assert set(dev).isdisjoint(test)
    assert set(dev) | set(test) == set(labels)
    assert sum(value == "rel" for value in dev.values()) == 3
    assert sum(value == "irrel" for value in dev.values()) == 3
    assert sum(value == "rel" for value in test.values()) == 2
    assert sum(value == "irrel" for value in test.values()) == 2


def test_select_best_weight_uses_only_passed_dev_results():
    weak = RankingMetrics(0.5, 0.5, 1.0, 1, 2, 2, 2)
    strong = RankingMetrics(1.0, 1.0, 1.0, 2, 2, 2, 2)

    weight, result = select_best_weight([(0.0, weak), (3.0, strong)])

    assert weight == 3.0
    assert result is strong


def test_split_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        deterministic_label_split({"a": "rel", "b": "rel"}, dev_ratio=1.0)


def test_snapshot_test_split_excludes_dev_and_anchor_but_keeps_unjudged(tmp_path):
    rows = [
        ("anchor", "rel", "anchor", True, 9),
        ("dev", "rel", "dev", False, 8),
        ("test", "irrel", "test", False, 2),
        ("unknown", None, "unjudged", False, 4),
    ]
    payload = {
        "schema_version": "paper-tracker.eval-snapshot/1.0.0",
        "dataset_version": "fixture",
        "records": [
            {
                "id": paper_id,
                "title": paper_id,
                "summary": "summary",
                "label": label_value,
                "is_anchor": is_anchor,
                "score": score,
                "reason": "reason",
                "split": split,
            }
            for paper_id, label_value, split, is_anchor, score in rows
        ],
    }
    payload["records_sha256"] = hashlib.sha256(
        json.dumps(
            payload["records"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_snapshot_input(path, "test")

    assert [record["id"] for record in loaded.records] == ["test", "unknown"]
    assert loaded.labels == {"test": "irrel"}
    assert loaded.anchors == {"anchor"}
    assert loaded.frozen_scores["test"] == (2, "reason")


def test_frozen_score_missing_entry_triggers_fallback_flag():
    papers = [_paper("present"), _paper("missing")]

    scores, failed = frozen_llm_score(papers, {"present": (8, "ok")})

    assert scores == [(8, "ok"), (0, "冻结分数缺失")]
    assert failed is True


def test_snapshot_rejects_tampered_records_hash(tmp_path):
    payload = {
        "schema_version": "paper-tracker.eval-snapshot/1.0.0",
        "dataset_version": "tampered",
        "records_sha256": "0" * 64,
        "records": [
            {
                "id": "p1",
                "title": "changed",
                "summary": "summary",
                "label": "rel",
                "is_anchor": False,
                "score": 8,
                "reason": "reason",
                "split": "test",
            }
        ],
    }
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="records_sha256"):
        load_snapshot_input(path, "test")
