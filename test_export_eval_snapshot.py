import json

import pytest

from scripts.export_eval_snapshot import (
    SnapshotError,
    build_snapshot,
    export_snapshot,
    verify_snapshot,
)


def _sources():
    papers = [
        {"id": paper_id, "title": f"Title {paper_id}", "summary": "Summary"}
        for paper_id in ("r0", "r1", "r2", "r3", "i0", "i1", "i2", "i3", "u0")
    ]
    labels = {
        "r0": "rel",
        "r1": "rel",
        "r2": "rel",
        "r3": "rel",
        "i0": "irrel",
        "i1": "irrel",
        "i2": "irrel",
        "i3": "irrel",
    }
    profile = {"interest": "private", "feedback": [{"id": "r0", "label": "up"}]}
    scores = {
        f"legacy:{paper['id']}": [index % 11, f"reason {paper['id']}"]
        for index, paper in enumerate(papers)
    }
    return papers, labels, profile, scores


def test_snapshot_is_deterministic_stratified_and_deidentified():
    papers, labels, profile, scores = _sources()

    first = build_snapshot(papers, labels, profile, scores, dev_ratio=0.5, seed=7)
    second = build_snapshot(
        list(reversed(papers)),
        dict(reversed(list(labels.items()))),
        profile,
        dict(reversed(list(scores.items()))),
        dev_ratio=0.5,
        seed=7,
    )

    assert first == second
    assert first["counts"]["papers"] == 9
    assert first["counts"]["judged"] == 8
    assert first["counts"]["splits"] == {
        "anchor": 1,
        "dev": 4,
        "test": 3,
        "unjudged": 1,
    }
    assert first["counts"]["split_labels"]["dev"] == {"rel": 2, "irrel": 2}
    assert first["counts"]["split_labels"]["test"] == {"rel": 1, "irrel": 2}
    assert next(row for row in first["records"] if row["id"] == "r0")["split"] == "anchor"
    assert next(row for row in first["records"] if row["id"] == "u0")["split"] == "unjudged"
    serialized = json.dumps(first, ensure_ascii=False)
    assert "private" not in serialized
    assert "legacy:" not in serialized
    verify_snapshot(first)


def test_export_is_atomic_and_check_detects_stale_snapshot(tmp_path):
    papers, labels, profile, scores = _sources()
    source_values = {
        "papers.json": papers,
        "labels.json": labels,
        "profile.json": profile,
        "scores.json": scores,
    }
    for name, value in source_values.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    output = tmp_path / "nested" / "eval_v1.json"
    export_snapshot(
        tmp_path / "papers.json",
        tmp_path / "labels.json",
        tmp_path / "profile.json",
        tmp_path / "scores.json",
        output,
        dev_ratio=0.5,
        seed=7,
    )

    verify_snapshot(output)
    assert not list(output.parent.glob(".eval_v1.json.*.tmp"))
    export_snapshot(
        tmp_path / "papers.json",
        tmp_path / "labels.json",
        tmp_path / "profile.json",
        tmp_path / "scores.json",
        output,
        dev_ratio=0.5,
        seed=7,
        check=True,
    )

    stored = json.loads(output.read_text(encoding="utf-8"))
    stored["records"][0]["title"] = "tampered"
    output.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(SnapshotError, match="stale"):
        export_snapshot(
            tmp_path / "papers.json",
            tmp_path / "labels.json",
            tmp_path / "profile.json",
            tmp_path / "scores.json",
            output,
            dev_ratio=0.5,
            seed=7,
            check=True,
        )


def test_export_rejects_new_cache_schema_mixed_into_legacy_snapshot():
    papers, labels, profile, scores = _sources()
    scores["legacy:r0"] = {
        "score": 8,
        "reason": "new schema",
        "model": "new-model",
    }

    with pytest.raises(SnapshotError, match="do not mix a newer cache schema"):
        build_snapshot(papers, labels, profile, scores)


def test_verifier_detects_record_tampering():
    snapshot = build_snapshot(*_sources(), dev_ratio=0.5, seed=7)
    snapshot["records"][0]["summary"] = "tampered"

    with pytest.raises(SnapshotError, match="records_sha256"):
        verify_snapshot(snapshot)
