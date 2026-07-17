#!/usr/bin/env python3
"""Export and verify the public ``eval_v1`` evidence snapshot.

The runtime JSON files remain local and untouched.  This script reads them,
validates their legacy schemas, removes profile text and cache namespaces, and
atomically writes a deterministic public snapshot.

Examples:
    python scripts/export_eval_snapshot.py
    python scripts/export_eval_snapshot.py --check
    python scripts/export_eval_snapshot.py --verify-only
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS = ROOT / "papers_store.json"
DEFAULT_LABELS = ROOT / "labels.json"
DEFAULT_PROFILE = ROOT / "profile.json"
DEFAULT_SCORES = ROOT / "scores_cache.json"
DEFAULT_OUTPUT = ROOT / "eval_data" / "eval_v1.json"

SCHEMA_VERSION = "paper-tracker.eval-snapshot/1.0.0"
DATASET_VERSION = "eval_v1"
VALID_LABELS = frozenset({"rel", "irrel"})
VALID_SPLITS = frozenset({"anchor", "dev", "test", "unjudged"})
DEFAULT_DEV_RATIO = 0.7
DEFAULT_SEED = 20260717


class SnapshotError(ValueError):
    """Raised when source data or a committed snapshot fails validation."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SnapshotError(f"missing input: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read JSON {path}: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write a complete JSON file before atomically replacing ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_papers(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise SnapshotError("papers_store.json must contain a JSON array")

    papers: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise SnapshotError(f"paper #{index} must be an object")
        values: dict[str, str] = {}
        for field in ("id", "title", "summary"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise SnapshotError(f"paper #{index} has invalid {field!r}")
            values[field] = value.strip()
        if values["id"] in seen_ids:
            raise SnapshotError(f"duplicate paper id: {values['id']}")
        seen_ids.add(values["id"])
        papers.append(values)
    return papers


def _validate_labels(raw: Any, paper_ids: set[str]) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise SnapshotError("labels.json must contain a JSON object")
    labels: dict[str, str] = {}
    for paper_id, label in raw.items():
        if not isinstance(paper_id, str) or label not in VALID_LABELS:
            raise SnapshotError(f"invalid label entry: {paper_id!r} -> {label!r}")
        if paper_id not in paper_ids:
            raise SnapshotError(f"label references unknown paper: {paper_id}")
        labels[paper_id] = label
    return labels


def _anchor_ids(raw_profile: Any, paper_ids: set[str]) -> set[str]:
    if not isinstance(raw_profile, Mapping):
        raise SnapshotError("profile.json must contain a JSON object")
    feedback = raw_profile.get("feedback", [])
    if not isinstance(feedback, list):
        raise SnapshotError("profile.feedback must be a JSON array")

    anchors: set[str] = set()
    for index, item in enumerate(feedback):
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise SnapshotError(f"profile feedback #{index} has no valid id")
        paper_id = item["id"]
        if paper_id not in paper_ids:
            raise SnapshotError(f"feedback anchor references unknown paper: {paper_id}")
        anchors.add(paper_id)
    return anchors


def _parse_legacy_score(value: Any, cache_key: str) -> tuple[int | float, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise SnapshotError(
            f"legacy score {cache_key!r} must be [score, reason]; "
            "do not mix a newer cache schema into eval_v1"
        )
    score, reason = value
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SnapshotError(f"legacy score {cache_key!r} is not numeric")
    if not math.isfinite(score) or not 0 <= score <= 10:
        raise SnapshotError(f"legacy score {cache_key!r} is outside [0, 10]")
    if not isinstance(reason, str) or not reason.strip():
        raise SnapshotError(f"legacy score {cache_key!r} has no reason")
    return score, reason.strip()


def _legacy_scores(raw: Any, paper_ids: set[str]) -> dict[str, tuple[int | float, str]]:
    if not isinstance(raw, Mapping):
        raise SnapshotError("scores_cache.json must contain a JSON object")

    scores: dict[str, tuple[int | float, str]] = {}
    for cache_key, value in raw.items():
        if not isinstance(cache_key, str) or ":" not in cache_key:
            raise SnapshotError(f"invalid legacy score cache key: {cache_key!r}")
        _, paper_id = cache_key.split(":", 1)
        if paper_id not in paper_ids:
            # A cache may contain stale papers.  They are not part of the snapshot.
            continue
        if paper_id in scores:
            raise SnapshotError(
                f"multiple legacy score versions found for {paper_id}; "
                "select one cache version before exporting"
            )
        scores[paper_id] = _parse_legacy_score(value, cache_key)
    return scores


def _stable_order(ids: Sequence[str], *, seed: int, label: str) -> list[str]:
    return sorted(
        ids,
        key=lambda paper_id: hashlib.sha256(
            f"{seed}:{label}:{paper_id}".encode("utf-8")
        ).digest(),
    )


def stratified_splits(
    labels: Mapping[str, str],
    anchor_ids: set[str],
    *,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Assign anchors separately, then split judged non-anchors by label."""

    if not 0 < dev_ratio < 1:
        raise SnapshotError("dev_ratio must be between 0 and 1")

    assignments = {paper_id: "anchor" for paper_id in anchor_ids}
    for label in ("rel", "irrel"):
        ids = [
            paper_id
            for paper_id, value in labels.items()
            if value == label and paper_id not in anchor_ids
        ]
        ordered = _stable_order(ids, seed=seed, label=label)
        if len(ordered) >= 2:
            dev_count = round(len(ordered) * dev_ratio)
            dev_count = min(max(dev_count, 1), len(ordered) - 1)
        elif ordered:
            dev_count = 1 if dev_ratio >= 0.5 else 0
        else:
            dev_count = 0
        assignments.update({paper_id: "dev" for paper_id in ordered[:dev_count]})
        assignments.update({paper_id: "test" for paper_id in ordered[dev_count:]})
    return assignments


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = Counter(record["label"] or "unjudged" for record in records)
    splits = Counter(record["split"] for record in records)
    split_labels: dict[str, dict[str, int]] = {}
    for split in ("anchor", "dev", "test", "unjudged"):
        values = Counter(
            record["label"] or "unjudged"
            for record in records
            if record["split"] == split
        )
        split_labels[split] = {
            label: values[label]
            for label in ("rel", "irrel", "unjudged")
            if values[label]
        }
    return {
        "papers": len(records),
        "judged": labels["rel"] + labels["irrel"],
        "unjudged": labels["unjudged"],
        "labels": {"rel": labels["rel"], "irrel": labels["irrel"]},
        "anchors": sum(bool(record["is_anchor"]) for record in records),
        "scores_present": sum(record["score"] is not None for record in records),
        "splits": {
            split: splits[split]
            for split in ("anchor", "dev", "test", "unjudged")
        },
        "split_labels": split_labels,
    }


def build_snapshot(
    raw_papers: Any,
    raw_labels: Any,
    raw_profile: Any,
    raw_scores: Any,
    *,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    papers = _validate_papers(raw_papers)
    paper_ids = {paper["id"] for paper in papers}
    labels = _validate_labels(raw_labels, paper_ids)
    anchors = _anchor_ids(raw_profile, paper_ids)
    scores = _legacy_scores(raw_scores, paper_ids)
    assignments = stratified_splits(
        labels, anchors, dev_ratio=dev_ratio, seed=seed
    )

    records: list[dict[str, Any]] = []
    for paper in sorted(papers, key=lambda item: item["id"]):
        paper_id = paper["id"]
        label = labels.get(paper_id)
        is_anchor = paper_id in anchors
        score_item = scores.get(paper_id)
        records.append(
            {
                "id": paper_id,
                "title": paper["title"],
                "summary": paper["summary"],
                "label": label,
                "is_anchor": is_anchor,
                "score": score_item[0] if score_item else None,
                "reason": score_item[1] if score_item else None,
                "split": assignments.get(paper_id, "unjudged"),
            }
        )

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "provenance": {
            "judgments": {
                "type": "silver",
                "method": "legacy LLM-as-a-judge",
                "per_record_human_review_provenance_available": False,
            },
            "scores": {
                "type": "legacy cached LLM relevance score",
                "range": [0, 10],
            },
            "privacy": {
                "excluded": [
                    "profile interest text",
                    "liked/disliked profile text",
                    "feedback dates",
                    "cache namespace",
                    "local paths and credentials",
                ]
            },
        },
        "split_policy": {
            "method": "label-stratified SHA-256 ordering after reserving anchors",
            "dev_ratio": dev_ratio,
            "seed": seed,
            "hash_input": "{seed}:{label}:{paper_id}",
            "reserved_splits": {
                "anchor": "papers referenced by profile.feedback",
                "unjudged": "papers without a rel/irrel judgment",
            },
        },
        "counts": _summarize(records),
        "records_sha256": _canonical_sha256(records),
        "records": records,
    }
    verify_snapshot(snapshot)
    return snapshot


def verify_snapshot(snapshot_or_path: Mapping[str, Any] | Path) -> dict[str, Any]:
    """Validate schema, derived counts, split invariants, and record hash."""

    snapshot = (
        _read_json(snapshot_or_path)
        if isinstance(snapshot_or_path, Path)
        else snapshot_or_path
    )
    if not isinstance(snapshot, Mapping):
        raise SnapshotError("snapshot must contain a JSON object")
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotError(f"unexpected schema_version: {snapshot.get('schema_version')!r}")
    if snapshot.get("dataset_version") != DATASET_VERSION:
        raise SnapshotError(
            f"unexpected dataset_version: {snapshot.get('dataset_version')!r}"
        )
    records = snapshot.get("records")
    if not isinstance(records, list):
        raise SnapshotError("snapshot.records must be an array")

    ids: set[str] = set()
    record_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SnapshotError(f"snapshot record #{index} must be an object")
        required = {
            "id",
            "title",
            "summary",
            "label",
            "is_anchor",
            "score",
            "reason",
            "split",
        }
        if set(record) != required:
            raise SnapshotError(f"snapshot record #{index} has unexpected fields")
        paper_id = record["id"]
        if not isinstance(paper_id, str) or not paper_id or paper_id in ids:
            raise SnapshotError(f"invalid or duplicate snapshot id: {paper_id!r}")
        ids.add(paper_id)
        record_ids.append(paper_id)
        for text_field in ("title", "summary"):
            if not isinstance(record[text_field], str) or not record[text_field]:
                raise SnapshotError(f"invalid {text_field} for {paper_id}")
        if record["label"] not in VALID_LABELS | {None}:
            raise SnapshotError(f"invalid snapshot label for {paper_id}")
        if not isinstance(record["is_anchor"], bool):
            raise SnapshotError(f"invalid anchor flag for {paper_id}")
        if record["split"] not in VALID_SPLITS:
            raise SnapshotError(f"invalid split for {paper_id}")
        if record["is_anchor"] != (record["split"] == "anchor"):
            raise SnapshotError(f"anchor flag/split mismatch for {paper_id}")
        # Anchor reservation takes precedence over judgment status.  A feedback
        # anchor may legitimately have no legacy silver label (one does in v1).
        if record["label"] is None and record["split"] not in {
            "anchor",
            "unjudged",
        }:
            raise SnapshotError(f"unjudged record has dev/test split: {paper_id}")
        if record["label"] is not None and record["split"] == "unjudged":
            raise SnapshotError(f"judged record has unjudged split: {paper_id}")
        score, reason = record["score"], record["reason"]
        if (score is None) != (reason is None):
            raise SnapshotError(f"score/reason presence mismatch for {paper_id}")
        if score is not None:
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 10
                or not isinstance(reason, str)
                or not reason
            ):
                raise SnapshotError(f"invalid score/reason for {paper_id}")

    if record_ids != sorted(record_ids):
        raise SnapshotError("snapshot records must be sorted by id")

    policy = snapshot.get("split_policy")
    if not isinstance(policy, Mapping):
        raise SnapshotError("snapshot.split_policy must be an object")
    dev_ratio = policy.get("dev_ratio")
    seed = policy.get("seed")
    if isinstance(dev_ratio, bool) or not isinstance(dev_ratio, (int, float)):
        raise SnapshotError("split_policy.dev_ratio must be numeric")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SnapshotError("split_policy.seed must be an integer")
    labels = {
        record["id"]: record["label"]
        for record in records
        if record["label"] in VALID_LABELS
    }
    anchors = {record["id"] for record in records if record["is_anchor"]}
    expected_assignments = stratified_splits(
        labels, anchors, dev_ratio=dev_ratio, seed=seed
    )
    for record in records:
        expected_split = expected_assignments.get(record["id"], "unjudged")
        if record["split"] != expected_split:
            raise SnapshotError(
                f"record split does not match deterministic policy: {record['id']}"
            )

    expected_counts = _summarize(records)
    if snapshot.get("counts") != expected_counts:
        raise SnapshotError("snapshot counts do not match records")
    expected_hash = _canonical_sha256(records)
    if snapshot.get("records_sha256") != expected_hash:
        raise SnapshotError("snapshot records_sha256 does not match records")
    return {
        "records": len(records),
        "records_sha256": expected_hash,
        "counts": expected_counts,
    }


def export_snapshot(
    papers_path: Path,
    labels_path: Path,
    profile_path: Path,
    scores_path: Path,
    output_path: Path,
    *,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    seed: int = DEFAULT_SEED,
    check: bool = False,
) -> dict[str, Any]:
    snapshot = build_snapshot(
        _read_json(papers_path),
        _read_json(labels_path),
        _read_json(profile_path),
        _read_json(scores_path),
        dev_ratio=dev_ratio,
        seed=seed,
    )
    if check:
        existing = _read_json(output_path)
        if existing != snapshot:
            raise SnapshotError(
                f"{output_path} is stale; regenerate it with this script"
            )
    else:
        _atomic_write_json(output_path, snapshot)
    return snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--scores-cache", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-ratio", type=float, default=DEFAULT_DEV_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and fail if the committed snapshot differs",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the public snapshot without local runtime source files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only:
        result = verify_snapshot(args.output)
    else:
        snapshot = export_snapshot(
            args.papers,
            args.labels,
            args.profile,
            args.scores_cache,
            args.output,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
            check=args.check,
        )
        result = verify_snapshot(snapshot)
    if args.output.exists():
        result["file_sha256"] = _file_sha256(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SnapshotError as exc:
        raise SystemExit(f"snapshot error: {exc}") from exc
