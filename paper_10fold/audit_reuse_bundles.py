#!/usr/bin/env python3
"""Audit that reused checkpoints, embeddings, results, and coarse graphs stay bundled."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: str | Path, cache: dict[str, str]) -> str:
    path = os.path.realpath(path)
    if path not in cache:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        cache[path] = digest.hexdigest()
    return cache[path]


def locate_dataset_file(entry: dict) -> str | None:
    data_dir = Path(entry["data_dir"])
    data_name = entry["data_name"]
    basename = Path(data_name).name
    candidates = (
        data_dir / f"{basename}_processed.pkl",
        data_dir / f"{data_name}_processed.pkl",
        data_dir / data_name / f"{basename}_processed.pkl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def issue(issues: list[dict], entry: dict, code: str, **details) -> None:
    issues.append(
        {
            "dataset": entry["dataset_key"],
            "branch": entry["branch"],
            "fold": int(entry["fold"]),
            "code": code,
            **details,
        }
    )


def audit(manifest_path: str | Path) -> dict:
    manifest = read_json(manifest_path)
    issues: list[dict] = []
    hashes: dict[str, str] = {}
    audited = 0
    for entry in manifest["entries"]:
        if entry["stage1"].get("status") != "reuse":
            continue
        audited += 1
        checkpoint = entry["stage1"].get("checkpoint")
        embeddings = entry["stage1"].get("embeddings")
        legacy_result_path = entry["stage1"].get("legacy_result")
        expected = (checkpoint, embeddings, legacy_result_path)
        if not all(path and os.path.isfile(path) for path in expected):
            issue(issues, entry, "missing_stage1_bundle_file")
            continue
        bundle_dirs = {os.path.realpath(os.path.dirname(path)) for path in expected}
        if len(bundle_dirs) != 1:
            issue(issues, entry, "stage1_bundle_directory_mismatch", paths=list(expected))

        legacy_result = read_json(legacy_result_path)
        legacy_coarse = legacy_result.get("paths", {}).get("coarse_adj")
        current_result_path = os.path.join(entry["canonical_dir"], "fold_result.json")
        if not os.path.isfile(current_result_path):
            issue(issues, entry, "missing_current_fold_result")
            continue
        current_result = read_json(current_result_path)
        current_coarse = current_result.get("coarse_adj")
        if legacy_coarse and current_coarse and os.path.isfile(legacy_coarse) and os.path.isfile(current_coarse):
            if sha256(legacy_coarse, hashes) != sha256(current_coarse, hashes):
                issue(
                    issues,
                    entry,
                    "coarse_graph_cross_version",
                    legacy_coarse=legacy_coarse,
                    current_coarse=current_coarse,
                )
        else:
            issue(issues, entry, "missing_coarse_bundle_file", legacy=legacy_coarse, current=current_coarse)

        dataset_file = locate_dataset_file(entry)
        created_at = legacy_result.get("created_at")
        if dataset_file and created_at:
            result_time = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").timestamp()
            if os.path.getmtime(dataset_file) > result_time + 1:
                issue(
                    issues,
                    entry,
                    "dataset_newer_than_reused_result",
                    dataset_file=dataset_file,
                    dataset_mtime=datetime.fromtimestamp(os.path.getmtime(dataset_file)).isoformat(timespec="seconds"),
                    result_created_at=created_at,
                )

    by_code = Counter(item["code"] for item in issues)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for item in issues:
        by_dataset[item["dataset"]].append(item["code"])
    return {
        "manifest": str(manifest_path),
        "reused_entries_audited": audited,
        "issue_count": len(issues),
        "issue_counts_by_code": dict(sorted(by_code.items())),
        "affected_datasets": {
            dataset: dict(sorted(Counter(codes).items()))
            for dataset, codes in sorted(by_dataset.items())
        },
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--allow-issues", action="store_true")
    args = parser.parse_args()
    report = audit(args.manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["issue_count"] and not args.allow_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
