#!/usr/bin/env python3
"""Reconstruct the Syn2 CV manifest from the splits stored with legacy results.

This preserves the already-trained Stage-1 checkpoints: only the manifest pointer is
repaired, and the runner will then reuse the frozen embeddings/checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/paper_10fold_20260814/execution_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = (root / args.manifest).resolve()
    manifest = read_json(manifest_path)

    entries = [e for e in manifest["entries"] if e["dataset_key"] == "syn2"]
    if len(entries) != 20:
        raise RuntimeError(f"Expected 20 Syn2 entries, found {len(entries)}")

    fold_tests: dict[int, list[int]] = {}
    for fold in range(10):
        fold_entries = [e for e in entries if int(e["fold"]) == fold]
        recorded_splits = []
        for entry in fold_entries:
            result = read_json(Path(entry["stage1"]["legacy_result"]))
            recorded_splits.append(result["split"])
        if recorded_splits[0] != recorded_splits[1]:
            raise RuntimeError(f"Mean/MIL legacy splits disagree for Syn2 fold {fold}")
        fold_tests[fold] = list(map(int, recorded_splits[0]["test_indices"]))

    flattened = [idx for fold in range(10) for idx in fold_tests[fold]]
    if len(flattened) != 1000 or len(set(flattened)) != 1000:
        raise RuntimeError("Syn2 legacy test folds do not form a 1000-sample partition")

    old_split_path = Path(entries[0]["split_manifest"])
    split_manifest = read_json(old_split_path)
    for fold_record in split_manifest["folds"]:
        fold = int(fold_record["fold_id"])
        fold_record["sample_indices"] = fold_tests[fold]

    repaired_path = root / "splits" / "synthetic_milinst_mil_weak_pos_strong_v2_cv10_seed1024_adjacent_legacy_20260810.json"
    write_json(repaired_path, split_manifest)
    for entry in entries:
        entry["split_manifest"] = str(repaired_path)
    write_json(manifest_path, manifest)
    print(repaired_path)
    print("validated 10 disjoint folds and matching Mean/MIL legacy splits")


if __name__ == "__main__":
    main()
