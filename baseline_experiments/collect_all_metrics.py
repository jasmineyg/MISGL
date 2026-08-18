"""Validate and combine MISGL plus three baseline result trees."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fair_protocol import aggregate_fold_records, binary_metrics, load_fixed_folds, save_json


BASELINES = ["Attention-based MIL", "RGMIL", "SubGNN"]
METRICS = [
    "acc", "precision", "recall", "specificity", "f1", "f1_macro",
    "balanced_acc", "roc_auc", "pr_auc",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--misgl-fold-csv", required=True)
    parser.add_argument("--attention-root", required=True)
    parser.add_argument("--rgmil-root", required=True)
    parser.add_argument("--subgnn-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        return
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_entries(execution_manifest):
    with Path(execution_manifest).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = {}
    for entry in payload["entries"]:
        entries.setdefault(
            entry["dataset_key"],
            {"data_name": entry["data_name"], "split_manifest": entry["split_manifest"]},
        )
    return payload, entries


def baseline_records(method, root, entries, missing):
    records = []
    for dataset_key, entry in entries.items():
        with Path(entry["split_manifest"]).open(encoding="utf-8") as handle:
            raw_split = json.load(handle)
        sample_count = sum(len(fold["sample_indices"]) for fold in raw_split["folds"])
        splits = load_fixed_folds(entry["split_manifest"], sample_count)
        for fold_id, split in enumerate(splits):
            path = Path(root) / dataset_key / f"fold_{fold_id}" / "test_predictions.npz"
            if not path.is_file():
                missing.append({"method": method, "dataset_key": dataset_key, "fold": fold_id, "path": str(path)})
                continue
            payload = np.load(path)
            orig_indices = payload["orig_indices"].astype(np.int64)
            if not np.array_equal(np.sort(orig_indices), np.sort(split["test_indices"])):
                raise ValueError(f"{path}: predictions do not match the exact test fold")
            labels = payload["labels"].astype(np.int64)
            probabilities = payload["probabilities"].astype(np.float64)
            metrics = binary_metrics(labels, probabilities)
            records.append(
                {
                    "dataset_key": dataset_key,
                    "data_name": entry["data_name"],
                    "method": method,
                    "fold": fold_id,
                    "fold_seed": int(split["fold_seed"]),
                    "test_n": len(labels),
                    **metrics,
                    "source": str(path),
                }
            )
    return records


def normalize_misgl(rows):
    numeric_float = set(METRICS)
    numeric_int = {"fold", "fold_seed", "test_n", "n", "tn", "fp", "fn", "tp"}
    output = []
    for row in rows:
        normalized = dict(row)
        for key in numeric_float:
            normalized[key] = float(normalized[key])
        for key in numeric_int:
            normalized[key] = int(normalized[key])
        normalized["source"] = normalized.pop("prediction_file")
        output.append(normalized)
    return output


def main():
    args = parse_args()
    execution, entries = canonical_entries(args.execution_manifest)
    missing = []
    records = normalize_misgl(read_csv(args.misgl_fold_csv))
    records.extend(baseline_records("Attention-based MIL", args.attention_root, entries, missing))
    records.extend(baseline_records("RGMIL", args.rgmil_root, entries, missing))
    records.extend(baseline_records("SubGNN", args.subgnn_root, entries, missing))
    if missing and not args.allow_incomplete:
        preview = "\n".join(str(item) for item in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} fold results:\n{preview}")
    method_order = {
        name: idx
        for idx, name in enumerate(
            ["GAT+mean pool", "MIL-HEAD", "POS-HEAD", "MISGL", *BASELINES]
        )
    }
    dataset_order = {name: idx for idx, name in enumerate(entries)}
    records.sort(key=lambda row: (dataset_order[row["dataset_key"]], method_order[row["method"]], int(row["fold"])))
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["dataset_key"], record["method"])].append(record)
    summary_rows = []
    summary_json = {}
    for (dataset_key, method), group in grouped.items():
        aggregate = aggregate_fold_records(group)
        row = {
            "dataset_key": dataset_key,
            "data_name": entries[dataset_key]["data_name"],
            "method": method,
            "fold_count": len(group),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = aggregate[metric]["mean"]
            row[f"{metric}_std"] = aggregate[metric]["std"]
        summary_rows.append(row)
        summary_json.setdefault(dataset_key, {})[method] = {
            "fold_count": len(group), "aggregate": aggregate
        }
    summary_rows.sort(key=lambda row: (dataset_order[row["dataset_key"]], method_order[row["method"]]))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fold_fields = [
        "dataset_key", "data_name", "method", "fold", "fold_seed", "test_n", "n",
        "acc", "precision", "recall", "specificity", "f1", "f1_macro", "balanced_acc",
        "tn", "fp", "fn", "tp", "roc_auc", "pr_auc", "source",
    ]
    write_csv(output_root / "all_fold_metrics.csv", records, fold_fields)
    write_csv(output_root / "summary_metrics.csv", summary_rows)
    write_csv(output_root / "missing_results.csv", missing, ["method", "dataset_key", "fold", "path"])
    save_json(
        output_root / "summary.json",
        {
            "protocol": execution["protocol"],
            "seed": execution["seed"],
            "expected_fold_records": len(entries) * 7 * 10,
            "actual_fold_records": len(records),
            "missing_count": len(missing),
            "datasets": summary_json,
        },
    )
    print(
        json.dumps(
            {"records": len(records), "groups": len(summary_rows), "missing": len(missing)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
