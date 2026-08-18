"""Recompute unified binary metrics for the four saved MISGL variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from fair_protocol import aggregate_fold_records, binary_metrics, load_fixed_folds, save_json


MODELS = {
    "GAT+mean pool": ("mean", "stage1_predictions.pt"),
    "MIL-HEAD": ("mil", "stage1_predictions.pt"),
    "POS-HEAD": ("mean", "stage2_predictions.pt"),
    "MISGL": ("mil", "stage2_predictions.pt"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def write_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    with Path(args.execution_manifest).open(encoding="utf-8") as handle:
        execution = json.load(handle)
    dataset_entries = {}
    for entry in execution["entries"]:
        dataset_entries.setdefault(
            entry["dataset_key"],
            {
                "data_name": entry["data_name"],
                "split_manifest": entry["split_manifest"],
            },
        )
    all_records = []
    summary = {}
    for dataset_key, entry in dataset_entries.items():
        with Path(entry["split_manifest"]).open(encoding="utf-8") as handle:
            raw_split = json.load(handle)
        sample_count = sum(len(fold["sample_indices"]) for fold in raw_split["folds"])
        splits = load_fixed_folds(entry["split_manifest"], sample_count)
        summary[dataset_key] = {}
        for model_name, (branch, filename) in MODELS.items():
            model_records = []
            for fold_id, split in enumerate(splits):
                path = Path(args.result_root) / branch / dataset_key / f"fold_{fold_id}" / filename
                payload = torch.load(path, map_location="cpu", weights_only=False)
                orig_indices = payload["orig_indices"].detach().cpu().numpy().astype(np.int64)
                labels = payload["labels"].detach().cpu().numpy().astype(np.int64)
                probabilities = payload["probabilities"].detach().cpu().numpy().astype(np.float64)
                if len(np.unique(orig_indices)) != sample_count:
                    raise ValueError(f"{path}: orig_indices are not a full unique dataset index")
                position = np.empty(sample_count, dtype=np.int64)
                position[orig_indices] = np.arange(sample_count, dtype=np.int64)
                selected = position[split["test_indices"]]
                selected_indices = orig_indices[selected]
                if not np.array_equal(selected_indices, split["test_indices"]):
                    raise ValueError(f"{path}: test index mapping mismatch")
                metrics = binary_metrics(labels[selected], probabilities[selected])
                record = {
                    "dataset_key": dataset_key,
                    "data_name": entry["data_name"],
                    "method": model_name,
                    "fold": fold_id,
                    "fold_seed": int(split["fold_seed"]),
                    "test_n": len(selected),
                    **metrics,
                    "prediction_file": str(path),
                }
                all_records.append(record)
                model_records.append(record)
            summary[dataset_key][model_name] = {
                "folds": model_records,
                "aggregate": aggregate_fold_records(model_records),
            }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "fold_metrics.csv", all_records)
    save_json(
        output_root / "summary.json",
        {
            "protocol": execution["protocol"],
            "seed": execution["seed"],
            "source_result_root": args.result_root,
            "models": list(MODELS),
            "datasets": summary,
        },
    )
    print(f"saved {len(all_records)} fold records to {output_root}")


if __name__ == "__main__":
    main()
