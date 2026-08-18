#!/usr/bin/env python3
"""Summarize fixed 10-fold results and OOF attention without model training."""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


MODEL_MAP = {
    ("mean", "stage1"): "GAT+mean pool",
    ("mil", "stage1"): "MIL-HEAD",
    ("mean", "stage2"): "POS-HEAD",
    ("mil", "stage2"): "MISGL",
}
METRICS = ("acc", "F1", "prec", "rec")


def stats(values):
    values = [float(value) for value in values]
    return {
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
        "n": len(values),
    }


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = os.path.dirname(os.path.abspath(args.manifest))
    summary_dir = os.path.join(root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    missing = []
    fold_rows = []
    attention_rows = []
    raw_attention_files = 0
    test_indices = defaultdict(list)
    population = {}

    for entry in manifest["entries"]:
        fold_dir = entry["canonical_dir"]
        result_path = os.path.join(fold_dir, "fold_result.json")
        if not os.path.isfile(result_path):
            missing.append({"entry": entry, "artifact": "fold_result.json"})
            continue
        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        for stage in ("stage1", "stage2"):
            row = {
                "dataset": entry["dataset_key"],
                "data_name": entry["data_name"],
                "model": MODEL_MAP[(entry["branch"], stage)],
                "branch": entry["branch"],
                "stage": stage,
                "fold": int(entry["fold"]),
            }
            test_metrics = result[stage]["metrics"]["test"]
            row.update({metric: test_metrics.get(metric) for metric in METRICS})
            fold_rows.append(row)

        if entry["branch"] != "mil":
            continue
        raw_path = os.path.join(fold_dir, "test_positive_attention.pt")
        metrics_path = os.path.join(fold_dir, "attention_metrics.json")
        predictions_path = os.path.join(fold_dir, "stage1_predictions.pt")
        for path in (raw_path, metrics_path, predictions_path):
            if not os.path.isfile(path):
                missing.append({"entry": entry, "artifact": os.path.basename(path)})
        if not all(os.path.isfile(path) for path in (raw_path, metrics_path, predictions_path)):
            continue
        raw = torch.load(raw_path, map_location="cpu")
        with open(metrics_path, "r", encoding="utf-8") as handle:
            attention = json.load(handle)
        raw_ids = [int(row["orig_graph_idx"]) for row in raw["positive_bags"]]
        metric_ids = [int(row["orig_graph_idx"]) for row in attention["positive_bags"]]
        if raw_ids != metric_ids:
            missing.append({"entry": entry, "artifact": "raw/metric attention id mismatch"})
            continue
        raw_attention_files += 1
        for row in attention["positive_bags"]:
            attention_rows.append({"dataset": entry["dataset_key"], **row})

        predictions = torch.load(predictions_path, map_location="cpu")
        index_to_label = {
            int(idx): int(label)
            for idx, label in zip(predictions["orig_indices"].tolist(), predictions["labels"].tolist())
        }
        population.setdefault(entry["dataset_key"], set(index_to_label))
        fold_test = [int(idx) for idx in result["split"]["test_indices"]]
        test_indices[entry["dataset_key"]].append(set(fold_test))
        expected_positive_ids = {idx for idx in fold_test if index_to_label[idx] == 1}
        if expected_positive_ids != set(metric_ids):
            missing.append({"entry": entry, "artifact": "positive test attention coverage mismatch"})

    for dataset, fold_sets in test_indices.items():
        seen = set()
        for fold_set in fold_sets:
            if seen.intersection(fold_set):
                missing.append({"dataset": dataset, "artifact": "duplicate OOF test bag"})
            seen.update(fold_set)
        if len(fold_sets) != 10 or seen != population.get(dataset, set()):
            missing.append({"dataset": dataset, "artifact": "incomplete OOF test coverage"})

    aggregate = []
    grouped = defaultdict(lambda: defaultdict(list))
    for row in fold_rows:
        for metric in METRICS:
            if row[metric] is not None:
                grouped[(row["dataset"], row["model"])][metric].append(float(row[metric]))
    for (dataset, model), values in sorted(grouped.items()):
        row = {"dataset": dataset, "model": model}
        for metric in METRICS:
            summary = stats(values[metric])
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
            row[f"{metric}_n"] = summary["n"]
        aggregate.append(row)

    attention_summary = []
    by_dataset = defaultdict(list)
    for row in attention_rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, rows in sorted(by_dataset.items()):
        enrichments = [row["positive_attention_enrichment"] for row in rows if row["positive_attention_enrichment"] is not None]
        aucs = [row["attention_ranking_auc"] for row in rows if row["attention_ranking_auc"] is not None]
        correct = [row["positive_attention_enrichment"] for row in rows if row["correct"] and row["positive_attention_enrichment"] is not None]
        wrong = [row["positive_attention_enrichment"] for row in rows if not row["correct"] and row["positive_attention_enrichment"] is not None]
        fold_enrichment = defaultdict(list)
        fold_auc = defaultdict(list)
        for row in rows:
            if row["positive_attention_enrichment"] is not None:
                fold_enrichment[int(row["fold_idx"])].append(row["positive_attention_enrichment"])
            if row["attention_ranking_auc"] is not None:
                fold_auc[int(row["fold_idx"])].append(row["attention_ranking_auc"])
        fold_e_means = [np.mean(values) for values in fold_enrichment.values() if values]
        fold_auc_means = [np.mean(values) for values in fold_auc.values() if values]
        attention_summary.append(
            {
                "dataset": dataset,
                "oof_positive_bags": len(rows),
                "enrichment_mean": stats(enrichments)["mean"],
                "ranking_auc_mean": stats(aucs)["mean"],
                "correct_enrichment_mean": stats(correct)["mean"],
                "correct_n": len(correct),
                "wrong_enrichment_mean": stats(wrong)["mean"],
                "wrong_n": len(wrong),
                "fold_enrichment_mean": stats(fold_e_means)["mean"],
                "fold_enrichment_std": stats(fold_e_means)["std"],
                "fold_ranking_auc_mean": stats(fold_auc_means)["mean"],
                "fold_ranking_auc_std": stats(fold_auc_means)["std"],
            }
        )

    write_csv(
        os.path.join(summary_dir, "all_fold_metrics.csv"),
        ["dataset", "data_name", "model", "branch", "stage", "fold", *METRICS],
        fold_rows,
    )
    aggregate_fields = ["dataset", "model"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std", "n")]
    write_csv(os.path.join(summary_dir, "main_results_summary.csv"), aggregate_fields, aggregate)
    attention_fields = [
        "dataset", "orig_graph_idx", "subgraph_id", "fold_idx", "bag_label", "logit",
        "probability", "prediction", "correct", "num_nodes", "positive_instance_count",
        "positive_instance_prevalence", "positive_attention_mass",
        "positive_attention_enrichment", "attention_ranking_auc",
    ]
    write_csv(os.path.join(summary_dir, "oof_positive_bag_attention.csv"), attention_fields, attention_rows)
    attention_summary_fields = list(attention_summary[0]) if attention_summary else ["dataset"]
    write_csv(os.path.join(summary_dir, "attention_summary.csv"), attention_summary_fields, attention_summary)

    models = ["GAT+mean pool", "MIL-HEAD", "POS-HEAD", "MISGL"]
    datasets = [item["dataset_key"] for item in manifest["entries"][::20]]
    lookup = {(row["dataset"], row["model"]): row for row in aggregate}
    with open(os.path.join(summary_dir, "paper_accuracy_table.md"), "w", encoding="utf-8") as handle:
        handle.write("| Dataset | " + " | ".join(models) + " |\n")
        handle.write("|---|" + "---|" * len(models) + "\n")
        for dataset in datasets:
            cells = []
            for model in models:
                row = lookup.get((dataset, model), {})
                mean, std = row.get("acc_mean"), row.get("acc_std")
                cells.append("" if mean is None else f"{mean:.4f} ± {std:.4f}")
            handle.write(f"| {dataset} | " + " | ".join(cells) + " |\n")

    if by_dataset:
        fig, axes = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True)
        for axis, dataset in zip(axes.flat, datasets):
            rows = by_dataset.get(dataset, [])
            correct = [row["positive_attention_enrichment"] for row in rows if row["correct"] and row["positive_attention_enrichment"] is not None]
            wrong = [row["positive_attention_enrichment"] for row in rows if not row["correct"] and row["positive_attention_enrichment"] is not None]
            values, labels = [], []
            if correct:
                values.append(correct)
                labels.append("Correct")
            if wrong:
                values.append(wrong)
                labels.append("Wrong")
            if values:
                axis.boxplot(values, labels=labels, showfliers=False)
            axis.set_title(dataset)
            axis.set_ylabel("Positive attention enrichment")
            axis.grid(axis="y", alpha=0.25)
        fig.savefig(os.path.join(summary_dir, "attention_enrichment_correct_vs_wrong.png"), dpi=220)
        plt.close(fig)

    completeness = {
        "expected_branch_folds": 300,
        "actual_branch_folds": len(fold_rows) // 2,
        "expected_model_test_observations": 600,
        "actual_model_test_observations": len(fold_rows),
        "expected_attention_fold_files": 150,
        "actual_attention_fold_files": raw_attention_files,
        "datasets_with_complete_oof_attention": sum(len(sets) == 10 for sets in test_indices.values()),
        "expected_datasets_with_complete_oof_attention": 15,
        "missing_count": len(missing),
        "missing": missing,
    }
    with open(os.path.join(summary_dir, "completeness.json"), "w", encoding="utf-8") as handle:
        json.dump(completeness, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(summary_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"main_results": aggregate, "attention": attention_summary, "completeness": completeness},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(completeness, ensure_ascii=False))
    raise SystemExit(1 if missing or len(fold_rows) != 600 or raw_attention_files != 150 else 0)


if __name__ == "__main__":
    main()
