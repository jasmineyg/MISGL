"""Build a local, auditable interim results report from available MISGL folds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = [
    "acc",
    "precision",
    "recall",
    "specificity",
    "f1",
    "f1_macro",
    "balanced_acc",
    "roc_auc",
    "pr_auc",
]
METHOD_ORDER = ["GAT+mean pool", "MIL-HEAD", "POS-HEAD", "MISGL"]
DATASET_ORDER = [
    "products",
    "products_oracle",
    "products_perturb50",
    "products_random",
    "reddit",
    "reddit_oracle",
    "reddit_perturb50",
    "reddit_random",
    "arxiv",
    "arxiv_oracle",
    "arxiv_perturb50",
    "arxiv_random",
    "syn1",
    "syn2",
    "syn3",
]


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    with Path(args.fold_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 600:
        raise ValueError(f"expected 600 local MISGL fold records, found {len(rows)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_key"], row["method"])].append(row)

    if len(grouped) != 60:
        raise ValueError(f"expected 60 MISGL dataset-method groups, found {len(grouped)}")

    summaries = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            folds = sorted(grouped[(dataset, method)], key=lambda item: int(item["fold"]))
            if len(folds) != 10:
                raise ValueError(f"{dataset}/{method}: expected 10 folds, found {len(folds)}")
            record = {
                "dataset_key": dataset,
                "data_name": folds[0]["data_name"],
                "method": method,
                "fold_count": len(folds),
            }
            for metric in METRICS:
                values = [float(item[metric]) for item in folds]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_std"] = statistics.stdev(values)
            summaries.append(record)

    progress = [
        {
            "method": "Attention-based MIL",
            "completed_datasets": 15,
            "expected_datasets": 15,
            "observed_metric_files": 151,
            "estimated_formal_folds": 150,
            "numeric_results_local": False,
            "completed_dataset_keys": "all 15 datasets",
            "status": "training complete; result files stranded on unavailable server",
        },
        {
            "method": "RGMIL",
            "completed_datasets": 7,
            "expected_datasets": 15,
            "observed_metric_files": 82,
            "estimated_formal_folds": 81,
            "numeric_results_local": False,
            "completed_dataset_keys": "arxiv, arxiv_oracle, arxiv_perturb50, arxiv_random, syn1, syn2, syn3",
            "status": "partial; Products and Reddit queues were active at last successful check",
        },
        {
            "method": "SubGNN",
            "completed_datasets": 4,
            "expected_datasets": 15,
            "observed_metric_files": 41,
            "estimated_formal_folds": 40,
            "numeric_results_local": False,
            "completed_dataset_keys": "arxiv, syn1, syn2, syn3",
            "status": "partial; Arxiv-Oracle, Products, and Reddit were active at last successful check",
        },
    ]

    output_summary = Path(args.output_summary_csv)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = ["dataset_key", "data_name", "method", "fold_count"] + [
        field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std")
    ]
    with output_summary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "snapshot_time": "2026-08-16 04:23 CST",
        "protocol": "fixed grouped-stratified 10-fold; test=f, val=(f+1) mod 10; seed=1024",
        "misgl_fold_records": len(rows),
        "misgl_groups": len(grouped),
        "progress": progress,
        "summaries": summaries,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    best_rows = []
    for dataset in DATASET_ORDER:
        candidates = [record for record in summaries if record["dataset_key"] == dataset]
        best = max(candidates, key=lambda record: record["acc_mean"])
        best_rows.append((dataset, best["method"], best["acc_mean"], best["acc_std"]))

    lines = [
        "# 当前实验结果（阶段性整理）",
        "",
        "> 截止时间：2026-08-16 04:23 CST（school 服务器最后一次成功监控）。",
        "> 本报告只写入本地可核验的数值；服务器未同步的 baseline 结果仅记录完成进度，不推测指标。",
        "",
        "## 1. 数据与评价协议",
        "",
        "- 所有方法共用 MISGL 的固定 grouped-stratified 10-fold 划分，seed=1024。",
        "- fold `f` 为 test，fold `(f+1) mod 10` 为 validation，其余 8 folds 为 train。",
        "- 二分类预测规则为 `probability > 0.5`，恰好等于 0.5 判为负类。",
        "- 汇总值为 10 个固定 test folds 的 mean ± sample std。",
        "",
        "## 2. Baseline 当前进度",
        "",
        "| 方法 | 完成数据集 | 监控到的 metrics 文件 | 估计正式 folds | 数值已在本地 | 状态 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in progress:
        local = "是" if item["numeric_results_local"] else "否"
        lines.append(
            f"| {item['method']} | {item['completed_datasets']}/{item['expected_datasets']} "
            f"| {item['observed_metric_files']} | {item['estimated_formal_folds']} | {local} | {item['status']} |"
        )
    lines.extend([
        "",
        "说明：每种 baseline 均执行过一次 smoke test，因此监控到的 `metrics.json` 文件数包含 1 个 smoke 结果；“估计正式 folds”已扣除该记录。只有存在完整 10-fold `summary.json` 的数据集计入“完成数据集”。",
        "",
        "- Attention-based MIL：15 个数据集均已完成训练，但数值文件尚未同步到本机。",
        "- RGMIL 已完成：arxiv、arxiv_oracle、arxiv_perturb50、arxiv_random、syn1、syn2、syn3。",
        "- SubGNN 已完成：arxiv、syn1、syn2、syn3。",
        "",
        "## 3. MISGL 四种设置的完整 10-fold 结果",
        "",
        "本地 `fold_metrics.csv` 共 600 条记录：15 个数据集 × 4 个方法 × 10 folds，完整性检查通过。",
        "",
        "| 数据集 | 方法 | ACC | Precision | Recall | Specificity | F1 | Macro-F1 | Balanced ACC | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in summaries:
        values = [
            record["dataset_key"],
            record["method"],
            *[
                fmt(record[f"{metric}_mean"], record[f"{metric}_std"])
                for metric in METRICS
            ],
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## 4. 各数据集按 ACC 选择的当前最佳 MISGL 设置",
        "",
        "| 数据集 | 最佳设置 | ACC |",
        "|---|---|---:|",
    ])
    for dataset, method, mean, std in best_rows:
        lines.append(f"| {dataset} | {method} | {fmt(mean, std)} |")

    lines.extend([
        "",
        "## 5. 结果解释边界与后续补全",
        "",
        "- 当前只能进行 MISGL 四种设置之间的完整比较，不能据此比较三种 baseline 的性能。",
        "- Attention 虽已完成 15/15，但在其 150 个正式 fold 数值同步前，不应把它写入论文结果表。",
        "- RGMIL 和 SubGNN 仍为不完整实验；部分 folds 不用于最终 mean ± std，也不与完整 10-fold 方法直接比较。",
        "- 服务器恢复后应先同步各 baseline 的 `metrics.json`、`fold_metrics.csv`、`summary.json` 和 config，再执行 1050 条 fold / 105 组的严格聚合并更新本报告。",
        "",
        "## 6. 本地可核验来源",
        "",
        "- `paper_10fold/results/unified_metrics_20260815/fold_metrics.csv`：600 条 MISGL fold 指标。",
        "- `paper_10fold/results/unified_metrics_20260815/summary.json`：MISGL 10-fold 汇总。",
        "- `paper_10fold/results/completeness.json`：MISGL 完整性检查，missing_count=0。",
        "- baseline 进度来自 2026-08-16 04:23 CST 的最后一次成功服务器监控；不包含未同步数值。",
        "",
    ])

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
