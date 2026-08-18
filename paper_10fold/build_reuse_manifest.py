#!/usr/bin/env python3
"""Create the immutable execution/reuse plan for the paper's single 10-fold CV."""

import argparse
import json
import os
from collections import Counter


SEED = 1024
FOLDS = 10
REMOTE_WORK = "/data/yg/Subgraph-MIL/diffpool2"
DATA_ROOT = "/data/yg/Subgraph-MIL/Data/processed_data"

DATASETS = [
    dict(key="products", name="ogbn_products", data_dir=DATA_ROOT, kind="real"),
    dict(key="products_oracle", name="ogbn_products/ogbn_products_semantic_oracle_striped", data_dir=f"{DATA_ROOT}/semantic_oracle_striped", kind="real"),
    dict(key="products_perturb50", name="ogbn_products/ogbn_products_metis_perturbed_50", data_dir=f"{DATA_ROOT}/partition_variants", kind="real"),
    dict(key="products_random", name="ogbn_products/ogbn_products_random_constrained", data_dir=f"{DATA_ROOT}/partition_variants", kind="real"),
    dict(key="reddit", name="reddit", data_dir=DATA_ROOT, kind="real"),
    dict(key="reddit_oracle", name="reddit/reddit_semantic_oracle_striped", data_dir=f"{DATA_ROOT}/semantic_oracle_striped", kind="real"),
    dict(key="reddit_perturb50", name="reddit/reddit_metis_perturbed_50", data_dir=f"{DATA_ROOT}/partition_variants", kind="real"),
    dict(key="reddit_random", name="reddit/reddit_random_constrained", data_dir=f"{DATA_ROOT}/partition_variants", kind="real"),
    dict(key="arxiv", name="ogbn_arxiv", data_dir=DATA_ROOT, kind="real"),
    dict(key="arxiv_oracle", name="ogbn_arxiv/ogbn_arxiv_semantic_oracle_striped", data_dir=f"{DATA_ROOT}/semantic_oracle_striped", kind="real"),
    dict(key="arxiv_perturb50", name="ogbn_arxiv/ogbn_arxiv_metis_perturbed_50", data_dir=f"{DATA_ROOT}/partition_variants", kind="real"),
    dict(key="arxiv_random", name="ogbn_arxiv/ogbn_arxiv_random_constrained", data_dir=f"{DATA_ROOT}/partition_variants", kind="real"),
    dict(key="syn1", name="synthetic_milinst_mil_strong_pos_random_v2", data_dir=f"{DATA_ROOT}/synthetic_mil_consistent/synthetic_milinst_mil_strong_pos_random_v2", kind="synthetic"),
    dict(key="syn2", name="synthetic_milinst_mil_weak_pos_strong_v2", data_dir=f"{DATA_ROOT}/synthetic_mil_consistent/synthetic_milinst_mil_weak_pos_strong_v2", kind="synthetic"),
    dict(key="syn3", name="synthetic_milinst_both_useful_v2", data_dir=f"{DATA_ROOT}/synthetic_mil_consistent/synthetic_milinst_both_useful_v2", kind="synthetic"),
]

STAGE2_PROTOCOL = {
    "epochs": 300,
    "patience": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "hidden_dim": None,
    "dropout": 0.5,
    "coarse_top_k": 16,
}


def legacy_run_dir(dataset_key, branch):
    if dataset_key == "products":
        return os.path.join(
            REMOTE_WORK,
            "results/paper_5x10_20260813",
            branch,
            "products/repeat_1/ogbn_products",
        )
    real_mil = {
        "products_oracle": "coarse_gcn_runs/semantic_oracle_20260722/ogbn_products/ogbn_products_semantic_oracle_striped",
        "products_perturb50": "coarse_gcn_runs/products_variants_20260707/ogbn_products/ogbn_products_metis_perturbed_50",
        "products_random": "coarse_gcn_runs/products_variants_20260707/ogbn_products/ogbn_products_random_constrained",
        "reddit_oracle": "coarse_gcn_runs/semantic_oracle_20260722/reddit/reddit_semantic_oracle_striped",
        "reddit_perturb50": "coarse_gcn_runs/metis50_20260705_1502/reddit/reddit_metis_perturbed_50",
        "reddit_random": "coarse_gcn_runs/metis50_20260705_1502/reddit/reddit_random_constrained",
        "arxiv_oracle": "coarse_gcn_runs/semantic_oracle_20260722/ogbn_arxiv/ogbn_arxiv_semantic_oracle_striped",
        "arxiv_perturb50": "coarse_gcn_runs/metis50_20260705_1502/ogbn_arxiv/ogbn_arxiv_metis_perturbed_50",
        "arxiv_random": "coarse_gcn_runs/metis50_20260705_1502/ogbn_arxiv/ogbn_arxiv_random_constrained",
    }
    if branch == "mil" and dataset_key in real_mil:
        return os.path.join(REMOTE_WORK, "results", real_mil[dataset_key])
    synthetic = {
        "syn1": "synthetic_milinst_mil_strong_pos_random_v2",
        "syn2": "synthetic_milinst_mil_weak_pos_strong_v2",
        "syn3": "synthetic_milinst_both_useful_v2",
    }
    if dataset_key in synthetic:
        result_root = (
            "mil_consistent_synthetic2_mixed_20260810"
            if dataset_key == "syn2"
            else "mil_consistent_synthetic_v2_20260810"
        )
        return os.path.join(
            REMOTE_WORK,
            f"results/{result_root}",
            "mean_pool" if branch == "mean" else "mil_head",
            synthetic[dataset_key],
        )
    return None


def split_path(data_name):
    return os.path.join(
        REMOTE_WORK,
        "splits",
        f"{data_name}_cv10_seed{SEED}_adjacent.json",
    )


def existing_file(path):
    return path if path and os.path.isfile(path) else None


def build_entry(root, dataset, branch, fold):
    canonical_dir = os.path.join(root, branch, dataset["key"], f"fold_{fold}")
    legacy_dir = legacy_run_dir(dataset["key"], branch)
    legacy_fold = os.path.join(legacy_dir, f"fold_{fold}") if legacy_dir else None
    checkpoint = existing_file(os.path.join(legacy_fold, "stage1_branch_b.pt")) if legacy_fold else None
    embeddings = existing_file(os.path.join(legacy_fold, "z_mil.pt")) if legacy_fold else None
    legacy_result = existing_file(os.path.join(legacy_fold, "coarse_gcn_results.json")) if legacy_fold else None
    legacy_stage2 = existing_file(os.path.join(legacy_fold, "stage2_coarse_gcn.pt")) if legacy_fold else None

    stage1_reusable = bool(checkpoint and embeddings and legacy_result)
    # July real partition runs used run_coarse_gcn.py defaults (300/50).
    # Synthetic runs and the stopped repeated run used different Stage-2 limits.
    stage2_reusable = bool(
        stage1_reusable
        and legacy_stage2
        and dataset["kind"] == "real"
        and dataset["key"] != "products"
    )
    return {
        "dataset_key": dataset["key"],
        "data_name": dataset["name"],
        "data_dir": dataset["data_dir"],
        "kind": dataset["kind"],
        "branch": branch,
        "fold": fold,
        "seed": SEED,
        "split_manifest": split_path(dataset["name"]),
        "canonical_dir": canonical_dir,
        "stage1": {
            "status": "reuse" if stage1_reusable else "train",
            "checkpoint": checkpoint,
            "embeddings": embeddings,
            "legacy_result": legacy_result,
        },
        "stage2": {
            "status": "reuse" if stage2_reusable else "train_from_frozen_embedding",
            "checkpoint": legacy_stage2 if stage2_reusable else None,
            "legacy_result": legacy_result if stage2_reusable else None,
            "protocol": STAGE2_PROTOCOL,
        },
        "attention": {
            "status": "offline_export" if branch == "mil" else "not_applicable",
            "requires_training": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=f"{REMOTE_WORK}/results/paper_10fold_20260814",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    entries = [
        build_entry(args.root, dataset, branch, fold)
        for dataset in DATASETS
        for branch in ("mean", "mil")
        for fold in range(FOLDS)
    ]
    counts = Counter(
        (entry["stage1"]["status"], entry["stage2"]["status"])
        for entry in entries
    )
    payload = {
        "protocol": "single_grouped_stratified_10fold_8_1_1",
        "seed": SEED,
        "folds": FOLDS,
        "stage2_protocol": STAGE2_PROTOCOL,
        "expected_branch_folds": len(DATASETS) * 2 * FOLDS,
        "expected_model_test_observations": len(DATASETS) * 4 * FOLDS,
        "expected_attention_folds": len(DATASETS) * FOLDS,
        "status_counts": {f"{a}+{b}": n for (a, b), n in sorted(counts.items())},
        "entries": entries,
    }
    output = args.output or os.path.join(args.root, "execution_manifest.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"output": output, **payload["status_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
