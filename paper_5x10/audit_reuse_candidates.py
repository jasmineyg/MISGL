#!/usr/bin/env python3
"""Build a compact, read-only compatibility inventory of prior fold outputs."""

import argparse
import glob
import hashlib
import json
import os

import torch


def digest_indices(split):
    packed = json.dumps(
        {
            key: split.get(key, [])
            for key in ("train_indices", "val_indices", "test_indices")
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    rows = []
    for root in args.roots:
        pattern = os.path.join(root, "**", "fold_0", "coarse_gcn_results.json")
        for result_path in glob.glob(pattern, recursive=True):
            fold_dir = os.path.dirname(result_path)
            checkpoint_path = os.path.join(fold_dir, "stage1_branch_b.pt")
            try:
                result = json.load(open(result_path, "r", encoding="utf-8"))
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                hparams = checkpoint.get("hparams", {})
                branch_b = hparams.get("branch_b") or {}
                rows.append(
                    {
                        "run_dir": os.path.dirname(fold_dir),
                        "data_name": result.get("data_name"),
                        "fold_count": len(
                            glob.glob(os.path.join(os.path.dirname(fold_dir), "fold_*", "coarse_gcn_results.json"))
                        ),
                        "cv_seed": hparams.get("cv_seed"),
                        "cv_num_folds": hparams.get("cv_num_folds"),
                        "branch": "mil" if branch_b.get("use", False) else "mean",
                        "epoch": hparams.get("epoch"),
                        "patience": hparams.get("patience"),
                        "batch_size": hparams.get("batch_size"),
                        "split_digest_fold0": digest_indices(result.get("split", {})),
                        "has_stage2": bool(result.get("stage2", {}).get("final_metrics")),
                        "has_embeddings": os.path.exists(os.path.join(fold_dir, "z_mil.pt")),
                        "has_attention": os.path.exists(os.path.join(fold_dir, "attention_metrics.json")),
                    }
                )
            except Exception as exc:
                rows.append({"run_dir": os.path.dirname(fold_dir), "error": repr(exc)})
    print(json.dumps(sorted(rows, key=lambda row: row["run_dir"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
