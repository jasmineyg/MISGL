#!/usr/bin/env python3
"""Read-only inventory for cached paper fold artifacts."""

import argparse
import json
import os

import torch


def shape_summary(payload):
    return {
        key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
        for key, value in payload.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold_dirs", nargs="+")
    args = parser.parse_args()

    for fold_dir in args.fold_dirs:
        print(f"FOLD {fold_dir}")
        result_path = os.path.join(fold_dir, "coarse_gcn_results.json")
        result = json.load(open(result_path, "r", encoding="utf-8"))
        print("result_keys", sorted(result))
        print("stage1_keys", sorted(result["stage1"]))
        print("stage2_keys", sorted(result["stage2"]))
        print("paths", result.get("paths"))

        embedding_path = os.path.join(fold_dir, "z_mil.pt")
        embeddings = torch.load(embedding_path, map_location="cpu")
        print("embedding", shape_summary(embeddings))

        checkpoint = torch.load(
            os.path.join(fold_dir, "stage1_branch_b.pt"), map_location="cpu"
        )
        print("stage1_checkpoint_keys", sorted(checkpoint))

        attention_path = os.path.join(fold_dir, "attention_metrics.json")
        if os.path.exists(attention_path):
            attention = json.load(open(attention_path, "r", encoding="utf-8"))
            bags = attention.get("positive_bags", [])
            print("attention_keys", sorted(attention))
            print("positive_bag_count", len(bags))
            print("positive_bag_keys", sorted(bags[0]) if bags else [])


if __name__ == "__main__":
    main()
