#!/usr/bin/env python3
"""Read-only audit of Syn1 mean/POS provenance and saved-checkpoint behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_coarse(path: str) -> sp.spmatrix:
    try:
        return sp.load_npz(path)
    except ValueError:
        sys.path.insert(0, str(Path.cwd()))
        from MISGL.utils.coarse_graph import load_coarse_adjacency

        matrix, _metadata = load_coarse_adjacency(path)
        return matrix


def acc(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits.reshape(-1)) > 0.5).to(torch.int64)
    truth = labels.reshape(-1).to(torch.int64)
    return float((pred[mask] == truth[mask]).float().mean().item())


def linear(x: torch.Tensor, state: dict, prefix: str) -> torch.Tensor:
    return F.linear(x, state[f"{prefix}.weight"], state.get(f"{prefix}.bias"))


def stage2_logits(z: torch.Tensor, adj: sp.spmatrix, state: dict) -> dict[str, torch.Tensor]:
    matrix = adj.tocsr().astype(np.float32, copy=True)
    matrix = matrix + sp.eye(matrix.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    inv_degree = np.zeros_like(degree)
    inv_degree[degree > 0] = 1.0 / degree[degree > 0]
    normalized = sp.diags(inv_degree, format="csr") @ matrix
    propagated = torch.from_numpy(np.asarray(normalized @ z.numpy(), dtype=np.float32))
    z_pos = torch.relu(linear(propagated, state, "gcn"))
    identity_pos = torch.relu(linear(z, state, "gcn"))
    zeros = torch.zeros_like(z_pos)

    def classify(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(linear(torch.cat([left, right], dim=-1), state, "classifier.0"))
        return linear(hidden, state, "classifier.3").reshape(-1)

    return {
        "full": classify(z, z_pos),
        "identity_only": classify(z, identity_pos),
        "z_mean_block_only": classify(z, zeros),
        "graph_block_only": classify(torch.zeros_like(z), z_pos),
    }


def homophily(adj: sp.spmatrix, labels: torch.Tensor) -> dict:
    coo = adj.tocoo()
    keep = coo.row != coo.col
    rows = coo.row[keep]
    cols = coo.col[keep]
    weights = np.asarray(coo.data[keep], dtype=np.float64)
    y = labels.reshape(-1).numpy().astype(np.int64)
    same = y[rows] == y[cols]
    return {
        "directed_nonself_edges": int(len(rows)),
        "unweighted": float(np.mean(same)) if len(rows) else None,
        "weighted": float(np.average(same, weights=weights)) if len(rows) and weights.sum() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", default="syn1")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = [e for e in manifest["entries"] if e["dataset_key"] == args.dataset]
    by_key = {(e["branch"], int(e["fold"])): e for e in entries}
    report = {"dataset": args.dataset, "folds": []}

    for fold in range(10):
        mean = by_key[("mean", fold)]
        mil = by_key[("mil", fold)]
        mean_ckpt = load_pt(mean["stage1"]["checkpoint"])
        mil_ckpt = load_pt(mil["stage1"]["checkpoint"])
        mean_payload = load_pt(mean["stage1"]["embeddings"])
        mil_payload = load_pt(mil["stage1"]["embeddings"])
        mean_z = mean_payload.get("z_mean", mean_payload.get("z_mil")).detach().cpu().float()
        mil_z = mil_payload.get("z_mil", mil_payload.get("z_mean")).detach().cpu().float()
        result_path = Path(mean["canonical_dir"]) / "fold_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        provenance = json.loads((Path(mean["canonical_dir"]) / "stage1_provenance.json").read_text(encoding="utf-8"))
        stage2_predictions = load_pt(str(Path(mean["canonical_dir"]) / "stage2_predictions.pt"))
        stage2_checkpoint = load_pt(result["stage2"]["checkpoint"])
        coarse = load_coarse(result["coarse_adj"])
        labels = mean_payload["labels"].reshape(-1).to(torch.int64)
        orig_indices = mean_payload["orig_indices"].reshape(-1).to(torch.int64)
        test_ids = {int(v) for v in result["split"]["test_indices"]}
        test_mask = torch.tensor([int(v) in test_ids for v in orig_indices], dtype=torch.bool)
        variants = stage2_logits(mean_z, coarse, stage2_checkpoint["state_dict"])
        saved_logits = stage2_predictions["logits"].reshape(-1).float()
        fold_report = {
            "fold": fold,
            "mean_checkpoint_branch_b_use": bool(mean_ckpt["hparams"]["branch_b"]["use"]),
            "mil_checkpoint_branch_b_use": bool(mil_ckpt["hparams"]["branch_b"]["use"]),
            "mean_embedding_keys": sorted(mean_payload.keys()),
            "mean_embedding_source": mean["stage1"]["embeddings"],
            "mean_embedding_sha256": file_sha256(mean["stage1"]["embeddings"]),
            "mil_embedding_sha256": file_sha256(mil["stage1"]["embeddings"]),
            "mean_mil_embeddings_same_file_hash": file_sha256(mean["stage1"]["embeddings"])
            == file_sha256(mil["stage1"]["embeddings"]),
            "mean_mil_embedding_max_abs_diff": float((mean_z - mil_z).abs().max().item()),
            "provenance": provenance,
            "coarse_adj": result["coarse_adj"],
            "coarse_homophily": homophily(coarse, labels),
            "test_n": int(test_mask.sum().item()),
            "test_positive_rate": float(labels[test_mask].float().mean().item()),
            "reported_stage1_test_acc": float(result["stage1"]["metrics"]["test"]["acc"]),
            "reported_stage2_test_acc": float(result["stage2"]["metrics"]["test"]["acc"]),
            "saved_vs_recomputed_full_max_logit_diff": float((saved_logits - variants["full"]).abs().max().item()),
            "offline_test_acc": {name: acc(logits, labels, test_mask) for name, logits in variants.items()},
        }
        report["folds"].append(fold_report)

    numeric = [
        "reported_stage1_test_acc",
        "reported_stage2_test_acc",
        "test_positive_rate",
        "mean_mil_embedding_max_abs_diff",
        "saved_vs_recomputed_full_max_logit_diff",
    ]
    report["mean_over_folds"] = {
        key: float(np.mean([row[key] for row in report["folds"]])) for key in numeric
    }
    report["mean_over_folds"]["coarse_homophily_unweighted"] = float(
        np.mean([row["coarse_homophily"]["unweighted"] for row in report["folds"]])
    )
    for variant in report["folds"][0]["offline_test_acc"]:
        report["mean_over_folds"][f"offline_{variant}_test_acc"] = float(
            np.mean([row["offline_test_acc"][variant] for row in report["folds"]])
        )
    report["all_mean_checkpoints_non_mil"] = all(
        not row["mean_checkpoint_branch_b_use"] for row in report["folds"]
    )
    report["all_mil_checkpoints_mil"] = all(
        row["mil_checkpoint_branch_b_use"] for row in report["folds"]
    )
    report["any_mean_mil_embedding_hash_equal"] = any(
        row["mean_mil_embeddings_same_file_hash"] for row in report["folds"]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
