"""Fair fixed-fold runner for the original three-channel SubGNN model."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch

SUBGNN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SUBGNN_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SUBGNN_DIR))

import SubGNN as md
from fair_protocol import (
    PROTOCOL_NAME,
    aggregate_fold_records,
    binary_metrics,
    dataset_fingerprint,
    load_dataset_entry,
    load_fixed_folds,
    runtime_info,
    save_json,
    seed_everything,
    split_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--fold-limit", type=int, default=10)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-anchor-patches", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-limit", type=int, default=128)
    parser.add_argument("--smoke-eval-limit", type=int, default=64)
    return parser.parse_args()


def hyperparameters(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "max_epochs": 5 if args.smoke else args.epochs,
        "seed": 1024,
        "device": args.device,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "grad_clip": 0.0,
        "node_embed_size": 128,
        "n_layers": args.n_layers,
        "n_anchor_patches_N_in": args.n_anchor_patches,
        "n_anchor_patches_N_out": args.n_anchor_patches,
        "n_anchor_patches_pos_in": args.n_anchor_patches,
        "n_anchor_patches_pos_out": args.n_anchor_patches,
        "n_anchor_patches_structure": args.n_anchor_patches,
        "linear_hidden_dim_1": 64,
        "linear_hidden_dim_2": 32,
        "lin_dropout": 0.2,
        "lstm_dropout": 0.0,
        "lstm_n_layers": 1,
        "lstm_aggregator": "last",
        "cc_aggregator": "max",
        "trainable_cc": False,
        "freeze_node_embeds": False,
        "use_mpn_projection": True,
        "use_neighborhood": True,
        "use_position": True,
        "use_structure": True,
        "resample_anchor_patches": False,
        "compute_similarities": False,
        "sample_walk_len": 8,
        "n_triangular_walks": 2,
        "random_walk_len": 8,
        "rw_beta": 0.65,
        "max_sim_epochs": 1,
        "structure_patch_type": "triangular_random_walk",
        "structure_similarity_fn": "dtw",
        "neigh_sample_border_size": 1,
        "n_processes": 1,
        "set2set": False,
        "ff_attn": False,
        "print_train_times": False,
        "auto_lr_find": False,
    }


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: (
            value
            if key == "subgraph_idx"
            else value.to(device) if torch.is_tensor(value) else value
        )
        for key, value in batch.items()
    }


def forward_for_split(model, batch, split_name: str):
    args = (
        batch["subgraph_ids"],
        batch["cc_ids"],
        batch["subgraph_idx"],
        batch["NP_sim"],
        batch["I_S_sim"],
        batch["B_S_sim"],
    )
    if split_name == "train":
        return model.forward(
            "train", model.train_N_I_cc_embed, model.train_N_B_cc_embed,
            model.train_S_I_cc_embed, model.train_S_B_cc_embed,
            model.train_P_I_cc_embed, model.train_P_B_cc_embed, *args
        )
    if split_name == "val":
        return model.forward(
            "val", model.val_N_I_cc_embed, model.val_N_B_cc_embed,
            model.val_S_I_cc_embed, model.val_S_B_cc_embed,
            model.val_P_I_cc_embed, model.val_P_B_cc_embed, *args
        )
    if split_name == "test":
        return model.forward(
            "test", model.test_N_I_cc_embed, model.test_N_B_cc_embed,
            model.test_S_I_cc_embed, model.test_S_B_cc_embed,
            model.test_P_I_cc_embed, model.test_P_B_cc_embed, *args
        )
    raise ValueError(split_name)


def run_train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    labels_all = []
    probabilities_all = []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_for_split(model, batch, "train")
        labels = batch["label"].squeeze(-1)
        loss = model.loss(logits, labels)
        loss.backward()
        optimizer.step()
        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        total_loss += float(loss.detach()) * len(labels)
        count += len(labels)
        labels_all.append(labels.detach().cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    metrics = binary_metrics(np.concatenate(labels_all), np.concatenate(probabilities_all))
    metrics["loss"] = total_loss / count
    return metrics


@torch.inference_mode()
def evaluate(model, loader, split_name, device, original_indices):
    model.eval()
    total_loss = 0.0
    count = 0
    labels_all = []
    probabilities_all = []
    logits_all = []
    indices_all = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = forward_for_split(model, batch, split_name)
        labels = batch["label"].squeeze(-1)
        loss = model.loss(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        local_indices = batch["subgraph_idx"].detach().cpu().numpy().reshape(-1)
        total_loss += float(loss.detach()) * len(labels)
        count += len(labels)
        labels_all.append(labels.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
        indices_all.append(np.asarray(original_indices)[local_indices])
    labels = np.concatenate(labels_all).astype(np.int64)
    logits = np.concatenate(logits_all).astype(np.float32)
    probabilities = np.concatenate(probabilities_all).astype(np.float32)
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = total_loss / count
    return {
        "metrics": metrics,
        "orig_indices": np.concatenate(indices_all).astype(np.int64),
        "labels": labels,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": (probabilities > 0.5).astype(np.int64),
    }


def build_model(args, entry, split, cache_root):
    data_config = {
        "pkl_path": entry["pkl_path"],
        "data_name": args.dataset_key,
        "cache_root": str(cache_root),
        "split_indices": split,
    }
    return md.SubGNN(
        hyperparameters(args),
        graph_path=None,
        subgraph_path=None,
        embedding_path=None,
        similarities_path=None,
        shortest_paths_path=None,
        degree_dict_path=None,
        ego_graph_path=None,
        data_config=data_config,
    )


def write_csv(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    records = list(records)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    entry = load_dataset_entry(args.execution_manifest, args.dataset_key)
    # Prepare once; every fold reuses the same relabeled base graph, node features,
    # subgraph-node lists, and labels in this process.
    from data_loading import _prepare_dataset
    prepared_dataset = _prepare_dataset(entry["pkl_path"])
    labels = prepared_dataset["labels"]
    splits = load_fixed_folds(entry["split_manifest"], len(labels))
    result_root = Path(args.result_root) / args.dataset_key
    cache_root = Path(args.cache_root)
    if args.smoke:
        result_root = result_root / "smoke"
        cache_root = cache_root / "smoke"
    result_root.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "SubGNN",
        "paper": "Alsentzer et al. (2020)",
        "protocol": PROTOCOL_NAME,
        "dataset_key": args.dataset_key,
        "data_name": entry["data_name"],
        "dataset": dataset_fingerprint(entry["pkl_path"]),
        "split_manifest": entry["split_manifest"],
        "input_adaptation": {
            "base_graph": "original_graph",
            "subgraphs": "subgraph_structures",
            "initial_node_embeddings": "provided node_features",
        },
        "model": {
            "channels": ["neighborhood", "position", "structure"],
            "subchannels": ["internal", "border"],
            "layers": args.n_layers,
            "anchors_per_channel": args.n_anchor_patches,
            "structure_similarity": "DTW over degree sequences",
            "structure_anchor": "triangular random walk",
            "classifier": "three-layer feed-forward network",
        },
        "training": {
            "loss": "cross entropy",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": 5 if args.smoke else args.epochs,
            "patience": 5 if args.smoke else args.patience,
            "selection": "minimum validation loss",
        },
        "scalability_adaptation": (
            "exact shortest-path distances are computed on demand for sampled anchors; "
            "the infeasible all-pairs tensor is not materialized"
        ),
        "runtime": runtime_info(torch),
    }
    save_json(result_root / "config.json", config)
    records = []
    stop = min(10, args.fold_start + (1 if args.smoke else args.fold_limit))
    for fold_id in range(args.fold_start, stop):
        split = dict(splits[fold_id])
        if args.smoke:
            split["train_indices"] = split["train_indices"][: args.smoke_train_limit]
            split["val_indices"] = split["val_indices"][: args.smoke_eval_limit]
            split["test_indices"] = split["test_indices"][: args.smoke_eval_limit]
        seed_everything(int(split["fold_seed"]), torch)
        started = time.time()
        model = build_model(args, entry, split, cache_root).to(device)
        model.prepare_data()
        train_loader = model.train_dataloader()
        val_loader = model.val_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        bad_epochs = 0
        history = []
        max_epochs = 5 if args.smoke else args.epochs
        patience = 5 if args.smoke else args.patience
        for epoch in range(1, max_epochs + 1):
            train_metrics = run_train_epoch(model, train_loader, optimizer, device)
            val_output = evaluate(
                model, val_loader, "val", device, model.val_orig_indices
            )
            val_metrics = val_output["metrics"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                }
            )
            print(
                f"{args.dataset_key} fold={fold_id} epoch={epoch} "
                f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}",
                flush=True,
            )
            if float(val_metrics["loss"]) < best_loss:
                best_loss = float(val_metrics["loss"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        if best_state is None:
            raise RuntimeError("No SubGNN checkpoint selected")
        model.load_state_dict(best_state)
        test_loader = model.test_dataloader()
        test_output = evaluate(
            model, test_loader, "test", device, model.test_orig_indices
        )
        record = {
            "dataset_key": args.dataset_key,
            "method": "SubGNN",
            "fold": fold_id,
            "fold_seed": int(split["fold_seed"]),
            "split_digest": split_digest(split),
            "best_epoch": best_epoch,
            "train_n": len(split["train_indices"]),
            "val_n": len(split["val_indices"]),
            "test_n": len(split["test_indices"]),
            **test_output["metrics"],
            "elapsed_seconds": float(time.time() - started),
        }
        records.append(record)
        fold_dir = result_root / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "fold": fold_id,
                "fold_seed": int(split["fold_seed"]),
                "split_digest": record["split_digest"],
                "best_epoch": best_epoch,
                "config": config,
            },
            fold_dir / "checkpoint.pt",
        )
        np.savez_compressed(
            fold_dir / "test_predictions.npz",
            orig_indices=test_output["orig_indices"],
            labels=test_output["labels"],
            logits=test_output["logits"],
            probabilities=test_output["probabilities"],
            predictions=test_output["predictions"],
        )
        save_json(fold_dir / "history.json", {"history": history})
        save_json(fold_dir / "metrics.json", record)
        if args.smoke:
            checks = {
                "finite_losses": bool(all(np.isfinite(row["train_loss"]) for row in history)),
                "loss_decreased": bool(min(row["train_loss"] for row in history) < history[0]["train_loss"]),
                "output_length_matches_test": bool(len(test_output["labels"]) == len(split["test_indices"])),
                "test_indices_exact_subset": bool(
                    np.array_equal(np.sort(test_output["orig_indices"]), np.sort(split["test_indices"]))
                ),
                "no_split_overlap": bool(
                    not set(split["train_indices"]).intersection(split["val_indices"])
                    and not set(split["train_indices"]).intersection(split["test_indices"])
                    and not set(split["val_indices"]).intersection(split["test_indices"])
                ),
            }
            save_json(result_root / "smoke_checks.json", checks)
            if not all(checks.values()):
                raise RuntimeError(f"Smoke checks failed: {checks}")
    write_csv(result_root / "fold_metrics.csv", records)
    if not args.smoke and len(records) == 10:
        save_json(
            result_root / "summary.json",
            {"folds": records, "aggregate": aggregate_fold_records(records), "config": config},
        )


if __name__ == "__main__":
    main()
