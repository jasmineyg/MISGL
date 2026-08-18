"""Fair 10-fold runner for Ilse et al. gated-attention MIL."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data_graph import build_data_loaders, load_graph_pickle
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
from model import GatedAttentionMIL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--cache-root", default="result/fair_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--fold-limit", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-limit", type=int, default=256)
    parser.add_argument("--smoke-eval-limit", type=int, default=128)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def move_batch(batch, device: torch.device):
    features, labels, _, indices, bag_index, _ = batch
    return (
        features.to(device, non_blocking=True),
        labels.to(device, dtype=torch.float32, non_blocking=True).view(-1),
        indices.detach().cpu().numpy().astype(np.int64),
        bag_index.to(device, non_blocking=True),
    )


def run_train_epoch(model, loader, optimizer, criterion, device: torch.device) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    count = 0
    labels_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    for batch in loader:
        features, labels, _, bag_index = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _, _ = model(features, bag_index, bag_count=len(labels))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * len(labels)
        count += len(labels)
        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits.detach()).cpu().numpy())
    metrics = binary_metrics(np.concatenate(labels_all), np.concatenate(probs_all))
    metrics["loss"] = loss_sum / count
    return metrics


@torch.inference_mode()
def evaluate(model, loader, criterion, device: torch.device) -> Dict[str, object]:
    model.eval()
    loss_sum = 0.0
    count = 0
    labels_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    indices_all: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    for batch in loader:
        features, labels, indices, bag_index = move_batch(batch, device)
        logits, _, _, _ = model(features, bag_index, bag_count=len(labels))
        loss = criterion(logits, labels)
        loss_sum += float(loss.detach()) * len(labels)
        count += len(labels)
        labels_all.append(labels.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        indices_all.append(indices)
    labels = np.concatenate(labels_all).astype(np.int64)
    logits = np.concatenate(logits_all).astype(np.float32)
    probabilities = np.concatenate(probs_all).astype(np.float32)
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = loss_sum / count
    return {
        "metrics": metrics,
        "orig_indices": np.concatenate(indices_all).astype(np.int64),
        "labels": labels,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": (probabilities > 0.5).astype(np.int64),
    }


def write_fold_csv(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    records = list(records)
    if not records:
        return
    fields = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    entry = load_dataset_entry(args.execution_manifest, args.dataset_key)
    graph_data = load_graph_pickle(
        entry["pkl_path"], cache_dir=args.cache_root, rebuild_cache=args.rebuild_cache
    )
    labels = graph_data["subgraph_labels"].cpu().numpy().astype(np.int64)
    splits = load_fixed_folds(entry["split_manifest"], len(labels))
    result_root = Path(args.result_root) / args.dataset_key
    if args.smoke:
        result_root = result_root / "smoke"
    result_root.mkdir(parents=True, exist_ok=True)

    config = {
        "method": "Attention-based MIL (gated, embedding-level)",
        "paper": "Ilse et al. (2018)",
        "protocol": PROTOCOL_NAME,
        "dataset_key": args.dataset_key,
        "data_name": entry["data_name"],
        "dataset": dataset_fingerprint(entry["pkl_path"]),
        "split_manifest": entry["split_manifest"],
        "model": {
            "instance_encoder": "Linear-ReLU",
            "hidden_dim": args.hidden_dim,
            "attention": "tanh(Vh) * sigmoid(Uh), softmax",
            "attention_dim": args.attention_dim,
            "dropout": args.dropout,
            "classifier": "linear bag classifier",
            "source_graph_edges_used": False,
        },
        "training": {
            "loss": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size_bags": args.batch_size,
            "max_epochs": 3 if args.smoke else args.epochs,
            "patience": 3 if args.smoke else args.patience,
            "selection": "minimum validation loss",
        },
        "runtime": runtime_info(torch),
    }
    save_json(result_root / "config.json", config)

    records: List[Dict[str, object]] = []
    fold_stop = min(10, args.fold_start + (1 if args.smoke else args.fold_limit))
    for fold_id in range(args.fold_start, fold_stop):
        split = dict(splits[fold_id])
        if args.smoke:
            split["train_indices"] = split["train_indices"][: args.smoke_train_limit]
            split["val_indices"] = split["val_indices"][: args.smoke_eval_limit]
            split["test_indices"] = split["test_indices"][: args.smoke_eval_limit]
        fold_seed = int(split["fold_seed"])
        seed_everything(fold_seed, torch)
        fold_dir = result_root / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader, test_loader = build_data_loaders(
            graph_data,
            split["train_indices"],
            split["val_indices"],
            split["test_indices"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = GatedAttentionMIL(
            input_dim=int(graph_data["feature_dimension"]),
            hidden_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            dropout=args.dropout,
        ).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        bad_epochs = 0
        history: List[Dict[str, float]] = []
        started = time.time()
        max_epochs = 3 if args.smoke else args.epochs
        patience = 3 if args.smoke else args.patience
        for epoch in range(1, max_epochs + 1):
            train_metrics = run_train_epoch(model, train_loader, optimizer, criterion, device)
            val_output = evaluate(model, val_loader, criterion, device)
            val_metrics = val_output["metrics"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_metrics["loss"]),
                    "train_acc": float(train_metrics["acc"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_acc": float(val_metrics["acc"]),
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
            raise RuntimeError("No model checkpoint was selected")
        model.load_state_dict(best_state)
        val_output = evaluate(model, val_loader, criterion, device)
        test_output = evaluate(model, test_loader, criterion, device)
        test_metrics = dict(test_output["metrics"])
        record: Dict[str, object] = {
            "dataset_key": args.dataset_key,
            "method": "Attention-based MIL",
            "fold": fold_id,
            "fold_seed": fold_seed,
            "split_digest": split_digest(split),
            "best_epoch": best_epoch,
            "train_n": len(split["train_indices"]),
            "val_n": len(split["val_indices"]),
            "test_n": len(split["test_indices"]),
            **test_metrics,
            "elapsed_seconds": float(time.time() - started),
        }
        records.append(record)
        torch.save(
            {
                "model_state_dict": best_state,
                "fold": fold_id,
                "fold_seed": fold_seed,
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
            smoke_checks = {
                "finite_losses": bool(
                    all(np.isfinite(row["train_loss"]) and np.isfinite(row["val_loss"]) for row in history)
                ),
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
            save_json(result_root / "smoke_checks.json", smoke_checks)
            if not all(smoke_checks.values()):
                raise RuntimeError(f"Smoke checks failed: {smoke_checks}")

    write_fold_csv(result_root / "fold_metrics.csv", records)
    if not args.smoke and len(records) == 10:
        save_json(
            result_root / "summary.json",
            {"folds": records, "aggregate": aggregate_fold_records(records), "config": config},
        )


if __name__ == "__main__":
    main()
