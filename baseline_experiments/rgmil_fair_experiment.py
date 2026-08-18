"""Fixed-protocol preprocessing, VDN search, and 10-fold training for RGMIL."""

from __future__ import annotations

import argparse
import copy
import csv
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

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
from rgmil_fair_model import GAT, RGMILSearch


THRESHOLD_SPACE = [round(value, 2) for value in np.arange(0.05, 1.01, 0.05)]
LAYER_SPACE = list(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--phase", choices=["all", "prepare", "search", "train"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-nodes-per-bag", type=int, default=500)
    parser.add_argument("--search-max-steps", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--agent-learning-rate", type=float, default=5e-4)
    parser.add_argument("--agent-weight-decay", type=float, default=1e-3)
    parser.add_argument("--policy-layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--negative-slope", type=float, default=0.1)
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--fold-limit", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-limit", type=int, default=128)
    parser.add_argument("--smoke-eval-limit", type=int, default=64)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def fit_bag(features: np.ndarray, max_nodes: int, rng: np.random.Generator) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if len(features) == 0:
        raise ValueError("RGMIL does not support empty bags")
    if len(features) == 1:
        features = np.repeat(features, 2, axis=0)
        features[1] += rng.normal(0.0, 0.1, size=features.shape[1]).astype(np.float32)
    elif len(features) > max_nodes:
        selection = np.linspace(0, len(features) - 1, max_nodes, dtype=np.int64)
        features = features[selection]
    return np.ascontiguousarray(features, dtype=np.float32)


def build_similarity(features: np.ndarray) -> np.ndarray:
    # Equations (6) and (10): exp(-pairwise Euclidean distance), then threshold.
    distances = euclidean_distances(features, features)
    similarity = np.exp(-distances).astype(np.float32)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def prepare_cache(entry, cache_path: Path, max_nodes: int, seed: int) -> Dict[str, object]:
    source_fingerprint = dataset_fingerprint(entry["pkl_path"])
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
        if cache.get("source_fingerprint") == source_fingerprint and cache.get("max_nodes") == max_nodes:
            print(f"Reusing RGMIL cache: {cache_path}", flush=True)
            return cache
    print(f"Loading source pickle: {entry['pkl_path']}", flush=True)
    with Path(entry["pkl_path"]).open("rb") as handle:
        payload = pickle.load(handle)
    node_features = np.asarray(payload["node_features"], dtype=np.float32)
    node_counts = np.asarray(payload["subgraph_node_counts"], dtype=np.int64)
    labels = np.asarray(payload["subgraph_labels"], dtype=np.int64)
    if int(node_counts.sum()) != len(node_features) or len(node_counts) != len(labels):
        raise ValueError("Flat node features, bag counts, and labels are not aligned")
    offsets = np.concatenate(([0], np.cumsum(node_counts)))
    rng = np.random.default_rng(seed)
    bags = []
    sampled_bags = 0
    similarity_min = 1.0
    similarity_max = 0.0
    for bag_idx in range(len(labels)):
        features = fit_bag(
            node_features[offsets[bag_idx] : offsets[bag_idx + 1]], max_nodes, rng
        )
        sampled_bags += int(node_counts[bag_idx] > max_nodes)
        similarity = build_similarity(features)
        similarity_min = min(similarity_min, float(similarity.min()))
        similarity_max = max(similarity_max, float(similarity.max()))
        bags.append((int(labels[bag_idx]), features, similarity, int(bag_idx)))
        if (bag_idx + 1) % 1000 == 0:
            print(f"prepared {bag_idx + 1}/{len(labels)} bags", flush=True)
    cache = {
        "source_fingerprint": source_fingerprint,
        "max_nodes": max_nodes,
        "feature_dim": int(node_features.shape[1]),
        "labels": labels,
        "bags": bags,
        "sampled_bags": sampled_bags,
        "similarity_min": similarity_min,
        "similarity_max": similarity_max,
        "graph_construction": "exp(-pairwise Euclidean distance)",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved RGMIL cache: {cache_path}", flush=True)
    return cache


def state_vectors(bags: Sequence[Tuple], max_size: int, feature_dim: int):
    # Eq. (7): one topology statistic for every possible bag size d.
    by_size: Dict[int, List[float]] = {}
    bag_means = []
    for _, features, similarity, _ in bags:
        size = len(features)
        by_size.setdefault(size, []).append(float(similarity.mean()))
        bag_means.append(features.mean(axis=0))
    state_1 = np.zeros(max_size, dtype=np.float32)
    for size, values in by_size.items():
        state_1[size - 1] = float(np.mean(values))
    state_2 = (
        np.mean(np.stack(bag_means), axis=0).astype(np.float32)
        if bag_means
        else np.zeros(feature_dim, dtype=np.float32)
    )
    return state_1, state_2


def build_search_blocks(cache, split, smoke: bool, train_limit: int, eval_limit: int):
    bags = cache["bags"]
    max_size = max(len(bag[1]) for bag in bags)
    feature_dim = int(cache["feature_dim"])
    blocks = []
    for canonical_fold in split["train_folds"].tolist():
        # Recover each canonical training fold so the RL state space follows the paper.
        indices = split["canonical_folds"][int(canonical_fold)]
        if smoke:
            indices = indices[: max(1, train_limit // len(split["train_folds"]))]
        block_bags = [bags[int(index)] for index in indices]
        blocks.append(block_bags + [state_vectors(block_bags, max_size, feature_dim)])
    val_indices = split["val_indices"][:eval_limit] if smoke else split["val_indices"]
    val_bags = [bags[int(index)] for index in val_indices]
    blocks.append(val_bags + [state_vectors(val_bags, max_size, feature_dim)])
    return blocks


def search_actions(args, cache, split, search_dir: Path, config) -> Dict[str, object]:
    result_path = search_dir / "action_search.json"
    if result_path.exists() and not args.smoke:
        import json
        with result_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    seed_everything(args.seed, torch)
    blocks = build_search_blocks(
        cache, split, args.smoke, args.smoke_train_limit, args.smoke_eval_limit
    )
    search = RGMILSearch(
        data_name=args.dataset_key,
        threshold_space=THRESHOLD_SPACE,
        layer_space=LAYER_SPACE,
        state_1_dim=len(blocks[0][-1][0]),
        state_2_dim=int(cache["feature_dim"]),
        gnn_learning_rate=args.learning_rate,
        gnn_weight_decay=args.weight_decay,
        agent_learning_rate=args.agent_learning_rate,
        agent_weight_decay=args.agent_weight_decay,
        policy_layer_num=args.policy_layers,
        drop_rate=args.dropout,
        slope_rate=args.negative_slope,
        discount_rate=0.95,
        epsilon_start=1.0,
        epsilon_end=0.0,
        epsilon_decay_steps=50,
        history_num=10,
        reward_tolerance=1e-4,
        memory_size=20,
        memory_batch_size=1,
        device=torch.device(args.device),
    )
    max_steps = 2 if args.smoke else args.search_max_steps
    converged = False
    started = time.time()
    for timestep in range(max_steps):
        converged = search.step(blocks, timestep)
        print(
            f"search {args.dataset_key} t={timestep} threshold={search.threshold:.2f} "
            f"layers={search.layer_num} val_acc={search.history_performance[-1]:.6f}",
            flush=True,
        )
        if converged:
            break
    result = {
        "dataset_key": args.dataset_key,
        "fold_used_for_action_search": 0,
        "threshold": float(search.threshold),
        "layer_num": int(search.layer_num),
        "steps": len(search.threshold_trace),
        "converged": bool(converged),
        "elapsed_seconds": float(time.time() - started),
        "threshold_trace": search.threshold_trace,
        "layer_trace": search.layer_trace,
        "validation_accuracy_trace": search.history_performance[10:],
        "reward_trace": search.reward_trace[10:],
        "gnn_loss_trace": search.loss_gnn_trace,
        "agent_loss_trace": search.loss_agent_trace[10:],
        "config": config,
    }
    search_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_path, result)
    torch.save(
        {
            "gnn": search.gnn.state_dict(),
            "agent_1": search.agent_1.state_dict(),
            "agent_2": search.agent_2.state_dict(),
            "result": result,
        },
        search_dir / "action_search_checkpoint.pt",
    )
    return result


def evaluate_gnn(model, bags, threshold: float, layer_num: int, device: torch.device):
    model.eval()
    labels = []
    logits = []
    probabilities = []
    indices = []
    loss_sum = 0.0
    with torch.inference_mode():
        for label, features, similarity, original_index in bags:
            target = torch.tensor([label], dtype=torch.float32, device=device)
            logit, _ = model((features, similarity, threshold, layer_num))
            loss_sum += float(model.loss_function(logit, target))
            probability = float(torch.sigmoid(logit).item())
            labels.append(int(label))
            logits.append(float(logit.item()))
            probabilities.append(probability)
            indices.append(int(original_index))
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = loss_sum / max(len(bags), 1)
    return {
        "metrics": metrics,
        "orig_indices": np.asarray(indices, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "logits": np.asarray(logits, dtype=np.float32),
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "predictions": (np.asarray(probabilities) > 0.5).astype(np.int64),
    }


def train_epoch(model, optimizer, bags, threshold, layer_num, device):
    model.train()
    loss_sum = 0.0
    labels = []
    probabilities = []
    for label, features, similarity, _ in bags:
        target = torch.tensor([label], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        logit, _ = model((features, similarity, threshold, layer_num))
        loss = model.loss_function(logit, target)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach())
        labels.append(int(label))
        probabilities.append(float(torch.sigmoid(logit.detach()).item()))
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = loss_sum / max(len(bags), 1)
    return metrics


def write_csv(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    records = list(records)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def train_folds(args, cache, splits, action_result, result_root: Path, config):
    device = torch.device(args.device)
    threshold = float(action_result["threshold"])
    layer_num = int(action_result["layer_num"])
    records = []
    stop = min(10, args.fold_start + (1 if args.smoke else args.fold_limit))
    for fold_id in range(args.fold_start, stop):
        split = dict(splits[fold_id])
        if args.smoke:
            split["train_indices"] = split["train_indices"][: args.smoke_train_limit]
            split["val_indices"] = split["val_indices"][: args.smoke_eval_limit]
            split["test_indices"] = split["test_indices"][: args.smoke_eval_limit]
        seed_everything(int(split["fold_seed"]), torch)
        train_bags = [cache["bags"][int(index)] for index in split["train_indices"]]
        val_bags = [cache["bags"][int(index)] for index in split["val_indices"]]
        test_bags = [cache["bags"][int(index)] for index in split["test_indices"]]
        model = GAT(
            args.dataset_key, int(cache["feature_dim"]), layer_num,
            args.dropout, args.negative_slope, device
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_acc = -1.0
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        bad_epochs = 0
        history = []
        max_epochs = 3 if args.smoke else args.epochs
        patience = 3 if args.smoke else args.patience
        started = time.time()
        for epoch in range(1, max_epochs + 1):
            train_metrics = train_epoch(
                model, optimizer, train_bags, threshold, layer_num, device
            )
            val_output = evaluate_gnn(model, val_bags, threshold, layer_num, device)
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
                f"train {args.dataset_key} fold={fold_id} epoch={epoch} "
                f"train_loss={train_metrics['loss']:.6f} val_acc={val_metrics['acc']:.6f}",
                flush=True,
            )
            improved = (
                float(val_metrics["acc"]) > best_acc
                or (
                    float(val_metrics["acc"]) == best_acc
                    and float(val_metrics["loss"]) < best_loss
                )
            )
            if improved:
                best_acc = float(val_metrics["acc"])
                best_loss = float(val_metrics["loss"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        if best_state is None:
            raise RuntimeError("No RGMIL checkpoint selected")
        model.load_state_dict(best_state)
        test_output = evaluate_gnn(model, test_bags, threshold, layer_num, device)
        record = {
            "dataset_key": args.dataset_key,
            "method": "RGMIL",
            "fold": fold_id,
            "fold_seed": int(split["fold_seed"]),
            "split_digest": split_digest(split),
            "best_epoch": best_epoch,
            "threshold": threshold,
            "layer_num": layer_num,
            "train_n": len(train_bags),
            "val_n": len(val_bags),
            "test_n": len(test_bags),
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
                "threshold": threshold,
                "layer_num": layer_num,
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
                "output_length_matches_test": bool(len(test_output["labels"]) == len(test_bags)),
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


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    entry = load_dataset_entry(args.execution_manifest, args.dataset_key)
    cache_path = Path(args.cache_root) / args.dataset_key / "bags.pkl"
    if args.rebuild_cache and cache_path.exists():
        cache_path.unlink()
    cache = prepare_cache(entry, cache_path, args.max_nodes_per_bag, args.seed)
    splits = load_fixed_folds(entry["split_manifest"], len(cache["labels"]))
    canonical_folds = []
    import json
    with Path(entry["split_manifest"]).open(encoding="utf-8") as handle:
        raw_manifest = json.load(handle)
    canonical_folds = [np.asarray(fold["sample_indices"], dtype=np.int64) for fold in raw_manifest["folds"]]
    for split in splits:
        split["canonical_folds"] = canonical_folds
    result_root = Path(args.result_root) / args.dataset_key
    if args.smoke:
        result_root = result_root / "smoke"
    result_root.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "RGMIL-VDN",
        "paper": "Zhao et al. (2024)",
        "protocol": PROTOCOL_NAME,
        "dataset_key": args.dataset_key,
        "data_name": entry["data_name"],
        "dataset": dataset_fingerprint(entry["pkl_path"]),
        "split_manifest": entry["split_manifest"],
        "input_adaptation": {
            "bag": "one dataset subgraph",
            "instance": "one node feature vector",
            "source_graph_edges_used": False,
            "internal_bag_graph": "exp(-pairwise Euclidean distance), threshold selected by VDN",
            "max_nodes_per_bag": args.max_nodes_per_bag,
            "sampled_bags": cache["sampled_bags"],
        },
        "search": {
            "method": "two-agent VDN",
            "only_fold": 0,
            "threshold_space": THRESHOLD_SPACE,
            "layer_space": LAYER_SPACE,
            "max_steps": 2 if args.smoke else args.search_max_steps,
            "epsilon": "1 to 0 over 50 steps",
            "discount": 0.95,
            "history": 10,
            "replay_capacity": 20,
            "reward_tolerance": 1e-4,
        },
        "training": {
            "loss": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": 3 if args.smoke else args.epochs,
            "patience": 3 if args.smoke else args.patience,
            "dropout": args.dropout,
            "negative_slope": args.negative_slope,
            "selection": "validation accuracy; validation loss tie-break",
        },
        "implementation_fixes": [
            "copy similarity before thresholding to prevent action-dependent cache corruption",
            "disable functional dropout during evaluation",
            "return raw classifier logits for BCEWithLogitsLoss",
            "use dynamic transition modulo for eight exact training folds",
        ],
        "runtime": runtime_info(torch),
    }
    save_json(result_root / "config.json", config)
    if args.phase == "prepare":
        return
    search_dir = result_root / "action_search"
    action_result = search_actions(args, cache, splits[0], search_dir, config)
    if args.phase == "search":
        return
    train_folds(args, cache, splits, action_result, result_root, config)


if __name__ == "__main__":
    main()
