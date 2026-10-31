"""Syn1--Syn3 synthetic data construction, including hard negative instances."""

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .dataset_settings import benchmark_catalog, sha256

SYNTHETIC_PRESETS = {
    "synthetic_milinst_mil_strong_pos_random_v2": {
        "description": "V2: clear positive instances, "
        "mostly single-positive bags, and "
        "random coarse graph.",
        "positive_count_low": 1,
        "positive_count_high": 1,
        "positive_count_mixture": [
            {"prob": 0.78, "low": 1, "high": 1, "name": "single_positive_bag"},
            {"prob": 0.22, "low": 9, "high": 12, "name": "dense_positive_bag"},
        ],
        "instance_signal": 2.25,
        "bag_noise": 0.55,
        "node_noise": 0.55,
        "hard_negative_rate": 0.02,
        "hard_negative_strength": 0.25,
        "same_neighbors": 8,
        "diff_neighbors": 8,
        "seed": 2026081001,
    },
    "synthetic_milinst_mil_weak_pos_strong_v2": {
        "description": "V2 control: mixed-count weak "
        "positive instances and a strongly "
        "homophilic coarse graph.",
        "positive_count_low": 1,
        "positive_count_high": 1,
        "positive_count_mixture": [
            {"prob": 0.78, "low": 1, "high": 1, "name": "single_positive_bag"},
            {"prob": 0.22, "low": 9, "high": 12, "name": "dense_positive_bag"},
        ],
        "instance_signal": 0.8,
        "bag_noise": 0.8,
        "node_noise": 1.05,
        "hard_negative_rate": 0.14,
        "hard_negative_strength": 0.55,
        "same_neighbors": 14,
        "diff_neighbors": 2,
        "seed": 2026081002,
    },
    "synthetic_milinst_both_useful_v2": {
        "description": "V2: clear but diluted internal evidence and "
        "a moderately homophilic coarse graph.",
        "positive_count_low": 1,
        "positive_count_high": 1,
        "positive_count_mixture": [
            {"prob": 0.78, "low": 1, "high": 1, "name": "single_positive_bag"},
            {"prob": 0.22, "low": 9, "high": 12, "name": "dense_positive_bag"},
        ],
        "instance_signal": 2.0,
        "bag_noise": 0.55,
        "node_noise": 0.65,
        "hard_negative_rate": 0.04,
        "hard_negative_strength": 0.35,
        "same_neighbors": 12,
        "diff_neighbors": 4,
        "seed": 2026081003,
    },
}


def add_internal_edges(graph, node_ids):
    n = len(node_ids)
    for i in range(n):
        graph.add_edge(node_ids[i], node_ids[(i + 1) % n], weight=1.0)
        graph.add_edge(node_ids[i], node_ids[(i + 3) % n], weight=1.0)


def choose_neighbors(rng, labels, same_k, diff_k):
    n = labels.shape[0]
    neighbors = [set() for _ in range(n)]
    by_label = {0: np.flatnonzero(labels == 0), 1: np.flatnonzero(labels == 1)}
    for i in range(n):
        same_pool = by_label[int(labels[i])]
        same_pool = same_pool[same_pool != i]
        diff_pool = by_label[1 - int(labels[i])]
        same = rng.choice(
            same_pool, size=min(int(same_k), same_pool.size), replace=False
        )
        diff = rng.choice(
            diff_pool, size=min(int(diff_k), diff_pool.size), replace=False
        )
        for j in np.concatenate([same, diff]):
            neighbors[i].add(int(j))
            neighbors[int(j)].add(i)
    return neighbors


def positive_instance_count(rng, label, low, high, mixture=None):
    if int(label) == 0:
        return 0, "negative_bag"
    if mixture:
        probs = np.asarray([float(item["prob"]) for item in mixture], dtype=np.float64)
        probs = probs / probs.sum()
        choice = int(rng.choice(len(mixture), p=probs))
        item = mixture[choice]
        return int(rng.integers(int(item["low"]), int(item["high"]) + 1)), str(
            item.get("name", f"mode_{choice}")
        )
    return int(rng.integers(int(low), int(high) + 1)), "positive_bag"


def build_synthetic_dataset(args, cfg):
    rng = np.random.default_rng(int(cfg["seed"]))
    n_sub = int(args.num_subgraphs)
    n_per = int(args.nodes_per_subgraph)
    fdim = int(args.feature_dim)
    total_nodes = n_sub * n_per

    bag_labels = np.zeros(n_sub, dtype=np.int64)
    bag_labels[n_sub // 2 :] = 1
    rng.shuffle(bag_labels)

    semantic = rng.normal(size=fdim).astype(np.float32)
    semantic /= np.linalg.norm(semantic) + 1e-12

    original_graph = nx.Graph()
    subgraphs = []
    node_features = np.zeros((total_nodes, fdim), dtype=np.float32)
    instance_labels = np.zeros(total_nodes, dtype=np.int64)
    node_categories = {}
    subgraph_assignment = np.empty(total_nodes, dtype=np.int64)
    positive_counts = np.zeros(n_sub, dtype=np.int64)
    positive_count_modes = []
    hard_negative_flags = np.zeros(total_nodes, dtype=np.int64)

    for sid in range(n_sub):
        bag_label = int(bag_labels[sid])
        pos_count, pos_mode = positive_instance_count(
            rng,
            bag_label,
            cfg["positive_count_low"],
            cfg["positive_count_high"],
            cfg.get("positive_count_mixture"),
        )
        positive_counts[sid] = pos_count
        positive_count_modes.append(pos_mode)
        pos_local_indices = set(
            rng.choice(n_per, size=pos_count, replace=False).astype(int).tolist()
        )
        bag_noise = rng.normal(scale=float(cfg["bag_noise"]), size=fdim).astype(
            np.float32
        )

        local = nx.Graph()
        local_nodes = []
        for k in range(n_per):
            nid = sid * n_per + k
            z = 1 if k in pos_local_indices else 0
            instance_labels[nid] = z
            if z == 1:
                center = float(cfg["instance_signal"]) * semantic
            else:
                hard = rng.random() < float(cfg["hard_negative_rate"])
                hard_negative_flags[nid] = int(hard)
                if hard:
                    center = (
                        float(cfg["hard_negative_strength"])
                        * float(cfg["instance_signal"])
                        * semantic
                    )
                else:
                    center = -float(cfg["instance_signal"]) * semantic
            feat = (
                center
                + bag_noise
                + rng.normal(scale=float(cfg["node_noise"]), size=fdim).astype(
                    np.float32
                )
            )
            node_features[nid] = feat.astype(np.float32)
            node_categories[nid] = int(z)
            subgraph_assignment[nid] = sid
            original_graph.add_node(
                nid,
                features=node_features[nid],
                instance_label=int(z),
                bag_label=bag_label,
            )
            local.add_node(
                nid,
                features=node_features[nid],
                instance_label=int(z),
                bag_label=bag_label,
            )
            local_nodes.append(nid)

        add_internal_edges(original_graph, local_nodes)
        add_internal_edges(local, local_nodes)
        local.graph["label"] = int(instance_labels[local_nodes].max())
        local.graph["subgraph_id"] = sid
        local.graph["orig_idx"] = sid
        local.graph["positive_instance_count"] = int(pos_count)
        local.graph["positive_instance_fraction"] = float(pos_count / n_per)
        local.graph["positive_count_mode"] = pos_mode
        subgraphs.append(local)

    derived_bag_labels = np.asarray(
        [int(g.graph["label"]) for g in subgraphs], dtype=np.int64
    )
    if not np.array_equal(derived_bag_labels, bag_labels):
        raise RuntimeError(
            "MIL consistency violated: bag label does not equal max instance label."
        )

    coarse_neighbors = choose_neighbors(
        rng, bag_labels, cfg["same_neighbors"], cfg["diff_neighbors"]
    )
    for i, nbrs in enumerate(coarse_neighbors):
        for j in nbrs:
            if i >= j:
                continue
            for _ in range(int(args.cross_edges_per_pair)):
                u = i * n_per + int(rng.integers(n_per))
                v = j * n_per + int(rng.integers(n_per))
                original_graph.add_edge(u, v, weight=1.0)

    rows = np.arange(total_nodes, dtype=np.int64)
    cols = subgraph_assignment.astype(np.int64)
    assignment_matrix = sp.csr_matrix(
        (np.ones(total_nodes, dtype=np.float32), (rows, cols)),
        shape=(total_nodes, n_sub),
    )

    indices = np.arange(n_sub, dtype=np.int64)
    rng.shuffle(indices)
    train_cut = int(round(n_sub * 0.8))

    metadata = {
        "dataset_name": args.preset,
        "dataset_type": "synthetic_mil_consistent",
        "description": cfg["description"],
        "num_subgraphs": int(n_sub),
        "total_nodes": int(total_nodes),
        "total_edges": int(original_graph.number_of_edges()),
        "nodes_per_subgraph": int(n_per),
        "feature_dim": int(fdim),
        "max_num_nodes": int(n_per),
        "target_category": 1,
        "positive_count_low": int(cfg["positive_count_low"]),
        "positive_count_high": int(cfg["positive_count_high"]),
        "positive_count_mixture": cfg.get("positive_count_mixture"),
        "instance_signal": float(cfg["instance_signal"]),
        "bag_noise": float(cfg["bag_noise"]),
        "node_noise": float(cfg["node_noise"]),
        "hard_negative_rate": float(cfg["hard_negative_rate"]),
        "hard_negative_strength": float(cfg["hard_negative_strength"]),
        "same_neighbors": int(cfg["same_neighbors"]),
        "diff_neighbors": int(cfg["diff_neighbors"]),
        "cross_edges_per_pair": int(args.cross_edges_per_pair),
        "seed": int(cfg["seed"]),
        "semantic_direction": semantic.astype(float).tolist(),
    }
    return {
        "subgraph_labels": bag_labels.astype(int).tolist(),
        "original_graph": original_graph,
        "node_features": node_features,
        "subgraph_structures": subgraphs,
        "assignment_matrix": assignment_matrix,
        "subgraph_assignment": subgraph_assignment,
        "dataset_metadata": metadata,
        "node_categories": node_categories,
        "train_test_split": {
            "train_indices": sorted(indices[:train_cut].astype(int).tolist()),
            "test_indices": sorted(indices[train_cut:].astype(int).tolist()),
        },
        "node_binary_labels": instance_labels.astype(int).tolist(),
        "instance_labels": instance_labels.astype(int).tolist(),
        "hard_negative_flags": hard_negative_flags.astype(int).tolist(),
        "positive_instance_counts": positive_counts.astype(int).tolist(),
        "positive_count_modes": positive_count_modes,
        "feature_dimension": int(fdim),
        "subgraph_node_counts": [n_per] * n_sub,
        "group_ids": list(range(n_sub)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("syn1", "syn2", "syn3"),
        default=["syn1", "syn2", "syn3"],
    )
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)
    catalog = benchmark_catalog()
    for key in args.datasets:
        entry = catalog[key]
        folder = Path(args.data_root) / entry["directory"]
        path = folder / (entry["dataset_name"] + "_processed.pkl")
        if path.exists():
            raise FileExistsError("Refusing to overwrite " + str(path))
        options = SimpleNamespace(
            preset=entry["dataset_name"],
            num_subgraphs=1000,
            nodes_per_subgraph=32 if key == "syn1" else 16,
            feature_dim=32,
            cross_edges_per_pair=2,
        )
        dataset = build_synthetic_dataset(
            options, SYNTHETIC_PRESETS[entry["dataset_name"]]
        )
        folder.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "parameters": dataset["dataset_metadata"],
            "processed_sha256": sha256(path),
            "original_processed_sha256": entry["processed_sha256"],
        }
        (folder / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(key, path, metadata["processed_sha256"])


if __name__ == "__main__":
    main()
