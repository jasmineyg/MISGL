"""Processed-pickle adapter for SubGNN with process-wide dataset reuse."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Sequence

import networkx as nx
import numpy as np
import torch


REQUIRED_KEYS = (
    "subgraph_structures",
    "subgraph_labels",
    "feature_dimension",
    "original_graph",
    "node_features",
)
N_SPLITS = 10
_PREPARED_DATASETS: Dict[str, Dict[str, Any]] = {}


def read_processed_payload(pkl_path: str) -> Dict[str, Any]:
    path = Path(pkl_path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Processed pickle is missing required keys: {missing}")
    if len(payload["subgraph_structures"]) != len(payload["subgraph_labels"]):
        raise ValueError("subgraph_structures and subgraph_labels must have equal length")
    return payload


def normalized_scalar_labels(raw_labels: Sequence[Any]) -> np.ndarray:
    labels = []
    for idx, label in enumerate(raw_labels):
        if isinstance(label, (list, tuple, set, np.ndarray)):
            raise TypeError(f"subgraph_labels[{idx}] is not scalar")
        labels.append(int(label))
    values = sorted(set(labels))
    if len(values) != 2:
        raise ValueError(f"Expected binary labels, got {values}")
    mapping = {value: idx for idx, value in enumerate(values)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def _prepare_dataset(pkl_path: str) -> Dict[str, Any]:
    key = str(Path(pkl_path).resolve())
    if key in _PREPARED_DATASETS:
        return _PREPARED_DATASETS[key]
    payload = read_processed_payload(key)
    original_graph = payload["original_graph"]
    if not isinstance(original_graph, nx.Graph):
        raise TypeError("original_graph is not a NetworkX graph")
    node_features = np.asarray(payload["node_features"], dtype=np.float32)
    feature_dim = int(payload["feature_dimension"])
    if node_features.ndim != 2 or node_features.shape[1] != feature_dim:
        raise ValueError(f"node_features must have shape (num_nodes, {feature_dim})")
    graph_nodes = list(original_graph.nodes())
    if len(graph_nodes) != len(node_features):
        raise ValueError("original_graph node count does not match node_features")
    if not graph_nodes or min(graph_nodes) != 0 or max(graph_nodes) != len(node_features) - 1:
        raise ValueError("SubGNN adapter requires contiguous zero-based original node IDs")

    mapping = {node_id: int(node_id) + 1 for node_id in graph_nodes}
    # The payload is private to this process, so in-place relabeling avoids duplicating a
    # multi-gigabyte NetworkX graph. The prepared graph is cached and reused by all folds.
    shifted_graph = nx.relabel_nodes(original_graph, mapping, copy=False)
    sample_nodes = []
    for sample_idx, graph in enumerate(payload["subgraph_structures"]):
        if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
            raise ValueError(f"subgraph_structures[{sample_idx}] is not a non-empty graph")
        nodes = [mapping[int(node_id)] for node_id in graph.nodes()]
        if any(node_id not in shifted_graph for node_id in nodes):
            raise KeyError(f"subgraph_structures[{sample_idx}] contains unknown nodes")
        sample_nodes.append(nodes)
    prepared = {
        "networkx_graph": shifted_graph,
        "node_features": torch.from_numpy(node_features),
        "feature_dim": feature_dim,
        "sample_nodes": sample_nodes,
        "labels": normalized_scalar_labels(payload["subgraph_labels"]),
    }
    _PREPARED_DATASETS[key] = prepared
    return prepared


def build_processed_pickle_spec(
    pkl_path: str,
    data_name: str,
    cache_root: str,
    split_indices: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    prepared = _prepare_dataset(pkl_path)
    labels = prepared["labels"]
    sample_nodes = prepared["sample_nodes"]
    train_indices = np.asarray(split_indices["train_indices"], dtype=np.int64)
    val_indices = np.asarray(split_indices["val_indices"], dtype=np.int64)
    test_indices = np.asarray(split_indices["test_indices"], dtype=np.int64)
    fold_id = int(split_indices["fold_id"])
    cache_dir = Path(cache_root) / data_name / f"fold_{fold_id}"
    similarities_path = cache_dir / "similarities"
    similarities_path.mkdir(parents=True, exist_ok=True)
    # A missing degree file intentionally selects exact NetworkX degree queries and avoids
    # materializing/loading a multi-million-entry JSON dictionary for every fold.
    degree_path = cache_dir / "degree_not_materialized.json"
    ego_graph_path = cache_dir / "ego_graph_not_materialized.json"
    return {
        "format": "processed_pickle",
        "data_name": data_name,
        "networkx_graph": prepared["networkx_graph"],
        "node_features": prepared["node_features"],
        "feature_dim": prepared["feature_dim"],
        "num_classes": int(labels.max()) + 1,
        "train_sub_G": [sample_nodes[index] for index in train_indices],
        "val_sub_G": [sample_nodes[index] for index in val_indices],
        "test_sub_G": [sample_nodes[index] for index in test_indices],
        "train_sub_G_label": torch.tensor(labels[train_indices], dtype=torch.long),
        "val_sub_G_label": torch.tensor(labels[val_indices], dtype=torch.long),
        "test_sub_G_label": torch.tensor(labels[test_indices], dtype=torch.long),
        "train_orig_indices": train_indices,
        "val_orig_indices": val_indices,
        "test_orig_indices": test_indices,
        "similarities_path": similarities_path,
        "degree_dict_path": degree_path,
        "shortest_paths_path": None,
        "ego_graph_path": ego_graph_path,
    }
