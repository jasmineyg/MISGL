"""Read IGB-tiny, OGBN-Arxiv/Products and Reddit, then construct MISGL data."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .dataset_settings import benchmark_catalog, real_graph_settings
from .dataset_export import export_partitioned_dataset
from .graph_partition import partition_graph


@dataclass
class GraphDataset:
    """A graph plus row-aligned features and node labels.

    ``partition[i]`` always describes ``nodes[i]``.  This ordering invariant is
    shared by every partitioner and by the processed-dataset writer.
    """

    name: str
    graph: nx.Graph
    nodes: List[int]
    node_features: np.ndarray
    node_categories: Dict[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "GraphDataset":
        node_count = len(self.nodes)
        if self.node_features.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature matrix, got shape {self.node_features.shape}"
            )
        if self.node_features.shape[0] != node_count:
            raise ValueError(
                "Feature/node mismatch: "
                f"features={self.node_features.shape[0]}, nodes={node_count}"
            )
        if len(self.node_categories) != node_count:
            raise ValueError(
                "Label/node mismatch: "
                f"labels={len(self.node_categories)}, nodes={node_count}"
            )
        if set(self.graph.nodes()) != set(self.nodes):
            raise ValueError("Graph nodes do not match the ordered node list")
        missing_labels = [
            node for node in self.nodes if node not in self.node_categories
        ]
        if missing_labels:
            raise ValueError(f"Missing labels for nodes: {missing_labels[:10]}")
        return self


# IGB-tiny processed arrays.

IGB_DEFAULT_PATH = "raw_data/igb-tiny"


IGB_LABEL_FILE = "node_label_19.npy"


def load_igb_tiny(
    name: str = "igb-tiny", dataset_path: str = None, label_file: str = None
):
    """Load IGB-tiny while preserving row-order node ids and an undirected graph."""
    dataset_path = Path(dataset_path or IGB_DEFAULT_PATH)
    processed_dir = dataset_path / "processed"
    paper_dir = processed_dir / "paper"
    edge_path = processed_dir / "paper__cites__paper" / "edge_index.npy"
    feature_path = paper_dir / "node_feat.npy"
    label_file = label_file or IGB_LABEL_FILE
    label_path = paper_dir / label_file

    required = (edge_path, feature_path, label_path)
    missing_paths = [str(path) for path in required if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required IGB-tiny files: {missing_paths}")

    node_features = np.load(feature_path, allow_pickle=False)
    raw_labels = np.load(label_path, allow_pickle=False)
    edges = np.load(edge_path, allow_pickle=False)

    if node_features.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix in {feature_path}, got {node_features.shape}"
        )
    num_nodes = int(node_features.shape[0])
    if raw_labels.ndim != 1 or raw_labels.shape[0] != num_nodes:
        raise ValueError(
            f"Feature/label mismatch: features={node_features.shape}, labels={raw_labels.shape}"
        )
    if not np.all(np.isfinite(raw_labels)):
        raise ValueError(f"Labels contain non-finite values: {label_path}")
    labels = np.asarray(raw_labels, dtype=np.int64)
    if not np.array_equal(raw_labels, labels):
        raise ValueError(f"Labels must be integer-valued: {label_path}")

    if edges.ndim != 2:
        raise ValueError(f"Expected a 2D edge array in {edge_path}, got {edges.shape}")
    if edges.shape[1] == 2:
        edge_rows = edges
    elif edges.shape[0] == 2:
        edge_rows = edges.T
    else:
        raise ValueError(
            f"Expected edge shape (E, 2) or (2, E) in {edge_path}, got {edges.shape}"
        )

    min_edge_index = int(edge_rows.min()) if edge_rows.size else 0
    max_edge_index = int(edge_rows.max()) if edge_rows.size else -1
    if min_edge_index < 0:
        raise ValueError(f"Edge references negative node id {min_edge_index}")
    if max_edge_index >= num_nodes:
        raise ValueError(
            f"Edge references node {max_edge_index}, but only {num_nodes} nodes are available"
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(src), int(dst)) for src, dst in edge_rows)
    nodes = list(range(num_nodes))
    node_categories = {node_id: int(label) for node_id, label in enumerate(labels)}

    return GraphDataset(
        name=name,
        graph=graph,
        nodes=nodes,
        node_features=node_features,
        node_categories=node_categories,
        metadata={
            "dataset_type": "igb",
            "dataset_path": str(dataset_path),
            "label_file": label_file,
        },
    ).validate()


# OGBN-Arxiv and OGBN-Products CSV files.

OGBN_DEFAULT_PATHS = {
    "ogbn-arxiv": "raw_data/ogbn-arxiv",
    "ogbn-products": "raw_data/ogbn-products",
}


def load_ogbn(name: str, dataset_path: str = None) -> GraphDataset:
    """Load OGBN CSV.GZ files without changing the original graph construction."""
    dataset_path = Path(dataset_path or OGBN_DEFAULT_PATHS[name])
    raw_dir = dataset_path / "raw"
    edge_path = raw_dir / "edge.csv.gz"
    feature_path = raw_dir / "node-feat.csv.gz"
    label_path = raw_dir / "node-label.csv.gz"

    required = (edge_path, feature_path, label_path)
    missing_paths = [str(path) for path in required if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required raw files: {missing_paths}")

    edge_frame = pd.read_csv(edge_path, header=None, dtype=np.int64)
    if edge_frame.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns in {edge_path}, found {edge_frame.shape[1]}"
        )
    edges = edge_frame.to_numpy(dtype=np.int64, copy=False)

    node_features = pd.read_csv(feature_path, header=None, dtype=np.float32).to_numpy(
        copy=False
    )
    if node_features.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix in {feature_path}, got {node_features.shape}"
        )

    label_frame = pd.read_csv(label_path, header=None, dtype=np.int64)
    if label_frame.shape[1] != 1:
        raise ValueError(
            f"Expected 1 label column in {label_path}, found {label_frame.shape[1]}"
        )
    labels = label_frame.iloc[:, 0].to_numpy(dtype=np.int64, copy=False)

    if node_features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Feature/label row mismatch: "
            f"features={node_features.shape[0]}, labels={labels.shape[0]}"
        )

    num_nodes = node_features.shape[0]
    max_edge_index = int(edges.max()) if edges.size else -1
    min_edge_index = int(edges.min()) if edges.size else 0
    if min_edge_index < 0:
        raise ValueError(f"Edge references negative node id {min_edge_index}")
    if max_edge_index >= num_nodes:
        raise ValueError(
            f"Edge references node {max_edge_index}, but only {num_nodes} nodes are available"
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(map(tuple, edges))
    nodes = list(range(num_nodes))
    node_categories = {node_id: int(label) for node_id, label in enumerate(labels)}

    return GraphDataset(
        name=name,
        graph=graph,
        nodes=nodes,
        node_features=node_features,
        node_categories=node_categories,
        metadata={"dataset_type": "ogbn", "dataset_path": str(dataset_path)},
    ).validate()


# Reddit feature/label NPZ and sparse adjacency.

REDDIT_DEFAULT_PATH = "raw_data/reddit"


def _validate_1d_array(name, values, expected_len):
    if values.ndim != 1:
        raise ValueError(f"Expected {name} to be 1D, got shape {values.shape}")
    if values.shape[0] != expected_len:
        raise ValueError(
            f"Expected {name} length {expected_len}, got {values.shape[0]}"
        )


def load_reddit(name: str = "reddit", dataset_path: str = None) -> GraphDataset:
    """Load PyG Reddit raw files using the validated original conversion rules."""
    dataset_path = Path(dataset_path or REDDIT_DEFAULT_PATH)
    raw_dir = dataset_path / "raw"
    data_path = raw_dir / "reddit_data.npz"
    graph_path = raw_dir / "reddit_graph.npz"

    required = (data_path, graph_path)
    missing_paths = [str(path) for path in required if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required Reddit files: {missing_paths}")

    with np.load(data_path, allow_pickle=False) as data:
        required_keys = {"feature", "label"}
        missing_keys = sorted(required_keys - set(data.files))
        if missing_keys:
            raise KeyError(f"Missing keys in {data_path}: {missing_keys}")

        node_features = data["feature"].astype(np.float32, copy=False)
        labels = data["label"].astype(np.int64, copy=False).reshape(-1)
        if node_features.ndim != 2:
            raise ValueError(
                f"Expected feature to be 2D, got shape {node_features.shape}"
            )

        num_nodes = node_features.shape[0]
        _validate_1d_array("label", labels, num_nodes)
        if "node_ids" in data.files:
            node_ids = data["node_ids"].astype(np.int64, copy=False).reshape(-1)
            _validate_1d_array("node_ids", node_ids, num_nodes)
            if not np.array_equal(node_ids, np.arange(num_nodes, dtype=np.int64)):
                raise ValueError(
                    "This loader expects Reddit node_ids to match feature row order 0..N-1"
                )

    adjacency = sp.load_npz(graph_path).tocoo()
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Graph/feature shape mismatch: graph={adjacency.shape}, features={node_features.shape}"
        )
    if adjacency.nnz and (int(adjacency.row.min()) < 0 or int(adjacency.col.min()) < 0):
        raise ValueError("Graph contains negative node ids")
    if adjacency.nnz and (
        int(adjacency.row.max()) >= num_nodes or int(adjacency.col.max()) >= num_nodes
    ):
        raise ValueError(f"Graph references node id outside 0..{num_nodes - 1}")

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (int(src), int(dst)) for src, dst in zip(adjacency.row, adjacency.col)
    )
    nodes = list(range(num_nodes))
    node_categories = {node_id: int(label) for node_id, label in enumerate(labels)}

    return GraphDataset(
        name=name,
        graph=graph,
        nodes=nodes,
        node_features=node_features,
        node_categories=node_categories,
        metadata={"dataset_type": "reddit", "dataset_path": str(dataset_path)},
    ).validate()


_ALIASES = {
    "ogbn_arxiv": "ogbn-arxiv",
    "ogbn-products": "ogbn-products",
    "ogbn_products": "ogbn-products",
    "ogbn-arxiv": "ogbn-arxiv",
    "reddit": "reddit",
    "igb-tiny": "igb-tiny",
    "igb_tiny": "igb-tiny",
}


def normalize_dataset_name(name: str) -> str:
    normalized = str(name).strip().lower()
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_ALIASES.values())))
        raise ValueError(
            f"Unsupported dataset {name!r}; choose one of: {supported}"
        ) from exc


def load_real_graph(config) -> GraphDataset:
    """Load one configured dataset through a stable interface."""
    name = normalize_dataset_name(config["dataset"])
    dataset_path = config.get("dataset_path")
    if name == "reddit":
        return load_reddit(name, dataset_path)
    if name == "igb-tiny":
        return load_igb_tiny(name, dataset_path, config.get("label_file"))
    return load_ogbn(name, dataset_path)


SUPPORTED_DATASETS = tuple(sorted(set(_ALIASES.values())))


# Real-dataset construction CLI.


def paper_dataset_config(
    benchmark, raw_dir, data_root="data", reference_partition=None
):
    """Resolve a named paper setting and its model-compatible output filename."""
    settings = real_graph_settings()
    if benchmark not in settings:
        raise ValueError(f"Unknown real-graph benchmark: {benchmark}")
    config = dict(settings[benchmark])
    entry = benchmark_catalog()[benchmark]
    config.update(
        dataset_path=str(raw_dir),
        output_name=entry["dataset_name"],
        output_dir=str(Path(data_root) / entry["directory"]),
        write_processed_pickle=True,
    )
    if reference_partition is not None:
        if config["partition_method"] != "metis_perturb":
            raise ValueError(
                "--reference-partition applies to Perturb-50 settings only"
            )
        config["reference_partition"] = str(reference_partition)
    elif config["partition_method"] == "metis_perturb":
        # Require the already constructed METIS assignment, so both settings
        # share exactly the same reference rather than invoking METIS twice.
        reference_key = benchmark.replace("metis_perturb_50", "metis")
        reference = benchmark_catalog()[reference_key]
        config["reference_partition"] = str(
            Path(data_root)
            / reference["directory"]
            / (reference["dataset_name"] + "_partition.npz")
        )
    return config


def build_real_dataset(config):
    """Load, partition and export one graph without replacing existing outputs."""
    method = config["partition_method"]
    name = config["output_name"]
    output_dir = Path(config["output_dir"])
    for suffix in ("_processed.pkl", "_summary.json", "_partition.npz"):
        path = output_dir / (name + suffix)
        if path.exists():
            raise FileExistsError("Refusing to overwrite " + str(path))
    if method == "metis_perturb" and not Path(config["reference_partition"]).is_file():
        raise FileNotFoundError(
            "Construct METIS first or supply --reference-partition: "
            + config["reference_partition"]
        )
    dataset = load_real_graph(config)
    num_parts = int(config["num_partitions"])
    if not 0 < num_parts <= len(dataset.nodes):
        raise ValueError("num_partitions must be between 1 and the number of nodes")
    result = partition_graph(dataset, method, config)
    # The training schema aligns bag IDs with membership-matrix columns.
    # Do not silently drop empty partitions and change that correspondence.
    if np.any(np.bincount(result.assignments, minlength=num_parts) == 0):
        raise ValueError("Empty partitions cannot be exported with aligned bag IDs")
    return export_partitioned_dataset(dataset, result, method, config)


def main(argv=None):
    settings = real_graph_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(settings))
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Dataset root containing raw/ (OGBN/Reddit) or processed/ (IGB)",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--reference-partition",
        help="Existing METIS assignment .npz or processed .pkl for Perturb-50",
    )
    args = parser.parse_args(argv)
    config = paper_dataset_config(
        args.benchmark, args.raw_dir, args.data_root, args.reference_partition
    )
    for name, path in build_real_dataset(config).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
