"""Build MIL labels and export processed subgraphs, assignments and metadata."""

import gc
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

SUBGRAPH_TRAIN_RATIO = 0.8


def build_partition_node_indices(partition, num_parts=None):
    """Build a node-index list for each partition exactly once."""
    partition_array = np.asarray(partition, dtype=np.int64)
    if partition_array.ndim != 1:
        raise ValueError("partition must be a 1D array-like object")

    if num_parts is None:
        num_parts = int(partition_array.max()) + 1 if partition_array.size else 0

    part_node_indices = [[] for _ in range(num_parts)]
    for node_idx, part_id in enumerate(partition_array):
        part_node_indices[int(part_id)].append(node_idx)

    return partition_array, part_node_indices


def build_partition_category_counts(nodes, node_categories, part_node_indices):
    """Count category frequency per partition for reuse across analyses."""
    part_category_counts = []

    for node_indices in part_node_indices:
        category_counts = defaultdict(int)
        for node_idx in node_indices:
            category_counts[node_categories[nodes[node_idx]]] += 1
        part_category_counts.append(category_counts)

    return part_category_counts


def calculate_max_subgraph_size(partition, num_parts):
    """Return the largest partition size."""
    partition_array = np.asarray(partition, dtype=np.int64)
    if partition_array.size == 0 or num_parts == 0:
        return 0
    return int(np.bincount(partition_array, minlength=num_parts).max())


def create_assignment_matrix(partition, num_parts):
    """Create a sparse one-hot assignment matrix from partition ids."""
    partition_array = np.asarray(partition, dtype=np.int64)
    if partition_array.size == 0:
        return csr_matrix((0, num_parts), dtype=np.uint8)

    row_indices = np.arange(partition_array.size, dtype=np.int64)
    data = np.ones(partition_array.size, dtype=np.uint8)
    return csr_matrix(
        (data, (row_indices, partition_array)),
        shape=(partition_array.size, num_parts),
        dtype=np.uint8,
    )


def find_optimal_target_category(
    nodes,
    partition,
    node_categories,
    num_parts=None,
    part_node_indices=None,
    part_category_counts=None,
):
    """Choose the target category with the most balanced positive/negative subgraphs."""
    if part_node_indices is None:
        _, part_node_indices = build_partition_node_indices(partition, num_parts)
    if part_category_counts is None:
        part_category_counts = build_partition_category_counts(
            nodes, node_categories, part_node_indices
        )

    non_empty_category_counts = [counts for counts in part_category_counts if counts]
    if not non_empty_category_counts:
        raise ValueError("node categories are empty, cannot choose a target category")

    unique_categories = sorted(
        {
            category
            for category_counts in non_empty_category_counts
            for category in category_counts
        }
    )
    total_subgraphs = len(non_empty_category_counts)
    category_presence = defaultdict(int)
    for category_counts in non_empty_category_counts:
        for category in category_counts:
            category_presence[category] += 1

    category_stats = {}
    best_category = None
    best_balance_score = float("inf")

    for category in unique_categories:
        positive_count = int(category_presence.get(category, 0))
        negative_count = total_subgraphs - positive_count
        balance_score = abs(positive_count - negative_count)
        category_stats[category] = {
            "positive_subgraphs": positive_count,
            "negative_subgraphs": negative_count,
            "balance_score": balance_score,
            "total_subgraphs": total_subgraphs,
        }

        if balance_score < best_balance_score:
            best_category = category
            best_balance_score = balance_score

    return best_category, category_stats


def create_subgraph_labels(
    nodes,
    partition,
    node_categories,
    target_category,
    num_parts=None,
    part_node_indices=None,
    part_category_counts=None,
):
    """Create binary subgraph labels for the chosen target category."""
    if part_node_indices is None:
        _, part_node_indices = build_partition_node_indices(partition, num_parts)
    if part_category_counts is None:
        part_category_counts = build_partition_category_counts(
            nodes, node_categories, part_node_indices
        )

    return [
        int(part_category_counts[part_id].get(target_category, 0) > 0)
        for part_id, node_indices in enumerate(part_node_indices)
        if node_indices
    ]


def create_subgraph_structures(
    graph,
    nodes,
    partition,
    node_features,
    node_categories,
    subgraph_labels,
    num_parts=None,
    part_node_indices=None,
):
    """Build subgraph objects with node features and metadata."""
    if part_node_indices is None:
        _, part_node_indices = build_partition_node_indices(partition, num_parts)

    subgraph_structures = []
    subgraph_id = 0

    for node_indices in part_node_indices:
        if not node_indices:
            continue

        part_nodes = [nodes[idx] for idx in node_indices]
        subgraph = graph.subgraph(part_nodes).copy()

        for node_idx, node in zip(node_indices, part_nodes):
            subgraph.nodes[node]["features"] = node_features[node_idx]
            subgraph.nodes[node]["category"] = node_categories[node]
            subgraph.nodes[node]["original_id"] = node

        subgraph.graph["label"] = subgraph_labels[subgraph_id]
        subgraph.graph["subgraph_id"] = subgraph_id
        subgraph_structures.append(subgraph)
        subgraph_id += 1

    return subgraph_structures


def create_train_test_split(
    subgraph_labels, train_ratio=SUBGRAPH_TRAIN_RATIO, random_seed=42
):
    """Create a stratified train/test split over subgraph labels."""
    np.random.seed(random_seed)

    positive_indices = [idx for idx, label in enumerate(subgraph_labels) if label == 1]
    negative_indices = [idx for idx, label in enumerate(subgraph_labels) if label == 0]
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    pos_train_size = int(len(positive_indices) * train_ratio)
    neg_train_size = int(len(negative_indices) * train_ratio)

    train_indices = (
        positive_indices[:pos_train_size] + negative_indices[:neg_train_size]
    )
    test_indices = positive_indices[pos_train_size:] + negative_indices[neg_train_size:]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def create_dataset_metadata(
    data_name,
    graph,
    nodes,
    node_features,
    subgraph_structures,
    target_category,
    subgraph_labels,
    category_stats,
    partition,
    num_parts,
    node_categories,
):
    """Collect dataset metadata for downstream use."""
    categories = np.fromiter(
        node_categories.values(), dtype=np.int64, count=len(node_categories)
    )
    unique_categories, counts = np.unique(categories, return_counts=True)
    category_distribution = {
        int(category): int(count) for category, count in zip(unique_categories, counts)
    }
    positive_subgraphs = int(sum(subgraph_labels))

    return {
        "dataset_name": data_name,
        "total_nodes": len(nodes),
        "total_edges": graph.number_of_edges(),
        "num_subgraphs": len(subgraph_structures),
        "avg_subgraph_size": (
            len(nodes) / len(subgraph_structures) if subgraph_structures else 0
        ),
        "max_num_nodes": calculate_max_subgraph_size(partition, num_parts),
        "feature_dim": node_features.shape[1],
        "num_categories": len(category_distribution),
        "category_distribution": category_distribution,
        "target_category": target_category,
        "positive_subgraphs": positive_subgraphs,
        "negative_subgraphs": len(subgraph_labels) - positive_subgraphs,
        "category_balance_stats": category_stats,
    }


def build_category_stats(part_category_counts):
    category_presence = {}
    non_empty_counts = [counts for counts in part_category_counts if counts]
    total_subgraphs = len(non_empty_counts)
    for counts in non_empty_counts:
        for category in counts:
            category_presence[category] = category_presence.get(category, 0) + 1

    category_stats = {}
    for category in sorted(category_presence):
        positive_count = int(category_presence[category])
        negative_count = total_subgraphs - positive_count
        category_stats[int(category)] = {
            "positive_subgraphs": positive_count,
            "negative_subgraphs": int(negative_count),
            "balance_score": int(abs(positive_count - negative_count)),
            "total_subgraphs": int(total_subgraphs),
        }
    return category_stats


def _target_label_mask(dataset, target_category):
    return np.asarray(
        [
            int(dataset.node_categories[node] == target_category)
            for node in dataset.nodes
        ],
        dtype=np.bool_,
    )


def _part_positive_counts(partition, target_label_mask, num_parts):
    counts = np.zeros(num_parts, dtype=np.int64)
    np.add.at(counts, partition[target_label_mask], 1)
    return counts


def _positive_count_histogram(positive_counts):
    unique_counts, frequencies = np.unique(positive_counts, return_counts=True)
    return {int(count): int(freq) for count, freq in zip(unique_counts, frequencies)}


def calculate_label_profile(partition, dataset, target_category, num_parts):
    target_label_mask = _target_label_mask(dataset, target_category)
    positive_counts = _part_positive_counts(partition, target_label_mask, num_parts)
    positive_subgraph_mask = positive_counts > 0
    positive_subgraph_counts = positive_counts[positive_subgraph_mask]
    positive_subgraphs = int(np.count_nonzero(positive_subgraph_mask))
    negative_subgraphs = int(num_parts - positive_subgraphs)

    if positive_subgraph_counts.size:
        quantiles = np.percentile(positive_subgraph_counts, [25, 50, 75]).tolist()
        minimum = int(positive_subgraph_counts.min())
        maximum = int(positive_subgraph_counts.max())
        mean = float(positive_subgraph_counts.mean())
        std = float(positive_subgraph_counts.std())
    else:
        quantiles = [0.0, 0.0, 0.0]
        minimum = maximum = 0
        mean = std = 0.0

    return {
        "target_category": int(target_category),
        "total_positive_nodes": int(target_label_mask.sum()),
        "positive_subgraphs": positive_subgraphs,
        "negative_subgraphs": negative_subgraphs,
        "positive_subgraph_ratio": (
            float(positive_subgraphs / num_parts) if num_parts else 0.0
        ),
        "negative_subgraph_ratio": (
            float(negative_subgraphs / num_parts) if num_parts else 0.0
        ),
        "positive_nodes_per_positive_subgraph_min": minimum,
        "positive_nodes_per_positive_subgraph_mean": mean,
        "positive_nodes_per_positive_subgraph_std": std,
        "positive_nodes_per_positive_subgraph_q25": float(quantiles[0]),
        "positive_nodes_per_positive_subgraph_median": float(quantiles[1]),
        "positive_nodes_per_positive_subgraph_q75": float(quantiles[2]),
        "positive_nodes_per_positive_subgraph_max": maximum,
        "positive_count_histogram": _positive_count_histogram(positive_counts),
        "positive_counts_by_part": positive_counts.astype(int).tolist(),
    }


def compare_label_profile(label_profile, reference_label_profile):
    current_counts = np.asarray(
        label_profile["positive_counts_by_part"], dtype=np.int64
    )
    reference_counts = np.asarray(
        reference_label_profile["positive_counts_by_part"], dtype=np.int64
    )
    current_hist = label_profile["positive_count_histogram"]
    reference_hist = reference_label_profile["positive_count_histogram"]
    histogram_keys = sorted(set(current_hist) | set(reference_hist))
    histogram_l1 = sum(
        abs(current_hist.get(key, 0) - reference_hist.get(key, 0))
        for key in histogram_keys
    )
    return {
        "reference_positive_subgraphs": int(
            reference_label_profile["positive_subgraphs"]
        ),
        "positive_subgraphs_delta": int(
            label_profile["positive_subgraphs"]
            - reference_label_profile["positive_subgraphs"]
        ),
        "positive_subgraph_ratio_delta": float(
            label_profile["positive_subgraph_ratio"]
            - reference_label_profile["positive_subgraph_ratio"]
        ),
        "positive_count_histogram_l1_delta": int(histogram_l1),
        "positive_count_by_part_max_abs_delta": int(
            np.max(np.abs(current_counts - reference_counts))
        ),
        "positive_count_by_part_mean_abs_delta": float(
            np.mean(np.abs(current_counts - reference_counts))
        ),
        "positive_count_profile_exact_match": bool(
            np.array_equal(current_counts, reference_counts)
        ),
    }


def calculate_partition_statistics(
    dataset, partition, num_parts, part_node_indices=None
):
    partition = np.asarray(partition, dtype=np.int64)
    node_to_idx = {node: idx for idx, node in enumerate(dataset.nodes)}
    part_sizes = np.bincount(partition, minlength=num_parts)
    non_empty_sizes = part_sizes[part_sizes > 0]
    non_empty_parts = set(np.flatnonzero(part_sizes > 0).tolist())
    if part_node_indices is None:
        _, part_node_indices = build_partition_node_indices(partition, num_parts)

    cross_edges = 0
    coarse_edges = set()
    coarse_degree_parts = set()
    for src, dst in dataset.graph.edges():
        src_part = int(partition[node_to_idx[src]])
        dst_part = int(partition[node_to_idx[dst]])
        if src_part == dst_part:
            continue
        cross_edges += 1
        edge = (src_part, dst_part) if src_part < dst_part else (dst_part, src_part)
        coarse_edges.add(edge)
        coarse_degree_parts.add(src_part)
        coarse_degree_parts.add(dst_part)

    total_edges = dataset.graph.number_of_edges()
    original_isolated_nodes = sum(
        1 for node in dataset.nodes if dataset.graph.degree(node) == 0
    )
    num_subgraphs = int(non_empty_sizes.size)
    isolated_coarse_subgraphs = len(non_empty_parts - coarse_degree_parts)
    edgeless_subgraphs = 0
    internal_isolated_nodes = 0
    for part_id in non_empty_parts:
        part_nodes = [dataset.nodes[idx] for idx in part_node_indices[part_id]]
        subgraph = dataset.graph.subgraph(part_nodes)
        if subgraph.number_of_edges() == 0:
            edgeless_subgraphs += 1
        internal_isolated_nodes += sum(
            1 for _, degree in subgraph.degree() if degree == 0
        )

    if num_subgraphs:
        size_quantiles = np.percentile(non_empty_sizes, [25, 50, 75]).tolist()
        avg_size = float(non_empty_sizes.mean())
        std_size = float(non_empty_sizes.std())
        min_size = int(non_empty_sizes.min())
        max_size = int(non_empty_sizes.max())
    else:
        size_quantiles = [0.0, 0.0, 0.0]
        avg_size = std_size = 0.0
        min_size = max_size = 0

    coarse_edge_count = len(coarse_edges)
    return {
        "requested_num_partitions": int(num_parts),
        "num_subgraphs": num_subgraphs,
        "empty_subgraphs": int(num_parts - num_subgraphs),
        "size_min": min_size,
        "size_max": max_size,
        "size_mean": avg_size,
        "size_std": std_size,
        "size_q25": float(size_quantiles[0]),
        "size_median": float(size_quantiles[1]),
        "size_q75": float(size_quantiles[2]),
        "cross_subgraph_edges": int(cross_edges),
        "cross_subgraph_edge_ratio": (
            float(cross_edges / total_edges) if total_edges else 0.0
        ),
        "coarse_graph_nodes": num_subgraphs,
        "coarse_graph_edges": int(coarse_edge_count),
        "coarse_graph_avg_degree": (
            float((2 * coarse_edge_count) / num_subgraphs) if num_subgraphs else 0.0
        ),
        "isolated_coarse_subgraphs": int(isolated_coarse_subgraphs),
        "isolated_coarse_subgraph_ratio": (
            float(isolated_coarse_subgraphs / num_subgraphs) if num_subgraphs else 0.0
        ),
        "edgeless_subgraphs": int(edgeless_subgraphs),
        "edgeless_subgraph_ratio": (
            float(edgeless_subgraphs / num_subgraphs) if num_subgraphs else 0.0
        ),
        "internal_isolated_nodes": int(internal_isolated_nodes),
        "original_isolated_nodes": int(original_isolated_nodes),
        "total_edges": int(total_edges),
    }


def _variant_name(method, config):
    if method == "metis_perturb":
        percentage = int(round(float(config.get("perturb_ratio", 0.5)) * 100))
        return f"metis_perturb_{percentage}"
    return method


def export_partitioned_dataset(dataset, result, method, config):
    """Write canonical files while preserving the original processed-pickle schema."""
    num_parts = int(config["num_partitions"])
    random_seed = int(config.get("seed", 42))
    write_pickle = bool(config.get("write_processed_pickle", True))
    partition = np.asarray(result.assignments, dtype=np.int64)
    _, part_node_indices = build_partition_node_indices(partition, num_parts)
    part_category_counts = build_partition_category_counts(
        dataset.nodes, dataset.node_categories, part_node_indices
    )
    category_stats = build_category_stats(part_category_counts)
    subgraph_labels = create_subgraph_labels(
        dataset.nodes,
        partition,
        dataset.node_categories,
        result.target_category,
        num_parts=num_parts,
        part_node_indices=part_node_indices,
        part_category_counts=part_category_counts,
    )

    if write_pickle:
        subgraph_structures = create_subgraph_structures(
            dataset.graph,
            dataset.nodes,
            partition,
            dataset.node_features,
            dataset.node_categories,
            subgraph_labels,
            num_parts=num_parts,
            part_node_indices=part_node_indices,
        )
        train_indices, test_indices = create_train_test_split(
            subgraph_labels,
            train_ratio=SUBGRAPH_TRAIN_RATIO,
            random_seed=random_seed,
        )
        assignment_matrix = create_assignment_matrix(partition, num_parts)
    else:
        subgraph_structures = [None for indices in part_node_indices if indices]
        train_indices = []
        test_indices = []
        assignment_matrix = None

    metadata = create_dataset_metadata(
        dataset.name,
        dataset.graph,
        dataset.nodes,
        dataset.node_features,
        subgraph_structures,
        result.target_category,
        subgraph_labels,
        category_stats,
        partition,
        num_parts,
        dataset.node_categories,
    )
    metadata.update(dataset.metadata)
    metadata.update(result.metadata)
    metadata["partition_method"] = method
    metadata["partition_statistics"] = calculate_partition_statistics(
        dataset, partition, num_parts, part_node_indices
    )
    label_profile = calculate_label_profile(
        partition, dataset, result.target_category, num_parts
    )
    metadata["label_profile"] = label_profile
    reference_profile = calculate_label_profile(
        result.reference_assignments, dataset, result.target_category, num_parts
    )
    reference_comparison = compare_label_profile(label_profile, reference_profile)
    reference_name = str(result.metadata.get("label_profile_reference", "metis"))
    metadata["label_profile_reference"] = reference_name
    metadata["label_profile_vs_reference"] = reference_comparison
    if reference_name == "metis":
        metadata["label_profile_vs_metis"] = reference_comparison

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        config.get("output_name") or f"{dataset.name}_{_variant_name(method, config)}"
    )
    summary_path = output_dir / f"{stem}_summary.json"
    partition_path = output_dir / f"{stem}_partition.npz"
    pickle_path = output_dir / f"{stem}_processed.pkl"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    subgraph_node_counts = (
        np.bincount(partition, minlength=num_parts).astype(int).tolist()
    )
    np.savez_compressed(
        partition_path,
        subgraph_assignment=partition,
        subgraph_node_counts=np.asarray(subgraph_node_counts, dtype=np.int64),
        target_category=np.asarray([result.target_category], dtype=np.int64),
    )

    if write_pickle:
        node_binary_labels = [
            int(dataset.node_categories[node] == result.target_category)
            for node in dataset.nodes
        ]
        complete_dataset = {
            "subgraph_labels": subgraph_labels,
            "original_graph": dataset.graph,
            "node_features": dataset.node_features,
            "subgraph_structures": subgraph_structures,
            "assignment_matrix": assignment_matrix,
            "subgraph_assignment": partition.copy(),
            "dataset_metadata": metadata,
            "node_categories": dataset.node_categories,
            "train_test_split": {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "train_ratio": SUBGRAPH_TRAIN_RATIO,
                "test_ratio": 1 - SUBGRAPH_TRAIN_RATIO,
            },
            "node_binary_labels": node_binary_labels,
            "feature_dimension": dataset.node_features.shape[1],
            "subgraph_node_counts": subgraph_node_counts,
        }
        with pickle_path.open("wb") as file:
            pickle.dump(complete_dataset, file)
        del complete_dataset, subgraph_structures, assignment_matrix
        gc.collect()
    else:
        pickle_path = None

    return {
        "summary": str(summary_path),
        "partition": str(partition_path),
        "processed_pickle": str(pickle_path) if pickle_path else "",
    }
