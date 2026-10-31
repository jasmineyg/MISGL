"""METIS, label-constrained Perturb-50 and balanced multi-source BFS partitions.

Each algorithm preserves the paper implementation, including the structure-only
isolate repair and the shared METIS reference used by Perturb-50.
"""

import math
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import dataset_export


@dataclass
class PartitionResult:
    assignments: np.ndarray
    target_category: int
    reference_assignments: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_assignments(assignments, dataset, num_parts):
    assignments = np.asarray(assignments, dtype=np.int64)
    if assignments.ndim != 1 or assignments.size != len(dataset.nodes):
        raise ValueError(
            "Partition must be a 1D assignment for every node: "
            f"shape={assignments.shape}, nodes={len(dataset.nodes)}"
        )
    if assignments.size and (assignments.min() < 0 or assignments.max() >= num_parts):
        raise ValueError(
            f"Partition ids must be within 0..{num_parts - 1}; "
            f"found {assignments.min()}..{assignments.max()}"
        )
    return assignments


def load_reference(path: str) -> Tuple[np.ndarray, Optional[int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference partition does not exist: {path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "subgraph_assignment" not in data.files:
                raise KeyError(f"Missing subgraph_assignment in {path}")
            assignments = np.asarray(data["subgraph_assignment"], dtype=np.int64)
            target = (
                int(np.asarray(data["target_category"]).reshape(-1)[0])
                if "target_category" in data.files
                else None
            )
        return assignments, target
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as file:
            dataset = pickle.load(file)
        assignments = np.asarray(dataset["subgraph_assignment"], dtype=np.int64)
        target = dataset.get("dataset_metadata", {}).get("target_category")
        return assignments, int(target) if target is not None else None
    raise ValueError("reference_partition must be a .npz, .pkl, or .pickle file")


def choose_target_category(dataset, assignments, num_parts):
    _, part_node_indices = dataset_export.build_partition_node_indices(
        assignments, num_parts
    )
    part_category_counts = dataset_export.build_partition_category_counts(
        dataset.nodes, dataset.node_categories, part_node_indices
    )
    target_category, _ = dataset_export.find_optimal_target_category(
        dataset.nodes,
        assignments,
        dataset.node_categories,
        num_parts=num_parts,
        part_node_indices=part_node_indices,
        part_category_counts=part_category_counts,
    )
    return int(target_category)


def resolve_reference(dataset, config):
    num_parts = int(config["num_partitions"])
    reference_path = config.get("reference_partition")
    if reference_path:
        assignments, target_category = load_reference(reference_path)
        assignments = validate_assignments(assignments, dataset, num_parts)
        if target_category is None:
            target_category = choose_target_category(dataset, assignments, num_parts)
        return (
            assignments,
            int(target_category),
            {
                "reference_partition": str(reference_path),
                "reference_generated": False,
            },
        )

    result = partition_metis(dataset, config)
    return (
        result.assignments,
        result.target_category,
        {
            "reference_partition": "",
            "reference_generated": True,
            **result.metadata,
        },
    )


# METIS and its structure-only internal-isolate repair.


def count_isolated_nodes(graph, nodes, partition, node_to_idx):
    """Count nodes that have no intra-partition neighbors."""
    isolated_count = 0

    for idx, node in enumerate(nodes):
        current_part = partition[idx]
        intra_degree = 0

        for neighbor in graph.neighbors(node):
            neighbor_idx = node_to_idx[neighbor]
            if partition[neighbor_idx] == current_part:
                intra_degree += 1

        if intra_degree == 0:
            isolated_count += 1

    return isolated_count


def collect_isolated_nodes(graph, nodes, partition, node_to_idx):
    """Split isolated nodes into repairable and globally isolated groups."""
    repairable_isolated_nodes = []
    globally_isolated_nodes = []

    for node_idx, node in enumerate(nodes):
        current_part = partition[node_idx]
        intra_degree = 0
        total_degree = 0

        for neighbor in graph.neighbors(node):
            total_degree += 1
            neighbor_idx = node_to_idx[neighbor]
            if partition[neighbor_idx] == current_part:
                intra_degree += 1

        if intra_degree == 0:
            if total_degree == 0:
                globally_isolated_nodes.append((node_idx, node))
            else:
                repairable_isolated_nodes.append((node_idx, node))

    return repairable_isolated_nodes, globally_isolated_nodes


def get_node_partition_connectivity(graph, node, current_part, partition, node_to_idx):
    """Return current intra-partition degree and cross-partition connections for a node."""
    intra_degree = 0
    total_degree = 0
    inter_connections = defaultdict(int)

    for neighbor in graph.neighbors(node):
        total_degree += 1
        neighbor_idx = node_to_idx[neighbor]
        neighbor_part = partition[neighbor_idx]
        if neighbor_part == current_part:
            intra_degree += 1
        else:
            inter_connections[neighbor_part] += 1

    return intra_degree, total_degree, inter_connections


def select_isolated_node_target(inter_connections, part_sizes, max_size):
    """Choose a target partition with a relaxed-capacity fallback."""
    candidates = []
    for target_part, connection_count in inter_connections.items():
        target_size = part_sizes[target_part]
        overflow = max(target_size + 1 - max_size, 0)
        candidates.append((target_part, connection_count, target_size, overflow))

    feasible_candidates = [candidate for candidate in candidates if candidate[3] == 0]
    if feasible_candidates:
        target_part, connection_count, target_size, overflow = min(
            feasible_candidates,
            key=lambda item: (-item[1], item[2], item[0]),
        )
        return {
            "target_part": target_part,
            "connection_count": connection_count,
            "target_size": target_size,
            "overflow": overflow,
            "used_relaxed_capacity": False,
        }

    target_part, connection_count, target_size, overflow = min(
        candidates,
        key=lambda item: (-item[1], item[3], item[2], item[0]),
    )
    return {
        "target_part": target_part,
        "connection_count": connection_count,
        "target_size": target_size,
        "overflow": overflow,
        "used_relaxed_capacity": True,
    }


def repair_metis_isolates(graph, nodes, partition, node_to_idx, ufactor=1.2):
    """Repair isolated nodes while preferring capacity-safe moves and falling back when needed."""
    print("\n=== 开始修复孤立点 ===")
    partition = partition.copy()
    num_parts = max(partition) + 1
    part_sizes = [0] * num_parts

    for part in partition:
        part_sizes[part] += 1

    max_size = max(1, int(len(nodes) / num_parts * ufactor))
    used_relaxed_capacity = False
    skipped_global_isolated_nodes = set()
    print(f"容量约束: max_size={max_size}, ufactor={ufactor}")

    for round_idx in range(20):
        isolated_nodes, globally_isolated_nodes = collect_isolated_nodes(
            graph,
            nodes,
            partition,
            node_to_idx,
        )

        current_global_isolated = {node for _, node in globally_isolated_nodes}
        new_global_isolated = current_global_isolated - skipped_global_isolated_nodes
        if new_global_isolated:
            skipped_global_isolated_nodes.update(new_global_isolated)
            print(
                f"第 {round_idx + 1} 轮: 跳过 {len(new_global_isolated)} 个全图孤立点"
            )

        if not isolated_nodes:
            print(f"第 {round_idx + 1} 轮: 不再存在孤立点")
            if skipped_global_isolated_nodes:
                print(
                    f"保留未修复的全图孤立点: {len(skipped_global_isolated_nodes)} 个"
                )
            print(f"孤立点修复完成，使用放宽容量约束: {used_relaxed_capacity}")
            return partition

        print(f"第 {round_idx + 1} 轮: 发现 {len(isolated_nodes)} 个孤立点")

        for node_idx, node in isolated_nodes:
            current_part = partition[node_idx]
            intra_degree, total_degree, inter_connections = (
                get_node_partition_connectivity(
                    graph,
                    node,
                    current_part,
                    partition,
                    node_to_idx,
                )
            )

            if intra_degree > 0:
                continue

            if not inter_connections:
                if total_degree == 0:
                    skipped_global_isolated_nodes.add(node)
                    continue
                raise RuntimeError(
                    f"节点 {node} 当前仍被判定为孤立点，但没有可迁移的跨分区邻居"
                )

            selected_target = select_isolated_node_target(
                inter_connections, part_sizes, max_size
            )
            target_part = selected_target["target_part"]

            if selected_target["used_relaxed_capacity"]:
                used_relaxed_capacity = True
                print(
                    "放宽容量约束以修复孤立点: "
                    f"node={node}, from_part={current_part}, to_part={target_part}, "
                    f"connections={selected_target['connection_count']}, "
                    f"target_size={selected_target['target_size']}, "
                    f"overflow_after_move={selected_target['overflow']}"
                )

            partition[node_idx] = target_part
            part_sizes[current_part] -= 1
            part_sizes[target_part] += 1

    remaining_isolated, globally_isolated_nodes = collect_isolated_nodes(
        graph,
        nodes,
        partition,
        node_to_idx,
    )
    remaining_isolated = len(remaining_isolated)
    if remaining_isolated == 0:
        if globally_isolated_nodes:
            print(f"保留未修复的全图孤立点: {len(globally_isolated_nodes)} 个")
        print(f"孤立点修复完成，使用放宽容量约束: {used_relaxed_capacity}")
        return partition

    raise RuntimeError(
        f"孤立点修复失败，仍剩余 {remaining_isolated} 个孤立点，"
        f"使用放宽容量约束: {used_relaxed_capacity}"
    )


def partition_metis(dataset, config) -> PartitionResult:
    """Run the paper METIS partition followed by internal-isolate repair."""
    try:
        import metis as metis_library
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'metis' package and METIS shared library are required for partition_method=metis"
        ) from exc

    num_parts = int(config["num_partitions"])
    random_seed = int(config.get("seed", 42))
    np.random.seed(random_seed)

    _, parts = metis_library.part_graph(dataset.graph, num_parts)
    parts = np.asarray(parts, dtype=np.int64)

    node_to_idx = {node: idx for idx, node in enumerate(dataset.nodes)}
    initial_isolated = count_isolated_nodes(
        dataset.graph, dataset.nodes, parts, node_to_idx
    )
    if initial_isolated > 0:
        parts = repair_metis_isolates(
            dataset.graph,
            dataset.nodes,
            parts,
            node_to_idx,
            ufactor=float(config.get("isolated_ufactor", 1.3)),
        )
        parts = np.asarray(parts, dtype=np.int64)

    parts = validate_assignments(parts, dataset, num_parts)
    target_category = choose_target_category(dataset, parts, num_parts)
    return PartitionResult(
        assignments=parts,
        target_category=target_category,
        reference_assignments=parts.copy(),
        metadata={
            "partition_method": "metis",
            "initial_isolated_nodes_repaired": int(initial_isolated),
            "post_optimization_enabled": False,
        },
    )


# Perturb-50: exchange assignments within each binary-label group.


def _target_label_mask(dataset, target_category):
    return np.asarray(
        [
            int(dataset.node_categories[node] == target_category)
            for node in dataset.nodes
        ],
        dtype=np.bool_,
    )


def _swap_selected_nodes_across_parts(partition, selected_nodes):
    swapped_nodes = 0
    rejected_same_partition_pairs = 0
    pos = 0
    while pos + 1 < selected_nodes.size:
        left_node = selected_nodes[pos]
        right_node = selected_nodes[pos + 1]
        if partition[left_node] == partition[right_node]:
            replacement_pos = None
            for candidate_pos in range(pos + 2, selected_nodes.size):
                candidate_node = selected_nodes[candidate_pos]
                if partition[candidate_node] != partition[left_node]:
                    replacement_pos = candidate_pos
                    break
            if replacement_pos is None:
                rejected_same_partition_pairs += 1
                pos += 2
                continue
            selected_nodes[pos + 1], selected_nodes[replacement_pos] = (
                selected_nodes[replacement_pos],
                selected_nodes[pos + 1],
            )
            right_node = selected_nodes[pos + 1]

        partition[left_node], partition[right_node] = (
            partition[right_node],
            partition[left_node],
        )
        swapped_nodes += 2
        pos += 2
    return swapped_nodes, rejected_same_partition_pairs


def perturb_within_labels(
    reference_partition,
    dataset,
    target_category,
    perturb_ratio,
    random_seed,
):
    if not 0 <= perturb_ratio <= 1:
        raise ValueError(f"perturb_ratio must be within [0, 1], got {perturb_ratio}")

    rng = np.random.default_rng(random_seed)
    result = np.asarray(reference_partition, dtype=np.int64).copy()
    target_label_mask = _target_label_mask(dataset, target_category)

    swapped_nodes = 0
    rejected_same_partition_pairs = 0
    target_swapped_nodes = 0
    for label_value in (True, False):
        candidates = np.flatnonzero(target_label_mask == label_value)
        label_target = int(round(candidates.size * perturb_ratio))
        label_target -= label_target % 2
        target_swapped_nodes += label_target
        if label_target <= 0:
            continue
        selected = rng.permutation(candidates)[:label_target].copy()
        label_swapped, label_rejected = _swap_selected_nodes_across_parts(
            result, selected
        )
        swapped_nodes += label_swapped
        rejected_same_partition_pairs += label_rejected

    diagnostics = {
        "target_swapped_nodes": int(target_swapped_nodes),
        "actual_swapped_nodes": int(swapped_nodes),
        "actual_swapped_node_ratio": float(swapped_nodes / len(reference_partition)),
        "rejected_same_partition_pairs": int(rejected_same_partition_pairs),
        "label_control": "same_label_swaps",
    }
    return result, diagnostics


def partition_metis_perturb(dataset, config) -> PartitionResult:
    num_parts = int(config["num_partitions"])
    perturb_ratio = float(config.get("perturb_ratio", 0.5))
    reference, target_category, reference_metadata = resolve_reference(dataset, config)
    method_seed = int(config.get("seed", 42)) + int(round(perturb_ratio * 1000))
    assignments, diagnostics = perturb_within_labels(
        reference,
        dataset,
        target_category,
        perturb_ratio,
        method_seed,
    )
    assignments = validate_assignments(assignments, dataset, num_parts)
    return PartitionResult(
        assignments=assignments,
        target_category=target_category,
        reference_assignments=reference,
        metadata={
            **reference_metadata,
            "partition_method": "metis_perturb",
            "legacy_algorithm_name": "metis_perturbed",
            "requested_perturb_ratio": perturb_ratio,
            "method_seed": method_seed,
            **diagnostics,
        },
    )


# Balanced multi-source BFS and its size-constrained isolate repair.

DEFAULT_BALANCE_TOLERANCE = 0.05


def balanced_multi_source_bfs(graph, nodes, num_parts, random_seed):
    """Return an exactly balanced assignment built by multi-source BFS.

    The traversal never inspects node features or labels.  If a frontier is
    exhausted before its partition reaches capacity, it is restarted at the
    next still-unassigned node in a seeded random ordering.  These restarts
    cover disconnected components, components without an initial source, and
    globally isolated nodes without a separate structure-blind cleanup pass.
    """
    node_count = len(nodes)
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if num_parts > node_count:
        raise ValueError(
            f"num_parts cannot exceed node count: {num_parts} > {node_count}"
        )

    rng = np.random.default_rng(int(random_seed))
    node_order = rng.permutation(node_count)
    part_order = rng.permutation(num_parts)

    capacities = np.full(num_parts, node_count // num_parts, dtype=np.int64)
    extra_capacity_parts = part_order[: node_count % num_parts]
    capacities[extra_capacity_parts] += 1

    assignments = np.full(node_count, -1, dtype=np.int64)
    part_sizes = np.zeros(num_parts, dtype=np.int64)
    frontiers = [deque() for _ in range(num_parts)]
    neighbor_iters = [None for _ in range(num_parts)]
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    next_source_offset = 0

    def next_unassigned_index():
        nonlocal next_source_offset
        while next_source_offset < node_count:
            node_idx = int(node_order[next_source_offset])
            next_source_offset += 1
            if assignments[node_idx] < 0:
                return node_idx
        return None

    def assign_node(node_idx, part_id):
        assignments[node_idx] = part_id
        part_sizes[part_id] += 1
        frontiers[part_id].append(node_idx)

    # Give every partition one distinct initial source.  Since K <= N, every
    # capacity is at least one and the resulting output has no empty parts.
    for part_id in part_order:
        source_idx = next_unassigned_index()
        if source_idx is None:
            raise RuntimeError("Unable to select an initial BFS source")
        assign_node(source_idx, int(part_id))

    initial_seed_count = num_parts
    reseed_count = 0
    active_parts = deque(int(part_id) for part_id in part_order)

    # Each partition claims at most one new node per turn.  Keeping the current
    # adjacency iterator between turns preserves ordinary BFS ordering while
    # preventing a high-degree source from monopolizing a component before the
    # other sources get a chance to expand.
    while active_parts:
        part_id = active_parts.popleft()
        if part_sizes[part_id] >= capacities[part_id]:
            frontiers[part_id].clear()
            neighbor_iters[part_id] = None
            continue

        claimed_node = False
        while not claimed_node:
            if neighbor_iters[part_id] is None:
                if not frontiers[part_id]:
                    source_idx = next_unassigned_index()
                    if source_idx is None:
                        raise RuntimeError(
                            "Multi-source BFS exhausted all nodes before filling capacities"
                        )
                    assign_node(source_idx, part_id)
                    reseed_count += 1
                    claimed_node = True
                    break

                source_idx = frontiers[part_id].popleft()
                source_node = nodes[source_idx]
                neighbor_iters[part_id] = iter(graph.adj[source_node])

            for neighbor in neighbor_iters[part_id]:
                neighbor_idx = node_to_idx[neighbor]
                if assignments[neighbor_idx] < 0:
                    assign_node(neighbor_idx, part_id)
                    claimed_node = True
                    break
            else:
                neighbor_iters[part_id] = None

        if part_sizes[part_id] < capacities[part_id]:
            active_parts.append(part_id)
        else:
            frontiers[part_id].clear()
            neighbor_iters[part_id] = None

    if np.any(assignments < 0):
        raise RuntimeError("Multi-source BFS left unassigned nodes")
    if not np.array_equal(part_sizes, capacities):
        raise RuntimeError("Multi-source BFS did not satisfy partition capacities")

    diagnostics = {
        "algorithm": "balanced_multi_source_bfs",
        "growth_policy": "round_robin_one_claim_per_partition",
        "method_seed": int(random_seed),
        "initial_seed_count": int(initial_seed_count),
        "reseed_count": int(reseed_count),
        "total_bfs_sources": int(initial_seed_count + reseed_count),
        "capacity_min": int(capacities.min()),
        "capacity_max": int(capacities.max()),
        "uses_node_features": False,
        "uses_node_labels": False,
    }
    return assignments, diagnostics


def repair_bfs_isolates(
    graph,
    nodes,
    assignments,
    num_parts,
    balance_tolerance=DEFAULT_BALANCE_TOLERANCE,
):
    """Move repairable internal isolates to adjacent parts within size bounds."""
    assignments = np.asarray(assignments, dtype=np.int64).copy()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    part_sizes = np.bincount(assignments, minlength=num_parts).astype(np.int64)
    average_size = len(nodes) / num_parts
    min_size = max(1, math.floor(average_size * (1.0 - balance_tolerance)))
    max_size = max(1, math.ceil(average_size * (1.0 + balance_tolerance)))

    initial_internal_isolates = 0
    moved_nodes = 0
    globally_isolated_nodes = 0

    for node_idx, node in enumerate(nodes):
        source_part = int(assignments[node_idx])
        neighbor_part_counts = {}
        has_internal_neighbor = False
        degree = 0
        for neighbor in graph.adj[node]:
            degree += 1
            neighbor_part = int(assignments[node_to_idx[neighbor]])
            if neighbor_part == source_part:
                has_internal_neighbor = True
                break
            neighbor_part_counts[neighbor_part] = (
                neighbor_part_counts.get(neighbor_part, 0) + 1
            )

        if has_internal_neighbor:
            continue
        initial_internal_isolates += 1
        if degree == 0:
            globally_isolated_nodes += 1
            continue
        if part_sizes[source_part] <= min_size:
            continue

        candidates = [
            part_id
            for part_id in neighbor_part_counts
            if part_sizes[part_id] < max_size
        ]
        if not candidates:
            continue
        target_part = max(
            candidates,
            key=lambda part_id: (
                neighbor_part_counts[part_id],
                -int(part_sizes[part_id]),
                -int(part_id),
            ),
        )
        assignments[node_idx] = target_part
        part_sizes[source_part] -= 1
        part_sizes[target_part] += 1
        moved_nodes += 1

    remaining_internal_isolates = 0
    for node_idx, node in enumerate(nodes):
        part_id = int(assignments[node_idx])
        if not any(
            int(assignments[node_to_idx[neighbor]]) == part_id
            for neighbor in graph.adj[node]
        ):
            remaining_internal_isolates += 1

    diagnostics = {
        "isolated_repair_balance_tolerance": float(balance_tolerance),
        "isolated_repair_min_size": int(min_size),
        "isolated_repair_max_size": int(max_size),
        "initial_internal_isolated_nodes": int(initial_internal_isolates),
        "globally_isolated_nodes_skipped": int(globally_isolated_nodes),
        "internal_isolated_nodes_moved": int(moved_nodes),
        "remaining_internal_isolated_nodes": int(remaining_internal_isolates),
        "final_size_min": int(part_sizes.min()),
        "final_size_max": int(part_sizes.max()),
    }
    return assignments, diagnostics


def partition_bfs(dataset, config) -> PartitionResult:
    num_parts = int(config["num_partitions"])
    random_seed = int(config.get("seed", 42))
    assignments, diagnostics = balanced_multi_source_bfs(
        dataset.graph,
        dataset.nodes,
        num_parts,
        random_seed,
    )
    assignments, repair_diagnostics = repair_bfs_isolates(
        dataset.graph,
        dataset.nodes,
        assignments,
        num_parts,
    )
    assignments = validate_assignments(assignments, dataset, num_parts)

    # Labels are consulted only after the structural partition is complete, to
    # preserve the repository's existing bag-label/output schema.
    target_category = choose_target_category(dataset, assignments, num_parts)
    return PartitionResult(
        assignments=assignments,
        target_category=target_category,
        reference_assignments=assignments.copy(),
        metadata={
            "partition_method": "bfs",
            "paper_basis": "BGL multi-source BFS block generation",
            "label_profile_reference": "self",
            **diagnostics,
            **repair_diagnostics,
        },
    )


# Public partition interface.

_PARTITION_METHODS = {
    "bfs": partition_bfs,
    "metis": partition_metis,
    "metis_perturb": partition_metis_perturb,
}

_ALIASES = {
    "bfs": "bfs",
    "msbfs": "bfs",
    "multi_source_bfs": "bfs",
    "multi-source-bfs": "bfs",
    "metis": "metis",
    "metis_perturb": "metis_perturb",
    "metis-perturb": "metis_perturb",
    "metis_perturbed": "metis_perturb",
}


def normalize_method(method: str) -> str:
    normalized = str(method).strip().lower()
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(_PARTITION_METHODS)
        raise ValueError(
            f"Unsupported partition method {method!r}; choose: {supported}"
        ) from exc


def partition_graph(dataset, method: str, config) -> PartitionResult:
    """Partition a loaded graph with BFS, METIS or the paper's Perturb-50 setting."""
    return _PARTITION_METHODS[normalize_method(method)](dataset, config)


SUPPORTED_METHODS = tuple(_PARTITION_METHODS)
