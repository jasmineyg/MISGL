"""Full nine-dimensional node structure used by the final MIL encoder."""

import numpy as np


PPR_ALPHA = 0.15
FEATURE_NAMES = (
    'degree_norm',
    'avg_neighbor_degree_norm',
    'two_hop_reach_norm',
    'ppr_self_return',
    'triangle_count_log_norm',
    'clustering_coeff',
    'core_number_norm',
    'pagerank_max_norm',
    'lappe_fiedler_abs_norm',
)


INPUT_DIM = len(FEATURE_NAMES)


def compute_structural_features(adjacency):
    """Return the nine fixed, normalized node descriptors in FEATURE_NAMES order."""
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError('adjacency must be a square matrix')
    if not np.isfinite(adjacency).all():
        raise ValueError('adjacency must contain finite values')
    node_count = int(adjacency.shape[0])
    if node_count == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    connected = np.logical_or(adjacency != 0, adjacency.T != 0)
    connected = connected.copy()
    np.fill_diagonal(connected, False)
    graph = connected.astype(np.float64, copy=False)
    max_degree = max(float(node_count - 1), 1.0)

    degree = graph.sum(axis=1)
    degree_norm = degree / max_degree
    neighbor_degree_sum = graph @ degree
    average_neighbor_degree = neighbor_degree_sum / np.maximum(degree, 1.0)
    average_neighbor_degree_norm = average_neighbor_degree / max_degree

    two_step = graph @ graph
    exactly_two = (two_step > 0) & (~connected)
    np.fill_diagonal(exactly_two, False)
    two_hop_reach_norm = exactly_two.sum(axis=1).astype(np.float64) / max_degree

    transition = _row_transition(graph)
    ppr_self_return = _ppr_self_return(transition, PPR_ALPHA)

    closed_wedge_count = (two_step * graph).sum(axis=1)
    triangle_count = closed_wedge_count / 2.0
    max_triangles = max(max_degree * (max_degree - 1.0) / 2.0, 1.0)
    triangle_count_log_norm = np.log1p(triangle_count) / np.log1p(max_triangles)
    possible_wedges = degree * (degree - 1.0)
    clustering = np.divide(
        closed_wedge_count,
        np.maximum(possible_wedges, 1.0),
        out=np.zeros_like(degree),
        where=possible_wedges > 0,
    )

    core_number_norm = _core_number(graph) / max_degree
    pagerank = _pagerank(transition, PPR_ALPHA)
    pagerank_max_norm = pagerank / max(float(pagerank.max()), np.finfo(np.float64).eps)
    lappe = _fiedler_absolute(graph)

    result = np.stack(
        (
            degree_norm,
            average_neighbor_degree_norm,
            two_hop_reach_norm,
            ppr_self_return,
            triangle_count_log_norm,
            clustering,
            core_number_norm,
            pagerank_max_norm,
            lappe,
        ),
        axis=1,
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError('structural descriptor computation produced non-finite values')
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def _row_transition(graph):
    node_count = graph.shape[0]
    degree = graph.sum(axis=1)
    transition = np.divide(
        graph,
        np.maximum(degree[:, None], 1.0),
        out=np.zeros_like(graph),
        where=degree[:, None] > 0,
    )
    isolated = degree == 0
    transition[isolated, isolated] = 1.0
    return transition


def _ppr_self_return(transition, alpha):
    node_count = transition.shape[0]
    system = np.eye(node_count, dtype=np.float64) - (1.0 - alpha) * transition
    inverse = np.linalg.solve(system, np.eye(node_count, dtype=np.float64))
    return np.clip(alpha * np.diag(inverse), 0.0, 1.0)


def _pagerank(transition, alpha, tolerance=1.0e-12, max_iterations=500):
    node_count = transition.shape[0]
    rank = np.full(node_count, 1.0 / node_count, dtype=np.float64)
    teleport = np.full(node_count, alpha / node_count, dtype=np.float64)
    for _ in range(max_iterations):
        updated = teleport + (1.0 - alpha) * (transition.T @ rank)
        if np.abs(updated - rank).sum() <= tolerance:
            rank = updated
            break
        rank = updated
    rank_sum = float(rank.sum())
    return rank / rank_sum if rank_sum > 0 else np.full(node_count, 1.0 / node_count)


def _core_number(graph):
    node_count = graph.shape[0]
    remaining = np.ones(node_count, dtype=bool)
    working_degree = graph.sum(axis=1).copy()
    values = np.zeros(node_count, dtype=np.float64)
    running_core = 0.0
    for _ in range(node_count):
        masked = np.where(remaining, working_degree, np.inf)
        index = int(np.argmin(masked))
        running_core = max(running_core, float(masked[index]))
        values[index] = running_core
        remaining[index] = False
        working_degree = np.maximum(working_degree - graph[:, index], 0.0)
    return values


def _fiedler_absolute(graph):
    node_count = graph.shape[0]
    if node_count < 2 or not np.any(graph):
        return np.zeros(node_count, dtype=np.float64)
    degree = graph.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    valid = degree > 0
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    laplacian = np.eye(node_count, dtype=np.float64) - (
        inv_sqrt[:, None] * graph * inv_sqrt[None, :]
    )
    _values, vectors = np.linalg.eigh(laplacian)
    vector = np.abs(vectors[:, 1])
    maximum = float(vector.max())
    return vector / maximum if maximum > 0 else np.zeros(node_count, dtype=np.float64)
