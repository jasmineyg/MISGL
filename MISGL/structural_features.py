"""Selected paper descriptors: C+M in the historical category9 interface."""

import numpy as np


FEATURE_NAMES = (
    "degree_norm", "avg_neighbor_degree_norm", "two_hop_reach_norm",
    "ppr_self_return", "triangle_count_log_norm", "clustering_coeff",
    "core_number_norm", "pagerank_max_norm", "lappe_fiedler_abs_norm",
)
CM_MASK = (1, 1, 0, 0, 1, 1, 0, 0, 0)
INPUT_DIM = len(FEATURE_NAMES)


def compute_cm_features(adjacency: np.ndarray) -> np.ndarray:
    """Compute only selected statistics, leaving unused category9 slots zero.

    Match the historical category9_v1 formulas: unweighted, undirected,
    loop-free topology; float64 arithmetic followed by float32 storage.
    """
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.isfinite(adjacency).all():
        raise ValueError("adjacency must contain finite values")
    n = adjacency.shape[0]
    result = np.zeros((n, INPUT_DIM), dtype=np.float32)
    if n == 0:
        return result
    connected = np.logical_or(adjacency != 0, adjacency.T != 0)
    np.fill_diagonal(connected, False)
    graph = connected.astype(np.float64)
    max_degree = max(float(n - 1), 1.0)
    degree = graph.sum(axis=1)
    result[:, 0] = degree / max_degree
    result[:, 1] = (graph @ degree) / np.maximum(degree, 1.0) / max_degree
    closed_wedges = ((graph @ graph) * graph).sum(axis=1)
    max_triangles = max(max_degree * (max_degree - 1.0) / 2.0, 1.0)
    result[:, 4] = np.log1p(closed_wedges / 2.0) / np.log1p(max_triangles)
    possible_wedges = degree * (degree - 1.0)
    result[:, 5] = np.divide(
        closed_wedges, np.maximum(possible_wedges, 1.0),
        out=np.zeros_like(degree), where=possible_wedges > 0,
    )
    return result
