"""Internal-Head: nine structural statistics and structure-aware gated attention."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


PPR_ALPHA = 0.15
STRUCTURAL_STATISTICS = (
    "degree_norm",
    "avg_neighbor_degree_norm",
    "two_hop_reach_norm",
    "ppr_self_return",
    "triangle_count_log_norm",
    "clustering_coeff",
    "core_number_norm",
    "pagerank_max_norm",
    "lappe_fiedler_abs_norm",
)


NUM_STRUCTURAL_STATISTICS = len(STRUCTURAL_STATISTICS)


def compute_structural_statistics(adjacency):
    """Return the nine fixed, normalized node descriptors in STRUCTURAL_STATISTICS order."""
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.isfinite(adjacency).all():
        raise ValueError("adjacency must contain finite values")
    node_count = int(adjacency.shape[0])
    if node_count == 0:
        return np.zeros((0, len(STRUCTURAL_STATISTICS)), dtype=np.float32)

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
        raise RuntimeError(
            "structural descriptor computation produced non-finite values"
        )
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


@dataclass(frozen=True)
class InternalHeadOutput:
    """Node attention_weights and the resulting bag embeddings."""

    z_int: torch.Tensor
    attention_weights: torch.Tensor
    subgraph_index: torch.Tensor


class InternalHead(nn.Module):
    """Paper Section 4.2: structure-aware attention and gated residual fusion.

    ``h`` contains GAT node representations, ``s`` contains nine structural
    statistics, and ``z_int`` is the internal subgraph representation. The
    learned residual scale, initialized to 0.1, preserves the experiment code.
    """

    def __init__(
        self,
        node_dim: int,
        attention_dim: int,
        structure_dim: int,
        structure_gate_dim: int,
        residual_init: float,
    ) -> None:
        super().__init__()
        if min(node_dim, attention_dim, structure_dim, structure_gate_dim) <= 0:
            raise ValueError("Internal-Head dimensions must be positive")

        if structure_dim != 32:
            raise ValueError("structure_dim must be 32 for the final encoder")
        self.mlp_s = nn.Sequential(
            nn.Linear(NUM_STRUCTURAL_STATISTICS, structure_dim),
            nn.LayerNorm(structure_dim),
            nn.ReLU(),
        )
        self.structure_projection = nn.Linear(structure_dim, node_dim)
        self.mlp_g = nn.Sequential(
            nn.Linear(node_dim * 2, structure_gate_dim),
            nn.ReLU(),
            nn.Linear(structure_gate_dim, node_dim),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_init), dtype=torch.float32)
        )

        score_dim = node_dim + structure_dim
        self.attention_v = nn.Linear(score_dim, attention_dim)
        self.attention_u = nn.Linear(score_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        subgraph_index: torch.Tensor,
    ) -> InternalHeadOutput:
        if h.ndim != 2:
            raise ValueError("h must have shape [nodes, features]")
        if s.ndim != 2 or s.shape != (h.size(0), NUM_STRUCTURAL_STATISTICS):
            raise ValueError("s must have shape [nodes, 9]")
        if subgraph_index.ndim != 1 or subgraph_index.size(0) != h.size(0):
            raise ValueError("subgraph_index must contain one id per node")
        if h.size(0) == 0:
            raise ValueError("InternalHead requires at least one node")
        if subgraph_index.dtype != torch.long:
            raise TypeError("subgraph_index must have dtype torch.long")
        if s.device != h.device:
            raise ValueError("s and h must share a device")
        if subgraph_index.device != h.device:
            raise ValueError("subgraph_index and h must share a device")
        if torch.any(subgraph_index < 0):
            raise ValueError("subgraph_index must be non-negative")

        subgraph_count = int(subgraph_index.max().item()) + 1
        if torch.any(torch.bincount(subgraph_index, minlength=subgraph_count) == 0):
            raise ValueError("subgraph ids must be contiguous and start at zero")

        # Eq. (6): encode the nine structural statistics.
        s_encoded = self.mlp_s(s)
        # Eq. (7): gated attention over node and structural representations.
        phi = torch.cat((h, s_encoded), dim=-1)
        gated_score = torch.tanh(self.attention_v(phi))
        gated_score = gated_score * torch.sigmoid(self.attention_u(phi))
        scores = self.attention_w(gated_score).squeeze(-1).clamp(-12.0, 12.0)

        maximum = scores.new_full((subgraph_count,), -torch.inf)
        maximum.scatter_reduce_(
            0,
            subgraph_index,
            scores,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(scores - maximum[subgraph_index])
        normalizer = scores.new_zeros(subgraph_count)
        normalizer.scatter_add_(0, subgraph_index, exponentials)
        attention_weights = exponentials / normalizer[subgraph_index]
        attention_weights = attention_weights.clamp(1.0e-6, 1.0 - 1.0e-6)

        # Eqs. (8)-(11): u, v, r, g and the internal representation z_int.
        u = self._weighted_sum(h, attention_weights, subgraph_index, subgraph_count)
        v = self._weighted_sum(
            s_encoded, attention_weights, subgraph_index, subgraph_count
        )
        r = self.structure_projection(v)
        g = self.mlp_g(torch.cat((u, r), dim=-1))
        z_int = u + self.residual_scale * g * r
        return InternalHeadOutput(
            z_int=z_int,
            attention_weights=attention_weights,
            subgraph_index=subgraph_index,
        )

    @staticmethod
    def _weighted_sum(
        values: torch.Tensor,
        attention_weights: torch.Tensor,
        subgraph_index: torch.Tensor,
        subgraph_count: int,
    ) -> torch.Tensor:
        weighted = attention_weights.unsqueeze(-1) * values
        pooled = values.new_zeros((subgraph_count, values.size(1)))
        pooled.index_add_(0, subgraph_index, weighted)
        return pooled
