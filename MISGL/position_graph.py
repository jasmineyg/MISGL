"""Sparse position-graph construction for the active subgraphs."""

from numbers import Integral
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


def _float_csr(value: Any, name: str) -> sp.csr_matrix:
    if sp.issparse(value):
        matrix = value.tocsr().astype(np.float32, copy=False)
    elif isinstance(value, np.ndarray):
        matrix = sp.csr_matrix(value, dtype=np.float32)
    else:
        raise TypeError('{} must be a SciPy sparse matrix or NumPy array.'.format(name))
    if matrix.ndim != 2:
        raise ValueError('{} must be two-dimensional.'.format(name))
    if not np.isfinite(matrix.data).all():
        raise ValueError('{} contains non-finite values.'.format(name))
    if np.any(matrix.data < 0):
        raise ValueError('{} must contain non-negative values.'.format(name))
    return matrix


def _original_adjacency(original_graph: Any, node_count: int) -> sp.csr_matrix:
    if sp.issparse(original_graph):
        adjacency = _float_csr(original_graph, 'original_graph')
    elif isinstance(original_graph, nx.Graph):
        if original_graph.number_of_nodes() != node_count:
            raise ValueError(
                'original_graph node count must match assignment_matrix rows.'
            )
        if set(original_graph.nodes()) != set(range(node_count)):
            raise ValueError(
                'original_graph node ids must be exactly 0..N-1 so they align with '
                'assignment_matrix rows.'
            )
        adjacency = sp.csr_matrix(
            nx.to_scipy_sparse_array(
                original_graph,
                nodelist=range(node_count),
                dtype=np.float32,
                weight='weight',
                format='csr',
            )
        )
        if not np.isfinite(adjacency.data).all():
            raise ValueError('original_graph contains non-finite edge weights.')
        if np.any(adjacency.data < 0):
            raise ValueError('original_graph must contain non-negative edge weights.')
    else:
        raise TypeError('original_graph must be a NetworkX graph or SciPy sparse matrix.')

    if adjacency.shape != (node_count, node_count):
        raise ValueError(
            'original_graph adjacency shape must match assignment_matrix rows.'
        )
    return adjacency


def _active_columns(active_ids: Any, column_count: int) -> np.ndarray:
    if not isinstance(active_ids, (list, tuple, np.ndarray)):
        raise TypeError('active_ids must be a one-dimensional integer sequence.')
    columns = np.asarray(active_ids)
    if columns.ndim != 1 or columns.size == 0:
        raise ValueError('active_ids must be a non-empty one-dimensional array.')
    if columns.dtype.kind not in 'iu':
        raise TypeError('active_ids must contain integers.')
    columns = columns.astype(np.int64, copy=False)
    if np.unique(columns).size != columns.size:
        raise ValueError('active_ids must not contain duplicates.')
    if columns.min() < 0 or columns.max() >= column_count:
        raise IndexError('active_ids contain values outside assignment_matrix columns.')
    return columns


def _row_top_k(matrix: sp.csr_matrix, top_k: int) -> sp.csr_matrix:
    indptr = [0]
    indices = []
    values = []

    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        row_indices = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if row_values.size > top_k:
            selected = np.argpartition(row_values, -top_k)[-top_k:]
            selected = selected[np.argsort(-row_values[selected], kind='stable')]
        else:
            selected = np.argsort(-row_values, kind='stable')
        indices.extend(row_indices[selected].tolist())
        values.extend(row_values[selected].tolist())
        indptr.append(len(indices))

    result = sp.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
    )
    result.sort_indices()
    return result


def build_position_adjacency(
    original_graph: Any,
    assignment_matrix: Any,
    active_ids: Any,
    top_k: int,
) -> sp.csr_matrix:
    """Build top-k ``S_active.T @ A @ S_active`` adjacency without self-edges."""
    if isinstance(top_k, bool) or not isinstance(top_k, Integral):
        raise TypeError('top_k must be an integer.')
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError('top_k must be positive.')

    assignment = _float_csr(assignment_matrix, 'assignment_matrix')
    node_count, column_count = assignment.shape
    if node_count == 0 or column_count == 0:
        raise ValueError('assignment_matrix must have non-zero dimensions.')
    columns = _active_columns(active_ids, column_count)
    adjacency = _original_adjacency(original_graph, node_count)

    active_assignment = assignment[:, columns]
    position_adjacency = (
        active_assignment.transpose() @ adjacency @ active_assignment
    ).tocsr().astype(np.float32, copy=False)
    if not np.isfinite(position_adjacency.data).all():
        raise ValueError('Position adjacency contains non-finite values.')

    diagonal = position_adjacency.diagonal()
    if np.any(diagonal):
        position_adjacency = (
            position_adjacency
            - sp.diags(
                diagonal,
                offsets=0,
                shape=position_adjacency.shape,
                dtype=np.float32,
                format='csr',
            )
        ).tocsr()
        position_adjacency.eliminate_zeros()
    position_adjacency = position_adjacency.maximum(
        position_adjacency.transpose()
    ).tocsr()
    position_adjacency = _row_top_k(position_adjacency, top_k)
    position_adjacency.eliminate_zeros()
    return position_adjacency


def row_normalize(adjacency: Any) -> sp.csr_matrix:
    """Add unit self-loops, then apply row normalization ``D^-1 A``."""
    matrix = _float_csr(adjacency, 'adjacency')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('adjacency must be square.')

    matrix = (
        matrix + sp.eye(matrix.shape[0], dtype=np.float32, format='csr')
    ).tocsr()
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    normalized = sp.diags(
        np.reciprocal(row_sums), dtype=np.float32, format='csr'
    ) @ matrix
    return normalized.tocsr().astype(np.float32, copy=False)


def to_torch_sparse(adjacency: Any, device: Any) -> torch.Tensor:
    """Convert a SciPy/NumPy adjacency to a coalesced torch COO tensor."""
    matrix = _float_csr(adjacency, 'adjacency').tocoo()
    indices = torch.from_numpy(
        np.vstack((matrix.row, matrix.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=matrix.shape,
        dtype=torch.float32,
        device=device,
    ).coalesce()
