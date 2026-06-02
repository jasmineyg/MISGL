# coding=utf-8

import numpy as np
import scipy.sparse as sp


def _as_csr_matrix(value, dtype=np.float32):
    if sp.issparse(value):
        return value.tocsr().astype(dtype, copy=False)
    return sp.csr_matrix(value, dtype=dtype)


def _graph_to_sparse_adj(graph, num_nodes, dtype=np.float32):
    if sp.issparse(graph):
        adj = graph.tocsr().astype(dtype, copy=False)
    else:
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError('networkx is required to convert original_graph to a sparse adjacency.') from exc

        graph_nodes = list(graph.nodes())
        if len(graph_nodes) != num_nodes:
            raise ValueError(
                'original_graph node count does not match assignment_matrix rows: '
                f'{len(graph_nodes)} vs {num_nodes}.'
            )
        if set(graph_nodes) == set(range(num_nodes)):
            nodelist = list(range(num_nodes))
        else:
            nodelist = graph_nodes

        if hasattr(nx, 'to_scipy_sparse_array'):
            adj = nx.to_scipy_sparse_array(graph, nodelist=nodelist, dtype=dtype, format='csr')
        else:
            adj = nx.to_scipy_sparse_matrix(graph, nodelist=nodelist, dtype=dtype, format='csr')
        adj = _as_csr_matrix(adj, dtype=dtype)

    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(
            'original_graph adjacency shape does not match assignment_matrix rows: '
            f'{adj.shape} vs ({num_nodes}, {num_nodes}).'
        )
    return adj


def _row_topk_csr(matrix, top_k):
    matrix = matrix.tocsr()
    if top_k is None:
        matrix.sort_indices()
        return matrix

    top_k = int(top_k)
    if top_k <= 0:
        return sp.csr_matrix(matrix.shape, dtype=matrix.dtype)

    indptr = [0]
    indices = []
    data = []
    for row_idx in range(matrix.shape[0]):
        start, end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
        row_indices = matrix.indices[start:end]
        row_data = matrix.data[start:end]
        if row_data.size > top_k:
            keep = np.argpartition(row_data, -top_k)[-top_k:]
            order = keep[np.argsort(-row_data[keep])]
        else:
            order = np.argsort(-row_data)
        indices.extend(row_indices[order].tolist())
        data.extend(row_data[order].tolist())
        indptr.append(len(indices))

    return sp.csr_matrix(
        (
            np.asarray(data, dtype=matrix.dtype),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
    )


def build_coarse_adjacency(
    original_graph,
    assignment_matrix,
    active_subgraph_ids=None,
    top_k=16,
    normalize=True,
    include_self=False,
    symmetrize=True,
    dtype=np.float32,
):
    """Build A_hat = D^-1 S^T A S D^-1 and keep row-wise top-k edges.

    Args:
        original_graph: NetworkX graph or sparse adjacency of the original graph.
        assignment_matrix: Sparse/dense node-to-subgraph assignment matrix S.
        active_subgraph_ids: Optional column ids to keep, in output row order.
        top_k: Number of highest-weight outgoing coarse edges to keep per row.
        normalize: Whether to divide by the two endpoint subgraph sizes.
        include_self: Whether to keep diagonal coarse weights.
        symmetrize: Whether to enforce symmetry before row-wise top-k pruning.
        dtype: Output floating dtype.

    Returns:
        (coarse_adj, metadata) where coarse_adj is a CSR matrix.
    """
    S = _as_csr_matrix(assignment_matrix, dtype=dtype)
    num_nodes, num_subgraphs = S.shape

    if active_subgraph_ids is None:
        active_subgraph_ids = np.arange(num_subgraphs, dtype=np.int64)
    else:
        active_subgraph_ids = np.asarray(active_subgraph_ids, dtype=np.int64)
        if active_subgraph_ids.ndim != 1:
            raise ValueError('active_subgraph_ids must be a 1-D sequence.')
        if active_subgraph_ids.size == 0:
            raise ValueError('active_subgraph_ids cannot be empty.')
        if np.unique(active_subgraph_ids).size != active_subgraph_ids.size:
            raise ValueError('active_subgraph_ids must be unique.')
        if active_subgraph_ids.min() < 0 or active_subgraph_ids.max() >= num_subgraphs:
            raise ValueError(
                'active_subgraph_ids contains values outside assignment_matrix columns: '
                f'valid=[0, {num_subgraphs}), got min={active_subgraph_ids.min()}, '
                f'max={active_subgraph_ids.max()}.'
            )
        S = S[:, active_subgraph_ids]

    A = _graph_to_sparse_adj(original_graph, num_nodes, dtype=dtype)
    coarse = (S.T @ A @ S).tocsr().astype(dtype, copy=False)

    subgraph_sizes = np.asarray(S.sum(axis=0)).reshape(-1).astype(dtype, copy=False)
    if normalize:
        inv_sizes = np.zeros_like(subgraph_sizes, dtype=dtype)
        valid = subgraph_sizes > 0
        inv_sizes[valid] = 1.0 / subgraph_sizes[valid]
        D_inv = sp.diags(inv_sizes, offsets=0, shape=coarse.shape, dtype=dtype, format='csr')
        coarse = (D_inv @ coarse @ D_inv).tocsr()

    if not include_self:
        diagonal = coarse.diagonal()
        if np.any(diagonal):
            coarse = (coarse - sp.diags(diagonal, offsets=0, shape=coarse.shape, dtype=dtype, format='csr')).tocsr()
            coarse.eliminate_zeros()

    if symmetrize:
        coarse = coarse.maximum(coarse.T).tocsr()

    coarse = _row_topk_csr(coarse, top_k).astype(dtype, copy=False)
    coarse.eliminate_zeros()

    metadata = {
        'num_original_nodes': int(num_nodes),
        'num_assignment_columns': int(num_subgraphs),
        'num_coarse_nodes': int(coarse.shape[0]),
        'active_subgraph_ids': active_subgraph_ids.astype(np.int64, copy=False),
        'subgraph_sizes': subgraph_sizes.astype(dtype, copy=False),
        'top_k': None if top_k is None else int(top_k),
        'normalize': bool(normalize),
        'include_self': bool(include_self),
        'symmetrize': bool(symmetrize),
    }
    return coarse, metadata
