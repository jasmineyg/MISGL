# coding=utf-8

import json

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
            raise ImportError('networkx is required to convert original_graph.') from exc

        graph_nodes = list(graph.nodes())
        if len(graph_nodes) != num_nodes:
            raise ValueError(
                'original_graph node count does not match assignment_matrix rows: '
                f'{len(graph_nodes)} vs {num_nodes}.'
            )
        nodelist = list(range(num_nodes)) if set(graph_nodes) == set(range(num_nodes)) else graph_nodes
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
    include_self=False,
    symmetrize=True,
    dtype=np.float32,
):
    """Build raw S^T A S coarse edges and keep row-wise top-k weights."""
    S = _as_csr_matrix(assignment_matrix, dtype=dtype)
    num_nodes, num_subgraphs = S.shape

    if active_subgraph_ids is None:
        active_subgraph_ids = np.arange(num_subgraphs, dtype=np.int64)
    else:
        active_subgraph_ids = np.asarray(active_subgraph_ids, dtype=np.int64)
        if active_subgraph_ids.ndim != 1:
            raise ValueError('active_subgraph_ids must be 1-D.')
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
        'top_k': None if top_k is None else int(top_k),
        'include_self': bool(include_self),
        'symmetrize': bool(symmetrize),
    }
    return coarse, metadata


def save_coarse_adjacency(path, coarse_adj, metadata=None):
    coarse_adj = coarse_adj.tocsr()
    metadata = dict(metadata or {})
    np.savez(
        path,
        indptr=coarse_adj.indptr.astype(np.int64, copy=False),
        indices=coarse_adj.indices.astype(np.int64, copy=False),
        data=coarse_adj.data.astype(np.float32, copy=False),
        shape=np.asarray(coarse_adj.shape, dtype=np.int64),
        active_subgraph_ids=np.asarray(metadata.get('active_subgraph_ids', []), dtype=np.int64),
        metadata_json=np.asarray(json.dumps(_json_safe_metadata(metadata), sort_keys=True)),
    )


def load_coarse_adjacency(path):
    with np.load(path, allow_pickle=False) as cached:
        shape = tuple(int(v) for v in cached['shape'].tolist())
        coarse_adj = sp.csr_matrix(
            (
                cached['data'].astype(np.float32, copy=False),
                cached['indices'].astype(np.int64, copy=False),
                cached['indptr'].astype(np.int64, copy=False),
            ),
            shape=shape,
        )
        metadata = json.loads(str(cached['metadata_json'].item()))
        metadata['active_subgraph_ids'] = cached['active_subgraph_ids'].astype(np.int64, copy=False)
    return coarse_adj, metadata


def _json_safe_metadata(metadata):
    out = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist() if value.size <= 64 else {'shape': list(value.shape), 'dtype': str(value.dtype)}
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out
