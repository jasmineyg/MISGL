# coding=utf-8

import logging
import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


LAPPE_CACHE_VERSION = 2


def default_lappe_cache_path(processed_data_dir, data_name, lap_pe_dim, coarse_topk):
  return os.path.join(
    processed_data_dir,
    '{}_lappe_dim{}_topk{}.pt'.format(data_name, int(lap_pe_dim), int(coarse_topk)),
  )


def resolve_lappe_cache_path(hparams, data_name):
  configured = getattr(hparams, 'lappe_cache_path', None)
  if configured:
    return configured
  processed_data_dir = getattr(hparams, 'processed_data_dir', '.')
  return default_lappe_cache_path(
    processed_data_dir,
    data_name,
    int(getattr(hparams, 'lap_pe_dim', 16)),
    int(getattr(hparams, 'coarse_topk', 20)),
  )


def load_lappe_cache(
    cache_path,
    expected_dim=None,
    expected_num_subgraphs=None,
    expected_cache_version=None,
):
  payload = torch.load(cache_path, map_location='cpu')
  if isinstance(payload, torch.Tensor):
    lap_pe = payload.to(dtype=torch.float32)
    payload = {'lap_pe': lap_pe}
  elif isinstance(payload, dict) and isinstance(payload.get('lap_pe'), torch.Tensor):
    payload = dict(payload)
    payload['lap_pe'] = payload['lap_pe'].to(dtype=torch.float32)
  else:
    raise ValueError('Invalid LapPE cache format: {}'.format(cache_path))

  lap_pe = payload['lap_pe']
  if expected_cache_version is not None:
    cache_version = payload.get('cache_version')
    if cache_version != expected_cache_version:
      raise ValueError(
        'LapPE cache version mismatch at {}: expected {}, got {}'.format(
          cache_path, expected_cache_version, cache_version
        )
      )
  if lap_pe.dim() != 2:
    raise ValueError('LapPE cache must have shape [num_subgraphs, lap_pe_dim], got {}'.format(tuple(lap_pe.shape)))
  if expected_dim is not None and int(lap_pe.size(1)) != int(expected_dim):
    raise ValueError(
      'LapPE cache dim mismatch at {}: expected {}, got {}'.format(
        cache_path, int(expected_dim), int(lap_pe.size(1))
      )
    )
  if expected_num_subgraphs is not None and int(lap_pe.size(0)) != int(expected_num_subgraphs):
    raise ValueError(
      'LapPE cache row mismatch at {}: expected {}, got {}'.format(
        cache_path, int(expected_num_subgraphs), int(lap_pe.size(0))
      )
    )
  return payload


def get_or_build_lappe(dataset, hparams, data_name):
  lap_pe_dim = int(getattr(hparams, 'lap_pe_dim', 16))
  coarse_topk = int(getattr(hparams, 'coarse_topk', 20))
  num_subgraphs = len(dataset.get('subgraph_structures', []))
  cache_path = resolve_lappe_cache_path(hparams, data_name)

  if os.path.exists(cache_path):
    try:
      logging.warning('Loading LapPE cache from {}'.format(cache_path))
      return load_lappe_cache(
        cache_path,
        expected_dim=lap_pe_dim,
        expected_num_subgraphs=num_subgraphs,
        expected_cache_version=LAPPE_CACHE_VERSION,
      )
    except ValueError as exc:
      logging.warning('Rebuilding incompatible LapPE cache: {}'.format(exc))

  logging.warning('Building LapPE cache at {}'.format(cache_path))
  payload = build_lappe_payload(
    original_graph=dataset.get('original_graph', None),
    assignment_matrix=dataset.get('assignment_matrix', None),
    num_subgraphs=num_subgraphs,
    subgraph_structures=dataset.get('subgraph_structures', None),
    lap_pe_dim=lap_pe_dim,
    coarse_topk=coarse_topk,
  )
  os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
  torch.save(payload, cache_path)
  return payload


def build_lappe_payload(
    original_graph,
    assignment_matrix,
    num_subgraphs,
    subgraph_structures=None,
    lap_pe_dim=16,
    coarse_topk=20,
):
  if original_graph is None:
    raise ValueError('original_graph is required to build LapPE.')
  if assignment_matrix is None:
    raise ValueError('assignment_matrix is required to build LapPE.')
  if num_subgraphs <= 0:
    raise ValueError('num_subgraphs must be positive.')

  source_cluster_ids = np.arange(num_subgraphs, dtype=np.int64)
  alignment_diagnostics = {}
  if subgraph_structures is not None:
    node_to_subgraph, node_counts, source_cluster_ids, alignment_diagnostics = (
      align_assignment_to_subgraphs(
        original_graph,
        assignment_matrix,
        subgraph_structures,
      )
    )
  else:
    node_to_subgraph, node_counts = _assignment_to_node_map(assignment_matrix, num_subgraphs)
  coarse_adj = build_coarse_adjacency(original_graph, node_to_subgraph, node_counts)
  coarse_adj = coarse_adj.tolil()
  coarse_adj.setdiag(0.0)
  coarse_adj = coarse_adj.tocsr()
  coarse_adj.eliminate_zeros()

  pruned_adj = topk_rows(coarse_adj, int(coarse_topk))
  sym_adj = pruned_adj.maximum(pruned_adj.T).tocsr()
  laplacian, zero_degree_count = normalized_laplacian(sym_adj)
  eigenvalues, lap_pe = compute_lappe(laplacian, int(lap_pe_dim))

  kept_eigenvalues = eigenvalues[1:1 + int(lap_pe_dim)]
  diagnostics = {
    'num_subgraphs': int(num_subgraphs),
    'lap_pe_dim': int(lap_pe_dim),
    'coarse_topk': int(coarse_topk),
    'zero_degree_coarse_nodes': int(zero_degree_count),
    'coarse_edges_before_topk': int(coarse_adj.nnz),
    'coarse_edges_after_topk': int(pruned_adj.nnz),
    'coarse_edges_after_sym': int(sym_adj.nnz),
  }
  diagnostics.update(alignment_diagnostics)
  return {
    'cache_version': LAPPE_CACHE_VERSION,
    'lap_pe': torch.tensor(lap_pe, dtype=torch.float32),
    'eigenvalues': torch.tensor(eigenvalues, dtype=torch.float32),
    'kept_eigenvalues': torch.tensor(kept_eigenvalues, dtype=torch.float32),
    'source_cluster_ids': torch.tensor(source_cluster_ids, dtype=torch.long),
    'diagnostics': diagnostics,
  }


def align_assignment_to_subgraphs(original_graph, assignment_matrix, subgraph_structures):
  if not subgraph_structures:
    raise ValueError('subgraph_structures must not be empty.')

  source_assignment, source_cluster_count = _assignment_to_source_map(assignment_matrix)
  graph_nodes = list(original_graph.nodes())
  use_direct_node_ids = all(_is_valid_node_index(node, len(source_assignment)) for node in graph_nodes)
  node_to_row = None if use_direct_node_ids else {node: idx for idx, node in enumerate(graph_nodes)}

  source_cluster_ids = []
  alignment_mismatch_count = 0
  aligned_node_count = 0
  for subgraph_idx, subgraph in enumerate(subgraph_structures):
    rows = []
    for node in subgraph.nodes():
      row_idx = (
        int(node)
        if use_direct_node_ids and _is_valid_node_index(node, len(source_assignment))
        else node_to_row.get(node, -1)
      )
      if row_idx < 0 or row_idx >= len(source_assignment):
        raise ValueError(
          'Subgraph {} contains node {!r} that cannot be mapped to assignment_matrix.'.format(
            subgraph_idx, node
          )
        )
      rows.append(row_idx)

    assigned = source_assignment[np.asarray(rows, dtype=np.int64)]
    assigned = assigned[assigned >= 0]
    if assigned.size == 0:
      raise ValueError('Subgraph {} has no valid assignment rows.'.format(subgraph_idx))

    values, counts = np.unique(assigned, return_counts=True)
    source_cluster = int(values[int(np.argmax(counts))])
    source_cluster_ids.append(source_cluster)
    alignment_mismatch_count += int(np.count_nonzero(assigned != source_cluster))
    aligned_node_count += int(assigned.size)

  if len(set(source_cluster_ids)) != len(source_cluster_ids):
    raise ValueError('Multiple subgraphs map to the same assignment cluster.')

  source_to_subgraph = np.full((source_cluster_count,), -1, dtype=np.int64)
  for subgraph_idx, source_cluster in enumerate(source_cluster_ids):
    if source_cluster < 0 or source_cluster >= source_cluster_count:
      raise ValueError(
        'Subgraph {} maps to invalid assignment cluster {}.'.format(
          subgraph_idx, source_cluster
        )
      )
    source_to_subgraph[source_cluster] = subgraph_idx

  node_to_subgraph = np.full(source_assignment.shape, -1, dtype=np.int64)
  valid = (source_assignment >= 0) & (source_assignment < source_cluster_count)
  node_to_subgraph[valid] = source_to_subgraph[source_assignment[valid]]
  mapped = node_to_subgraph >= 0
  node_counts = np.bincount(
    node_to_subgraph[mapped],
    minlength=len(subgraph_structures),
  ).astype(np.float64, copy=False)

  active_source_clusters = np.unique(source_assignment[source_assignment >= 0])
  diagnostics = {
    'assignment_cluster_count': int(source_cluster_count),
    'active_assignment_clusters': int(active_source_clusters.size),
    'mapped_assignment_clusters': int(len(source_cluster_ids)),
    'unmapped_assignment_clusters': int(
      np.count_nonzero(~np.isin(active_source_clusters, source_cluster_ids))
    ),
    'alignment_mismatch_nodes': int(alignment_mismatch_count),
    'alignment_checked_nodes': int(aligned_node_count),
  }
  return (
    node_to_subgraph,
    node_counts,
    np.asarray(source_cluster_ids, dtype=np.int64),
    diagnostics,
  )


def _assignment_to_source_map(assignment_matrix):
  assignment = assignment_matrix
  if sp.issparse(assignment):
    assignment = assignment.tocsr()
    if assignment.ndim != 2:
      raise ValueError('Sparse assignment_matrix must be 2D.')
    source_assignment = np.full((assignment.shape[0],), -1, dtype=np.int64)
    for row_idx in range(assignment.shape[0]):
      start, end = assignment.indptr[row_idx], assignment.indptr[row_idx + 1]
      if start == end:
        continue
      local_data = assignment.data[start:end]
      local_cols = assignment.indices[start:end]
      source_assignment[row_idx] = int(local_cols[int(np.argmax(local_data))])
    return source_assignment, int(assignment.shape[1])

  assignment = np.asarray(assignment)
  if assignment.ndim == 1:
    source_assignment = assignment.astype(np.int64, copy=True)
    valid = source_assignment >= 0
    source_cluster_count = int(source_assignment[valid].max()) + 1 if np.any(valid) else 0
    return source_assignment, source_cluster_count
  if assignment.ndim == 2:
    row_sum = assignment.sum(axis=1)
    source_assignment = assignment.argmax(axis=1).astype(np.int64, copy=False)
    source_assignment = np.asarray(source_assignment, dtype=np.int64)
    source_assignment[row_sum <= 0] = -1
    return source_assignment, int(assignment.shape[1])
  raise ValueError('assignment_matrix must be 1D or 2D.')


def _assignment_to_node_map(assignment_matrix, num_subgraphs):
  assignment = assignment_matrix
  if sp.issparse(assignment):
    assignment = assignment.tocsr()
    if assignment.ndim != 2:
      raise ValueError('Sparse assignment_matrix must be 2D.')
    if assignment.shape[1] < num_subgraphs:
      raise ValueError('assignment_matrix has fewer columns than num_subgraphs.')
    node_to_subgraph = np.full((assignment.shape[0],), -1, dtype=np.int64)
    for row_idx in range(assignment.shape[0]):
      start, end = assignment.indptr[row_idx], assignment.indptr[row_idx + 1]
      if start == end:
        continue
      local_data = assignment.data[start:end]
      local_cols = assignment.indices[start:end]
      chosen = int(local_cols[int(np.argmax(local_data))])
      if 0 <= chosen < num_subgraphs:
        node_to_subgraph[row_idx] = chosen
  else:
    assignment = np.asarray(assignment)
    if assignment.ndim == 1:
      node_to_subgraph = assignment.astype(np.int64, copy=True)
      node_to_subgraph[(node_to_subgraph < 0) | (node_to_subgraph >= num_subgraphs)] = -1
    elif assignment.ndim == 2:
      if assignment.shape[1] < num_subgraphs:
        raise ValueError('assignment_matrix has fewer columns than num_subgraphs.')
      active = assignment[:, :num_subgraphs]
      row_sum = active.sum(axis=1)
      node_to_subgraph = active.argmax(axis=1).astype(np.int64, copy=False)
      node_to_subgraph = np.asarray(node_to_subgraph, dtype=np.int64)
      node_to_subgraph[row_sum <= 0] = -1
    else:
      raise ValueError('assignment_matrix must be 1D or 2D.')

  valid = (node_to_subgraph >= 0) & (node_to_subgraph < num_subgraphs)
  node_counts = np.bincount(node_to_subgraph[valid], minlength=num_subgraphs).astype(np.float64, copy=False)
  return node_to_subgraph, node_counts


def build_coarse_adjacency(original_graph, node_to_subgraph, node_counts):
  num_subgraphs = int(node_counts.shape[0])
  row, col, data = [], [], []
  graph_nodes = list(original_graph.nodes())
  use_direct_node_ids = all(_is_valid_node_index(node, len(node_to_subgraph)) for node in graph_nodes)
  node_to_row = None if use_direct_node_ids else {node: idx for idx, node in enumerate(graph_nodes)}

  def _subgraph_for_node(node):
    if use_direct_node_ids:
      row_idx = int(node)
    else:
      row_idx = node_to_row.get(node, -1)
    if row_idx < 0 or row_idx >= len(node_to_subgraph):
      return -1
    return int(node_to_subgraph[row_idx])

  for u, v, attrs in original_graph.edges(data=True):
    src = _subgraph_for_node(u)
    dst = _subgraph_for_node(v)
    if src < 0 or dst < 0:
      continue
    weight = float(attrs.get('weight', 1.0)) if isinstance(attrs, dict) else 1.0
    row.append(src)
    col.append(dst)
    data.append(weight)
    if not original_graph.is_directed() and src != dst:
      row.append(dst)
      col.append(src)
      data.append(weight)

  coarse_counts = sp.coo_matrix(
    (np.asarray(data, dtype=np.float64), (np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64))),
    shape=(num_subgraphs, num_subgraphs),
    dtype=np.float64,
  ).tocsr()
  inv_counts = np.zeros_like(node_counts, dtype=np.float64)
  np.divide(1.0, node_counts, out=inv_counts, where=node_counts > 0)
  return sp.diags(inv_counts, format='csr') @ coarse_counts @ sp.diags(inv_counts, format='csr')


def _is_valid_node_index(node, node_count):
  return isinstance(node, (int, np.integer)) and 0 <= int(node) < node_count


def topk_rows(matrix, topk):
  matrix = matrix.tocsr()
  if topk <= 0 or matrix.nnz == 0:
    return sp.csr_matrix(matrix.shape, dtype=matrix.dtype)

  rows, cols, values = [], [], []
  for row_idx in range(matrix.shape[0]):
    start, end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
    if start == end:
      continue
    row_cols = matrix.indices[start:end]
    row_vals = matrix.data[start:end]
    if row_vals.size > topk:
      keep = np.argpartition(row_vals, -topk)[-topk:]
      row_cols = row_cols[keep]
      row_vals = row_vals[keep]
    rows.extend([row_idx] * row_vals.size)
    cols.extend(row_cols.tolist())
    values.extend(row_vals.tolist())

  return sp.coo_matrix((values, (rows, cols)), shape=matrix.shape, dtype=matrix.dtype).tocsr()


def normalized_laplacian(adjacency):
  adjacency = adjacency.tocsr()
  degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
  inv_sqrt_degree = np.zeros_like(degree, dtype=np.float64)
  np.divide(1.0, np.sqrt(degree), out=inv_sqrt_degree, where=degree > 0)
  norm_adj = sp.diags(inv_sqrt_degree, format='csr') @ adjacency @ sp.diags(inv_sqrt_degree, format='csr')
  laplacian = sp.eye(adjacency.shape[0], dtype=np.float64, format='csr') - norm_adj
  return laplacian, int(np.count_nonzero(degree <= 0))


def compute_lappe(laplacian, lap_pe_dim):
  num_nodes = laplacian.shape[0]
  requested = int(lap_pe_dim) + 1
  if num_nodes <= 0:
    raise ValueError('Laplacian must contain at least one node.')

  if num_nodes == 1:
    eigenvalues = np.asarray([1.0], dtype=np.float64)
    eigenvectors = np.ones((1, 1), dtype=np.float64)
  elif requested >= num_nodes:
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
  else:
    eigenvalues, eigenvectors = spla.eigsh(laplacian, k=requested, which='SM')
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

  full_eigenvalues = _pad_1d(eigenvalues[:requested], requested, fill_value=np.nan)
  pe = eigenvectors[:, 1:1 + int(lap_pe_dim)]
  if pe.shape[1] < int(lap_pe_dim):
    pe = np.concatenate(
      [pe, np.zeros((num_nodes, int(lap_pe_dim) - pe.shape[1]), dtype=pe.dtype)],
      axis=1,
    )
  return full_eigenvalues.astype(np.float32, copy=False), pe.astype(np.float32, copy=False)


def _pad_1d(values, target_size, fill_value=np.nan):
  values = np.asarray(values, dtype=np.float64)
  if values.size >= target_size:
    return values[:target_size]
  out = np.full((target_size,), fill_value, dtype=np.float64)
  out[:values.size] = values
  return out
