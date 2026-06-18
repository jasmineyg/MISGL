# coding=utf-8

"""Boundary/context feature extraction for subgraph-level analysis."""

from __future__ import annotations

import math
import logging
import time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import networkx as nx
import numpy as np


ORIGINAL_NODE_KEYS = (
    'original_id',
    'original_index',
    'orig_id',
    'orig_idx',
    'node_index',
    'node_id',
    'original_node_id',
)

NODE_FEATURE_KEYS = (
    'features',
    'feature',
    'x',
    'feat',
    'node_features',
)

CONTEXT_FEATURE_NAMES = (
    'boundary_node_count',
    'boundary_ratio',
    'cross_edge_count',
    'cross_edge_density',
    'context_node_count',
    'context_ratio',
    'boundary_degree_mean',
    'boundary_degree_max',
    'context_degree_mean',
    'context_degree_max',
    'boundary_feature_mean',
    'context_feature_mean',
    'boundary_feature_norm',
    'context_feature_norm',
    'context_minus_boundary_feature_norm',
    'context_to_boundary_count_ratio',
)

DETAIL_FEATURE_NAMES = (
    'subgraph_node_count',
    'internal_edge_count',
    'external_neighbor_count',
    'boundary_internal_edge_count',
)


def _safe_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_divide(numerator, denominator):
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return float(numerator) / denominator


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _node_feature_from_attrs(attrs):
    for key in NODE_FEATURE_KEYS:
        if key not in attrs:
            continue
        value = attrs.get(key)
        if value is None:
            continue
        arr = _to_numpy(value)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return arr
    return None


def _resolve_original_node_id(node_id, attrs=None):
    attrs = attrs or {}
    for key in ORIGINAL_NODE_KEYS:
        if key in attrs and attrs[key] is not None:
            try:
                return int(attrs[key])
            except (TypeError, ValueError):
                return attrs[key]
    if isinstance(node_id, (int, np.integer)):
        return int(node_id)
    return node_id


def _assignment_membership(assignment_matrix, original_nodes, subgraph_count):
    if assignment_matrix is None:
        raise ValueError('dataset_raw must contain assignment_matrix for boundary/context analysis.')

    node_to_subgraphs: Dict[object, Set[int]] = {}
    subgraph_to_nodes: List[Set[object]] = [set() for _ in range(subgraph_count)]

    if hasattr(assignment_matrix, 'tocoo'):
        coo = assignment_matrix.tocoo()
        rows = np.asarray(coo.row)
        cols = np.asarray(coo.col)
        data = np.asarray(coo.data)
        active = data != 0
        row_col_iter = zip(rows[active], cols[active])
    else:
        matrix = _to_numpy(assignment_matrix)
        if matrix is None or matrix.ndim != 2:
            raise ValueError('assignment_matrix must be a 2-D matrix.')
        rows, cols = np.nonzero(matrix)
        row_col_iter = zip(rows, cols)

    for row_idx, subgraph_idx in row_col_iter:
        subgraph_idx = int(subgraph_idx)
        if subgraph_idx < 0 or subgraph_idx >= subgraph_count:
            continue
        row_idx = int(row_idx)
        node_id = original_nodes[row_idx] if row_idx < len(original_nodes) else row_idx
        node_to_subgraphs.setdefault(node_id, set()).add(subgraph_idx)
        subgraph_to_nodes[subgraph_idx].add(node_id)

    return node_to_subgraphs, subgraph_to_nodes


def _augment_membership_from_subgraphs(subgraphs, node_to_subgraphs, subgraph_to_nodes):
    for subgraph_idx, subgraph in enumerate(subgraphs):
        for node_id, attrs in subgraph.nodes(data=True):
            orig_node_id = _resolve_original_node_id(node_id, attrs)
            node_to_subgraphs.setdefault(orig_node_id, set()).add(subgraph_idx)
            subgraph_to_nodes[subgraph_idx].add(orig_node_id)


def _build_node_feature_map(original_graph, subgraphs):
    node_feature_map: Dict[object, np.ndarray] = {}

    for node_id, attrs in original_graph.nodes(data=True):
        feature = _node_feature_from_attrs(attrs)
        if feature is not None:
            node_feature_map[node_id] = feature

    for subgraph in subgraphs:
        for node_id, attrs in subgraph.nodes(data=True):
            orig_node_id = _resolve_original_node_id(node_id, attrs)
            feature = _node_feature_from_attrs(attrs)
            if feature is not None and orig_node_id not in node_feature_map:
                node_feature_map[orig_node_id] = feature

    return node_feature_map


def _mean_feature(nodes, node_feature_map):
    features = [
        np.asarray(node_feature_map[node_id], dtype=np.float32).reshape(-1)
        for node_id in nodes
        if node_id in node_feature_map
    ]
    if not features:
        return None

    min_dim = min(feature.size for feature in features)
    if min_dim <= 0:
        return None
    trimmed = np.stack([feature[:min_dim] for feature in features], axis=0)
    return np.mean(trimmed, axis=0)


def _feature_scalar_stats(nodes, node_feature_map):
    mean_vec = _mean_feature(nodes, node_feature_map)
    if mean_vec is None:
        return 0.0, 0.0, None
    return _safe_float(np.mean(mean_vec)), _safe_float(np.linalg.norm(mean_vec)), mean_vec


def _degrees(graph, nodes):
    values = []
    for node_id in nodes:
        if node_id in graph:
            values.append(float(graph.degree(node_id)))
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.max(values))


def _undirected_edge_key(src, dst):
    if src == dst:
        return (src, dst)
    return frozenset((src, dst))


def compute_boundary_context_rows(dataset_raw, log_progress=False, progress_interval=1000):
    """Return one boundary/context stats row per subgraph.

    The returned rows are keyed by subgraph index through the ``orig_graph_idx``
    column. Node-to-subgraph membership allows multiple subgraph assignments.
    """
    if dataset_raw is None:
        raise ValueError('dataset_raw is required for boundary/context analysis.')

    original_graph = dataset_raw.get('original_graph', None)
    if original_graph is None:
        raise ValueError('dataset_raw must contain original_graph for boundary/context analysis.')
    if not isinstance(original_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError('dataset_raw["original_graph"] must be a NetworkX graph.')

    subgraphs = dataset_raw.get('subgraph_structures', None)
    if subgraphs is None:
        raise ValueError('dataset_raw must contain subgraph_structures for boundary/context analysis.')

    original_nodes = list(original_graph.nodes())
    subgraph_count = len(subgraphs)
    start_time = time.perf_counter()
    if log_progress:
        assignment = dataset_raw.get('assignment_matrix', None)
        assignment_shape = getattr(assignment, 'shape', None)
        logging.warning(
            '[BoundaryContext] start: original_nodes=%d, original_edges=%d, subgraphs=%d, assignment_shape=%s',
            len(original_nodes),
            original_graph.number_of_edges(),
            subgraph_count,
            assignment_shape,
        )
    node_to_subgraphs, subgraph_to_nodes = _assignment_membership(
        dataset_raw.get('assignment_matrix', None),
        original_nodes,
        subgraph_count,
    )
    if log_progress:
        assigned_count = sum(len(nodes) for nodes in subgraph_to_nodes)
        logging.warning(
            '[BoundaryContext] assignment membership ready: assigned_pairs=%d, elapsed=%.1fs',
            assigned_count,
            time.perf_counter() - start_time,
        )
    _augment_membership_from_subgraphs(subgraphs, node_to_subgraphs, subgraph_to_nodes)
    if log_progress:
        assigned_count = sum(len(nodes) for nodes in subgraph_to_nodes)
        logging.warning(
            '[BoundaryContext] subgraph membership augmented: assigned_pairs=%d, elapsed=%.1fs',
            assigned_count,
            time.perf_counter() - start_time,
        )

    work_graph = original_graph.to_undirected(as_view=True) if original_graph.is_directed() else original_graph
    node_feature_map = _build_node_feature_map(original_graph, subgraphs)
    subgraph_labels = dataset_raw.get('subgraph_labels', None)
    if log_progress:
        logging.warning(
            '[BoundaryContext] node feature map ready: feature_nodes=%d, elapsed=%.1fs',
            len(node_feature_map),
            time.perf_counter() - start_time,
        )

    rows = []
    for subgraph_idx, subgraph in enumerate(subgraphs):
        members = set(subgraph_to_nodes[subgraph_idx])
        if not members:
            for node_id, attrs in subgraph.nodes(data=True):
                members.add(_resolve_original_node_id(node_id, attrs))

        boundary_nodes = set()
        context_nodes = set()
        external_neighbors = set()
        cross_edges = set()
        boundary_internal_edges = set()

        for node_id in members:
            if node_id not in work_graph:
                continue
            for neighbor_id in work_graph.neighbors(node_id):
                if neighbor_id in members:
                    boundary_internal_edges.add(_undirected_edge_key(node_id, neighbor_id))
                    continue
                boundary_nodes.add(node_id)
                context_nodes.add(neighbor_id)
                external_neighbors.add(neighbor_id)
                cross_edges.add(_undirected_edge_key(node_id, neighbor_id))

        boundary_feature_mean, boundary_feature_norm, boundary_mean_vec = _feature_scalar_stats(
            boundary_nodes,
            node_feature_map,
        )
        context_feature_mean, context_feature_norm, context_mean_vec = _feature_scalar_stats(
            context_nodes,
            node_feature_map,
        )
        if boundary_mean_vec is not None and context_mean_vec is not None:
            min_dim = min(boundary_mean_vec.size, context_mean_vec.size)
            context_minus_boundary_feature_norm = _safe_float(
                np.linalg.norm(context_mean_vec[:min_dim] - boundary_mean_vec[:min_dim])
            )
        else:
            context_minus_boundary_feature_norm = 0.0

        boundary_degree_mean, boundary_degree_max = _degrees(work_graph, boundary_nodes)
        context_degree_mean, context_degree_max = _degrees(work_graph, context_nodes)

        subgraph_node_count = len(members)
        boundary_node_count = len(boundary_nodes)
        context_node_count = len(context_nodes)
        cross_edge_count = len(cross_edges)
        subgraph_id = subgraph.graph.get('subgraph_id', subgraph_idx)
        try:
            subgraph_id = int(subgraph_id)
        except (TypeError, ValueError):
            pass

        label = subgraph.graph.get('label', None)
        if subgraph_labels is not None and 0 <= subgraph_idx < len(subgraph_labels):
            try:
                label = int(subgraph_labels[subgraph_idx])
            except (TypeError, ValueError):
                pass

        row = {
            'orig_graph_idx': int(subgraph_idx),
            'subgraph_id': subgraph_id,
            'label_from_context_table': label,
            'subgraph_node_count': int(subgraph_node_count),
            'internal_edge_count': int(len(boundary_internal_edges)),
            'external_neighbor_count': int(len(external_neighbors)),
            'boundary_internal_edge_count': int(len(boundary_internal_edges)),
            'boundary_node_count': int(boundary_node_count),
            'boundary_ratio': _safe_divide(boundary_node_count, subgraph_node_count),
            'cross_edge_count': int(cross_edge_count),
            'cross_edge_density': _safe_divide(
                cross_edge_count,
                max(boundary_node_count * max(context_node_count, 1), 1),
            ),
            'context_node_count': int(context_node_count),
            'context_ratio': _safe_divide(context_node_count, subgraph_node_count + context_node_count),
            'boundary_degree_mean': boundary_degree_mean,
            'boundary_degree_max': boundary_degree_max,
            'context_degree_mean': context_degree_mean,
            'context_degree_max': context_degree_max,
            'boundary_feature_mean': boundary_feature_mean,
            'context_feature_mean': context_feature_mean,
            'boundary_feature_norm': boundary_feature_norm,
            'context_feature_norm': context_feature_norm,
            'context_minus_boundary_feature_norm': context_minus_boundary_feature_norm,
            'context_to_boundary_count_ratio': _safe_divide(context_node_count, max(boundary_node_count, 1)),
        }
        rows.append(row)
        if log_progress and (subgraph_idx + 1) % int(progress_interval) == 0:
            logging.warning(
                '[BoundaryContext] processed %d/%d subgraphs, elapsed=%.1fs',
                subgraph_idx + 1,
                subgraph_count,
                time.perf_counter() - start_time,
            )

    if log_progress:
        logging.warning(
            '[BoundaryContext] done: rows=%d, elapsed=%.1fs',
            len(rows),
            time.perf_counter() - start_time,
        )
    return rows


def rows_by_orig_graph_idx(rows):
    return {int(row['orig_graph_idx']): dict(row) for row in rows}


def context_feature_vector(row):
    return np.asarray([_safe_float(row.get(name, 0.0)) for name in CONTEXT_FEATURE_NAMES], dtype=np.float32)


__all__ = [
    'CONTEXT_FEATURE_NAMES',
    'DETAIL_FEATURE_NAMES',
    'compute_boundary_context_rows',
    'context_feature_vector',
    'rows_by_orig_graph_idx',
]
