# coding=utf-8

"""Boundary Interaction Profile topology and feature helpers."""

from __future__ import annotations

import logging
import math
import time

import networkx as nx
import numpy as np

from MISGL.utils.boundary_context import (
    CONTEXT_FEATURE_NAMES,
    _assignment_membership,
    _augment_membership_from_subgraphs,
    _build_node_feature_map,
    _mean_feature,
    _resolve_original_node_id,
    _safe_divide,
    _safe_float,
)


BOUNDARY_PROFILE_FEATURE_NAMES = (
    'bip_boundary_node_count',
    'bip_boundary_ratio',
    'bip_boundary_cross_degree_mean',
    'bip_boundary_cross_degree_max',
    'bip_boundary_cross_degree_std',
    'bip_boundary_cross_degree_entropy',
    'bip_boundary_cross_degree_entropy_norm',
    'bip_boundary_effective_count',
    'bip_boundary_raw_feature_mean',
    'bip_boundary_raw_feature_norm',
    'bip_inner_raw_feature_mean',
    'bip_inner_raw_feature_norm',
    'bip_boundary_inner_raw_distance',
    'bip_boundary_inner_raw_cosine',
    'bip_boundary_embedding_mean',
    'bip_boundary_embedding_norm',
    'bip_inner_embedding_mean',
    'bip_inner_embedding_norm',
    'bip_boundary_inner_embedding_distance',
    'bip_boundary_inner_embedding_cosine',
    'bip_boundary_attention_mean',
    'bip_boundary_attention_max',
    'bip_boundary_attention_std',
    'bip_boundary_attention_sum',
    'bip_boundary_attention_share',
    'bip_inner_attention_mean',
    'bip_boundary_inner_attention_difference',
    'bip_top1_attention_is_boundary',
    'bip_top5_attention_boundary_ratio',
)

SEMANTIC_PROFILE_FEATURE_NAMES = (
    'bip_context_raw_feature_mean',
    'bip_context_raw_feature_norm',
    'bip_context_embedding_mean',
    'bip_context_embedding_norm',
    'bip_context_z_cosine',
    'bip_context_z_cosine_mean',
    'bip_context_z_cosine_std',
    'bip_context_z_cosine_max',
    'bip_context_boundary_embedding_cosine',
    'bip_cross_edge_embedding_cosine_mean',
    'bip_cross_edge_embedding_cosine_std',
    'bip_context_pseudo_prob_mean',
    'bip_context_pseudo_prob_std',
    'bip_context_pseudo_prob_min',
    'bip_context_pseudo_prob_max',
    'bip_context_pseudo_positive_ratio',
    'bip_context_embedding_coverage',
    'bip_context_pseudo_prob_coverage',
)

INTERACTION_PROFILE_FEATURE_NAMES = (
    'bip_cross_edge_count',
    'bip_context_node_count',
    'bip_bipartite_density',
    'bip_top1_cross_edge_share',
    'bip_top5_cross_edge_share',
    'bip_cross_edge_concentration_hhi',
    'bip_context_boundary_support_mean',
    'bip_context_boundary_support_max',
    'bip_context_multi_boundary_ratio',
    'bip_component_count',
    'bip_largest_component_node_ratio',
    'bip_largest_component_edge_ratio',
)

BIP_FEATURE_NAMES = (
    BOUNDARY_PROFILE_FEATURE_NAMES
    + SEMANTIC_PROFILE_FEATURE_NAMES
    + INTERACTION_PROFILE_FEATURE_NAMES
)


def safe_cosine(left, right):
    if left is None or right is None:
        return 0.0
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    dim = min(left.size, right.size)
    if dim <= 0:
        return 0.0
    left = left[:dim]
    right = right[:dim]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return _safe_float(np.dot(left, right) / denom)


def safe_distance(left, right):
    if left is None or right is None:
        return 0.0
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    dim = min(left.size, right.size)
    if dim <= 0:
        return 0.0
    return _safe_float(np.linalg.norm(left[:dim] - right[:dim]))


def distribution_stats(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {
            'mean': 0.0,
            'max': 0.0,
            'std': 0.0,
            'entropy': 0.0,
            'entropy_norm': 0.0,
            'effective_count': 0.0,
            'top1_share': 0.0,
            'top5_share': 0.0,
            'hhi': 0.0,
        }
    total = float(values.sum())
    probabilities = values / total if total > 0.0 else np.zeros_like(values)
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
    entropy_norm = _safe_divide(entropy, math.log(values.size)) if values.size > 1 else 0.0
    sorted_values = np.sort(values)[::-1]
    return {
        'mean': float(np.mean(values)),
        'max': float(np.max(values)),
        'std': float(np.std(values)),
        'entropy': entropy,
        'entropy_norm': entropy_norm,
        'effective_count': float(math.exp(entropy)) if total > 0.0 else 0.0,
        'top1_share': _safe_divide(sorted_values[:1].sum(), total),
        'top5_share': _safe_divide(sorted_values[:5].sum(), total),
        'hhi': float(np.sum(probabilities ** 2)) if total > 0.0 else 0.0,
    }


class _UnionFind(object):

    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]


def bipartite_component_stats(cross_edges):
    if not cross_edges:
        return {
            'component_count': 0,
            'largest_component_node_ratio': 0.0,
            'largest_component_edge_ratio': 0.0,
        }

    union_find = _UnionFind()
    tagged_edges = []
    for boundary_node, context_node in cross_edges:
        left = ('b', boundary_node)
        right = ('c', context_node)
        union_find.union(left, right)
        tagged_edges.append((left, right))

    component_nodes = {}
    for node in union_find.parent:
        root = union_find.find(node)
        component_nodes[root] = component_nodes.get(root, 0) + 1

    component_edges = {}
    for left, _ in tagged_edges:
        root = union_find.find(left)
        component_edges[root] = component_edges.get(root, 0) + 1

    total_nodes = len(union_find.parent)
    total_edges = len(tagged_edges)
    return {
        'component_count': len(component_nodes),
        'largest_component_node_ratio': _safe_divide(max(component_nodes.values()), total_nodes),
        'largest_component_edge_ratio': _safe_divide(max(component_edges.values()), total_edges),
    }


def _raw_feature_stats(nodes, node_feature_map):
    mean_vector = _mean_feature(nodes, node_feature_map)
    if mean_vector is None:
        return 0.0, 0.0, None
    return (
        _safe_float(np.mean(mean_vector)),
        _safe_float(np.linalg.norm(mean_vector)),
        mean_vector,
    )


def build_boundary_interaction_topology(
    dataset_raw,
    log_progress=False,
    progress_interval=1000,
    max_cross_edge_samples=2048,
):
    """Build reusable boundary/context topology and static BIP features."""
    if dataset_raw is None:
        raise ValueError('dataset_raw is required for Boundary Interaction Profile analysis.')

    original_graph = dataset_raw.get('original_graph', None)
    subgraphs = dataset_raw.get('subgraph_structures', None)
    assignment_matrix = dataset_raw.get('assignment_matrix', None)
    if original_graph is None:
        raise ValueError('dataset_raw must contain original_graph for BIP analysis.')
    if subgraphs is None:
        raise ValueError('dataset_raw must contain subgraph_structures for BIP analysis.')
    if assignment_matrix is None:
        raise ValueError('dataset_raw must contain assignment_matrix for BIP analysis.')
    if not isinstance(original_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError('dataset_raw["original_graph"] must be a NetworkX graph.')

    start_time = time.perf_counter()
    original_nodes = list(original_graph.nodes())
    node_to_position = {node_id: idx for idx, node_id in enumerate(original_nodes)}
    node_to_subgraphs, subgraph_to_nodes = _assignment_membership(
        assignment_matrix,
        original_nodes,
        len(subgraphs),
    )
    _augment_membership_from_subgraphs(subgraphs, node_to_subgraphs, subgraph_to_nodes)
    node_feature_map = _build_node_feature_map(original_graph, subgraphs)
    work_graph = original_graph.to_undirected(as_view=True) if original_graph.is_directed() else original_graph

    if log_progress:
        logging.warning(
            '[BIP] topology start: nodes=%d, edges=%d, subgraphs=%d',
            len(original_nodes),
            original_graph.number_of_edges(),
            len(subgraphs),
        )

    profiles = {}
    for subgraph_idx, subgraph in enumerate(subgraphs):
        members = set(subgraph_to_nodes[subgraph_idx])
        if not members:
            members = {
                _resolve_original_node_id(node_id, attrs)
                for node_id, attrs in subgraph.nodes(data=True)
            }

        boundary_nodes = set()
        context_nodes = set()
        cross_edges = []
        boundary_cross_degree = {}
        context_boundary_support = {}

        for node_id in members:
            if node_id not in work_graph:
                continue
            for neighbor_id in work_graph.neighbors(node_id):
                if neighbor_id in members:
                    continue
                boundary_nodes.add(node_id)
                context_nodes.add(neighbor_id)
                cross_edges.append((node_id, neighbor_id))
                boundary_cross_degree[node_id] = boundary_cross_degree.get(node_id, 0) + 1
                context_boundary_support.setdefault(neighbor_id, set()).add(node_id)

        inner_nodes = members - boundary_nodes
        cross_degree_stats = distribution_stats(boundary_cross_degree.values())
        support_values = [len(nodes) for nodes in context_boundary_support.values()]
        component_stats = bipartite_component_stats(cross_edges)

        boundary_raw_mean_scalar, boundary_raw_norm, boundary_raw_mean = _raw_feature_stats(
            boundary_nodes,
            node_feature_map,
        )
        inner_raw_mean_scalar, inner_raw_norm, inner_raw_mean = _raw_feature_stats(
            inner_nodes,
            node_feature_map,
        )
        context_raw_mean_scalar, context_raw_norm, context_raw_mean = _raw_feature_stats(
            context_nodes,
            node_feature_map,
        )
        boundary_degrees = [float(work_graph.degree(node_id)) for node_id in boundary_nodes]
        context_degrees = [float(work_graph.degree(node_id)) for node_id in context_nodes]

        boundary_count = len(boundary_nodes)
        context_count = len(context_nodes)
        cross_edge_count = len(cross_edges)
        static_features = {
            'bip_boundary_node_count': float(boundary_count),
            'bip_boundary_ratio': _safe_divide(boundary_count, len(members)),
            'bip_boundary_cross_degree_mean': cross_degree_stats['mean'],
            'bip_boundary_cross_degree_max': cross_degree_stats['max'],
            'bip_boundary_cross_degree_std': cross_degree_stats['std'],
            'bip_boundary_cross_degree_entropy': cross_degree_stats['entropy'],
            'bip_boundary_cross_degree_entropy_norm': cross_degree_stats['entropy_norm'],
            'bip_boundary_effective_count': cross_degree_stats['effective_count'],
            'bip_boundary_raw_feature_mean': boundary_raw_mean_scalar,
            'bip_boundary_raw_feature_norm': boundary_raw_norm,
            'bip_inner_raw_feature_mean': inner_raw_mean_scalar,
            'bip_inner_raw_feature_norm': inner_raw_norm,
            'bip_boundary_inner_raw_distance': safe_distance(boundary_raw_mean, inner_raw_mean),
            'bip_boundary_inner_raw_cosine': safe_cosine(boundary_raw_mean, inner_raw_mean),
            'bip_context_raw_feature_mean': context_raw_mean_scalar,
            'bip_context_raw_feature_norm': context_raw_norm,
            'bip_cross_edge_count': float(cross_edge_count),
            'bip_context_node_count': float(context_count),
            'bip_bipartite_density': _safe_divide(
                cross_edge_count,
                boundary_count * context_count,
            ),
            'bip_top1_cross_edge_share': cross_degree_stats['top1_share'],
            'bip_top5_cross_edge_share': cross_degree_stats['top5_share'],
            'bip_cross_edge_concentration_hhi': cross_degree_stats['hhi'],
            'bip_context_boundary_support_mean': float(np.mean(support_values)) if support_values else 0.0,
            'bip_context_boundary_support_max': float(np.max(support_values)) if support_values else 0.0,
            'bip_context_multi_boundary_ratio': (
                float(np.mean(np.asarray(support_values) > 1)) if support_values else 0.0
            ),
            'bip_component_count': float(component_stats['component_count']),
            'bip_largest_component_node_ratio': component_stats['largest_component_node_ratio'],
            'bip_largest_component_edge_ratio': component_stats['largest_component_edge_ratio'],
        }
        current_context_features = {
            'boundary_node_count': float(boundary_count),
            'boundary_ratio': _safe_divide(boundary_count, len(members)),
            'cross_edge_count': float(cross_edge_count),
            'cross_edge_density': _safe_divide(cross_edge_count, boundary_count * context_count),
            'context_node_count': float(context_count),
            'context_ratio': _safe_divide(context_count, len(members) + context_count),
            'boundary_degree_mean': float(np.mean(boundary_degrees)) if boundary_degrees else 0.0,
            'boundary_degree_max': float(np.max(boundary_degrees)) if boundary_degrees else 0.0,
            'context_degree_mean': float(np.mean(context_degrees)) if context_degrees else 0.0,
            'context_degree_max': float(np.max(context_degrees)) if context_degrees else 0.0,
            'boundary_feature_mean': boundary_raw_mean_scalar,
            'context_feature_mean': context_raw_mean_scalar,
            'boundary_feature_norm': boundary_raw_norm,
            'context_feature_norm': context_raw_norm,
            'context_minus_boundary_feature_norm': safe_distance(
                context_raw_mean,
                boundary_raw_mean,
            ),
            'context_to_boundary_count_ratio': _safe_divide(context_count, boundary_count),
        }

        member_positions = np.asarray(
            [node_to_position[node_id] for node_id in members if node_id in node_to_position],
            dtype=np.int32,
        )
        boundary_positions = np.asarray(
            [node_to_position[node_id] for node_id in boundary_nodes if node_id in node_to_position],
            dtype=np.int32,
        )
        inner_positions = np.asarray(
            [node_to_position[node_id] for node_id in inner_nodes if node_id in node_to_position],
            dtype=np.int32,
        )
        context_positions = np.asarray(
            [node_to_position[node_id] for node_id in context_nodes if node_id in node_to_position],
            dtype=np.int32,
        )
        sampled_cross_edges = cross_edges
        if len(sampled_cross_edges) > int(max_cross_edge_samples):
            sample_indices = np.linspace(
                0,
                len(sampled_cross_edges) - 1,
                num=int(max_cross_edge_samples),
                dtype=np.int64,
            )
            sampled_cross_edges = [sampled_cross_edges[int(idx)] for idx in sample_indices]
        sampled_cross_edge_positions = np.asarray(
            [
                (node_to_position[left], node_to_position[right])
                for left, right in sampled_cross_edges
                if left in node_to_position and right in node_to_position
            ],
            dtype=np.int32,
        )
        if sampled_cross_edge_positions.size == 0:
            sampled_cross_edge_positions = np.zeros((0, 2), dtype=np.int32)
        else:
            sampled_cross_edge_positions = sampled_cross_edge_positions.reshape(-1, 2)

        profiles[int(subgraph_idx)] = {
            'orig_graph_idx': int(subgraph_idx),
            'member_positions': member_positions,
            'boundary_positions': boundary_positions,
            'inner_positions': inner_positions,
            'context_positions': context_positions,
            'sampled_cross_edge_positions': sampled_cross_edge_positions,
            'static_features': static_features,
            'current_context_features': current_context_features,
        }

        if log_progress and (subgraph_idx + 1) % max(int(progress_interval), 1) == 0:
            logging.warning(
                '[BIP] topology processed %d/%d subgraphs, elapsed=%.1fs',
                subgraph_idx + 1,
                len(subgraphs),
                time.perf_counter() - start_time,
            )

    if log_progress:
        logging.warning(
            '[BIP] topology ready: profiles=%d, elapsed=%.1fs',
            len(profiles),
            time.perf_counter() - start_time,
        )

    return {
        'profiles': profiles,
        'original_nodes': original_nodes,
        'node_to_position': node_to_position,
        'node_feature_map': node_feature_map,
        'node_to_subgraphs': node_to_subgraphs,
        'subgraph_to_nodes': subgraph_to_nodes,
    }


def feature_vector(row, feature_names):
    return np.asarray(
        [_safe_float(row.get(name, 0.0)) for name in feature_names],
        dtype=np.float32,
    )


def current_context_vector(row):
    return feature_vector(row, CONTEXT_FEATURE_NAMES)


__all__ = [
    'BIP_FEATURE_NAMES',
    'BOUNDARY_PROFILE_FEATURE_NAMES',
    'SEMANTIC_PROFILE_FEATURE_NAMES',
    'INTERACTION_PROFILE_FEATURE_NAMES',
    'build_boundary_interaction_topology',
    'bipartite_component_stats',
    'current_context_vector',
    'distribution_stats',
    'feature_vector',
    'safe_cosine',
    'safe_distance',
]
