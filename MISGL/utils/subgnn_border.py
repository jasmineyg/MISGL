# coding=utf-8

"""SubGNN border-structure utilities for MISGL.

This module follows the SubGNN structure-channel design more closely than the
first lightweight router: structure anchors are sampled by triangular random
walks, border walks are generated over the anchor border/external neighborhood,
and anchor representations are later encoded from walk node features by a
BiLSTM. The graph-only work remains in data loading.
"""

import json
import os
import random

import numpy as np


_ORIGINAL_NODE_KEYS = (
    'original_id',
    'original_index',
    'orig_id',
    'orig_idx',
    'node_index',
    'node_id',
    'original_node_id',
)


_PAD_NODE = None


def _cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if text in ('0', 'false', 'no', 'n', 'off'):
        return False
    return default


def subgnn_border_enabled(hparams):
    cfg = getattr(hparams, 'subgnn_border', None)
    return _as_bool(_cfg_get(cfg, 'use', False), default=False)


def _canonical_node_id(node_id, attrs=None):
    attrs = attrs or {}
    for key in _ORIGINAL_NODE_KEYS:
        if key in attrs and attrs[key] is not None:
            return attrs[key]
    return node_id


def _node_neighbors(graph, node_id):
    if node_id not in graph:
        return set()
    if graph.is_directed():
        return set(graph.successors(node_id)).union(set(graph.predecessors(node_id)))
    return set(graph.neighbors(node_id))


def _graph_degree(graph, node_id):
    return len(_node_neighbors(graph, node_id))


def _local_internal_degrees(subgraph):
    return {node_id: len(_node_neighbors(subgraph, node_id)) for node_id in subgraph.nodes()}


def subgraph_original_nodes(subgraph):
    return [_canonical_node_id(node_id, attrs) for node_id, attrs in subgraph.nodes(data=True)]


def external_degree_sequence(original_graph, subgraph, max_sequence_length=64):
    """Return descending positive external degrees for a subgraph."""
    local_degrees = _local_internal_degrees(subgraph)
    values = []
    for node_id, attrs in subgraph.nodes(data=True):
        original_id = _canonical_node_id(node_id, attrs)
        external_degree = _graph_degree(original_graph, original_id) - int(local_degrees.get(node_id, 0))
        values.append(max(0.0, float(external_degree)))

    values = [value for value in values if value > 0.0]
    values.sort(reverse=True)
    if max_sequence_length is not None and int(max_sequence_length) > 0:
        values = values[:int(max_sequence_length)]
    if not values:
        values = [0.0]
    return np.asarray(values, dtype=np.float32)


def _external_degree_sequence_for_node_set(original_graph, node_set, max_sequence_length=64):
    node_set = set(node_set)
    values = []
    for node_id in node_set:
        neighbors = _node_neighbors(original_graph, node_id)
        internal_degree = len(neighbors.intersection(node_set))
        values.append(max(0.0, float(len(neighbors) - internal_degree)))
    values = [value for value in values if value > 0.0]
    values.sort(reverse=True)
    if max_sequence_length is not None and int(max_sequence_length) > 0:
        values = values[:int(max_sequence_length)]
    if not values:
        values = [0.0]
    return np.asarray(values, dtype=np.float32)


def _is_triangle(graph, a, b, c):
    if a is _PAD_NODE or b is _PAD_NODE or c is _PAD_NODE:
        return False
    if a not in graph or b not in graph or c not in graph:
        return False
    return c in _node_neighbors(graph, a) and c in _node_neighbors(graph, b)


def _split_triangular_neighbors(networkx_graph, anchor_patch_subgraph, all_valid_nodes, prev_node, curr_node, inside):
    graph = anchor_patch_subgraph if inside else networkx_graph
    neighbors = list(_node_neighbors(graph, curr_node))
    if not inside and all_valid_nodes is not None:
        valid = set(all_valid_nodes)
        neighbors = [node for node in neighbors if node in valid]

    triangular_neighbors = []
    non_triangular_neighbors = []
    for node in neighbors:
        if _is_triangle(graph, prev_node, curr_node, node):
            triangular_neighbors.append(node)
        else:
            non_triangular_neighbors.append(node)
    return triangular_neighbors, non_triangular_neighbors


def _choose(rng, values):
    if not values:
        return _PAD_NODE
    return values[rng.randrange(len(values))]


def triangular_random_walk(
    original_graph,
    anchor_patch_nodes=None,
    walk_len=8,
    rng=None,
    rw_beta=0.5,
    inside=True,
    all_valid_nodes=None,
    start_node=None,
):
    """Perform SubGNN-style triangular random walk.

    When inside=True, the walk is restricted to the anchor patch subgraph. When
    inside=False, the walk starts from an in-border node and can move through
    valid border/external nodes in the base graph.
    """
    rng = rng or random.Random()
    walk_len = max(1, int(walk_len))
    if anchor_patch_nodes is None:
        anchor_patch_nodes = list(original_graph.nodes())
    anchor_patch_nodes = [node for node in anchor_patch_nodes if node in original_graph]
    if not anchor_patch_nodes:
        nodes = list(original_graph.nodes())
        return [_choose(rng, nodes)] if nodes else []

    anchor_patch_subgraph = original_graph.subgraph(anchor_patch_nodes)
    if inside:
        start_candidates = list(anchor_patch_subgraph.nodes())
    else:
        start_candidates = [node for node in anchor_patch_nodes if _node_neighbors(original_graph, node) - set(anchor_patch_nodes)]
        if not start_candidates:
            start_candidates = list(anchor_patch_subgraph.nodes())
        if all_valid_nodes is None:
            external = set()
            for node in anchor_patch_nodes:
                external.update(_node_neighbors(original_graph, node) - set(anchor_patch_nodes))
            all_valid_nodes = set(start_candidates).union(external)

    prev_node = start_node if start_node in start_candidates else _choose(rng, start_candidates)
    if prev_node is _PAD_NODE:
        return []

    first_graph = anchor_patch_subgraph if inside else original_graph
    first_neighbors = list(_node_neighbors(first_graph, prev_node))
    if not inside and all_valid_nodes is not None:
        first_neighbors = [node for node in first_neighbors if node in set(all_valid_nodes)]
    curr_node = _choose(rng, first_neighbors)
    if curr_node is _PAD_NODE:
        return [prev_node]

    visited = [prev_node, curr_node]
    while len(visited) < walk_len:
        triangular_neighbors, non_triangular_neighbors = _split_triangular_neighbors(
            original_graph,
            anchor_patch_subgraph,
            all_valid_nodes,
            prev_node,
            curr_node,
            inside=inside,
        )
        if not triangular_neighbors and not non_triangular_neighbors:
            break
        if not triangular_neighbors:
            next_node = _choose(rng, non_triangular_neighbors)
        elif not non_triangular_neighbors:
            next_node = _choose(rng, triangular_neighbors)
        elif rng.random() <= float(rw_beta):
            next_node = _choose(rng, triangular_neighbors)
        else:
            next_node = _choose(rng, non_triangular_neighbors)
        if next_node is _PAD_NODE:
            break
        prev_node, curr_node = curr_node, next_node
        visited.append(next_node)
    return visited


def _border_valid_nodes(original_graph, patch):
    patch_set = set(patch)
    in_border = set()
    external = set()
    for node in patch_set:
        outside = _node_neighbors(original_graph, node) - patch_set
        if outside:
            in_border.add(node)
            external.update(outside)
    return in_border, external, in_border.union(external)


def _sample_anchor_patch(original_graph, start_node, anchor_size, rng, walk_len=None, rw_beta=0.5, patch_type='triangular_random_walk'):
    if patch_type == 'ego_graph':
        # Fallback option retained for experiments; triangular_random_walk is the SubGNN default.
        visited = [start_node]
        queue = [start_node]
        seen = {start_node}
        while queue and len(visited) < anchor_size:
            node_id = queue.pop(0)
            for neighbor in sorted(_node_neighbors(original_graph, node_id), key=str):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                visited.append(neighbor)
                queue.append(neighbor)
                if len(visited) >= anchor_size:
                    break
        return visited[:anchor_size]

    walk = triangular_random_walk(
        original_graph,
        anchor_patch_nodes=list(original_graph.nodes()),
        walk_len=walk_len if walk_len is not None else anchor_size,
        rng=rng,
        rw_beta=rw_beta,
        inside=True,
        start_node=start_node,
    )
    # Preserve first occurrence order, as SubGNN samples a walk and then pads it as an anchor patch.
    patch = []
    seen = set()
    for node in walk:
        if node not in seen:
            patch.append(node)
            seen.add(node)
        if len(patch) >= anchor_size:
            break
    return patch or [start_node]


def _build_feature_lookup(subgraphs, feature_dim):
    lookup = {}
    for graph in subgraphs:
        for node_id, attrs in graph.nodes(data=True):
            feature = attrs.get('features')
            if feature is None:
                continue
            original_id = _canonical_node_id(node_id, attrs)
            arr = np.asarray(feature, dtype=np.float32).reshape(-1)
            out = np.zeros((feature_dim,), dtype=np.float32)
            out[: min(feature_dim, arr.size)] = arr[: min(feature_dim, arr.size)]
            lookup[original_id] = out
    return lookup


def _walks_to_features(walks, feature_lookup, feature_dim, walk_len):
    tensor = np.zeros((len(walks), int(walk_len), int(feature_dim)), dtype=np.float32)
    for walk_idx, walk in enumerate(walks):
        for step_idx, node_id in enumerate(walk[: int(walk_len)]):
            feature = feature_lookup.get(node_id)
            if feature is not None:
                tensor[walk_idx, step_idx, :] = feature
    return tensor


def sample_anchor_sequences(
    original_graph,
    num_anchors,
    anchor_size,
    seed=1024,
    max_sequence_length=64,
    n_triangular_walks=4,
    random_walk_len=8,
    sample_walk_len=None,
    rw_beta=0.5,
    structure_patch_type='triangular_random_walk',
    feature_lookup=None,
    feature_dim=None,
    return_walk_features=False,
):
    nodes = list(original_graph.nodes())
    if not nodes:
        raise ValueError('Cannot sample border anchors from an empty original_graph.')

    rng = random.Random(int(seed))
    starts = [nodes[rng.randrange(len(nodes))] for _ in range(int(num_anchors))]
    sequences = []
    patches = []
    all_walk_features = []
    sample_walk_len = int(sample_walk_len if sample_walk_len is not None else anchor_size)

    for start_node in starts:
        patch = _sample_anchor_patch(
            original_graph,
            start_node,
            int(anchor_size),
            rng,
            walk_len=sample_walk_len,
            rw_beta=rw_beta,
            patch_type=structure_patch_type,
        )
        patches.append([str(node_id) for node_id in patch])
        sequences.append(_external_degree_sequence_for_node_set(original_graph, patch, max_sequence_length))

        if return_walk_features:
            _, _, valid_nodes = _border_valid_nodes(original_graph, patch)
            walks = []
            for _ in range(int(n_triangular_walks)):
                walk = triangular_random_walk(
                    original_graph,
                    anchor_patch_nodes=patch,
                    walk_len=int(random_walk_len),
                    rng=rng,
                    rw_beta=rw_beta,
                    inside=False,
                    all_valid_nodes=valid_nodes,
                )
                walks.append(walk)
            all_walk_features.append(_walks_to_features(walks, feature_lookup or {}, int(feature_dim), int(random_walk_len)))

    if return_walk_features:
        return sequences, patches, np.stack(all_walk_features, axis=0).astype(np.float32, copy=False)
    return sequences, patches


def dtw_distance(seq_a, seq_b):
    a = np.asarray(seq_a, dtype=np.float32).reshape(-1)
    b = np.asarray(seq_b, dtype=np.float32).reshape(-1)
    if a.size == 0:
        a = np.asarray([0.0], dtype=np.float32)
    if b.size == 0:
        b = np.asarray([0.0], dtype=np.float32)

    prev = np.full((b.size + 1,), np.inf, dtype=np.float64)
    curr = np.full((b.size + 1,), np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, a.size + 1):
        curr.fill(np.inf)
        for j in range(1, b.size + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[b.size]) / float(max(a.size + b.size, 1))


def dtw_similarity_vector(sequence, anchor_sequences, temperature=1.0):
    temperature = max(float(temperature), 1e-6)
    distances = np.asarray([dtw_distance(sequence, anchor_sequence) for anchor_sequence in anchor_sequences], dtype=np.float32)
    return np.exp(-distances / temperature).astype(np.float32, copy=False)


def _anchor_size_from_cfg(cfg, subgraphs):
    raw = _cfg_get(cfg, 'anchor_size', None)
    if raw is not None:
        return max(1, int(raw))
    sizes = [len(graph.nodes()) for graph in subgraphs]
    if not sizes:
        return 1
    return max(4, min(32, int(np.median(np.asarray(sizes)))))


def build_subgnn_border_features(dataset_raw, hparams, data_name=None):
    cfg = getattr(hparams, 'subgnn_border', None)
    if not subgnn_border_enabled(hparams):
        return None

    original_graph = dataset_raw.get('original_graph', None)
    subgraphs = dataset_raw.get('subgraph_structures', None)
    if original_graph is None:
        raise ValueError('subgnn_border requires dataset_raw[\'original_graph\'].')
    if not subgraphs:
        raise ValueError('subgnn_border requires non-empty subgraph_structures.')

    feature_dim = int(dataset_raw.get('feature_dimension', dataset_raw.get('dataset_metadata', {}).get('feature_dim', 0)))
    if feature_dim <= 0:
        first_graph = subgraphs[0]
        first_node = next(iter(first_graph.nodes()))
        feature_dim = int(np.asarray(first_graph.nodes[first_node].get('features')).reshape(-1).size)

    num_anchors = max(1, int(_cfg_get(cfg, 'num_anchors', 16)))
    anchor_size = _anchor_size_from_cfg(cfg, subgraphs)
    max_sequence_length = int(_cfg_get(cfg, 'max_sequence_length', 64))
    seed = int(_cfg_get(cfg, 'anchor_seed', getattr(hparams, 'cv_seed', 1024)))
    temperature = float(_cfg_get(cfg, 'dtw_temperature', 1.0))
    shuffle = _as_bool(_cfg_get(cfg, 'shuffle', False), default=False)
    shuffle_seed = int(_cfg_get(cfg, 'shuffle_seed', seed + 7919))
    n_triangular_walks = max(1, int(_cfg_get(cfg, 'n_triangular_walks', 4)))
    random_walk_len = max(1, int(_cfg_get(cfg, 'random_walk_len', 8)))
    sample_walk_len = max(1, int(_cfg_get(cfg, 'sample_walk_len', anchor_size)))
    rw_beta = float(_cfg_get(cfg, 'rw_beta', 0.5))
    structure_patch_type = str(_cfg_get(cfg, 'structure_patch_type', 'triangular_random_walk'))

    feature_lookup = _build_feature_lookup(subgraphs, feature_dim)
    anchor_sequences, anchor_patches, anchor_walk_features = sample_anchor_sequences(
        original_graph,
        num_anchors=num_anchors,
        anchor_size=anchor_size,
        seed=seed,
        max_sequence_length=max_sequence_length,
        n_triangular_walks=n_triangular_walks,
        random_walk_len=random_walk_len,
        sample_walk_len=sample_walk_len,
        rw_beta=rw_beta,
        structure_patch_type=structure_patch_type,
        feature_lookup=feature_lookup,
        feature_dim=feature_dim,
        return_walk_features=True,
    )

    features = np.zeros((len(subgraphs), num_anchors), dtype=np.float32)
    external_counts = np.zeros((len(subgraphs),), dtype=np.float32)
    sequence_lengths = np.zeros((len(subgraphs),), dtype=np.float32)
    sequence_stds = np.zeros((len(subgraphs),), dtype=np.float32)

    for idx, graph in enumerate(subgraphs):
        sequence = external_degree_sequence(original_graph, graph, max_sequence_length=max_sequence_length)
        features[idx, :] = dtw_similarity_vector(sequence, anchor_sequences, temperature=temperature)
        external_counts[idx] = float(np.sum(sequence))
        sequence_lengths[idx] = float(len(sequence))
        sequence_stds[idx] = float(np.std(sequence)) if len(sequence) > 1 else 0.0

    if shuffle and len(features) > 1:
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(features))
        features = features[perm]

    diagnostics = {
        'data_name': data_name,
        'num_subgraphs': int(len(subgraphs)),
        'num_anchors': int(num_anchors),
        'anchor_size': int(anchor_size),
        'max_sequence_length': int(max_sequence_length),
        'anchor_seed': int(seed),
        'structure_patch_type': structure_patch_type,
        'sample_walk_len': int(sample_walk_len),
        'n_triangular_walks': int(n_triangular_walks),
        'random_walk_len': int(random_walk_len),
        'rw_beta': float(rw_beta),
        'anchor_encoder': 'BiLSTM over border triangular-random-walk node features',
        'feature_lookup_size': int(len(feature_lookup)),
        'feature_dim': int(feature_dim),
        'anchor_walk_features_shape': [int(v) for v in anchor_walk_features.shape],
        'shuffle': bool(shuffle),
        'shuffle_seed': int(shuffle_seed),
        'external_count_mean': float(np.mean(external_counts)),
        'external_count_std': float(np.std(external_counts)),
        'external_count_min': float(np.min(external_counts)),
        'external_count_max': float(np.max(external_counts)),
        'sequence_length_mean': float(np.mean(sequence_lengths)),
        'sequence_length_std': float(np.std(sequence_lengths)),
        'sequence_std_mean': float(np.mean(sequence_stds)),
        'anchor_sequence_lengths': [int(len(seq)) for seq in anchor_sequences],
        'anchor_patches_preview': anchor_patches[: min(3, len(anchor_patches))],
    }
    return {
        'features': features.astype(np.float32, copy=False),
        'external_counts': external_counts.astype(np.float32, copy=False),
        'anchor_walk_features': anchor_walk_features.astype(np.float32, copy=False),
        'diagnostics': diagnostics,
    }


def maybe_write_subgnn_border_diagnostics(hparams, data_name, diagnostics):
    if not diagnostics:
        return None
    cfg = getattr(hparams, 'subgnn_border', None)
    if not _as_bool(_cfg_get(cfg, 'export_diagnostics', False), default=False):
        return None
    output_dir = getattr(hparams, 'model_save_path', None) or 'results'
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{data_name}_subgnn_border_diagnostics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    return path
