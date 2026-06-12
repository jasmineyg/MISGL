# coding=utf-8

"""Offline diagnostics for Laplacian positional encodings.

The script is intentionally independent from the training data loader. It reads
the processed pickle directly, reconstructs the coarse graph, and writes a
compact result bundle that can be copied back from a remote server.
"""

import argparse
import csv
import json
import logging
import math
import os
import pickle
import shutil
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.stats import spearmanr
import torch
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from MISGL.utils.lappe import align_assignment_to_subgraphs
from MISGL.utils.lappe import build_coarse_adjacency
from MISGL.utils.lappe import compute_lappe
from MISGL.utils.lappe import normalized_laplacian
from MISGL.utils.lappe import topk_rows


SCRIPT_VERSION = 1


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(',') if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose LapPE information, coarse-graph health, top-k damage, and fusion use.'
    )
    parser.add_argument('--data_name_set', nargs='+', default=['ogbn_arxiv', 'reddit'])
    parser.add_argument('--processed_data_dir', default='/data/yg/Subgraph-MIL/Data/processed_data')
    parser.add_argument('--output_dir', default='result/lappe_diagnostics')
    parser.add_argument('--lap_pe_dim', type=int, default=16)
    parser.add_argument('--coarse_topk', type=int, default=20)
    parser.add_argument('--topk_list', default='5,10,20,40,80')
    parser.add_argument('--knn_k_list', default='5,10,20')
    parser.add_argument('--near_zero_tol', type=float, default=1e-6)
    parser.add_argument('--spectrum_size', type=int, default=64)
    parser.add_argument('--pair_sample_size', type=int, default=200000)
    parser.add_argument('--probe_splits', type=int, default=5)
    parser.add_argument('--probe_repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument(
        '--z_mil_path',
        default=None,
        help='Optional template, e.g. result/coarse_relation/{data_name}_zmil_fold0.pt.',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Optional trained LapPE checkpoint template. Enables direct model-use ablations.',
    )
    parser.add_argument(
        '--hparam_path',
        default=None,
        help='YAML used by --checkpoint. May also contain {data_name}.',
    )
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model_batch_size', type=int, default=64)
    parser.add_argument('--no_zip', action='store_true')
    parser.add_argument('--synthetic_smoke', action='store_true')
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(path, rows):
    ensure_dir(os.path.dirname(path) or '.')
    if not rows:
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            handle.write('')
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False)


def format_template(value, data_name):
    if value is None:
        return None
    return str(value).format(data_name=data_name)


def load_dataset(processed_data_dir, data_name):
    path = os.path.join(processed_data_dir, '{}_processed.pkl'.format(data_name))
    if not os.path.exists(path):
        raise FileNotFoundError('Processed dataset not found: {}'.format(path))
    logging.info('Loading processed dataset: %s', path)
    with open(path, 'rb') as handle:
        dataset = pickle.load(handle)
    required = ('original_graph', 'assignment_matrix', 'subgraph_structures')
    missing = [key for key in required if key not in dataset]
    if missing:
        raise KeyError('Dataset {} is missing keys: {}'.format(data_name, missing))
    return dataset, path


def resolve_labels(dataset):
    subgraphs = dataset['subgraph_structures']
    labels = dataset.get('subgraph_labels')
    if labels is None:
        labels = [graph.graph.get('label', 0) for graph in subgraphs]
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != len(subgraphs):
        raise ValueError('Label count does not match subgraph count.')
    if np.unique(labels).size != 2:
        raise ValueError('This diagnostic currently expects binary labels, got {}.'.format(np.unique(labels)))
    return labels


def build_graph_inputs(dataset):
    subgraphs = dataset['subgraph_structures']
    node_to_subgraph, node_counts, source_ids, alignment = align_assignment_to_subgraphs(
        dataset['original_graph'],
        dataset['assignment_matrix'],
        subgraphs,
    )
    coarse = build_coarse_adjacency(
        dataset['original_graph'],
        node_to_subgraph,
        node_counts,
    ).tocsr()
    coarse = coarse.tolil()
    coarse.setdiag(0.0)
    coarse = coarse.tocsr()
    coarse.eliminate_zeros()
    return coarse, source_ids, alignment


def symmetrize_after_topk(coarse_adj, topk):
    if topk is None:
        directed = coarse_adj.copy().tocsr()
    else:
        directed = topk_rows(coarse_adj, int(topk))
    sym = directed.maximum(directed.T).tolil()
    sym.setdiag(0.0)
    sym = sym.tocsr()
    sym.eliminate_zeros()
    return directed, sym


def graph_health(adjacency):
    adjacency = adjacency.tocsr()
    binary = adjacency.copy()
    binary.data = np.ones_like(binary.data)
    degree = np.diff(binary.indptr).astype(np.float64)
    weighted_degree = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float64)
    component_count, component_labels = csgraph.connected_components(
        binary,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(component_labels, minlength=component_count)
    isolated = int(np.count_nonzero(degree == 0))
    nonisolated_components = int(component_count - isolated)
    return {
        'num_nodes': int(adjacency.shape[0]),
        'directed_nnz': int(adjacency.nnz),
        'undirected_edges': int(sp.triu(binary, k=1).nnz),
        'connected_components': int(component_count),
        'nonisolated_components': nonisolated_components,
        'largest_component_size': int(component_sizes.max()) if component_sizes.size else 0,
        'largest_component_ratio': (
            float(component_sizes.max() / adjacency.shape[0]) if adjacency.shape[0] else 0.0
        ),
        'isolated_nodes': isolated,
        'isolated_ratio': float(isolated / adjacency.shape[0]) if adjacency.shape[0] else 0.0,
        'degree_mean': float(degree.mean()) if degree.size else 0.0,
        'degree_std': float(degree.std()) if degree.size else 0.0,
        'degree_min': float(degree.min()) if degree.size else 0.0,
        'degree_median': float(np.median(degree)) if degree.size else 0.0,
        'degree_max': float(degree.max()) if degree.size else 0.0,
        'weighted_degree_mean': float(weighted_degree.mean()) if weighted_degree.size else 0.0,
        'weighted_degree_std': float(weighted_degree.std()) if weighted_degree.size else 0.0,
        'weighted_degree_min': float(weighted_degree.min()) if weighted_degree.size else 0.0,
        'weighted_degree_median': float(np.median(weighted_degree)) if weighted_degree.size else 0.0,
        'weighted_degree_max': float(weighted_degree.max()) if weighted_degree.size else 0.0,
    }


def spectrum_and_pe(adjacency, lap_pe_dim, spectrum_size, near_zero_tol):
    laplacian, zero_degree_count = normalized_laplacian(adjacency)
    node_count = int(adjacency.shape[0])
    requested_dim = max(int(lap_pe_dim), int(spectrum_size) - 1)
    requested_dim = min(requested_dim, max(node_count - 1, 0))
    eigenvalues, eigenvectors = compute_lappe(laplacian, requested_dim)
    finite_values = eigenvalues[np.isfinite(eigenvalues)].astype(np.float64, copy=False)
    kept_pe = eigenvectors[:, :int(lap_pe_dim)]
    health = graph_health(adjacency)
    theoretical_zero_count = int(health['nonisolated_components'])
    positive = finite_values[finite_values > near_zero_tol]
    return {
        'laplacian': laplacian,
        'eigenvalues': finite_values,
        'lap_pe': kept_pe,
        'zero_degree_count': int(zero_degree_count),
        'computed_near_zero_count': int(np.count_nonzero(np.abs(finite_values) <= near_zero_tol)),
        'theoretical_zero_eigenvalue_count': theoretical_zero_count,
        'trivial_vectors_inside_lappe': int(min(max(theoretical_zero_count - 1, 0), lap_pe_dim)),
        'spectral_gap': float(positive[0]) if positive.size else None,
    }


def edge_homophily(adjacency, labels):
    upper = sp.triu(adjacency, k=1).tocoo()
    if upper.nnz == 0:
        return {
            'num_edges': 0,
            'edge_homophily': None,
            'weighted_edge_homophily': None,
        }
    same = (labels[upper.row] == labels[upper.col]).astype(np.float64)
    weights = upper.data.astype(np.float64)
    return {
        'num_edges': int(upper.nnz),
        'edge_homophily': float(same.mean()),
        'weighted_edge_homophily': float(np.sum(same * weights) / max(np.sum(weights), 1e-12)),
    }


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        'acc': float(accuracy_score(labels, predictions)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'auc': safe_auc(labels, probabilities),
    }


def summarize_metric_runs(rows):
    out = {}
    for metric in ('acc', 'f1', 'auc'):
        values = [row[metric] for row in rows if row.get(metric) is not None]
        out[metric] = {
            'mean': float(np.mean(values)) if values else None,
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
        }
    return out


def probe_space(features, labels, splits, repeats, seed, feature_name):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    splitter = RepeatedStratifiedKFold(
        n_splits=int(splits),
        n_repeats=int(repeats),
        random_state=int(seed),
    )
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='liblinear',
            random_state=int(seed),
        ),
    )
    dummy = DummyClassifier(strategy='stratified', random_state=int(seed))
    runs = []
    dummy_runs = []
    permutation_drops = []
    rng = np.random.default_rng(int(seed))
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(features, labels)):
        model = clone(logistic)
        model.fit(features[train_idx], labels[train_idx])
        probability = model.predict_proba(features[test_idx])[:, 1]
        metrics = binary_metrics(labels[test_idx], probability)
        metrics['split_id'] = int(split_id)
        runs.append(metrics)

        baseline = clone(dummy)
        baseline.fit(features[train_idx], labels[train_idx])
        dummy_probability = baseline.predict_proba(features[test_idx])[:, 1]
        if dummy_probability.ndim > 1:
            dummy_probability = dummy_probability[:, 1]
        dummy_metrics = binary_metrics(labels[test_idx], dummy_probability)
        dummy_metrics['split_id'] = int(split_id)
        dummy_runs.append(dummy_metrics)

        permuted = features[test_idx].copy()
        rng.shuffle(permuted, axis=0)
        permuted_probability = model.predict_proba(permuted)[:, 1]
        permuted_metrics = binary_metrics(labels[test_idx], permuted_probability)
        permutation_drops.append({
            metric: (
                metrics[metric] - permuted_metrics[metric]
                if metrics[metric] is not None and permuted_metrics[metric] is not None
                else None
            )
            for metric in ('acc', 'f1', 'auc')
        })

    return {
        'feature': feature_name,
        'num_samples': int(features.shape[0]),
        'num_features': int(features.shape[1]),
        'metrics': summarize_metric_runs(runs),
        'dummy_metrics': summarize_metric_runs(dummy_runs),
        'permutation_drop': summarize_metric_runs(permutation_drops),
        'runs': runs,
    }


def pair_distance_diagnostics(features, labels, sample_size, seed, feature_name):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    scaled = StandardScaler().fit_transform(features)
    rng = np.random.default_rng(int(seed))
    pair_count = int(sample_size)
    src = rng.integers(0, features.shape[0], size=pair_count * 2)
    dst = rng.integers(0, features.shape[0], size=pair_count * 2)
    valid = src != dst
    src = src[valid][:pair_count]
    dst = dst[valid][:pair_count]
    same = labels[src] == labels[dst]
    diff = ~same
    euclidean = np.linalg.norm(scaled[src] - scaled[dst], axis=1)
    normalized = scaled / np.maximum(np.linalg.norm(scaled, axis=1, keepdims=True), 1e-12)
    cosine = 1.0 - np.sum(normalized[src] * normalized[dst], axis=1)
    rows = []
    for metric_name, values in (('euclidean_standardized', euclidean), ('cosine_standardized', cosine)):
        same_values = values[same]
        diff_values = values[diff]
        pooled_var = 0.5 * (same_values.var() + diff_values.var())
        effect = (
            float((diff_values.mean() - same_values.mean()) / math.sqrt(max(pooled_var, 1e-12)))
            if same_values.size and diff_values.size else None
        )
        rows.append({
            'feature': feature_name,
            'distance': metric_name,
            'pair_count': int(values.size),
            'same_count': int(same_values.size),
            'different_count': int(diff_values.size),
            'same_mean': float(same_values.mean()) if same_values.size else None,
            'same_median': float(np.median(same_values)) if same_values.size else None,
            'different_mean': float(diff_values.mean()) if diff_values.size else None,
            'different_median': float(np.median(diff_values)) if diff_values.size else None,
            'different_over_same_mean': (
                float(diff_values.mean() / max(same_values.mean(), 1e-12))
                if same_values.size and diff_values.size else None
            ),
            'cohen_d_different_minus_same': effect,
            'auc_same_label_from_negative_distance': safe_auc(same.astype(np.int64), -values),
        })
    return rows


def knn_consistency(features, labels, k_list, feature_name):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    max_k = min(max(k_list), features.shape[0] - 1)
    if max_k <= 0:
        return []
    neighbors = NearestNeighbors(n_neighbors=max_k + 1, metric='cosine')
    neighbors.fit(features)
    indices = neighbors.kneighbors(features, return_distance=False)[:, 1:]
    class_counts = np.bincount(labels)
    chance = float(np.sum(np.square(class_counts / class_counts.sum())))
    rows = []
    for k in k_list:
        local_k = min(int(k), max_k)
        local = indices[:, :local_k]
        consistency = np.mean(labels[local] == labels[:, None], axis=1)
        rows.append({
            'feature': feature_name,
            'k': int(local_k),
            'label_consistency_mean': float(consistency.mean()),
            'label_consistency_std': float(consistency.std()),
            'chance_consistency': chance,
            'lift_over_chance': float(consistency.mean() - chance),
        })
    return rows


def l2_normalize_rows(features):
    features = np.asarray(features, dtype=np.float64)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def sampled_pairwise_distance_correlation(reference, candidate, sample_size, seed):
    rng = np.random.default_rng(int(seed))
    node_count = reference.shape[0]
    src = rng.integers(0, node_count, size=int(sample_size) * 2)
    dst = rng.integers(0, node_count, size=int(sample_size) * 2)
    valid = src != dst
    src = src[valid][:int(sample_size)]
    dst = dst[valid][:int(sample_size)]
    ref_distance = np.linalg.norm(reference[src] - reference[dst], axis=1)
    candidate_distance = np.linalg.norm(candidate[src] - candidate[dst], axis=1)
    correlation = spearmanr(ref_distance, candidate_distance).statistic
    return float(correlation) if np.isfinite(correlation) else None


def eigenspace_similarity(reference, candidate):
    if reference.shape[1] == 0 or candidate.shape[1] == 0:
        return None
    q_ref, _ = np.linalg.qr(reference)
    q_candidate, _ = np.linalg.qr(candidate)
    singular_values = np.linalg.svd(q_ref.T @ q_candidate, compute_uv=False)
    return float(np.mean(np.clip(singular_values, 0.0, 1.0)))


def topk_diagnostics(coarse_adj, labels, args):
    topk_values = sorted(set(parse_int_list(args.topk_list) + [int(args.coarse_topk)]))
    variants = [('unpruned', None)] + [('topk_{}'.format(k), k) for k in topk_values]
    graph_rows = []
    eigen_rows = []
    payloads = {}
    for variant_name, topk in variants:
        directed, sym = symmetrize_after_topk(coarse_adj, topk)
        health = graph_health(sym)
        spectrum = spectrum_and_pe(
            sym,
            lap_pe_dim=args.lap_pe_dim,
            spectrum_size=args.spectrum_size,
            near_zero_tol=args.near_zero_tol,
        )
        homophily = edge_homophily(sym, labels)
        row = {
            'variant': variant_name,
            'topk': topk,
            'directed_edges_before_sym': int(directed.nnz),
            **health,
            **homophily,
            'computed_near_zero_count': spectrum['computed_near_zero_count'],
            'theoretical_zero_eigenvalue_count': spectrum['theoretical_zero_eigenvalue_count'],
            'trivial_vectors_inside_lappe': spectrum['trivial_vectors_inside_lappe'],
            'spectral_gap': spectrum['spectral_gap'],
        }
        graph_rows.append(row)
        for eigen_idx, value in enumerate(spectrum['eigenvalues']):
            eigen_rows.append({
                'variant': variant_name,
                'topk': topk,
                'eigen_index': int(eigen_idx),
                'eigenvalue': float(value),
                'near_zero': int(abs(value) <= args.near_zero_tol),
            })
        payloads[variant_name] = {
            'adjacency': sym,
            'lap_pe': spectrum['lap_pe'],
            'eigenvalues': spectrum['eigenvalues'],
            'health': health,
        }

    reference = payloads['unpruned']
    reference_edges = max(int(reference['health']['undirected_edges']), 1)
    damage_rows = []
    for variant_name, topk in variants[1:]:
        candidate = payloads[variant_name]
        common_eigs = min(reference['eigenvalues'].size, candidate['eigenvalues'].size)
        eigen_delta = (
            np.linalg.norm(
                reference['eigenvalues'][:common_eigs] - candidate['eigenvalues'][:common_eigs]
            )
            / max(np.linalg.norm(reference['eigenvalues'][:common_eigs]), 1e-12)
        )
        damage_rows.append({
            'variant': variant_name,
            'topk': topk,
            'edge_retention_ratio': float(candidate['health']['undirected_edges'] / reference_edges),
            'component_increase': int(
                candidate['health']['connected_components']
                - reference['health']['connected_components']
            ),
            'isolated_node_increase': int(
                candidate['health']['isolated_nodes'] - reference['health']['isolated_nodes']
            ),
            'largest_component_ratio_drop': float(
                reference['health']['largest_component_ratio']
                - candidate['health']['largest_component_ratio']
            ),
            'eigenvalue_relative_l2_change': float(eigen_delta),
            'lappe_eigenspace_mean_cosine': eigenspace_similarity(
                reference['lap_pe'],
                candidate['lap_pe'],
            ),
            'lappe_pairwise_distance_spearman': sampled_pairwise_distance_correlation(
                reference['lap_pe'],
                candidate['lap_pe'],
                min(args.pair_sample_size, 50000),
                args.seed + int(topk),
            ),
        })
    selected_name = 'topk_{}'.format(int(args.coarse_topk))
    return {
        'graph_rows': graph_rows,
        'eigen_rows': eigen_rows,
        'damage_rows': damage_rows,
        'selected_lap_pe': payloads[selected_name]['lap_pe'],
        'selected_adjacency': payloads[selected_name]['adjacency'],
    }


def load_feature_payload(path, expected_rows):
    if not os.path.exists(path):
        raise FileNotFoundError('Feature payload not found: {}'.format(path))
    if path.lower().endswith('.npz'):
        raw = dict(np.load(path, allow_pickle=False))
    else:
        raw = torch.load(path, map_location='cpu')
        if isinstance(raw, torch.Tensor):
            raw = {'features': raw}
    if not isinstance(raw, dict):
        raise ValueError('Feature payload must be a dict or tensor: {}'.format(path))

    def as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    features = None
    for key in ('features', 'z_mil', 'z_B', 'mean_vec'):
        if key in raw:
            features = as_numpy(raw[key]).astype(np.float32, copy=False)
            break
    if features is None:
        raise KeyError('No z_mil-like features found in {}.'.format(path))
    indices = raw.get('orig_indices', raw.get('indices'))
    if indices is None:
        indices = np.arange(features.shape[0], dtype=np.int64)
    else:
        indices = as_numpy(indices).astype(np.int64, copy=False).reshape(-1)
    ordered = np.zeros((expected_rows, features.shape[1]), dtype=np.float32)
    present = np.zeros((expected_rows,), dtype=bool)
    for row_idx, orig_idx in enumerate(indices):
        if 0 <= int(orig_idx) < expected_rows:
            ordered[int(orig_idx)] = features[row_idx]
            present[int(orig_idx)] = True
    if not np.all(present):
        raise ValueError(
            'Feature payload misses {} rows; first missing rows: {}'.format(
                int(np.count_nonzero(~present)),
                np.where(~present)[0][:10].tolist(),
            )
        )
    extras = {}
    for key in ('pos_emb', 'logits', 'logits_full', 'logits_zero_pos', 'logits_shuffled_pos'):
        if key in raw:
            values = as_numpy(raw[key])
            if values.shape[0] == features.shape[0]:
                target_shape = (expected_rows,) + tuple(values.shape[1:])
                aligned = np.zeros(target_shape, dtype=values.dtype)
                for row_idx, orig_idx in enumerate(indices):
                    if 0 <= int(orig_idx) < expected_rows:
                        aligned[int(orig_idx)] = values[row_idx]
                extras[key] = aligned
    return ordered, extras


def feature_scale_row(feature_name, features):
    features = np.asarray(features, dtype=np.float64)
    row_norm = np.linalg.norm(features, axis=1)
    feature_std = features.std(axis=0)
    return {
        'feature': feature_name,
        'num_features': int(features.shape[1]),
        'row_l2_mean': float(row_norm.mean()),
        'row_l2_std': float(row_norm.std()),
        'element_abs_mean': float(np.abs(features).mean()),
        'per_dimension_std_mean': float(feature_std.mean()),
        'per_dimension_std_rms': float(np.sqrt(np.mean(np.square(feature_std)))),
    }


def fusion_probe(z_mil, lap_pe, labels, args):
    splitter = RepeatedStratifiedKFold(
        n_splits=int(args.probe_splits),
        n_repeats=int(args.probe_repeats),
        random_state=int(args.seed),
    )
    runs = []
    rng = np.random.default_rng(int(args.seed))
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(z_mil, labels)):
        scaler_z = StandardScaler().fit(z_mil[train_idx])
        scaler_p = StandardScaler().fit(lap_pe[train_idx])
        z_train = scaler_z.transform(z_mil[train_idx])
        z_test = scaler_z.transform(z_mil[test_idx])
        p_train = scaler_p.transform(lap_pe[train_idx])
        p_test = scaler_p.transform(lap_pe[test_idx])
        combined_train = np.concatenate([z_train, p_train], axis=1)
        combined_test = np.concatenate([z_test, p_test], axis=1)
        model = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='liblinear',
            random_state=int(args.seed),
        )
        model.fit(combined_train, labels[train_idx])
        full_probability = model.predict_proba(combined_test)[:, 1]
        full_metrics = binary_metrics(labels[test_idx], full_probability)
        zero_pos = combined_test.copy()
        zero_pos[:, z_test.shape[1]:] = 0.0
        zero_metrics = binary_metrics(labels[test_idx], model.predict_proba(zero_pos)[:, 1])
        shuffled_pos = combined_test.copy()
        shuffled_pos[:, z_test.shape[1]:] = p_test[rng.permutation(p_test.shape[0])]
        shuffled_metrics = binary_metrics(
            labels[test_idx],
            model.predict_proba(shuffled_pos)[:, 1],
        )
        coef = model.coef_[0]
        z_coef_norm = float(np.linalg.norm(coef[:z_test.shape[1]]))
        pos_coef_norm = float(np.linalg.norm(coef[z_test.shape[1]:]))
        runs.append({
            'split_id': int(split_id),
            'full_acc': full_metrics['acc'],
            'full_f1': full_metrics['f1'],
            'full_auc': full_metrics['auc'],
            'zero_pos_acc_drop': full_metrics['acc'] - zero_metrics['acc'],
            'zero_pos_f1_drop': full_metrics['f1'] - zero_metrics['f1'],
            'zero_pos_auc_drop': (
                full_metrics['auc'] - zero_metrics['auc']
                if full_metrics['auc'] is not None and zero_metrics['auc'] is not None else None
            ),
            'shuffle_pos_acc_drop': full_metrics['acc'] - shuffled_metrics['acc'],
            'shuffle_pos_f1_drop': full_metrics['f1'] - shuffled_metrics['f1'],
            'shuffle_pos_auc_drop': (
                full_metrics['auc'] - shuffled_metrics['auc']
                if full_metrics['auc'] is not None and shuffled_metrics['auc'] is not None else None
            ),
            'z_coefficient_l2': z_coef_norm,
            'pos_coefficient_l2': pos_coef_norm,
            'pos_over_z_coefficient_l2': float(pos_coef_norm / max(z_coef_norm, 1e-12)),
        })
    aggregate = {}
    for key in runs[0]:
        if key == 'split_id':
            continue
        values = [row[key] for row in runs if row[key] is not None]
        aggregate[key] = {
            'mean': float(np.mean(values)) if values else None,
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
        }
    return {'runs': runs, 'aggregate': aggregate}


def extract_logits(model_output):
    if isinstance(model_output, dict):
        model_output = model_output.get('ypred_A')
    if not isinstance(model_output, torch.Tensor):
        raise TypeError('Model output does not contain tensor logits.')
    return model_output.view(-1)


def load_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if isinstance(checkpoint.get(key), dict):
                state_dict = checkpoint[key]
                break
    if not isinstance(state_dict, dict):
        raise ValueError('Checkpoint does not contain a state_dict: {}'.format(checkpoint_path))
    return state_dict


def model_utilization_diagnostics(dataset, labels, lap_pe, checkpoint_path, hparam_path, args, data_name):
    if checkpoint_path is None:
        return {'status': 'skipped', 'reason': '--checkpoint was not provided.'}, None
    if hparam_path is None:
        return {'status': 'skipped', 'reason': '--hparam_path is required with --checkpoint.'}, None
    if not os.path.exists(checkpoint_path):
        return {'status': 'skipped', 'reason': 'Checkpoint not found: {}'.format(checkpoint_path)}, None
    if not os.path.exists(hparam_path):
        return {'status': 'skipped', 'reason': 'Hparam YAML not found: {}'.format(hparam_path)}, None

    import networkx as nx
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    from MISGL.models.encoder import MISGLEncoder
    from MISGL.utils import hparam
    from MISGL.utils import hparams_lib
    from MISGL.utils.global_variables import g_key

    class DiagnosticGraphDataset(Dataset):
        def __init__(self, hparams, graphs):
            self.rows = []
            feature_dim = int(hparams.channel_list[0])
            max_num_nodes = int(hparams.max_num_nodes)
            for graph_idx, graph in enumerate(graphs):
                nodelist = list(graph.nodes())
                num_nodes = len(nodelist)
                adjacency = nx.to_numpy_array(graph, nodelist=nodelist, dtype=np.float32)
                x = np.zeros((max_num_nodes, feature_dim), dtype=np.float32)
                for node_idx, node_id in enumerate(nodelist):
                    features = graph.nodes[node_id].get('features')
                    if features is None:
                        raise ValueError('Node {!r} is missing features.'.format(node_id))
                    x[node_idx] = np.asarray(features, dtype=np.float32)[:feature_dim]
                padded_adj = np.zeros((max_num_nodes, max_num_nodes), dtype=np.float32)
                padded_adj[:num_nodes, :num_nodes] = adjacency
                self.rows.append({
                    g_key.x: torch.tensor(x, dtype=torch.float32),
                    g_key.y: torch.tensor(int(labels[graph_idx]), dtype=torch.long),
                    g_key.node_num: torch.tensor(num_nodes, dtype=torch.int16),
                    g_key.adj_mat: torch.tensor(padded_adj, dtype=torch.float32),
                    g_key.orig_graph_idx: torch.tensor(graph_idx, dtype=torch.long),
                    g_key.subgraph_id: torch.tensor(graph_idx, dtype=torch.long),
                })

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    hp = hparam.HParams()
    hp.from_yaml(hparam_path)
    hparams_lib.apply_defaults(hp)
    hp.data_name = data_name
    hp.device = args.device
    hp.preload_data_to_gpu = False
    hp.channel_list[0] = int(
        dataset.get('feature_dimension', dataset.get('dataset_metadata', {}).get('feature_dim'))
    )
    max_nodes = int(
        dataset.get('dataset_metadata', {}).get(
            'max_num_nodes',
            max(len(graph.nodes()) for graph in dataset['subgraph_structures']),
        )
    )
    if 'max_num_nodes' in hp:
        hp.set_hparam('max_num_nodes', max_nodes)
    else:
        hp.add_hparam('max_num_nodes', max_nodes)

    subgraphs = dataset['subgraph_structures']
    for idx, graph in enumerate(subgraphs):
        graph.graph['orig_idx'] = int(idx)
        graph.graph['subgraph_id'] = int(idx)
        graph.graph['label'] = int(labels[idx])
    graph_dataset = DiagnosticGraphDataset(hp, subgraphs)
    graph_loader = DataLoader(graph_dataset, batch_size=int(args.model_batch_size), shuffle=False)
    device = torch.device(args.device)
    model = MISGLEncoder(hp, data_name=data_name).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device), strict=True)
    model.eval()

    shuffle_index = np.random.default_rng(int(args.seed)).permutation(lap_pe.shape[0])
    true_lappe = torch.tensor(lap_pe, dtype=torch.float32)
    shuffled_lappe = true_lappe[torch.tensor(shuffle_index, dtype=torch.long)]
    all_labels = []
    all_indices = []
    all_logits = []
    all_zero_logits = []
    all_shuffle_logits = []
    all_z = []
    all_pos = []
    gradient_norms = []
    for raw_batch in graph_loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        indices = batch[g_key.orig_graph_idx].view(-1).long()
        pe = true_lappe[indices.cpu()].to(device).requires_grad_(True)
        batch[g_key.lap_pe] = pe
        model_output, embeddings = model.forward_with_embeddings(batch)
        logits = extract_logits(model_output)
        gradient = torch.autograd.grad(logits.sum(), pe, retain_graph=False, create_graph=False)[0]

        zero_batch = dict(batch)
        zero_batch[g_key.lap_pe] = torch.zeros_like(pe)
        shuffled_batch = dict(batch)
        shuffled_batch[g_key.lap_pe] = shuffled_lappe[indices.cpu()].to(device)
        with torch.inference_mode():
            zero_output, _ = model.forward_with_embeddings(zero_batch)
            shuffled_output, _ = model.forward_with_embeddings(shuffled_batch)

        all_labels.append(batch[g_key.y].view(-1).detach().cpu())
        all_indices.append(indices.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_zero_logits.append(extract_logits(zero_output).detach().cpu())
        all_shuffle_logits.append(extract_logits(shuffled_output).detach().cpu())
        semantic = embeddings.get('z_B', embeddings.get('mean_vec'))
        all_z.append(semantic.detach().cpu())
        all_pos.append(embeddings['pos_emb'].detach().cpu())
        gradient_norms.append(gradient.norm(dim=1).detach().cpu())

    y = torch.cat(all_labels).numpy()
    logits = torch.cat(all_logits).numpy()
    zero_logits = torch.cat(all_zero_logits).numpy()
    shuffle_logits = torch.cat(all_shuffle_logits).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    zero_probabilities = 1.0 / (1.0 + np.exp(-zero_logits))
    shuffle_probabilities = 1.0 / (1.0 + np.exp(-shuffle_logits))
    full_metrics = binary_metrics(y, probabilities)
    zero_metrics = binary_metrics(y, zero_probabilities)
    shuffle_metrics = binary_metrics(y, shuffle_probabilities)

    first_layer = model.classifier[0]
    classifier_weights = first_layer.weight.detach().cpu().numpy()
    pos_dim = int(model.pos_dim)
    semantic_dim = int(classifier_weights.shape[1] - pos_dim)
    semantic_weight_norm = float(np.linalg.norm(classifier_weights[:, :semantic_dim]))
    position_weight_norm = float(np.linalg.norm(classifier_weights[:, semantic_dim:]))
    result = {
        'status': 'ok',
        'checkpoint': checkpoint_path,
        'classifier_input_mode': model.classifier_input_mode,
        'full_metrics': full_metrics,
        'zero_lappe_metrics': zero_metrics,
        'shuffled_lappe_metrics': shuffle_metrics,
        'zero_lappe_metric_drop': {
            key: (
                full_metrics[key] - zero_metrics[key]
                if full_metrics[key] is not None and zero_metrics[key] is not None else None
            )
            for key in ('acc', 'f1', 'auc')
        },
        'shuffled_lappe_metric_drop': {
            key: (
                full_metrics[key] - shuffle_metrics[key]
                if full_metrics[key] is not None and shuffle_metrics[key] is not None else None
            )
            for key in ('acc', 'f1', 'auc')
        },
        'logit_absolute_change_zero_mean': float(np.mean(np.abs(logits - zero_logits))),
        'logit_absolute_change_shuffle_mean': float(np.mean(np.abs(logits - shuffle_logits))),
        'lappe_gradient_l2_mean': float(torch.cat(gradient_norms).mean().item()),
        'semantic_classifier_weight_l2': semantic_weight_norm,
        'position_classifier_weight_l2': position_weight_norm,
        'position_over_semantic_weight_l2': float(
            position_weight_norm / max(semantic_weight_norm, 1e-12)
        ),
    }
    artifact = {
        'features': torch.cat(all_z),
        'pos_emb': torch.cat(all_pos),
        'labels': torch.cat(all_labels),
        'orig_indices': torch.cat(all_indices),
        'logits_full': torch.cat(all_logits),
        'logits_zero_pos': torch.cat(all_zero_logits),
        'logits_shuffled_pos': torch.cat(all_shuffle_logits),
    }
    return result, artifact


def conclusion_signals(summary):
    probes = summary.get('probes', {})
    lap_probe = probes.get('lap_pe', {})
    lap_auc = lap_probe.get('metrics', {}).get('auc', {}).get('mean')
    dummy_auc = lap_probe.get('dummy_metrics', {}).get('auc', {}).get('mean')
    fusion = summary.get('fusion_probe', {}).get('aggregate', {})
    fusion_shuffle_auc = fusion.get('shuffle_pos_auc_drop', {}).get('mean')
    model_use = summary.get('model_utilization', {})
    selected_graph = summary.get('selected_graph_health', {})
    damage = summary.get('selected_topk_damage', {})
    return {
        'lap_pe_label_signal': {
            'status': (
                'present' if lap_auc is not None and dummy_auc is not None and lap_auc - dummy_auc >= 0.05
                else 'weak_or_absent'
            ),
            'lap_pe_auc': lap_auc,
            'dummy_auc': dummy_auc,
        },
        'coarse_graph_health': {
            'status': (
                'fragmented'
                if selected_graph.get('largest_component_ratio', 1.0) < 0.9
                or selected_graph.get('isolated_ratio', 0.0) > 0.01
                else 'connected_enough'
            ),
            'largest_component_ratio': selected_graph.get('largest_component_ratio'),
            'isolated_ratio': selected_graph.get('isolated_ratio'),
            'connected_components': selected_graph.get('connected_components'),
        },
        'topk_spectral_damage': {
            'status': (
                'substantial'
                if damage.get('largest_component_ratio_drop', 0.0) > 0.05
                or damage.get('lappe_pairwise_distance_spearman', 1.0) < 0.8
                else 'limited'
            ),
            **damage,
        },
        'fusion_can_use_lappe': {
            'status': (
                'unknown' if summary.get('fusion_probe', {}).get('status') != 'ok'
                else 'yes' if fusion_shuffle_auc is not None and fusion_shuffle_auc >= 0.01
                else 'little_evidence'
            ),
            'shuffle_pos_auc_drop': fusion_shuffle_auc,
        },
        'trained_model_uses_lappe': {
            'status': (
                'unknown' if model_use.get('status') != 'ok'
                else 'yes' if model_use.get('shuffled_lappe_metric_drop', {}).get('auc', 0.0) >= 0.01
                or model_use.get('logit_absolute_change_shuffle_mean', 0.0) >= 0.01
                else 'little_evidence'
            ),
            'evidence': model_use,
        },
    }


def diagnose_dataset(dataset, data_name, args, out_dir):
    labels = resolve_labels(dataset)
    coarse_adj, source_ids, alignment = build_graph_inputs(dataset)
    topk = topk_diagnostics(coarse_adj, labels, args)
    lap_pe = topk['selected_lap_pe']
    feature_spaces = {'lap_pe': lap_pe}
    optional_features = {}

    z_mil_path = format_template(args.z_mil_path, data_name)
    if z_mil_path:
        z_mil, optional_features = load_feature_payload(z_mil_path, labels.shape[0])
        feature_spaces['z_mil'] = z_mil
        if 'pos_emb' in optional_features and optional_features['pos_emb'].ndim == 2:
            feature_spaces['pos_emb'] = optional_features['pos_emb']

    checkpoint_path = format_template(args.checkpoint, data_name)
    hparam_path = format_template(args.hparam_path, data_name)
    model_use, model_artifact = model_utilization_diagnostics(
        dataset,
        labels,
        lap_pe,
        checkpoint_path,
        hparam_path,
        args,
        data_name,
    )
    if model_artifact is not None:
        artifact_path = os.path.join(out_dir, 'model_diagnostic_artifact.pt')
        torch.save(model_artifact, artifact_path)
        z_model, model_extras = load_feature_payload(artifact_path, labels.shape[0])
        feature_spaces['z_mil_model'] = z_model
        feature_spaces['pos_emb_model'] = model_extras['pos_emb']

    probe_rows = []
    probe_summary = {}
    distance_rows = []
    knn_rows = []
    scale_rows = []
    for feature_name, features in feature_spaces.items():
        probe = probe_space(
            features,
            labels,
            args.probe_splits,
            args.probe_repeats,
            args.seed,
            feature_name,
        )
        probe_summary[feature_name] = probe
        for run in probe['runs']:
            probe_rows.append({'feature': feature_name, **run})
        distance_rows.extend(
            pair_distance_diagnostics(
                features,
                labels,
                args.pair_sample_size,
                args.seed,
                feature_name,
            )
        )
        knn_rows.extend(
            knn_consistency(
                features,
                labels,
                parse_int_list(args.knn_k_list),
                feature_name,
            )
        )
        scale_rows.append(feature_scale_row(feature_name, features))

    fusion = {'status': 'skipped', 'reason': 'No z_mil feature payload was provided.'}
    z_for_fusion = feature_spaces.get('z_mil', feature_spaces.get('z_mil_model'))
    if z_for_fusion is not None:
        fusion = fusion_probe(z_for_fusion, lap_pe, labels, args)
        fusion['status'] = 'ok'

    selected_variant = 'topk_{}'.format(int(args.coarse_topk))
    selected_graph_health = next(
        row for row in topk['graph_rows'] if row['variant'] == selected_variant
    )
    selected_damage = next(
        row for row in topk['damage_rows'] if row['variant'] == selected_variant
    )
    summary = {
        'script_version': SCRIPT_VERSION,
        'data_name': data_name,
        'num_subgraphs': int(labels.shape[0]),
        'label_histogram': {str(k): int(v) for k, v in Counter(labels.tolist()).items()},
        'lap_pe_dim': int(args.lap_pe_dim),
        'coarse_topk': int(args.coarse_topk),
        'source_cluster_ids_identity': bool(np.array_equal(source_ids, np.arange(source_ids.size))),
        'alignment_diagnostics': alignment,
        'known_prior_work': {
            'data_information_md_already_covers': [
                'coarse graph edge homophily',
                'weighted edge homophily',
                'z_mil kNN label consistency',
                'coarse edge weight usefulness',
            ],
            'reason_recomputed_here': (
                'The values are cheap and needed to align the exact LapPE top-k graph with '
                'the spectral and connectivity diagnostics.'
            ),
        },
        'selected_graph_health': selected_graph_health,
        'selected_topk_damage': selected_damage,
        'probes': probe_summary,
        'fusion_probe': fusion,
        'model_utilization': model_use,
    }
    summary['conclusion_signals'] = conclusion_signals(summary)

    write_csv(os.path.join(out_dir, 'probe_runs.csv'), probe_rows)
    write_csv(os.path.join(out_dir, 'distance_metrics.csv'), distance_rows)
    write_csv(os.path.join(out_dir, 'knn_consistency.csv'), knn_rows)
    write_csv(os.path.join(out_dir, 'feature_scales.csv'), scale_rows)
    write_csv(os.path.join(out_dir, 'graph_health.csv'), topk['graph_rows'])
    write_csv(os.path.join(out_dir, 'eigenvalues.csv'), topk['eigen_rows'])
    write_csv(os.path.join(out_dir, 'topk_spectral_damage.csv'), topk['damage_rows'])
    write_csv(os.path.join(out_dir, 'fusion_probe_runs.csv'), fusion.get('runs', []))
    save_json(os.path.join(out_dir, 'summary.json'), summary)
    return summary


def synthetic_dataset(seed=1024):
    import networkx as nx

    rng = np.random.default_rng(int(seed))
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 30)
    graph = nx.Graph()
    graph.add_nodes_from(range(60))
    for node in range(60):
        same_candidates = np.where(labels == labels[node])[0]
        for neighbor in rng.choice(same_candidates, size=4, replace=False):
            if node != int(neighbor):
                graph.add_edge(node, int(neighbor))
        graph.add_edge(node, int(rng.integers(0, 60)))
    subgraphs = []
    for node in range(60):
        subgraph = nx.Graph()
        subgraph.add_node(node, features=np.asarray([float(labels[node]), rng.normal()]))
        subgraph.graph['subgraph_id'] = node
        subgraph.graph['label'] = int(labels[node])
        subgraphs.append(subgraph)
    assignment = np.eye(60, dtype=np.float32)
    return {
        'original_graph': graph,
        'assignment_matrix': assignment,
        'subgraph_structures': subgraphs,
        'subgraph_labels': labels.tolist(),
        'feature_dimension': 2,
        'dataset_metadata': {'feature_dim': 2, 'max_num_nodes': 1},
    }


def write_return_manifest(root_dir, summaries):
    lines = [
        'LapPE diagnostics result bundle',
        '',
        'Return the generated ZIP file, or the complete directory if ZIP creation was disabled.',
        'Primary file: <dataset>/summary.json',
        'Supporting files: graph_health.csv, eigenvalues.csv, topk_spectral_damage.csv,',
        'probe_runs.csv, distance_metrics.csv, knn_consistency.csv, feature_scales.csv,',
        'fusion_probe_runs.csv, and optional model_diagnostic_artifact.pt.',
        '',
        'Datasets: {}'.format(', '.join(summary['data_name'] for summary in summaries)),
    ]
    with open(os.path.join(root_dir, 'RETURN_THIS_DIRECTORY.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    setup_logging()
    ensure_dir(args.output_dir)
    summaries = []
    data_names = ['synthetic'] if args.synthetic_smoke else args.data_name_set
    for data_name in data_names:
        dataset = synthetic_dataset(args.seed) if args.synthetic_smoke else load_dataset(
            args.processed_data_dir,
            data_name,
        )[0]
        dataset_out = os.path.join(args.output_dir, data_name)
        ensure_dir(dataset_out)
        logging.info('Running LapPE diagnostics for %s', data_name)
        summaries.append(diagnose_dataset(dataset, data_name, args, dataset_out))
        logging.info('Saved dataset diagnostics: %s', dataset_out)

    save_json(
        os.path.join(args.output_dir, 'all_datasets_summary.json'),
        {
            'script_version': SCRIPT_VERSION,
            'datasets': summaries,
        },
    )
    write_return_manifest(args.output_dir, summaries)
    if not args.no_zip:
        archive = shutil.make_archive(args.output_dir.rstrip('/\\'), 'zip', args.output_dir)
        logging.info('Created return bundle: %s', archive)


if __name__ == '__main__':
    main()
