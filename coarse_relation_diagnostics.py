# coding=utf-8

"""Diagnostics for fixed-embedding coarse-graph relation models.

This script does not train a model. It reads per-fold z_mil embeddings,
relation_predictions.csv files, and the coarse graph, then exports diagnostics
that explain when coarse-graph relation modeling helps or hurts.
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from sklearn import metrics

from MISGL.utils import hparam
from MISGL.utils import hparams_lib
from coarse_relation_experiment import build_split
from coarse_relation_experiment import prepare_data
from coarse_relation_experiment import sync_hparams_from_loader


def parse_int_list(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(',') if v.strip()]


def parse_str_list(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(',') if v.strip()]


def add_bool_arg(parser, name):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f'--{name}', dest=name, action='store_true', default=None)
    group.add_argument(f'--no_{name}', dest=name, action='store_false')


def cli_option_names(argv):
    names = set()
    for token in argv:
        if not token.startswith('--'):
            continue
        name = token[2:].split('=', 1)[0].replace('-', '_')
        if name.startswith('no_'):
            name = name[3:]
        names.add(name)
    return names


def load_yaml_blocks(hparam_path):
    hp = hparam.HParams()
    hp.from_yaml(hparam_path)
    diag_cfg = getattr(hp, 'coarse_relation_diagnostics', None)
    exp_cfg = getattr(hp, 'coarse_relation_experiment', None)
    return (
        diag_cfg if isinstance(diag_cfg, dict) else {},
        exp_cfg if isinstance(exp_cfg, dict) else {},
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose coarse relation outputs.')
    parser.add_argument('--hparam_path', default='config/b_on.yml')
    parser.add_argument('--data_name', default=None)
    parser.add_argument('--processed_data_dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--fold_idx', type=int, default=None)

    parser.add_argument('--split_source', choices=['fixed_cv', 'dataset'], default=None)
    parser.add_argument('--split_path', default=None)
    add_bool_arg(parser, 'create_split_if_missing')
    parser.add_argument('--dataset_val_frac', type=float, default=None)

    parser.add_argument('--relation_out_dir', default=None)
    parser.add_argument('--relation_graph', default=None)
    parser.add_argument('--diagnostics_out_dir', default=None)
    parser.add_argument('--embeddings_path', default=None)
    parser.add_argument('--predictions_path', default=None)
    parser.add_argument('--models', default=None)
    parser.add_argument('--baseline_model', default=None)

    parser.add_argument('--bin_count', type=int, default=None)
    parser.add_argument('--edge_sample_max_per_fold', type=int, default=None)
    parser.add_argument('--node_case_splits', default=None)
    add_bool_arg(parser, 'run_knn_diagnostics')
    parser.add_argument('--knn_k_list', default=None)
    parser.add_argument('--knn_batch_size', type=int, default=None)
    parser.add_argument('--knn_max_source_nodes', type=int, default=None)

    add_bool_arg(parser, 'synthetic_smoke')
    parser.add_argument('--smoke_num_nodes', type=int, default=80)
    args = parser.parse_args()

    cli_names = cli_option_names(sys.argv[1:])
    diag_cfg, exp_cfg = load_yaml_blocks(args.hparam_path)
    apply_config_defaults(args, exp_cfg, cli_names, prefix='experiment')
    apply_config_defaults(args, diag_cfg, cli_names, prefix='diagnostics')
    apply_hard_defaults(args)
    return args


def apply_config_defaults(args, cfg, cli_names, prefix):
    if not cfg:
        return
    aliases = {
        'out_dir': 'relation_out_dir',
    }
    for key, value in cfg.items():
        target = aliases.get(key, key)
        if not hasattr(args, target):
            continue
        if target in cli_names:
            continue
        current = getattr(args, target)
        if prefix == 'diagnostics' or current is None:
            setattr(args, target, value)


def apply_hard_defaults(args):
    if args.data_name is None:
        args.data_name = 'ogbn_arxiv'
    if args.processed_data_dir is None:
        args.processed_data_dir = '/data/yg/Subgraph-MIL/Data/processed_data'
    if args.device is None:
        args.device = 'cpu'
    if args.seed is None:
        args.seed = 1024
    if args.split_source is None:
        args.split_source = 'fixed_cv'
    if args.create_split_if_missing is None:
        args.create_split_if_missing = True
    if args.dataset_val_frac is None:
        args.dataset_val_frac = 0.2
    if args.relation_out_dir is None:
        args.relation_out_dir = 'result/coarse_relation'
    if args.relation_graph is None:
        args.relation_graph = 'coarse'
    if args.diagnostics_out_dir is None:
        args.diagnostics_out_dir = 'result/coarse_relation_diagnostics/{data_name}'
    if args.embeddings_path is None:
        args.embeddings_path = 'result/coarse_relation/{data_name}_zmil_fold{fold_idx}.pt'
    if args.predictions_path is None:
        args.predictions_path = '{relation_out_dir}/{data_name}/fold_{fold_idx}/relation_predictions.csv'
    if args.baseline_model is None:
        args.baseline_model = 'mlp'
    if args.bin_count is None:
        args.bin_count = 10
    if args.edge_sample_max_per_fold is None:
        args.edge_sample_max_per_fold = 200000
    if args.node_case_splits is None:
        args.node_case_splits = 'val,test'
    if args.run_knn_diagnostics is None:
        args.run_knn_diagnostics = True
    if args.knn_k_list is None:
        args.knn_k_list = '8,16,32,64'
    if args.knn_batch_size is None:
        args.knn_batch_size = 512
    if args.knn_max_source_nodes is None:
        args.knn_max_source_nodes = 5000
    if args.synthetic_smoke is None:
        args.synthetic_smoke = False


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def template_path(path, args, fold_idx):
    values = {
        'data_name': args.data_name,
        'fold_idx': fold_idx,
        'seed': args.seed,
        'relation_out_dir': args.relation_out_dir,
        'relation_graph': args.relation_graph,
    }
    return str(path).format(**values)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_hparams(args):
    hparams = hparam.HParams()
    hparams.from_yaml(args.hparam_path)
    hparams_lib.apply_defaults(hparams)
    hparams.data_name = args.data_name
    hparams.processed_data_dir = args.processed_data_dir
    hparams.device = args.device
    hparams.preload_data_to_gpu = False
    return hparams


def resolve_fold_indices(loader, args):
    if args.synthetic_smoke:
        return [0]
    if args.split_source != 'fixed_cv':
        return [None]
    if args.fold_idx is not None:
        return [int(args.fold_idx)]
    return list(range(int(loader.cv_num_folds)))


def build_split_args(args, fold_idx):
    return argparse.Namespace(
        split_source=args.split_source,
        split_path=template_path(args.split_path, args, fold_idx) if args.split_path else None,
        fold_idx=fold_idx,
        create_split_if_missing=bool(args.create_split_if_missing),
        dataset_val_frac=float(args.dataset_val_frac),
        seed=int(args.seed),
    )


def load_embedding_payload(path, num_nodes):
    payload = torch.load(path, map_location='cpu')
    features = payload['features'].detach().cpu().float().numpy()
    labels = payload['labels'].detach().cpu().long().numpy()
    orig_indices = payload.get('orig_indices')
    if orig_indices is None:
        orig_indices = np.arange(features.shape[0], dtype=np.int64)
    elif isinstance(orig_indices, torch.Tensor):
        orig_indices = orig_indices.detach().cpu().long().numpy()
    else:
        orig_indices = np.asarray(orig_indices, dtype=np.int64)

    ordered_features = np.zeros((num_nodes, features.shape[1]), dtype=np.float32)
    ordered_labels = np.full((num_nodes,), -1, dtype=np.int64)
    present = np.zeros((num_nodes,), dtype=bool)
    for row_idx, orig_idx in enumerate(orig_indices):
        node_idx = int(orig_idx)
        if 0 <= node_idx < num_nodes:
            ordered_features[node_idx] = features[row_idx]
            ordered_labels[node_idx] = int(labels[row_idx])
            present[node_idx] = True
    if not np.all(present):
        missing = np.where(~present)[0][:10].tolist()
        raise ValueError(f'Embeddings are missing coarse nodes, examples: {missing}')
    return ordered_features, ordered_labels


def read_predictions_csv(path, num_nodes):
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    models = sorted({name[:-5] for name in fieldnames if name.endswith('_prob')})
    predictions = {
        model: {
            'prob': np.full((num_nodes,), np.nan, dtype=np.float32),
            'pred': np.full((num_nodes,), -1, dtype=np.int64),
            'diagnostics': defaultdict(lambda: np.full((num_nodes,), np.nan, dtype=np.float32)),
        }
        for model in models
    }
    split = np.full((num_nodes,), 'unknown', dtype=object)
    csv_labels = np.full((num_nodes,), -1, dtype=np.int64)

    for row in rows:
        if not row.get('orig_idx', ''):
            continue
        node_idx = int(row['orig_idx'])
        if node_idx < 0 or node_idx >= num_nodes:
            continue
        split[node_idx] = row.get('split', 'unknown') or 'unknown'
        if row.get('label', '') != '':
            csv_labels[node_idx] = int(float(row['label']))
        for model in models:
            prob_key = f'{model}_prob'
            pred_key = f'{model}_pred'
            if row.get(prob_key, '') != '':
                predictions[model]['prob'][node_idx] = float(row[prob_key])
            if row.get(pred_key, '') != '':
                predictions[model]['pred'][node_idx] = int(float(row[pred_key]))
            prefix = f'{model}_'
            for key, value in row.items():
                if key in (prob_key, pred_key) or not key.startswith(prefix):
                    continue
                diag_name = key[len(prefix):]
                if value != '':
                    predictions[model]['diagnostics'][diag_name][node_idx] = float(value)

    for model in models:
        predictions[model]['diagnostics'] = dict(predictions[model]['diagnostics'])
    return predictions, split, csv_labels, models


def l2_normalize(features):
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return features / denom


def safe_auc(target, score):
    target = np.asarray(target)
    score = np.asarray(score)
    valid = np.isfinite(score)
    target = target[valid]
    score = score[valid]
    if target.size == 0 or np.unique(target).size < 2:
        return None
    return float(metrics.roc_auc_score(target, score))


def summarize_numeric(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'mean': None, 'std': None, 'min': None, 'max': None}
    return {
        'mean': float(values.mean()),
        'std': float(values.std(ddof=0)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def build_edge_arrays(coarse_adj, labels, features_norm, predictions):
    coo = coarse_adj.tocoo()
    src = coo.row.astype(np.int64, copy=False)
    dst = coo.col.astype(np.int64, copy=False)
    weight = coo.data.astype(np.float32, copy=False)
    valid = (
        (src >= 0) & (dst >= 0)
        & (src < labels.shape[0]) & (dst < labels.shape[0])
        & (labels[src] >= 0) & (labels[dst] >= 0)
        & (src != dst)
    )
    src = src[valid]
    dst = dst[valid]
    weight = weight[valid]
    same_label = (labels[src] == labels[dst]).astype(np.int64)
    cosine = np.sum(features_norm[src] * features_norm[dst], axis=1).astype(np.float32)

    feature_scores = {
        'coarse_weight': weight,
        'z_mil_cosine': cosine,
        'weight_times_positive_cosine': weight * np.maximum(cosine, 0.0),
    }
    for model, payload in predictions.items():
        prob = payload['prob']
        if np.isfinite(prob[src]).any() and np.isfinite(prob[dst]).any():
            feature_scores[f'{model}_prob_agreement'] = -np.abs(prob[src] - prob[dst])
            feature_scores[f'{model}_prob_mean'] = 0.5 * (prob[src] + prob[dst])
    return {
        'src': src,
        'dst': dst,
        'weight': weight,
        'same_label': same_label,
        'cosine': cosine,
        'feature_scores': feature_scores,
    }


def edge_feature_auc_rows(data_name, fold_idx, edge_arrays):
    rows = []
    same_label = edge_arrays['same_label']
    for feature_name, values in edge_arrays['feature_scores'].items():
        auc = safe_auc(same_label, values)
        rows.append({
            'data_name': data_name,
            'fold_idx': fold_idx,
            'feature': feature_name,
            'auc_predict_same_label': auc,
            'num_edges': int(same_label.size),
        })
    return rows


def edge_bin_rows(data_name, fold_idx, edge_arrays, bin_count):
    rows = []
    same_label = edge_arrays['same_label']
    weight = edge_arrays['weight']
    for feature_name, values in edge_arrays['feature_scores'].items():
        values = np.asarray(values)
        valid = np.where(np.isfinite(values))[0]
        if valid.size == 0:
            continue
        order = valid[np.argsort(values[valid])]
        chunks = np.array_split(order, int(bin_count))
        for bin_id, idx in enumerate(chunks):
            if idx.size == 0:
                continue
            local_same = same_label[idx].astype(np.float64)
            local_weight = weight[idx].astype(np.float64)
            rows.append({
                'data_name': data_name,
                'fold_idx': fold_idx,
                'feature': feature_name,
                'bin_id': int(bin_id),
                'count': int(idx.size),
                'feature_min': float(np.min(values[idx])),
                'feature_max': float(np.max(values[idx])),
                'feature_mean': float(np.mean(values[idx])),
                'same_label_rate': float(np.mean(local_same)),
                'weighted_same_label_rate': float(np.sum(local_same * local_weight) / max(np.sum(local_weight), 1e-12)),
                'edge_weight_mean': float(np.mean(local_weight)),
            })
    return rows


def sample_edge_rows(data_name, fold_idx, edge_arrays, labels, split, predictions, max_rows, seed):
    src = edge_arrays['src']
    dst = edge_arrays['dst']
    total = src.size
    if max_rows is not None and total > int(max_rows):
        rng = np.random.default_rng(int(seed) + int(fold_idx or 0))
        chosen = np.sort(rng.choice(total, size=int(max_rows), replace=False))
    else:
        chosen = np.arange(total)

    rows = []
    for idx in chosen:
        row = {
            'data_name': data_name,
            'fold_idx': fold_idx,
            'src': int(src[idx]),
            'dst': int(dst[idx]),
            'src_split': split[src[idx]],
            'dst_split': split[dst[idx]],
            'src_label': int(labels[src[idx]]),
            'dst_label': int(labels[dst[idx]]),
            'same_label': int(edge_arrays['same_label'][idx]),
            'coarse_weight': float(edge_arrays['weight'][idx]),
            'z_mil_cosine': float(edge_arrays['cosine'][idx]),
        }
        for model, payload in predictions.items():
            prob = payload['prob']
            if np.isfinite(prob[src[idx]]) and np.isfinite(prob[dst[idx]]):
                row[f'{model}_prob_gap_abs'] = float(abs(prob[src[idx]] - prob[dst[idx]]))
        rows.append(row)
    return rows


def node_neighbor_stats(coarse_adj, labels, predictions):
    adj = coarse_adj.tocsr()
    num_nodes = adj.shape[0]
    degree = np.diff(adj.indptr).astype(np.float32)
    weighted_degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    labels_float = labels.astype(np.float32)
    binary_adj = adj.copy()
    binary_adj.data = np.ones_like(binary_adj.data)

    pos_neighbor_count = np.asarray(binary_adj @ labels_float).reshape(-1)
    pos_neighbor_weight = np.asarray(adj @ labels_float).reshape(-1)
    neighbor_positive_ratio = pos_neighbor_count / np.maximum(degree, 1.0)
    weighted_neighbor_positive_ratio = pos_neighbor_weight / np.maximum(weighted_degree, 1e-12)
    neighbor_same_label_ratio = np.where(
        labels > 0,
        neighbor_positive_ratio,
        (degree - pos_neighbor_count) / np.maximum(degree, 1.0),
    )
    weighted_neighbor_same_label_ratio = np.where(
        labels > 0,
        weighted_neighbor_positive_ratio,
        (weighted_degree - pos_neighbor_weight) / np.maximum(weighted_degree, 1e-12),
    )

    stats = {
        'degree': degree,
        'weighted_degree': weighted_degree,
        'neighbor_positive_label_ratio': neighbor_positive_ratio,
        'weighted_neighbor_positive_label_ratio': weighted_neighbor_positive_ratio,
        'neighbor_same_label_ratio': neighbor_same_label_ratio,
        'weighted_neighbor_same_label_ratio': weighted_neighbor_same_label_ratio,
    }
    for model, payload in predictions.items():
        prob = payload['prob'].astype(np.float32)
        prob_filled = np.nan_to_num(prob, nan=0.0)
        finite = np.isfinite(prob).astype(np.float32)
        denom = np.asarray(adj @ finite).reshape(-1)
        weighted_sum = np.asarray(adj @ prob_filled).reshape(-1)
        stats[f'{model}_weighted_neighbor_prob'] = weighted_sum / np.maximum(denom, 1e-12)
    return stats


def change_group(baseline_correct, model_correct):
    if baseline_correct and model_correct:
        return 'both_correct'
    if (not baseline_correct) and model_correct:
        return 'baseline_wrong_model_right'
    if baseline_correct and (not model_correct):
        return 'baseline_right_model_wrong'
    return 'both_wrong'


def build_node_case_rows(data_name, fold_idx, labels, split, predictions, neighbor_stats, models, baseline, splits_to_keep):
    rows = []
    if baseline not in predictions:
        return rows
    baseline_pred = predictions[baseline]['pred']
    baseline_correct = baseline_pred == labels
    splits_to_keep = set(splits_to_keep)
    for node_idx in range(labels.shape[0]):
        if split[node_idx] not in splits_to_keep:
            continue
        row = {
            'data_name': data_name,
            'fold_idx': fold_idx,
            'orig_idx': int(node_idx),
            'split': split[node_idx],
            'label': int(labels[node_idx]),
        }
        for key, values in neighbor_stats.items():
            row[key] = float(values[node_idx])
        for model in models:
            if model not in predictions:
                continue
            payload = predictions[model]
            row[f'{model}_prob'] = float(payload['prob'][node_idx]) if np.isfinite(payload['prob'][node_idx]) else ''
            row[f'{model}_pred'] = int(payload['pred'][node_idx]) if payload['pred'][node_idx] >= 0 else ''
            row[f'{model}_correct'] = int(payload['pred'][node_idx] == labels[node_idx])
            if model != baseline and payload['pred'][node_idx] >= 0 and baseline_pred[node_idx] >= 0:
                row[f'{model}_vs_{baseline}'] = change_group(
                    bool(baseline_correct[node_idx]),
                    bool(payload['pred'][node_idx] == labels[node_idx]),
                )
            for diag_name, diag_values in payload.get('diagnostics', {}).items():
                value = diag_values[node_idx]
                row[f'{model}_{diag_name}'] = float(value) if np.isfinite(value) else ''
        rows.append(row)
    return rows


def summarize_node_cases(node_rows, models, baseline):
    grouped = defaultdict(list)
    for row in node_rows:
        for model in models:
            if model == baseline:
                continue
            group = row.get(f'{model}_vs_{baseline}')
            if group:
                grouped[(row['data_name'], row['fold_idx'], row['split'], model, group)].append(row)

    summary_rows = []
    numeric_keys = [
        'degree',
        'weighted_degree',
        'neighbor_same_label_ratio',
        'weighted_neighbor_same_label_ratio',
    ]
    for key, rows in grouped.items():
        data_name, fold_idx, split_name, model, group = key
        out = {
            'data_name': data_name,
            'fold_idx': fold_idx,
            'split': split_name,
            'model': model,
            'group': group,
            'count': len(rows),
        }
        for numeric_key in numeric_keys:
            values = [float(r[numeric_key]) for r in rows if r.get(numeric_key, '') != '']
            out[f'{numeric_key}_mean'] = summarize_numeric(values)['mean']
        for candidate in (f'{model}_gate1', f'{model}_gate2'):
            values = [float(r[candidate]) for r in rows if r.get(candidate, '') != '']
            if values:
                out[f'{candidate}_mean'] = summarize_numeric(values)['mean']
                out[f'{candidate}_std'] = summarize_numeric(values)['std']
        summary_rows.append(out)
    return summary_rows


def gate_summary_rows(node_rows, models):
    rows = []
    grouped = defaultdict(list)
    for row in node_rows:
        for model in models:
            for gate_name in (f'{model}_gate1', f'{model}_gate2'):
                if row.get(gate_name, '') != '':
                    grouped[(row['data_name'], row['fold_idx'], row['split'], model, gate_name)].append(float(row[gate_name]))
    for (data_name, fold_idx, split_name, model, gate_name), values in grouped.items():
        stats = summarize_numeric(values)
        rows.append({
            'data_name': data_name,
            'fold_idx': fold_idx,
            'split': split_name,
            'model': model,
            'gate': gate_name.replace(f'{model}_', ''),
            'count': len(values),
            **stats,
        })
    return rows


def knn_graph_summary(data_name, fold_idx, features_norm, labels, coarse_adj, k_list, batch_size, max_source_nodes, seed):
    num_nodes = features_norm.shape[0]
    k_list = sorted(set(int(k) for k in k_list if int(k) > 0))
    if not k_list:
        return []
    max_k = min(max(k_list), max(num_nodes - 1, 1))
    if max_source_nodes is not None and num_nodes > int(max_source_nodes):
        rng = np.random.default_rng(int(seed) + int(fold_idx or 0))
        source_nodes = np.sort(rng.choice(num_nodes, size=int(max_source_nodes), replace=False))
    else:
        source_nodes = np.arange(num_nodes)

    same_counts = {k: 0 for k in k_list}
    total_counts = {k: 0 for k in k_list}
    overlap_counts = {k: 0 for k in k_list}
    coarse_same_counts = {k: 0 for k in k_list}
    coarse_total_counts = {k: 0 for k in k_list}
    adj = coarse_adj.tocsr()

    coarse_top_sets = {}
    for node_idx in source_nodes:
        start, end = adj.indptr[node_idx], adj.indptr[node_idx + 1]
        row_indices = adj.indices[start:end]
        row_data = adj.data[start:end]
        if row_indices.size:
            order = np.argsort(-row_data)
            row_indices = row_indices[order]
        coarse_top_sets[int(node_idx)] = row_indices
        for k in k_list:
            local = row_indices[:k]
            if local.size:
                coarse_same_counts[k] += int(np.sum(labels[local] == labels[node_idx]))
                coarse_total_counts[k] += int(local.size)

    for offset in range(0, source_nodes.size, int(batch_size)):
        batch_nodes = source_nodes[offset:offset + int(batch_size)]
        sims = features_norm[batch_nodes] @ features_norm.T
        for local_row, node_idx in enumerate(batch_nodes):
            sims[local_row, int(node_idx)] = -np.inf
        if max_k >= sims.shape[1]:
            top_idx = np.argsort(-sims, axis=1)[:, :max_k]
        else:
            top_unsorted = np.argpartition(-sims, kth=max_k - 1, axis=1)[:, :max_k]
            top_scores = np.take_along_axis(sims, top_unsorted, axis=1)
            order = np.argsort(-top_scores, axis=1)
            top_idx = np.take_along_axis(top_unsorted, order, axis=1)
        for local_row, node_idx in enumerate(batch_nodes):
            coarse_set_cache = {}
            for k in k_list:
                neigh = top_idx[local_row, :k]
                same_counts[k] += int(np.sum(labels[neigh] == labels[node_idx]))
                total_counts[k] += int(neigh.size)
                coarse_top = coarse_top_sets[int(node_idx)][:k]
                coarse_key = int(k)
                if coarse_key not in coarse_set_cache:
                    coarse_set_cache[coarse_key] = set(int(v) for v in coarse_top.tolist())
                if coarse_set_cache[coarse_key]:
                    overlap_counts[k] += sum(int(v) in coarse_set_cache[coarse_key] for v in neigh.tolist())

    rows = []
    for k in k_list:
        rows.append({
            'data_name': data_name,
            'fold_idx': fold_idx,
            'k': int(k),
            'num_source_nodes': int(source_nodes.size),
            'zmil_knn_same_label_rate': float(same_counts[k] / max(total_counts[k], 1)),
            'coarse_topk_same_label_rate': float(coarse_same_counts[k] / max(coarse_total_counts[k], 1)),
            'knn_coarse_overlap_rate': float(overlap_counts[k] / max(total_counts[k], 1)),
        })
    return rows


def write_csv(path, rows):
    ensure_parent(path)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metric_rows(rows, group_keys, value_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(row)
    out = []
    for key, local_rows in grouped.items():
        item = {group_keys[i]: key[i] for i in range(len(group_keys))}
        item['num_rows'] = len(local_rows)
        for value_key in value_keys:
            values = [r.get(value_key) for r in local_rows if r.get(value_key) is not None]
            stats = summarize_numeric(values)
            item[f'{value_key}_mean'] = stats['mean']
            item[f'{value_key}_std'] = stats['std']
        out.append(item)
    return out


def diagnose_fold(args, loader, fold_idx, out_dir):
    split_args = build_split_args(args, fold_idx)
    split_meta = build_split(loader, split_args)
    num_nodes = int(loader.coarse_adj.shape[0])
    embedding_path = template_path(args.embeddings_path, args, fold_idx)
    predictions_path = template_path(args.predictions_path, args, fold_idx)
    logging.info('Fold %s embeddings: %s', fold_idx, embedding_path)
    logging.info('Fold %s predictions: %s', fold_idx, predictions_path)
    features, labels = load_embedding_payload(embedding_path, num_nodes)
    predictions, split, csv_labels, detected_models = read_predictions_csv(predictions_path, num_nodes)
    del csv_labels
    models = parse_str_list(args.models, default=detected_models) if args.models else detected_models
    models = [model for model in models if model in predictions]
    baseline = args.baseline_model
    features_norm = l2_normalize(features)
    edge_arrays = build_edge_arrays(loader.coarse_adj, labels, features_norm, predictions)

    edge_homophily = {
        'data_name': args.data_name,
        'fold_idx': fold_idx,
        'num_edges': int(edge_arrays['same_label'].size),
        'edge_homophily': float(np.mean(edge_arrays['same_label'])) if edge_arrays['same_label'].size else None,
        'weighted_edge_homophily': float(
            np.sum(edge_arrays['same_label'] * edge_arrays['weight']) / max(np.sum(edge_arrays['weight']), 1e-12)
        ) if edge_arrays['same_label'].size else None,
    }
    edge_auc = edge_feature_auc_rows(args.data_name, fold_idx, edge_arrays)
    edge_bins = edge_bin_rows(args.data_name, fold_idx, edge_arrays, int(args.bin_count))
    edge_sample = sample_edge_rows(
        args.data_name,
        fold_idx,
        edge_arrays,
        labels,
        split,
        predictions,
        args.edge_sample_max_per_fold,
        args.seed,
    )

    neighbor_stats = node_neighbor_stats(loader.coarse_adj, labels, predictions)
    node_rows = build_node_case_rows(
        args.data_name,
        fold_idx,
        labels,
        split,
        predictions,
        neighbor_stats,
        models,
        baseline,
        parse_str_list(args.node_case_splits, default=['val', 'test']),
    )
    node_summary = summarize_node_cases(node_rows, models, baseline)
    gate_rows = gate_summary_rows(node_rows, models)
    if args.run_knn_diagnostics:
        knn_rows = knn_graph_summary(
            args.data_name,
            fold_idx,
            features_norm,
            labels,
            loader.coarse_adj,
            parse_int_list(args.knn_k_list, default=[8, 16, 32, 64]),
            int(args.knn_batch_size),
            int(args.knn_max_source_nodes),
            int(args.seed),
        )
    else:
        knn_rows = []

    fold_summary = {
        'data_name': args.data_name,
        'fold_idx': fold_idx,
        'split': split_meta,
        'edge_homophily': edge_homophily,
        'edge_feature_auc': edge_auc,
        'num_node_case_rows': len(node_rows),
    }
    with open(os.path.join(out_dir, f'fold_{fold_idx}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(json_safe(fold_summary), f, indent=2, ensure_ascii=False)
    return {
        'summary': fold_summary,
        'edge_homophily': [edge_homophily],
        'edge_auc': edge_auc,
        'edge_bins': edge_bins,
        'edge_sample': edge_sample,
        'node_cases': node_rows,
        'node_summary': node_summary,
        'gate_summary': gate_rows,
        'knn_summary': knn_rows,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def synthetic_smoke(args):
    rng = np.random.default_rng(int(args.seed))
    num_nodes = int(args.smoke_num_nodes)
    labels = rng.integers(0, 2, size=num_nodes, dtype=np.int64)
    features = rng.normal(size=(num_nodes, 16)).astype(np.float32)
    rows = []
    cols = []
    for node_idx in range(num_nodes):
        neigh = rng.choice(num_nodes, size=min(8, num_nodes), replace=False)
        rows.extend([node_idx] * len(neigh))
        cols.extend(neigh.tolist())
    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(num_nodes, num_nodes))
    features_norm = l2_normalize(features)
    predictions = {}
    for model in ('mlp', 'sage', 'gated_sage'):
        prob = rng.random(num_nodes).astype(np.float32)
        predictions[model] = {
            'prob': prob,
            'pred': (prob > 0.5).astype(np.int64),
            'diagnostics': {},
        }
    predictions['gated_sage']['diagnostics'] = {
        'gate1': rng.random(num_nodes).astype(np.float32),
        'gate2': rng.random(num_nodes).astype(np.float32),
    }
    split = np.asarray(['train'] * num_nodes, dtype=object)
    split[int(num_nodes * 0.6):int(num_nodes * 0.8)] = 'val'
    split[int(num_nodes * 0.8):] = 'test'
    edge_arrays = build_edge_arrays(adj, labels, features_norm, predictions)
    neighbor_stats = node_neighbor_stats(adj, labels, predictions)
    node_rows = build_node_case_rows(
        'synthetic',
        0,
        labels,
        split,
        predictions,
        neighbor_stats,
        ['mlp', 'sage', 'gated_sage'],
        'mlp',
        ['val', 'test'],
    )
    out_dir = template_path(args.diagnostics_out_dir, args, 'smoke')
    ensure_dir(out_dir)
    write_csv(os.path.join(out_dir, 'edge_feature_auc.csv'), edge_feature_auc_rows('synthetic', 0, edge_arrays))
    write_csv(os.path.join(out_dir, 'edge_reliability_bins.csv'), edge_bin_rows('synthetic', 0, edge_arrays, 4))
    write_csv(os.path.join(out_dir, 'node_error_cases.csv'), node_rows)
    write_csv(os.path.join(out_dir, 'node_flip_summary.csv'), summarize_node_cases(node_rows, ['mlp', 'sage', 'gated_sage'], 'mlp'))
    write_csv(os.path.join(out_dir, 'gate_summary.csv'), gate_summary_rows(node_rows, ['gated_sage']))
    write_csv(os.path.join(out_dir, 'knn_graph_summary.csv'), knn_graph_summary('synthetic', 0, features_norm, labels, adj, [4, 8], 16, num_nodes, args.seed))
    logging.info('Synthetic smoke outputs: %s', out_dir)


def main():
    args = parse_args()
    setup_logging()
    if args.synthetic_smoke:
        synthetic_smoke(args)
        return

    hparams = load_hparams(args)
    loader = prepare_data(hparams)
    sync_hparams_from_loader(hparams, loader)
    fold_indices = resolve_fold_indices(loader, args)
    out_dir = template_path(args.diagnostics_out_dir, args, 'all')
    ensure_dir(out_dir)
    logging.info('Diagnostics output directory: %s', out_dir)
    logging.info('Running folds: %s', fold_indices)

    combined = defaultdict(list)
    fold_summaries = []
    for fold_idx in fold_indices:
        fold_out = diagnose_fold(args, loader, fold_idx, out_dir)
        fold_summaries.append(fold_out['summary'])
        for key, rows in fold_out.items():
            if key == 'summary':
                continue
            combined[key].extend(rows)

    write_csv(os.path.join(out_dir, 'edge_homophily.csv'), combined['edge_homophily'])
    write_csv(os.path.join(out_dir, 'edge_feature_auc.csv'), combined['edge_auc'])
    write_csv(os.path.join(out_dir, 'edge_reliability_bins.csv'), combined['edge_bins'])
    write_csv(os.path.join(out_dir, 'edge_reliability_sample.csv'), combined['edge_sample'])
    write_csv(os.path.join(out_dir, 'node_error_cases.csv'), combined['node_cases'])
    write_csv(os.path.join(out_dir, 'node_flip_summary.csv'), combined['node_summary'])
    write_csv(os.path.join(out_dir, 'gate_summary.csv'), combined['gate_summary'])
    write_csv(os.path.join(out_dir, 'knn_graph_summary.csv'), combined['knn_summary'])

    aggregate = {
        'data_name': args.data_name,
        'fold_indices': fold_indices,
        'fold_summaries': fold_summaries,
        'edge_feature_auc_mean': aggregate_metric_rows(
            combined['edge_auc'],
            ['feature'],
            ['auc_predict_same_label'],
        ),
        'edge_homophily_mean': aggregate_metric_rows(
            combined['edge_homophily'],
            ['data_name'],
            ['edge_homophily', 'weighted_edge_homophily'],
        ),
        'knn_summary_mean': aggregate_metric_rows(
            combined['knn_summary'],
            ['k'],
            ['zmil_knn_same_label_rate', 'coarse_topk_same_label_rate', 'knn_coarse_overlap_rate'],
        ),
    }
    summary_path = os.path.join(out_dir, 'diagnostics_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(aggregate), f, indent=2, ensure_ascii=False)
    logging.info('Saved diagnostics summary: %s', summary_path)


if __name__ == '__main__':
    main()
