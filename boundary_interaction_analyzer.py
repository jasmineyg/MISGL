# coding=utf-8

"""Boundary Interaction Profile validation for MISGL."""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch

from boundary_context_analyzer import (
    _build_export_loader,
    _classification_metrics,
    _clear_cuda_cache,
    _copy_dataset_hparams,
    _extract_logits,
    _fit_probe,
    _move_batch_to_device,
    _parse_data_name_set,
    _parse_fold_selection,
    _positive_probabilities,
    _predict_probe,
    _resolve_experiment_timestamp,
    _summary_metrics,
    _write_excel_if_reasonable,
)
from MISGL.bin.train_eval import train_eval_iter
from MISGL.models.encoder import MISGLEncoder
from MISGL.utils import hparam, hparams_lib, reproducibility
from MISGL.utils.boundary_context import CONTEXT_FEATURE_NAMES
from MISGL.utils.boundary_interaction import (
    BIP_FEATURE_NAMES,
    BOUNDARY_PROFILE_FEATURE_NAMES,
    INTERACTION_PROFILE_FEATURE_NAMES,
    SEMANTIC_PROFILE_FEATURE_NAMES,
    build_boundary_interaction_topology,
    feature_vector,
    safe_cosine,
    safe_distance,
)
from MISGL.utils.global_variables import g_key
from MISGL.utils.load_data import GraphDataLoaderWrapper


PROBE_CONFIGS = (
    'z_mil_only',
    'current_context_only',
    'z_mil_current_context',
    'z_mil_shuffled_current_context',
    'boundary_profile_only',
    'semantic_profile_only',
    'interaction_pattern_only',
    'bip_only',
    'z_mil_boundary_profile',
    'z_mil_semantic_profile',
    'z_mil_interaction_pattern',
    'z_mil_bip',
    'z_mil_full_bip',
    'z_mil_shuffled_bip',
)

FEATURE_GROUP_ABLATION_CONFIGS = (
    'boundary_profile_only',
    'semantic_profile_only',
    'interaction_pattern_only',
    'bip_only',
    'z_mil_boundary_profile',
    'z_mil_semantic_profile',
    'z_mil_interaction_pattern',
    'z_mil_full_bip',
)

FIX_BREAK_CONFIGS = (
    'z_mil_bip',
    'z_mil_shuffled_bip',
    'z_mil_boundary_profile',
    'z_mil_semantic_profile',
    'z_mil_interaction_pattern',
)

FINAL_SPLITS = ('train', 'val', 'test')


def _local_original_positions(dataset_raw, orig_idx, num_nodes, node_to_position):
    subgraphs = dataset_raw.get('subgraph_structures', [])
    if orig_idx < 0 or orig_idx >= len(subgraphs):
        return np.full((num_nodes,), -1, dtype=np.int64)

    positions = []
    subgraph = subgraphs[orig_idx]
    for node_id, attrs in list(subgraph.nodes(data=True))[:num_nodes]:
        original_id = node_id
        for key in (
            'original_id',
            'original_index',
            'orig_id',
            'orig_idx',
            'node_index',
            'node_id',
            'original_node_id',
        ):
            if key in attrs and attrs[key] is not None:
                original_id = attrs[key]
                break
        try:
            original_id = int(original_id)
        except (TypeError, ValueError):
            pass
        positions.append(int(node_to_position.get(original_id, -1)))

    if len(positions) < num_nodes:
        positions.extend([-1] * (num_nodes - len(positions)))
    return np.asarray(positions[:num_nodes], dtype=np.int64)


def _extract_attention(model_output, model, num_nodes_list):
    if isinstance(model_output, dict):
        branch_output = model_output.get('branch_b', None)
        if branch_output is not None:
            attention = branch_output.get('a_pad', None)
            if attention is not None:
                attention = attention.detach().cpu().numpy()
                return [
                    np.asarray(attention[idx, :num_nodes], dtype=np.float32)
                    for idx, num_nodes in enumerate(num_nodes_list)
                ]

    attention = getattr(getattr(model, 'gat_layer', None), 'last_attention_summary', None)
    if attention is not None:
        attention = attention.detach().cpu().numpy()
        return [
            np.asarray(attention[idx, :num_nodes], dtype=np.float32)
            for idx, num_nodes in enumerate(num_nodes_list)
        ]
    return [np.zeros((num_nodes,), dtype=np.float32) for num_nodes in num_nodes_list]


def _projection_matrix(input_dim, output_dim, seed):
    output_dim = max(1, min(int(output_dim), int(input_dim)))
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(max(input_dim, 1)),
        size=(input_dim, output_dim),
    )
    return matrix.astype(np.float32)


def _mean_rows(matrix, mask):
    if matrix.size == 0 or not np.any(mask):
        return np.zeros((matrix.shape[1],), dtype=np.float32)
    return np.asarray(matrix[mask].mean(axis=0), dtype=np.float32)


def _attention_features(attention, boundary_mask):
    attention = np.asarray(attention, dtype=np.float32).reshape(-1)
    boundary_mask = np.asarray(boundary_mask, dtype=bool).reshape(-1)
    boundary_attention = attention[boundary_mask]
    inner_attention = attention[~boundary_mask]
    total_attention = float(attention.sum())
    if boundary_attention.size == 0:
        mean_value = max_value = std_value = sum_value = share = 0.0
    else:
        mean_value = float(boundary_attention.mean())
        max_value = float(boundary_attention.max())
        std_value = float(boundary_attention.std())
        sum_value = float(boundary_attention.sum())
        share = sum_value / total_attention if total_attention > 0 else 0.0
    inner_mean = float(inner_attention.mean()) if inner_attention.size else 0.0

    if attention.size == 0:
        top1_hit = top5_ratio = 0.0
    else:
        order = np.argsort(-attention)
        top1_hit = float(boundary_mask[order[0]])
        top5 = order[:min(5, attention.size)]
        top5_ratio = float(boundary_mask[top5].mean()) if top5.size else 0.0

    return {
        'bip_boundary_attention_mean': mean_value,
        'bip_boundary_attention_max': max_value,
        'bip_boundary_attention_std': std_value,
        'bip_boundary_attention_sum': sum_value,
        'bip_boundary_attention_share': share,
        'bip_inner_attention_mean': inner_mean,
        'bip_boundary_inner_attention_difference': mean_value - inner_mean,
        'bip_top1_attention_is_boundary': top1_hit,
        'bip_top5_attention_boundary_ratio': top5_ratio,
    }


def _initialize_fold_state(topology, semantic_dim):
    node_count = len(topology['original_nodes'])
    return {
        'projection': None,
        'common_dim': None,
        'semantic_dim': int(semantic_dim),
        'node_embedding_sum': None,
        'node_embedding_count': np.zeros((node_count,), dtype=np.int32),
        'node_probability_sum': np.zeros((node_count,), dtype=np.float64),
        'node_probability_count': np.zeros((node_count,), dtype=np.int32),
    }


def _ensure_projection_and_storage(state, node_dim, z_dim, seed):
    if state['projection'] is not None:
        return
    common_dim = min(int(node_dim), int(z_dim))
    projection = _projection_matrix(common_dim, state['semantic_dim'], seed)
    state['projection'] = projection
    state['common_dim'] = common_dim
    state['node_embedding_sum'] = np.zeros(
        (state['node_embedding_count'].shape[0], projection.shape[1]),
        dtype=np.float32,
    )


def _collect_split_records(
    model,
    loader,
    hparams,
    dataset_name,
    fold_idx,
    split_name,
    dataset_raw,
    topology,
    state,
    projection_seed,
):
    model.eval()
    device = torch.device(hparams.device)
    export_loader = _build_export_loader(loader)
    records = []

    with torch.inference_mode():
        for raw_batch in export_loader:
            batch = _move_batch_to_device(raw_batch, device)
            model_output, embeddings = model.forward_with_embeddings(batch)
            logits = _extract_logits(model_output)
            probabilities = _positive_probabilities(logits)
            if probabilities is None:
                raise RuntimeError('MISGL model did not return logits for BIP export.')

            labels = batch[g_key.y].view(-1).detach().cpu().numpy().astype(np.int64)
            probs = probabilities.detach().cpu().numpy().reshape(-1)
            preds = (probs > 0.5).astype(np.int64)
            orig_indices = batch[g_key.orig_graph_idx].view(-1).detach().cpu().numpy().astype(np.int64)
            subgraph_ids = batch[g_key.subgraph_id].view(-1).detach().cpu().numpy().astype(np.int64)
            num_nodes_list = batch[g_key.node_num].view(-1).detach().cpu().numpy().astype(np.int64)
            z_values = embeddings['graph_emb_classifier'].detach().cpu().numpy().astype(np.float32)
            node_embeddings = embeddings['h'].detach().cpu().numpy().astype(np.float32)
            attention_list = _extract_attention(model_output, model, num_nodes_list)

            _ensure_projection_and_storage(
                state,
                node_embeddings.shape[-1],
                z_values.shape[-1],
                projection_seed,
            )
            common_dim = state['common_dim']
            projection = state['projection']

            for idx, num_nodes in enumerate(num_nodes_list):
                orig_idx = int(orig_indices[idx])
                profile = topology['profiles'][orig_idx]
                local_positions = _local_original_positions(
                    dataset_raw,
                    orig_idx,
                    int(num_nodes),
                    topology['node_to_position'],
                )
                valid_mask = local_positions >= 0
                projected_nodes = (
                    node_embeddings[idx, :num_nodes, :common_dim] @ projection
                ).astype(np.float32)
                z_projection = (
                    z_values[idx, :common_dim] @ projection
                ).astype(np.float32)

                valid_positions = local_positions[valid_mask]
                if valid_positions.size:
                    np.add.at(
                        state['node_embedding_sum'],
                        valid_positions,
                        projected_nodes[valid_mask],
                    )
                    np.add.at(state['node_embedding_count'], valid_positions, 1)
                    np.add.at(state['node_probability_sum'], valid_positions, float(probs[idx]))
                    np.add.at(state['node_probability_count'], valid_positions, 1)

                boundary_position_set = set(profile['boundary_positions'].tolist())
                boundary_mask = np.asarray(
                    [position in boundary_position_set for position in local_positions],
                    dtype=bool,
                )
                boundary_mask &= valid_mask
                inner_mask = valid_mask & (~boundary_mask)
                boundary_embedding = _mean_rows(projected_nodes, boundary_mask)
                inner_embedding = _mean_rows(projected_nodes, inner_mask)

                row = {
                    'dataset': dataset_name,
                    'fold': int(fold_idx),
                    'fold_idx': int(fold_idx),
                    'split': split_name,
                    'orig_graph_idx': orig_idx,
                    'subgraph_id': int(subgraph_ids[idx]),
                    'label': int(labels[idx]),
                    'misgl_original_prob': float(probs[idx]),
                    'misgl_original_pred': int(preds[idx]),
                    'misgl_original_correct': int(preds[idx] == labels[idx]),
                    'z_mil_norm': float(np.linalg.norm(z_values[idx])),
                    '_z_mil': np.asarray(z_values[idx], dtype=np.float32),
                    '_z_projection': z_projection,
                    '_boundary_embedding': boundary_embedding,
                    '_inner_embedding': inner_embedding,
                    '_current_context': feature_vector(
                        profile['current_context_features'],
                        CONTEXT_FEATURE_NAMES,
                    ),
                }
                row.update(profile['current_context_features'])
                row.update(profile['static_features'])
                row.update(_attention_features(attention_list[idx], boundary_mask))
                records.append(row)

    return records


def _sample_positions(positions, max_count):
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size <= int(max_count):
        return positions
    indices = np.linspace(0, positions.size - 1, num=int(max_count), dtype=np.int64)
    return positions[indices]


def _rowwise_cosine(matrix, vector):
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    vector_norm = float(np.linalg.norm(vector))
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * vector_norm
    values = np.zeros((matrix.shape[0],), dtype=np.float32)
    valid = denom > 0
    if np.any(valid):
        values[valid] = (matrix[valid] @ vector) / denom[valid]
    return values


def _enrich_bip_features(records, topology, state, max_context_samples):
    counts = state['node_embedding_count']
    embedding_map = state['node_embedding_sum']
    covered = counts > 0
    embedding_map[covered] /= counts[covered, None]

    probability_map = np.zeros_like(state['node_probability_sum'], dtype=np.float64)
    probability_covered = state['node_probability_count'] > 0
    probability_map[probability_covered] = (
        state['node_probability_sum'][probability_covered]
        / state['node_probability_count'][probability_covered]
    )

    for row in records:
        profile = topology['profiles'][int(row['orig_graph_idx'])]
        context_positions_all = profile['context_positions'].astype(np.int64, copy=False)
        context_positions = context_positions_all[covered[context_positions_all]]
        context_positions = _sample_positions(context_positions, max_context_samples)
        context_embeddings = embedding_map[context_positions]
        context_mean = (
            context_embeddings.mean(axis=0)
            if context_embeddings.size else np.zeros((embedding_map.shape[1],), dtype=np.float32)
        )

        z_projection = row['_z_projection']
        boundary_embedding = row['_boundary_embedding']
        inner_embedding = row['_inner_embedding']
        context_z_cosines = _rowwise_cosine(context_embeddings, z_projection)

        cross_edges = profile['sampled_cross_edge_positions'].astype(np.int64, copy=False)
        if cross_edges.size:
            cross_valid = covered[cross_edges[:, 0]] & covered[cross_edges[:, 1]]
            cross_edges = cross_edges[cross_valid]
        if cross_edges.size:
            left_embeddings = embedding_map[cross_edges[:, 0]]
            right_embeddings = embedding_map[cross_edges[:, 1]]
            numerators = np.sum(left_embeddings * right_embeddings, axis=1)
            denominators = (
                np.linalg.norm(left_embeddings, axis=1)
                * np.linalg.norm(right_embeddings, axis=1)
            )
            edge_cosines = np.zeros((cross_edges.shape[0],), dtype=np.float32)
            valid = denominators > 0
            edge_cosines[valid] = numerators[valid] / denominators[valid]
        else:
            edge_cosines = np.zeros((0,), dtype=np.float32)

        pseudo_positions = context_positions_all[probability_covered[context_positions_all]]
        pseudo_positions = _sample_positions(pseudo_positions, max_context_samples)
        pseudo_probs = probability_map[pseudo_positions]

        semantic_features = {
            'bip_boundary_embedding_mean': float(np.mean(boundary_embedding)),
            'bip_boundary_embedding_norm': float(np.linalg.norm(boundary_embedding)),
            'bip_inner_embedding_mean': float(np.mean(inner_embedding)),
            'bip_inner_embedding_norm': float(np.linalg.norm(inner_embedding)),
            'bip_boundary_inner_embedding_distance': safe_distance(
                boundary_embedding,
                inner_embedding,
            ),
            'bip_boundary_inner_embedding_cosine': safe_cosine(
                boundary_embedding,
                inner_embedding,
            ),
            'bip_context_embedding_mean': float(np.mean(context_mean)),
            'bip_context_embedding_norm': float(np.linalg.norm(context_mean)),
            'bip_context_z_cosine': safe_cosine(context_mean, z_projection),
            'bip_context_z_cosine_mean': (
                float(context_z_cosines.mean()) if context_z_cosines.size else 0.0
            ),
            'bip_context_z_cosine_std': (
                float(context_z_cosines.std()) if context_z_cosines.size else 0.0
            ),
            'bip_context_z_cosine_max': (
                float(context_z_cosines.max()) if context_z_cosines.size else 0.0
            ),
            'bip_context_boundary_embedding_cosine': safe_cosine(
                context_mean,
                boundary_embedding,
            ),
            'bip_cross_edge_embedding_cosine_mean': (
                float(edge_cosines.mean()) if edge_cosines.size else 0.0
            ),
            'bip_cross_edge_embedding_cosine_std': (
                float(edge_cosines.std()) if edge_cosines.size else 0.0
            ),
            'bip_context_pseudo_prob_mean': (
                float(pseudo_probs.mean()) if pseudo_probs.size else 0.0
            ),
            'bip_context_pseudo_prob_std': (
                float(pseudo_probs.std()) if pseudo_probs.size else 0.0
            ),
            'bip_context_pseudo_prob_min': (
                float(pseudo_probs.min()) if pseudo_probs.size else 0.0
            ),
            'bip_context_pseudo_prob_max': (
                float(pseudo_probs.max()) if pseudo_probs.size else 0.0
            ),
            'bip_context_pseudo_positive_ratio': (
                float(np.mean(pseudo_probs > 0.5)) if pseudo_probs.size else 0.0
            ),
            'bip_context_embedding_coverage': (
                float(np.mean(covered[context_positions_all]))
                if context_positions_all.size else 0.0
            ),
            'bip_context_pseudo_prob_coverage': (
                float(np.mean(probability_covered[context_positions_all]))
                if context_positions_all.size else 0.0
            ),
        }
        row.update(semantic_features)
        for feature_name in BIP_FEATURE_NAMES:
            row.setdefault(feature_name, 0.0)
        row['_boundary_profile'] = feature_vector(row, BOUNDARY_PROFILE_FEATURE_NAMES)
        row['_semantic_profile'] = feature_vector(row, SEMANTIC_PROFILE_FEATURE_NAMES)
        row['_interaction_profile'] = feature_vector(row, INTERACTION_PROFILE_FEATURE_NAMES)
        row['_bip'] = feature_vector(row, BIP_FEATURE_NAMES)


def _matrix(records, key):
    return np.stack([np.asarray(row[key], dtype=np.float32) for row in records], axis=0)


def _concat(records, keys):
    return np.concatenate([_matrix(records, key) for key in keys], axis=1)


def _shuffled_concat(records, base_key, extra_key, seed):
    base = _matrix(records, base_key)
    extra = _matrix(records, extra_key)
    if extra.shape[0] > 1:
        rng = np.random.default_rng(int(seed))
        extra = extra[rng.permutation(extra.shape[0])]
    return np.concatenate([base, extra], axis=1)


def _feature_matrix(records, config_name, seed):
    config_map = {
        'z_mil_only': ('_z_mil',),
        'current_context_only': ('_current_context',),
        'z_mil_current_context': ('_z_mil', '_current_context'),
        'boundary_profile_only': ('_boundary_profile',),
        'semantic_profile_only': ('_semantic_profile',),
        'interaction_pattern_only': ('_interaction_profile',),
        'bip_only': ('_bip',),
        'z_mil_boundary_profile': ('_z_mil', '_boundary_profile'),
        'z_mil_semantic_profile': ('_z_mil', '_semantic_profile'),
        'z_mil_interaction_pattern': ('_z_mil', '_interaction_profile'),
        'z_mil_bip': ('_z_mil', '_bip'),
        'z_mil_full_bip': ('_z_mil', '_bip'),
    }
    if config_name == 'z_mil_shuffled_current_context':
        return _shuffled_concat(records, '_z_mil', '_current_context', seed)
    if config_name == 'z_mil_shuffled_bip':
        return _shuffled_concat(records, '_z_mil', '_bip', seed)
    if config_name not in config_map:
        raise ValueError('Unsupported BIP probe config: {}'.format(config_name))
    return _concat(records, config_map[config_name])


def _labels(records):
    return np.asarray([int(row['label']) for row in records], dtype=np.int64)


def _apply_predictions(records, config_name, predictions, probabilities):
    for row, prediction, probability in zip(records, predictions, probabilities):
        prediction = int(prediction)
        row['{}_pred'.format(config_name)] = prediction
        row['{}_prob'.format(config_name)] = float(probability)
        row['{}_correct'.format(config_name)] = int(prediction == int(row['label']))


def _run_probes(records_by_split, class_weight, base_seed):
    metric_rows = []
    for split_name, records in records_by_split.items():
        labels = _labels(records)
        predictions = np.asarray([row['misgl_original_pred'] for row in records])
        probabilities = np.asarray([row['misgl_original_prob'] for row in records])
        metric_rows.append({
            'split': split_name,
            'config': 'misgl_original',
            **_classification_metrics(labels, predictions, probabilities),
        })

    train_records = records_by_split['train']
    train_labels = _labels(train_records)
    for config_idx, config_name in enumerate(PROBE_CONFIGS):
        train_seed = int(base_seed) + 1009 * config_idx
        train_matrix = _feature_matrix(train_records, config_name, train_seed)
        probe = _fit_probe(train_matrix, train_labels, class_weight)
        for split_idx, (split_name, records) in enumerate(records_by_split.items()):
            split_seed = train_seed + 97 * split_idx
            split_matrix = _feature_matrix(records, config_name, split_seed)
            predictions, probabilities = _predict_probe(probe, split_matrix)
            _apply_predictions(records, config_name, predictions, probabilities)
            metric_rows.append({
                'split': split_name,
                'config': config_name,
                **_classification_metrics(_labels(records), predictions, probabilities),
            })
    return metric_rows


def _add_metric_deltas(metrics_df):
    metrics_df = metrics_df.copy()
    baseline = metrics_df[metrics_df['config'].eq('z_mil_only')][
        ['dataset', 'fold_idx', 'split', 'acc', 'F1', 'AUC']
    ].rename(columns={
        'acc': 'baseline_acc',
        'F1': 'baseline_F1',
        'AUC': 'baseline_AUC',
    })
    metrics_df = metrics_df.merge(
        baseline,
        on=['dataset', 'fold_idx', 'split'],
        how='left',
    )
    for metric in ('acc', 'F1', 'AUC'):
        delta_values = (
            metrics_df[metric] - metrics_df['baseline_{}'.format(metric)]
        )
        metrics_df['delta_{}_vs_z_mil'.format(metric)] = delta_values
        metrics_df['delta_{}'.format(metric.lower())] = delta_values
    return metrics_df.drop(columns=['baseline_acc', 'baseline_F1', 'baseline_AUC'])


def _metric_summary_with_deltas(metrics_df):
    summary = _summary_metrics(metrics_df)
    delta_rows = []
    for keys, group in metrics_df.groupby(['dataset', 'split', 'config'], dropna=False):
        row = {'dataset': keys[0], 'split': keys[1], 'config': keys[2]}
        for metric in ('acc', 'F1', 'AUC'):
            column = 'delta_{}_vs_z_mil'.format(metric)
            values = pd.to_numeric(group[column], errors='coerce')
            row['{}_mean'.format(column)] = float(values.mean())
            row['{}_std'.format(column)] = (
                float(values.std(ddof=1)) if values.count() > 1 else 0.0
            )
            row['delta_{}_mean'.format(metric.lower())] = float(values.mean())
            row['delta_{}_std'.format(metric.lower())] = (
                float(values.std(ddof=1)) if values.count() > 1 else 0.0
            )
        delta_rows.append(row)
    return summary.merge(
        pd.DataFrame(delta_rows),
        on=['dataset', 'split', 'config'],
        how='left',
    )


def _fix_break_rows(records, dataset, fold_idx):
    rows = []
    baseline_correct = np.asarray(
        [row['z_mil_only_correct'] for row in records],
        dtype=np.int64,
    )
    baseline_wrong_count = int(np.sum(baseline_correct == 0))
    baseline_right_count = int(np.sum(baseline_correct == 1))

    for config_name in FIX_BREAK_CONFIGS:
        config_correct = np.asarray(
            [row['{}_correct'.format(config_name)] for row in records],
            dtype=np.int64,
        )
        fixes = (baseline_correct == 0) & (config_correct == 1)
        breaks = (baseline_correct == 1) & (config_correct == 0)
        rows.append({
            'dataset': dataset,
            'fold_idx': int(fold_idx),
            'split': 'test',
            'config': config_name,
            'fix_count': int(fixes.sum()),
            'break_count': int(breaks.sum()),
            'net_fix': int(fixes.sum() - breaks.sum()),
            'fix_rate': float(fixes.sum() / baseline_wrong_count) if baseline_wrong_count else 0.0,
            'break_rate': float(breaks.sum() / baseline_right_count) if baseline_right_count else 0.0,
            'baseline_wrong_count': baseline_wrong_count,
            'baseline_right_count': baseline_right_count,
        })
    return rows


def _mark_fix_break(records):
    for row in records:
        baseline_correct = int(row['z_mil_only_correct'])
        bip_correct = int(row['z_mil_bip_correct'])
        row['fix_or_not'] = int(baseline_correct == 0 and bip_correct == 1)
        row['break_or_not'] = int(baseline_correct == 1 and bip_correct == 0)
        row['z_mil_prob'] = row['z_mil_only_prob']
        row['z_mil_pred'] = row['z_mil_only_pred']
        row['z_mil_correct'] = row['z_mil_only_correct']
        row['bip_prob'] = row['z_mil_bip_prob']
        row['bip_pred'] = row['z_mil_bip_pred']
        row['bip_correct'] = row['z_mil_bip_correct']


def _safe_corr(left, right, method):
    left = pd.to_numeric(pd.Series(left), errors='coerce')
    right = pd.to_numeric(pd.Series(right), errors='coerce')
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 2:
        return np.nan
    if left[valid].nunique() < 2 or right[valid].nunique() < 2:
        return np.nan
    return float(left[valid].corr(right[valid], method=method))


def _correlations(per_subgraph_df):
    rows = []
    group_specs = [
        (['dataset', 'fold_idx', 'split'], 'fold'),
        (['dataset', 'split'], 'all_folds'),
    ]
    for group_columns, scope in group_specs:
        for keys, group in per_subgraph_df.groupby(group_columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(group_columns, keys))
            targets = {
                'label': group['label'],
                'z_mil_prob': group['z_mil_prob'],
                'z_mil_correct': group['z_mil_correct'],
                'z_mil_prediction_error': 1 - group['z_mil_correct'],
                'fix_or_not': group['fix_or_not'],
                'break_or_not': group['break_or_not'],
            }
            for feature_name in BIP_FEATURE_NAMES:
                for target_name, target_values in targets.items():
                    rows.append({
                        'dataset': key_map['dataset'],
                        'fold_idx': key_map.get('fold_idx', 'all'),
                        'split': key_map['split'],
                        'scope': scope,
                        'feature_group': (
                            'boundary'
                            if feature_name in BOUNDARY_PROFILE_FEATURE_NAMES
                            else 'semantic'
                            if feature_name in SEMANTIC_PROFILE_FEATURE_NAMES
                            else 'interaction'
                        ),
                        'feature': feature_name,
                        'target': target_name,
                        'pearson': _safe_corr(group[feature_name], target_values, 'pearson'),
                        'spearman': _safe_corr(group[feature_name], target_values, 'spearman'),
                        'n': int(len(group)),
                    })
    return pd.DataFrame(rows)


def _public_dataframe(records):
    return pd.DataFrame([
        {key: value for key, value in row.items() if not key.startswith('_')}
        for row in records
    ])


def _write_outputs(output_dir, records, metric_rows, fix_break_rows, fold_details):
    os.makedirs(output_dir, exist_ok=True)
    per_subgraph_df = _public_dataframe(records)
    metrics_by_fold_df = _add_metric_deltas(pd.DataFrame(metric_rows))
    metrics_summary_df = _metric_summary_with_deltas(metrics_by_fold_df)
    ablation_by_fold_df = metrics_by_fold_df[
        metrics_by_fold_df['config'].isin(FEATURE_GROUP_ABLATION_CONFIGS)
    ].copy()
    ablation_by_fold_df.insert(0, 'scope', 'fold')
    ablation_summary_df = metrics_summary_df[
        metrics_summary_df['config'].isin(FEATURE_GROUP_ABLATION_CONFIGS)
    ].copy()
    ablation_summary_df.insert(0, 'scope', 'summary')
    ablation_df = pd.concat(
        [ablation_by_fold_df, ablation_summary_df],
        ignore_index=True,
        sort=False,
    )

    fix_break_fold_df = pd.DataFrame(fix_break_rows)
    fix_break_fold_df.insert(0, 'scope', 'fold')
    fix_break_summary_rows = []
    for (dataset, config_name), group in fix_break_fold_df.groupby(
        ['dataset', 'config'],
        dropna=False,
    ):
        row = {
            'scope': 'summary',
            'dataset': dataset,
            'fold_idx': 'all',
            'split': 'test',
            'config': config_name,
            'fold_count': int(group['fold_idx'].nunique()),
        }
        for column in (
            'fix_count',
            'break_count',
            'net_fix',
            'fix_rate',
            'break_rate',
            'baseline_wrong_count',
            'baseline_right_count',
        ):
            values = pd.to_numeric(group[column], errors='coerce')
            row['{}_mean'.format(column)] = float(values.mean())
            row['{}_std'.format(column)] = (
                float(values.std(ddof=1)) if values.count() > 1 else 0.0
            )
            if column in ('fix_count', 'break_count', 'net_fix'):
                row['{}_sum'.format(column)] = float(values.sum())
        fix_break_summary_rows.append(row)
    fix_break_df = pd.concat(
        [fix_break_fold_df, pd.DataFrame(fix_break_summary_rows)],
        ignore_index=True,
        sort=False,
    )
    correlations_df = _correlations(per_subgraph_df)

    outputs = {
        'metrics_summary': metrics_summary_df,
        'metrics_by_fold': metrics_by_fold_df,
        'feature_group_ablation': ablation_df,
        'fix_break_analysis': fix_break_df,
        'correlations_BIP': correlations_df,
        'per_subgraph_BIP': per_subgraph_df,
    }
    for name, dataframe in outputs.items():
        dataframe.to_csv(
            os.path.join(output_dir, '{}.csv'.format(name)),
            index=False,
            encoding='utf-8-sig',
        )
        _write_excel_if_reasonable(
            os.path.join(output_dir, '{}.xlsx'.format(name)),
            {name[:31]: dataframe},
        )

    with open(os.path.join(output_dir, 'fold_details.json'), 'w', encoding='utf-8') as file_obj:
        json.dump(fold_details, file_obj, indent=2, ensure_ascii=False)
    return {name: os.path.join(output_dir, '{}.csv'.format(name)) for name in outputs}


def _run_dataset(hparams, data_name, args):
    logging.warning('Boundary Interaction Profile analysis: %s', data_name)
    data_loader = GraphDataLoaderWrapper(hparams, data_name=data_name)
    split_path = data_loader.get_cv_split_path(ensure_dir=False)
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            'CV split manifest not found: {}. Run prepare_cv_split.py first.'.format(split_path)
        )
    split_manifest = data_loader.load_cv_split_manifest(split_path)
    selected_folds = _parse_fold_selection(args.folds, int(split_manifest['cv_num_folds']))
    topology = build_boundary_interaction_topology(
        data_loader._dataset_raw,
        log_progress=True,
        progress_interval=args.progress_interval,
        max_cross_edge_samples=args.max_cross_edge_samples,
    )

    all_records = []
    all_metrics = []
    all_fix_break = []
    fold_details = {
        'dataset': data_name,
        'split_path': split_path,
        'selected_folds': selected_folds,
        'semantic_projection_dim': int(args.semantic_projection_dim),
        'max_context_samples': int(args.max_context_samples),
        'max_cross_edge_samples': int(args.max_cross_edge_samples),
        'feature_groups': {
            'boundary': list(BOUNDARY_PROFILE_FEATURE_NAMES),
            'semantic': list(SEMANTIC_PROFILE_FEATURE_NAMES),
            'interaction': list(INTERACTION_PROFILE_FEATURE_NAMES),
        },
        'folds': [],
    }

    for fold_idx in selected_folds:
        seed = int(hparams.cv_seed) + int(fold_idx)
        reproducibility.set_seed(seed, cuda_deterministic=(hparams.device == 'cuda'))
        logging.warning('[BIP] fold %d: train best-val MISGL model', fold_idx)
        train_loader, val_loader, test_loader, split_meta = (
            data_loader.get_cv_loaders_from_manifest(split_manifest, fold_idx)
        )
        model = MISGLEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
        model, _, best_val = train_eval_iter(
            model,
            train_loader,
            val_loader,
            None,
            hparams,
            dataset_raw=data_loader._dataset_raw,
        )

        state = _initialize_fold_state(topology, args.semantic_projection_dim)
        records_by_split = {}
        for split_name, loader in (
            ('train', train_loader),
            ('val', val_loader),
            ('test', test_loader),
        ):
            logging.warning('[BIP] fold %d: export %s embeddings', fold_idx, split_name)
            records_by_split[split_name] = _collect_split_records(
                model,
                loader,
                hparams,
                data_name,
                fold_idx,
                split_name,
                data_loader._dataset_raw,
                topology,
                state,
                projection_seed=seed + 7919,
            )

        fold_records = [
            row
            for split_name in FINAL_SPLITS
            for row in records_by_split[split_name]
        ]
        _enrich_bip_features(
            fold_records,
            topology,
            state,
            max_context_samples=args.max_context_samples,
        )
        fold_metric_rows = _run_probes(
            records_by_split,
            args.probe_class_weight,
            seed,
        )
        for row in fold_metric_rows:
            row.update({'dataset': data_name, 'fold_idx': int(fold_idx)})
        _mark_fix_break(fold_records)

        all_metrics.extend(fold_metric_rows)
        all_fix_break.extend(
            _fix_break_rows(records_by_split['test'], data_name, fold_idx)
        )
        all_records.extend(fold_records)
        fold_details['folds'].append({
            'fold_idx': int(fold_idx),
            'seed': int(seed),
            'best_val': best_val,
            'split': split_meta,
            'record_count': len(fold_records),
        })

        del model, train_loader, val_loader, test_loader, records_by_split, state
        _clear_cuda_cache(hparams)

    output_dir = os.path.join(
        hparams.model_save_path,
        'boundary_interaction',
        str(args.output_name or _resolve_experiment_timestamp(hparams)),
    )
    paths = _write_outputs(
        output_dir,
        all_records,
        all_metrics,
        all_fix_break,
        fold_details,
    )
    logging.warning('BIP outputs written to %s', output_dir)
    return paths


def main(args):
    logging.getLogger().setLevel(logging.INFO)
    base_hparams = hparam.HParams()
    base_hparams.from_yaml(args.hparam_path)
    hparams_lib.apply_defaults(base_hparams)
    dataset_names = _parse_data_name_set(args.data_name_set)
    if not dataset_names:
        dataset_names = _parse_data_name_set(getattr(base_hparams, 'data_name_set', None))
    if not dataset_names:
        raise ValueError('No dataset specified.')

    outputs = {}
    for data_name in dataset_names:
        hparams = _copy_dataset_hparams(base_hparams, data_name, args)
        if getattr(hparams, 'cuda_visible_devices', None) is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.cuda_visible_devices)
        outputs[data_name] = _run_dataset(hparams, data_name, args)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MISGL Boundary Interaction Profile validation.')
    parser.add_argument('--hparam_path', type=str, default='./config/b_on.yml')
    parser.add_argument('--data_name_set', nargs='*', default=None)
    parser.add_argument('--processed_data_dir', type=str, default=None)
    parser.add_argument('--folds', type=str, default=None)
    parser.add_argument('--output_name', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--probe_class_weight', choices=('balanced', 'none'), default='balanced')
    parser.add_argument('--no_preload_data_to_gpu', action='store_true')
    parser.add_argument('--semantic_projection_dim', type=int, default=32)
    parser.add_argument('--max_context_samples', type=int, default=2048)
    parser.add_argument('--max_cross_edge_samples', type=int, default=2048)
    parser.add_argument('--progress_interval', type=int, default=500)
    main(parser.parse_args())
