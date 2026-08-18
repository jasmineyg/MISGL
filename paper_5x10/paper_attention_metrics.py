# coding=utf-8

"""Per-fold MIL attention metrics required by paper section 6.1."""

import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from MISGL.utils.global_variables import g_key


def _move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _map_subgraph_nodes_to_labels(subgraph, node_binary_labels, num_nodes):
    nodes = list(subgraph.nodes())
    labels = np.zeros(len(nodes), dtype=np.int64)
    total = len(node_binary_labels)
    for idx, node_id in enumerate(nodes):
        orig_id = None
        if isinstance(node_id, (int, np.integer)):
            orig_id = int(node_id)
        else:
            attrs = subgraph.nodes[node_id]
            for key in ('original_index', 'orig_id', 'node_index', 'original_id'):
                if attrs.get(key) is not None:
                    try:
                        orig_id = int(attrs[key])
                        break
                    except (TypeError, ValueError):
                        pass
        if orig_id is not None and 0 <= orig_id < total:
            labels[idx] = int(node_binary_labels[orig_id])
    return labels[:num_nodes]


def _resolve_node_binary_labels(dataset_raw, orig_idx, num_nodes):
    if dataset_raw is None:
        return np.zeros(num_nodes, dtype=np.int64)
    labels = dataset_raw.get('node_binary_labels')
    subgraphs = dataset_raw.get('subgraph_structures')
    if labels is None:
        return np.zeros(num_nodes, dtype=np.int64)
    if subgraphs is not None and 0 <= orig_idx < len(subgraphs):
        return _map_subgraph_nodes_to_labels(subgraphs[orig_idx], labels, num_nodes)
    if 0 <= orig_idx < len(labels):
        return np.asarray(labels[orig_idx], dtype=np.int64)[:num_nodes]
    return np.zeros(num_nodes, dtype=np.int64)


def _safe_mean(values):
    return float(np.mean(values)) if values else None


def export_fold_attention_metrics(
    model, test_loader, hparams, dataset_raw, output_path, data_name, cv_seed, fold_idx,
):
    """Export positive attention mass, enrichment, and ranking AUC for test bags."""
    device = torch.device(hparams.device)
    model.eval()
    rows = []
    with torch.inference_mode():
        for raw_batch in test_loader:
            batch = _move_batch_to_device(raw_batch, device)
            output = model(batch)
            if not isinstance(output, dict) or output.get('branch_b') is None:
                raise RuntimeError('MIL branch attention is unavailable in model output.')
            branch = output['branch_b']
            attention_flat = branch['a'].detach().cpu().numpy().reshape(-1)
            logits = output['ypred_A'].detach().cpu().reshape(-1)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(np.int64)
            labels = batch[g_key.y].detach().cpu().numpy().reshape(-1).astype(np.int64)
            node_counts = batch[g_key.node_num].detach().cpu().numpy().reshape(-1).astype(np.int64)
            orig_indices = batch[g_key.orig_graph_idx].detach().cpu().numpy().reshape(-1).astype(np.int64)
            subgraph_tensor = batch.get(g_key.subgraph_id)
            if isinstance(subgraph_tensor, torch.Tensor):
                subgraph_ids = subgraph_tensor.detach().cpu().numpy().reshape(-1).astype(np.int64)
            else:
                subgraph_ids = np.full(labels.shape, -1, dtype=np.int64)

            cursor = 0
            for idx, num_nodes in enumerate(node_counts.tolist()):
                weights = attention_flat[cursor:cursor + num_nodes].astype(np.float64, copy=False)
                cursor += num_nodes
                if int(labels[idx]) != 1 or num_nodes <= 0:
                    continue
                node_labels = _resolve_node_binary_labels(dataset_raw, int(orig_indices[idx]), num_nodes)
                positive_count = int(np.sum(node_labels == 1))
                positive_mass = float(np.sum(weights[node_labels == 1])) if positive_count else 0.0
                prevalence = float(positive_count) / float(num_nodes)
                enrichment = positive_mass / prevalence if prevalence > 0 else None
                ranking_auc = None
                if 0 < positive_count < num_nodes:
                    ranking_auc = float(roc_auc_score(node_labels, weights))
                rows.append({
                    'orig_graph_idx': int(orig_indices[idx]),
                    'subgraph_id': int(subgraph_ids[idx]),
                    'num_nodes': int(num_nodes),
                    'positive_instance_count': positive_count,
                    'positive_instance_prevalence': prevalence,
                    'positive_attention_mass': positive_mass,
                    'positive_attention_enrichment': enrichment,
                    'attention_ranking_auc': ranking_auc,
                    'y_prob': float(probs[idx]),
                    'correct': bool(int(preds[idx]) == int(labels[idx])),
                })

    enrichment_values = [row['positive_attention_enrichment'] for row in rows if row['positive_attention_enrichment'] is not None]
    auc_values = [row['attention_ranking_auc'] for row in rows if row['attention_ranking_auc'] is not None]
    mass_values = [row['positive_attention_mass'] for row in rows]
    correct_enrichment = [row['positive_attention_enrichment'] for row in rows if row['correct'] and row['positive_attention_enrichment'] is not None]
    wrong_enrichment = [row['positive_attention_enrichment'] for row in rows if not row['correct'] and row['positive_attention_enrichment'] is not None]
    payload = {
        'data_name': data_name,
        'cv_seed': int(cv_seed),
        'fold_idx': int(fold_idx),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'positive_bag_count': len(rows),
            'positive_attention_mass_mean': _safe_mean(mass_values),
            'positive_attention_enrichment_mean': _safe_mean(enrichment_values),
            'attention_ranking_auc_mean': _safe_mean(auc_values),
            'correct_positive_bag_count': sum(int(row['correct']) for row in rows),
            'incorrect_positive_bag_count': sum(int(not row['correct']) for row in rows),
            'correct_positive_bag_enrichment_mean': _safe_mean(correct_enrichment),
            'incorrect_positive_bag_enrichment_mean': _safe_mean(wrong_enrichment),
        },
        'positive_bags': rows,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output_path

