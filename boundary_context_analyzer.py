# coding=utf-8

"""Post-hoc Boundary-Context analysis for MISGL.

This entry point keeps the MISGL model unchanged. It trains each CV fold with
the existing best-validation protocol, freezes z_mil embeddings, and then fits
lightweight logistic probes on z_mil/context feature combinations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from MISGL.bin.train_eval import train_eval_iter
from MISGL.models.encoder import MISGLEncoder
from MISGL.utils import hparam
from MISGL.utils import hparams_lib
from MISGL.utils import reproducibility
from MISGL.utils.boundary_context import (
    CONTEXT_FEATURE_NAMES,
    DETAIL_FEATURE_NAMES,
    compute_boundary_context_rows,
    context_feature_vector,
    rows_by_orig_graph_idx,
)
from MISGL.utils.global_variables import g_key
from MISGL.utils.load_data import GraphDataLoaderWrapper


PROBE_CONFIGS = (
    'z_mil_only',
    'context_only',
    'z_mil_context',
    'z_mil_shuffled_context',
)

ALL_CONFIGS = ('misgl_original',) + PROBE_CONFIGS
FINAL_SPLITS = ('train', 'val', 'test')
EXCEL_MAX_ROWS = 1048576


class ConstantProbabilityProbe(object):
    """Fallback classifier for single-class training folds."""

    def __init__(self, positive_probability):
        self.positive_probability = float(positive_probability)

    def predict_proba(self, x):
        pos = np.full((x.shape[0],), self.positive_probability, dtype=np.float64)
        return np.stack([1.0 - pos, pos], axis=1)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if lowered in ('0', 'false', 'no', 'n', 'off'):
        return False
    return default


def _parse_data_name_set(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    names = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        if ',' in text:
            names.extend([part.strip() for part in text.split(',') if part.strip()])
        else:
            names.append(text)
    deduped = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _parse_fold_selection(raw, fold_count):
    if raw is None or str(raw).strip() == '':
        return list(range(fold_count))

    selected = []
    for token in str(raw).split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            start_text, end_text = token.split('-', 1)
            start = int(start_text)
            end = int(end_text)
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(token))

    out = []
    seen = set()
    for fold_idx in selected:
        if fold_idx < 0 or fold_idx >= fold_count:
            raise ValueError(f'Fold index out of range: {fold_idx}')
        if fold_idx not in seen:
            seen.add(fold_idx)
            out.append(fold_idx)
    return out


def _resolve_experiment_timestamp(hparams):
    timestamp = getattr(hparams, 'timestamp', None)
    timestamp = str(timestamp).strip() if timestamp is not None else ''
    if timestamp:
        return timestamp
    return time.strftime('%Y%m%d_%H%M%S')


def _set_if_present_or_add(hparams, name, value):
    if name in hparams:
        hparams.set_hparam(name, value)
    else:
        hparams.add_hparam(name, value)


def _copy_dataset_hparams(base_hparams, data_name, args):
    hparams = hparam.HParams()
    for name, value in base_hparams.values().items():
        hparams.add_hparam(name, value)
    hparams_lib.apply_defaults(hparams)
    hparams.data_name = data_name

    if args.processed_data_dir:
        hparams.processed_data_dir = args.processed_data_dir
    if args.device:
        hparams.device = args.device
    if args.batch_size is not None:
        hparams.batch_size = int(args.batch_size)
    if args.epoch is not None:
        hparams.epoch = int(args.epoch)
    if args.no_preload_data_to_gpu:
        _set_if_present_or_add(hparams, 'preload_data_to_gpu', False)

    hparams.final_eval_splits = list(FINAL_SPLITS)
    hparams.tb_unique_run_dir = False
    hparams.timestamp = f'{data_name}_{_resolve_experiment_timestamp(base_hparams)}'

    base_save_path = getattr(hparams, 'model_save_path', None)
    if base_save_path:
        hparams.model_save_path = os.path.join(base_save_path, data_name)
    else:
        hparams.model_save_path = os.path.join('results', data_name)
    os.makedirs(hparams.model_save_path, exist_ok=True)
    return hparams


def _needs_device_move(value, device):
    if not isinstance(value, torch.Tensor):
        return False
    if value.device.type != device.type:
        return True
    return device.index is not None and value.device.index != device.index


def _move_batch_to_device(data, device):
    return {
        key: value.to(device, non_blocking=True) if _needs_device_move(value, device) else value
        for key, value in data.items()
    }


def _build_export_loader(loader):
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        worker_init_fn=loader.worker_init_fn,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=False,
        timeout=loader.timeout,
    )


def _extract_logits(model_output):
    if isinstance(model_output, dict) and 'ypred_A' in model_output:
        return model_output['ypred_A']
    if isinstance(model_output, dict) and 'ypred' in model_output:
        return model_output['ypred']
    if isinstance(model_output, torch.Tensor):
        return model_output
    return None


def _positive_probabilities(logits):
    if logits is None:
        return None
    if logits.dim() == 2 and logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits.view(-1))


def _safe_auc(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(labels, probabilities))
    except ValueError:
        return np.nan


def _classification_metrics(labels, predictions, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0:
        return {'acc': np.nan, 'F1': np.nan, 'AUC': np.nan}
    return {
        'acc': float(accuracy_score(labels, predictions)),
        'F1': float(f1_score(labels, predictions, average='binary', zero_division=0)),
        'AUC': _safe_auc(labels, probabilities),
    }


def _collect_split_records(model, loader, hparams, dataset_name, fold_idx, split_name, context_by_orig):
    model.eval()
    device = torch.device(getattr(hparams, 'device', 'cpu'))
    export_loader = _build_export_loader(loader)
    records = []

    with torch.inference_mode():
        for raw_batch in export_loader:
            batch = _move_batch_to_device(raw_batch, device)
            out, embeddings = model.forward_with_embeddings(batch)
            logits = _extract_logits(out)
            probabilities = _positive_probabilities(logits)
            if probabilities is None:
                raise RuntimeError('MISGL model did not return logits for prediction export.')

            labels = batch[g_key.y].view(-1).detach().cpu().numpy().astype(np.int64)
            probs_np = probabilities.detach().cpu().numpy().reshape(-1)
            preds_np = (probs_np > 0.5).astype(np.int64)
            z_mil_np = embeddings['graph_emb_classifier'].detach().cpu().numpy()

            orig_indices = batch[g_key.orig_graph_idx].view(-1).detach().cpu().numpy().astype(np.int64)
            if g_key.subgraph_id in batch:
                subgraph_ids = batch[g_key.subgraph_id].view(-1).detach().cpu().numpy().astype(np.int64)
            else:
                subgraph_ids = np.full((labels.shape[0],), -1, dtype=np.int64)

            for idx in range(labels.shape[0]):
                orig_idx = int(orig_indices[idx])
                context_row = dict(context_by_orig.get(orig_idx, {}))
                context_vec = context_feature_vector(context_row)
                z_vec = np.asarray(z_mil_np[idx], dtype=np.float32).reshape(-1)
                label = int(labels[idx])
                pred = int(preds_np[idx])
                probability = float(probs_np[idx])

                row = {
                    'dataset': dataset_name,
                    'fold_idx': int(fold_idx),
                    'split': split_name,
                    'orig_graph_idx': orig_idx,
                    'subgraph_id': int(subgraph_ids[idx]),
                    'label': label,
                    'prediction': pred,
                    'probability': probability,
                    'correct_or_not': int(pred == label),
                    'z_mil_norm': float(np.linalg.norm(z_vec)),
                    'context_vector_norm': float(np.linalg.norm(context_vec)),
                    '_z_mil': z_vec,
                    '_context': context_vec,
                }
                for name in DETAIL_FEATURE_NAMES + CONTEXT_FEATURE_NAMES:
                    row[name] = context_row.get(name, 0.0)
                records.append(row)

    return records


def _records_to_matrix(records, feature_kind):
    if feature_kind == 'z':
        return np.stack([row['_z_mil'] for row in records], axis=0)
    if feature_kind == 'context':
        return np.stack([row['_context'] for row in records], axis=0)
    if feature_kind == 'z_context':
        z = _records_to_matrix(records, 'z')
        context = _records_to_matrix(records, 'context')
        return np.concatenate([z, context], axis=1)
    raise ValueError(f'Unsupported feature kind: {feature_kind}')


def _labels_from_records(records):
    return np.asarray([int(row['label']) for row in records], dtype=np.int64)


def _shuffle_context_matrix(records, seed):
    z = _records_to_matrix(records, 'z')
    context = _records_to_matrix(records, 'context')
    if context.shape[0] > 1:
        rng = np.random.default_rng(seed)
        context = context[rng.permutation(context.shape[0])]
    return np.concatenate([z, context], axis=1)


def _fit_probe(x_train, y_train, class_weight):
    y_train = np.asarray(y_train, dtype=np.int64)
    if len(np.unique(y_train)) < 2:
        return ConstantProbabilityProbe(float(np.mean(y_train)) if y_train.size else 0.0)

    logistic_kwargs = {
        'max_iter': 1000,
        'solver': 'liblinear',
        'random_state': 0,
    }
    if class_weight and class_weight != 'none':
        logistic_kwargs['class_weight'] = class_weight

    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**logistic_kwargs)),
    ]).fit(x_train, y_train)


def _predict_probe(probe, x):
    probabilities = probe.predict_proba(x)[:, 1]
    predictions = (probabilities > 0.5).astype(np.int64)
    return predictions, probabilities


def _feature_matrix_for_config(records, config_name, seed):
    if config_name == 'z_mil_only':
        return _records_to_matrix(records, 'z')
    if config_name == 'context_only':
        return _records_to_matrix(records, 'context')
    if config_name == 'z_mil_context':
        return _records_to_matrix(records, 'z_context')
    if config_name == 'z_mil_shuffled_context':
        return _shuffle_context_matrix(records, seed)
    raise ValueError(f'Unsupported config: {config_name}')


def _apply_probe_outputs(records, config_name, predictions, probabilities):
    pred_key = f'{config_name}_prediction'
    prob_key = f'{config_name}_probability'
    correct_key = f'{config_name}_correct_or_not'
    for row, pred, prob in zip(records, predictions, probabilities):
        label = int(row['label'])
        pred = int(pred)
        row[pred_key] = pred
        row[prob_key] = float(prob)
        row[correct_key] = int(pred == label)


def _run_probe_configs(records_by_split, class_weight, base_seed):
    metrics_rows = []
    train_records = records_by_split['train']
    y_train = _labels_from_records(train_records)

    for split_name, records in records_by_split.items():
        labels = _labels_from_records(records)
        original_probs = np.asarray([float(row['probability']) for row in records], dtype=np.float64)
        original_preds = np.asarray([int(row['prediction']) for row in records], dtype=np.int64)
        metrics = _classification_metrics(labels, original_preds, original_probs)
        metrics_rows.append({
            'split': split_name,
            'config': 'misgl_original',
            **metrics,
        })

    for config_idx, config_name in enumerate(PROBE_CONFIGS):
        train_seed = int(base_seed) + config_idx * 1009
        x_train = _feature_matrix_for_config(train_records, config_name, train_seed)
        probe = _fit_probe(x_train, y_train, class_weight)

        for split_idx, (split_name, records) in enumerate(records_by_split.items()):
            seed = int(base_seed) + config_idx * 1009 + split_idx * 97
            x_split = _feature_matrix_for_config(records, config_name, seed)
            labels = _labels_from_records(records)
            preds, probs = _predict_probe(probe, x_split)
            _apply_probe_outputs(records, config_name, preds, probs)
            metrics = _classification_metrics(labels, preds, probs)
            metrics_rows.append({
                'split': split_name,
                'config': config_name,
                **metrics,
            })

    return metrics_rows


def _summary_metrics(metrics_df):
    group_cols = ['dataset', 'split', 'config']
    rows = []
    for keys, group in metrics_df.groupby(group_cols, dropna=False):
        dataset, split_name, config_name = keys
        row = {
            'dataset': dataset,
            'split': split_name,
            'config': config_name,
            'fold_count': int(group['fold_idx'].nunique()),
        }
        for metric in ('acc', 'F1', 'AUC'):
            values = pd.to_numeric(group[metric], errors='coerce')
            row[f'{metric}_mean'] = float(values.mean(skipna=True))
            row[f'{metric}_std'] = float(values.std(skipna=True, ddof=1)) if values.count() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _corr_pair(x, y, method):
    x = pd.to_numeric(pd.Series(x), errors='coerce')
    y = pd.to_numeric(pd.Series(y), errors='coerce')
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 2:
        return np.nan
    if x[valid].nunique(dropna=True) < 2 or y[valid].nunique(dropna=True) < 2:
        return np.nan
    return float(x[valid].corr(y[valid], method=method))


def _correlation_rows_for_group(group, dataset, fold_idx, split_name, config_name):
    if config_name == 'misgl_original':
        prediction_col = 'prediction'
        correct_col = 'correct_or_not'
    else:
        prediction_col = f'{config_name}_prediction'
        correct_col = f'{config_name}_correct_or_not'

    if prediction_col not in group.columns or correct_col not in group.columns:
        return []

    target_values = {
        'label': group['label'],
        'prediction': group[prediction_col],
        'prediction_error': 1 - pd.to_numeric(group[correct_col], errors='coerce'),
    }

    rows = []
    for feature_name in CONTEXT_FEATURE_NAMES:
        for target_name, target_series in target_values.items():
            rows.append({
                'dataset': dataset,
                'fold_idx': fold_idx,
                'split': split_name,
                'config': config_name,
                'feature': feature_name,
                'target': target_name,
                'pearson': _corr_pair(group[feature_name], target_series, 'pearson'),
                'spearman': _corr_pair(group[feature_name], target_series, 'spearman'),
                'n': int(len(group)),
            })
    return rows


def _build_correlations(per_subgraph_df):
    rows = []
    for (dataset, fold_idx, split_name), group in per_subgraph_df.groupby(['dataset', 'fold_idx', 'split'], dropna=False):
        for config_name in ALL_CONFIGS:
            rows.extend(_correlation_rows_for_group(group, dataset, int(fold_idx), split_name, config_name))
    return pd.DataFrame(rows)


def _build_correlations_all_folds(per_subgraph_df):
    rows = []
    for (dataset, split_name), group in per_subgraph_df.groupby(['dataset', 'split'], dropna=False):
        for config_name in ALL_CONFIGS:
            rows.extend(_correlation_rows_for_group(group, dataset, 'all', split_name, config_name))
    return pd.DataFrame(rows)


def _public_per_subgraph_df(records):
    public_rows = []
    for row in records:
        public_row = {
            key: value
            for key, value in row.items()
            if not key.startswith('_')
        }
        public_rows.append(public_row)
    return pd.DataFrame(public_rows)


def _write_excel_if_reasonable(path, sheets):
    max_rows = max((len(df) for df in sheets.values()), default=0)
    if max_rows > EXCEL_MAX_ROWS:
        logging.warning('Skipping Excel export for %s because it has %d rows.', path, max_rows)
        return False
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return True


def _write_outputs(output_dir, per_subgraph_records, metrics_rows, fold_details):
    os.makedirs(output_dir, exist_ok=True)

    per_subgraph_df = _public_per_subgraph_df(per_subgraph_records)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_summary_df = _summary_metrics(metrics_df) if not metrics_df.empty else pd.DataFrame()
    correlations_df = _build_correlations(per_subgraph_df) if not per_subgraph_df.empty else pd.DataFrame()
    correlations_all_df = _build_correlations_all_folds(per_subgraph_df) if not per_subgraph_df.empty else pd.DataFrame()

    per_subgraph_csv = os.path.join(output_dir, 'per_subgraph.csv')
    metrics_csv = os.path.join(output_dir, 'metrics_summary.csv')
    correlations_csv = os.path.join(output_dir, 'correlations.csv')
    fold_details_json = os.path.join(output_dir, 'fold_details.json')

    per_subgraph_df.to_csv(per_subgraph_csv, index=False, encoding='utf-8-sig')
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_by_fold.csv'), index=False, encoding='utf-8-sig')
    metrics_summary_df.to_csv(metrics_csv, index=False, encoding='utf-8-sig')
    correlations_df.to_csv(correlations_csv, index=False, encoding='utf-8-sig')
    correlations_all_df.to_csv(os.path.join(output_dir, 'correlations_all_folds.csv'), index=False, encoding='utf-8-sig')

    _write_excel_if_reasonable(os.path.join(output_dir, 'per_subgraph.xlsx'), {'per_subgraph': per_subgraph_df})
    _write_excel_if_reasonable(
        os.path.join(output_dir, 'metrics_summary.xlsx'),
        {
            'summary': metrics_summary_df,
            'by_fold': metrics_df,
        },
    )
    _write_excel_if_reasonable(
        os.path.join(output_dir, 'correlations.xlsx'),
        {
            'by_fold': correlations_df,
            'all_folds': correlations_all_df,
        },
    )

    with open(fold_details_json, 'w', encoding='utf-8') as f:
        json.dump(fold_details, f, indent=2, ensure_ascii=False)

    return {
        'per_subgraph_csv': per_subgraph_csv,
        'metrics_csv': metrics_csv,
        'correlations_csv': correlations_csv,
        'fold_details_json': fold_details_json,
    }


def _clear_cuda_cache(hparams):
    if getattr(hparams, 'device', None) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_dataset(hparams, data_name, args):
    logging.warning('Boundary-Context analysis for dataset: %s', data_name)
    data_loader = GraphDataLoaderWrapper(hparams, data_name=data_name)
    raw_dataset = data_loader._dataset_raw
    raw_subgraphs = raw_dataset.get('subgraph_structures', []) if raw_dataset is not None else []
    raw_original_graph = raw_dataset.get('original_graph', None) if raw_dataset is not None else None
    original_node_count = raw_original_graph.number_of_nodes() if raw_original_graph is not None else 0
    original_edge_count = raw_original_graph.number_of_edges() if raw_original_graph is not None else 0
    logging.warning(
        'Dataset loaded: subgraphs=%d, original_nodes=%d, original_edges=%d',
        len(raw_subgraphs),
        original_node_count,
        original_edge_count,
    )
    split_path = data_loader.get_cv_split_path(ensure_dir=False)
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            'CV split manifest not found: {}. Please run prepare_cv_split.py first.'.format(split_path)
        )
    split_manifest = data_loader.load_cv_split_manifest(split_path)
    selected_folds = _parse_fold_selection(args.folds, int(split_manifest['cv_num_folds']))
    logging.warning('CV split loaded: folds=%d, selected=%s', int(split_manifest['cv_num_folds']), selected_folds)

    context_rows = compute_boundary_context_rows(
        data_loader._dataset_raw,
        log_progress=True,
        progress_interval=int(args.context_progress_interval),
    )
    context_by_orig = rows_by_orig_graph_idx(context_rows)
    logging.warning('Boundary-Context table ready: rows=%d', len(context_rows))

    all_records = []
    all_metrics = []
    fold_details = {
        'dataset': data_name,
        'split_path': split_path,
        'selected_folds': selected_folds,
        'context_feature_names': list(CONTEXT_FEATURE_NAMES),
        'probe_configs': list(PROBE_CONFIGS),
        'folds': [],
    }

    for fold_idx in selected_folds:
        seed = int(getattr(hparams, 'cv_seed', 1024)) + int(fold_idx)
        reproducibility.set_seed(seed, cuda_deterministic=(hparams.device == 'cuda'))
        logging.warning('Fold %d: training MISGL best-val model (seed=%d)', fold_idx, seed)

        training_loader, validation_loader, test_loader, split_meta = data_loader.get_cv_loaders_from_manifest(
            split_manifest,
            fold_idx,
        )
        model = MISGLEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
        model, _, best_val_result = train_eval_iter(
            model,
            training_loader,
            validation_loader,
            None,
            hparams,
            dataset_raw=data_loader._dataset_raw,
        )

        loaders_by_split = {
            'train': training_loader,
            'val': validation_loader,
            'test': test_loader,
        }
        records_by_split = {}
        for split_name, loader in loaders_by_split.items():
            records_by_split[split_name] = _collect_split_records(
                model,
                loader,
                hparams,
                data_name,
                fold_idx,
                split_name,
                context_by_orig,
            )

        fold_metrics = _run_probe_configs(
            records_by_split,
            class_weight=args.probe_class_weight,
            base_seed=seed,
        )
        for row in fold_metrics:
            row.update({
                'dataset': data_name,
                'fold_idx': int(fold_idx),
            })
            all_metrics.append(row)

        fold_records = []
        for split_name in FINAL_SPLITS:
            fold_records.extend(records_by_split[split_name])
        all_records.extend(fold_records)

        fold_details['folds'].append({
            'fold_idx': int(fold_idx),
            'seed': int(seed),
            'best_val': best_val_result,
            'split': split_meta,
            'record_count': len(fold_records),
        })

        del model, training_loader, validation_loader, test_loader, loaders_by_split, records_by_split
        _clear_cuda_cache(hparams)

    output_dir = os.path.join(
        hparams.model_save_path,
        'boundary_context',
        str(args.output_name or _resolve_experiment_timestamp(hparams)),
    )
    paths = _write_outputs(output_dir, all_records, all_metrics, fold_details)
    logging.warning('Boundary-Context outputs written to %s', output_dir)
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
        raise ValueError('No dataset specified. Please set data_name_set in yaml or pass --data_name_set.')

    all_paths = {}
    for data_name in dataset_names:
        hparams = _copy_dataset_hparams(base_hparams, data_name, args)
        if getattr(hparams, 'cuda_visible_devices', None) is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.cuda_visible_devices)
        if hparams.device == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        all_paths[data_name] = _run_dataset(hparams, data_name, args)

    logging.warning('Completed Boundary-Context analysis: %s', all_paths)
    return all_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run post-hoc Boundary-Context analysis for MISGL.')
    parser.add_argument('--hparam_path', type=str, default='./config/b_on.yml')
    parser.add_argument('--data_name_set', nargs='*', default=None)
    parser.add_argument('--processed_data_dir', type=str, default=None)
    parser.add_argument('--folds', type=str, default=None, help='Fold selection, e.g. "0", "0,1,2", or "0-2".')
    parser.add_argument('--output_name', type=str, default=None)
    parser.add_argument('--device', type=str, default=None, help='Override hparams.device, e.g. cpu or cuda.')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument(
        '--probe_class_weight',
        type=str,
        default='balanced',
        choices=('balanced', 'none'),
        help='Class weighting for Logistic Regression probes.',
    )
    parser.add_argument(
        '--no_preload_data_to_gpu',
        action='store_true',
        help='Keep preprocessed graph tensors on CPU until each batch is moved to the target device.',
    )
    parser.add_argument(
        '--context_progress_interval',
        type=int,
        default=1000,
        help='Log Boundary-Context extraction progress every N subgraphs.',
    )
    main(parser.parse_args())
