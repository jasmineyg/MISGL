# coding=utf-8

import os
import logging
import json
import csv
import gc

import numpy as np

import torch
from MISGL.utils import get_loss
from MISGL.utils import reproducibility
from MISGL.utils.global_variables import *
from MISGL.utils.evaluate import evaluate
from MISGL.utils import load_data
from MISGL.models import encoder


_LIGHTWEIGHT_ATTENTION_EXPORTER = None


def _attention_train_output_path(path):
  base, ext = os.path.splitext(path)
  return f'{base}_train10p{ext}'


_METRIC_KEYS = (
  'acc', 'prec', 'rec', 'F1', 'balanced_acc', 'roc_auc', 'pr_auc',
  'tn', 'fp', 'fn', 'tp',
)
_FINAL_EVAL_SPLITS = ('train', 'val', 'test')


def _basic_metrics(result):
  return {
    key: None if result.get(key) is None else float(result[key])
    for key in _METRIC_KEYS
  }


def _format_metrics(result):
  return ', '.join(
    '{}: {}'.format(
      key,
      'n/a' if result.get(key) is None else '{:.4f}'.format(float(result[key])),
    )
    for key in _METRIC_KEYS
  )


def _metric_summary(values):
  finite_values = [
    float(value) for value in values
    if value is not None and np.isfinite(float(value))
  ]
  if not finite_values:
    return {'mean': None, 'std': None, 'count': 0}
  return {
    'mean': float(np.mean(finite_values)),
    'std': float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0,
    'count': len(finite_values),
  }


def _training_label_stats(training_loader):
  dataset = getattr(training_loader, 'dataset', None)
  examples = getattr(dataset, 'processed_graph_list', None)
  if examples is None:
    raise ValueError('Training dataset does not expose processed_graph_list.')

  negative_count = 0
  positive_count = 0
  for example in examples:
    label = int(example[g_key.y].detach().cpu().item())
    if label == 0:
      negative_count += 1
    elif label == 1:
      positive_count += 1
    else:
      raise ValueError('Binary loss received unsupported training label: {}'.format(label))

  if positive_count == 0:
    raise ValueError('Training fold has no positive examples.')
  if negative_count == 0:
    raise ValueError('Training fold has no negative examples.')
  return {
    'negative_count': int(negative_count),
    'positive_count': int(positive_count),
    'pos_weight': float(negative_count) / float(positive_count),
  }


def _configure_fold_loss(hparams, training_loader):
  loss_type = str(getattr(hparams, 'loss_type', 'bce')).strip().lower()
  label_stats = _training_label_stats(training_loader)
  pos_weight = label_stats['pos_weight'] if loss_type == 'weighted_bce' else None
  hparams.loss_pos_weight = pos_weight
  return {
    'loss_type': loss_type,
    'focal_gamma': float(getattr(hparams, 'focal_gamma', 2.0)),
    'label_smoothing': float(getattr(hparams, 'label_smoothing', 0.0)),
    'negative_count': label_stats['negative_count'],
    'positive_count': label_stats['positive_count'],
    'pos_weight': pos_weight,
  }


def _final_eval_splits(hparams):
  raw = getattr(hparams, 'final_eval_splits', list(_FINAL_EVAL_SPLITS))
  if isinstance(raw, str):
    splits = [s.strip() for s in raw.split(',') if s.strip()]
  elif isinstance(raw, (list, tuple, set)):
    splits = [str(s).strip() for s in raw if str(s).strip()]
  else:
    splits = list(_FINAL_EVAL_SPLITS)

  valid = {'train', 'val', 'test'}
  selected = []
  for split_name in splits:
    if split_name not in valid:
      raise ValueError(f'Unsupported final_eval_splits item: {split_name!r}')
    if split_name not in selected:
      selected.append(split_name)
  if 'test' not in selected:
    selected.append('test')
  return tuple(selected)


def _should_export_attention(hparams):
  return bool(getattr(hparams, 'export_attention', False) or getattr(hparams, 'analyze_attention', False))


def _attention_sample_frac(hparams, split_name):
  if split_name == 'train':
    return float(getattr(hparams, 'attention_train_sample_frac', 0.1))
  return float(getattr(hparams, 'attention_sample_frac', 0.1))


def _attention_top_k(hparams):
  return int(getattr(hparams, 'attention_top_k', 20))


def _should_export_train_attention(hparams):
  return bool(getattr(hparams, 'attention_export_train', True))


def _should_export_predictions(hparams):
  return bool(getattr(hparams, 'export_predictions', False))


def _extract_prediction_logits(model_output):
  if isinstance(model_output, dict) and 'ypred_A' in model_output:
    return model_output['ypred_A']
  if isinstance(model_output, dict) and 'ypred' in model_output:
    return model_output['ypred']
  if isinstance(model_output, torch.Tensor):
    return model_output
  return None


def _to_cpu_list(value, batch_size, default=None):
  if value is None:
    return [default] * batch_size
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().view(-1).tolist()
  return list(value)


def _embedding_scalar_list(emb, key, batch_size, reducer=None):
  if emb is None or key not in emb:
    return [None] * batch_size
  value = emb[key]
  if not isinstance(value, torch.Tensor):
    return [None] * batch_size
  value = value.detach().cpu()
  if reducer == 'mean' and value.dim() > 1:
    value = value.mean(dim=-1)
  elif reducer == 'norm' and value.dim() > 1:
    value = value.norm(dim=-1)
  return value.view(-1).tolist()


def _collect_prediction_rows(loader, model, hparams, split_name, fold_idx=None, run_idx=None):
  model.eval()
  device = torch.device(hparams.device)
  rows = []
  with torch.inference_mode():
    for graph_data in loader:
      batch = _move_batch_to_device(graph_data, device)
      if hasattr(model, 'forward_with_embeddings'):
        model_output, emb = model.forward_with_embeddings(batch)
      else:
        model_output = model(batch)
        emb = None
      logits = _extract_prediction_logits(model_output)
      if logits is None:
        raise ValueError('Cannot export predictions because model output has no logits.')
      probs = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
      labels = batch[g_key.y].view(-1).detach().cpu().numpy()
      preds = (probs > 0.5).astype(np.int64)
      batch_size = int(labels.shape[0])
      orig_indices = _to_cpu_list(batch.get(g_key.orig_graph_idx, None), batch_size, default=-1)
      subgraph_ids = _to_cpu_list(batch.get(g_key.subgraph_id, None), batch_size, default=-1)
      external_counts = _to_cpu_list(batch.get(g_key.border_external_count, None), batch_size, default=None)
      border_entropy = _embedding_scalar_list(emb, 'border_anchor_entropy', batch_size)
      border_residual_ratio = _embedding_scalar_list(emb, 'border_residual_ratio', batch_size)
      border_gate_mean = _embedding_scalar_list(emb, 'border_gate', batch_size, reducer='mean')
      z_border_norm = _embedding_scalar_list(emb, 'z_border', batch_size, reducer='norm')

      for i in range(batch_size):
        rows.append({
          'fold_idx': fold_idx,
          'run_idx': run_idx,
          'split': split_name,
          'orig_idx': int(orig_indices[i]) if orig_indices[i] is not None else -1,
          'subgraph_id': int(subgraph_ids[i]) if subgraph_ids[i] is not None else -1,
          'label': int(labels[i]),
          'prob': float(probs[i]),
          'pred': int(preds[i]),
          'correct': int(preds[i] == labels[i]),
          'border_external_count': None if external_counts[i] is None else float(external_counts[i]),
          'border_anchor_entropy': None if border_entropy[i] is None else float(border_entropy[i]),
          'border_residual_ratio': None if border_residual_ratio[i] is None else float(border_residual_ratio[i]),
          'border_gate_mean': None if border_gate_mean[i] is None else float(border_gate_mean[i]),
          'z_border_norm': None if z_border_norm[i] is None else float(z_border_norm[i]),
        })
  return rows


def _write_prediction_csv(rows, path):
  if not rows:
    return
  os.makedirs(os.path.dirname(path), exist_ok=True)
  fieldnames = [
    'fold_idx', 'run_idx', 'split', 'orig_idx', 'subgraph_id', 'label', 'prob', 'pred', 'correct',
    'border_external_count', 'border_anchor_entropy', 'border_residual_ratio', 'border_gate_mean',
    'z_border_norm',
  ]
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _export_predictions_for_splits(loaders_by_split, model, hparams, final_splits, fold_idx=None, run_idx=None):
  if not _should_export_predictions(hparams):
    return []
  paths = []
  for split_name in final_splits:
    rows = _collect_prediction_rows(
      loaders_by_split[split_name], model, hparams, split_name, fold_idx=fold_idx, run_idx=run_idx
    )
    index_name = 'fold_{}'.format(fold_idx) if fold_idx is not None else 'run_{}'.format(run_idx)
    out_path = os.path.join(
      hparams.model_save_path,
      '{}_{}_{}_predictions.csv'.format(hparams.timestamp, index_name, split_name),
    )
    _write_prediction_csv(rows, out_path)
    logging.warning('Saved prediction export to {}'.format(out_path))
    paths.append(out_path)
  return paths


def _get_lightweight_attention_exporter():
  global _LIGHTWEIGHT_ATTENTION_EXPORTER
  if _LIGHTWEIGHT_ATTENTION_EXPORTER is not None:
    return _LIGHTWEIGHT_ATTENTION_EXPORTER

  try:
    from attention_analyzer import export_lightweight_attention_from_model
  except ImportError as exc:
    raise ImportError(
      'Cannot import export_lightweight_attention_from_model from attention_analyzer.'
    ) from exc
  except AttributeError as exc:
    raise ImportError(
      'attention_analyzer does not define export_lightweight_attention_from_model.'
    ) from exc

  _LIGHTWEIGHT_ATTENTION_EXPORTER = export_lightweight_attention_from_model
  return _LIGHTWEIGHT_ATTENTION_EXPORTER


def _move_batch_to_device(data, device):
  def _needs_device_move(value):
    if not isinstance(value, torch.Tensor):
      return False
    if value.device.type != device.type:
      return True
    return device.index is not None and value.device.index != device.index

  return {
    key: value.to(device, non_blocking=True) if _needs_device_move(value) else value
    for key, value in data.items()
  }


def _clear_cuda_cache(hparams):
  gc.collect()
  if getattr(hparams, 'device', None) == 'cuda' and torch.cuda.is_available():
    torch.cuda.empty_cache()


def train_eval(hparams, data_name=None):
  """
  Repeated holdout training and evaluation entry point.

  - Use hparams.holdout_seeds when provided, otherwise derive seeds from
    holdout_runs and cv_seed.
  - Split train/val/test for each seed, train with validation early stopping,
    and evaluate the requested final splits.

  Returns:
    dict with per-run results and aggregate summary metrics.
  """
  data_loader = load_data.GraphDataLoaderWrapper(hparams, data_name=data_name)

  # 璇诲彇閲嶅鐣欏嚭閰嶇疆锛氫紭鍏堜娇鐢?holdout_seeds锛屽惁鍒欎娇鐢?holdout_runs/cv_seed 鐢熸垚
  holdout_seeds = getattr(hparams, 'holdout_seeds', None)
  if isinstance(holdout_seeds, list) and len(holdout_seeds) > 0:
    seeds = [int(s) for s in holdout_seeds]
  else:
    holdout_runs = int(getattr(hparams, 'holdout_runs', getattr(hparams, 'fold_num', 10)))
    base_seed = int(getattr(hparams, 'cv_seed', 1024))
    seeds = [base_seed + i for i in range(holdout_runs)]

  final_splits = _final_eval_splits(hparams)
  test_metrics = {key: [] for key in _METRIC_KEYS}
  split_metrics = {
    split_name: {key: [] for key in _METRIC_KEYS}
    for split_name in final_splits
  }
  all_results = []

  for run_idx, seed in enumerate(seeds):
    logging.warning('* holdout run: {} (seed={})'.format(run_idx, seed))

    reproducibility.set_seed(seed, cuda_deterministic=(hparams.device == 'cuda'))

    # Return train/val/test loaders only.
    training_loader, validation_loader, test_loader = data_loader.get_holdout_loaders(
      seed=seed, train_frac=0.6, val_frac=0.2, test_frac=0.2
    )
    loss_config = _configure_fold_loss(hparams, training_loader)

    summary_writer = None

    model = encoder.MISGLEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
    # Train with validation early stopping.
    model, _, best_val_result = train_eval_iter(
      model, training_loader, validation_loader, summary_writer, hparams, dataset_raw=data_loader._dataset_raw
    )

    loaders_by_split = {
      'train': training_loader,
      'val': validation_loader,
      'test': test_loader,
    }
    metrics_by_split = {}
    for split_name in final_splits:
      split_result = evaluate(loaders_by_split[split_name], model, hparams, dataset_name=split_name)
      metrics_by_split[split_name] = _basic_metrics(split_result)
    result = metrics_by_split['test']
    all_results.append({
      'run_idx': int(run_idx),
      'seed': int(seed),
      'best_val': best_val_result,
      'loss_config': loss_config,
      'metrics': dict(result),
      'split_metrics': metrics_by_split,
    })
    for key in test_metrics.keys():
      test_metrics[key].append(result[key])
    for split_name, split_result in metrics_by_split.items():
      for key in _METRIC_KEYS:
        split_metrics[split_name][key].append(split_result[key])
    logging.warning('Holdout {} selected model metrics => {}'.format(
      run_idx,
      '; '.join(
        '{} [{}]'.format(split_name, _format_metrics(metrics_by_split[split_name]))
        for split_name in final_splits
      ),
    ))

    _export_predictions_for_splits(loaders_by_split, model, hparams, final_splits, run_idx=run_idx)

    if _should_export_attention(hparams):
      out_path = os.path.join(hparams.model_save_path, f'{hparams.timestamp}_holdout_{run_idx}_analyze_attention.xlsx')
      logging.warning(f'[DEBUG] Exporting attention to: {out_path}')
      attention_exporter = _get_lightweight_attention_exporter()
      attention_exporter(
        model,
        test_loader,
        hparams,
        data_loader._dataset_raw,
        out_path,
        sample_frac=_attention_sample_frac(hparams, 'test'),
        split_name='test',
        sample_seed=seed,
        top_k=_attention_top_k(hparams),
      )
      if _should_export_train_attention(hparams):
        attention_exporter(
          model,
          training_loader,
          hparams,
          data_loader._dataset_raw,
          _attention_train_output_path(out_path),
          sample_frac=_attention_sample_frac(hparams, 'train'),
          split_name='train',
          sample_seed=seed,
          top_k=_attention_top_k(hparams),
        )
    else:
      logging.warning('[DEBUG] export_attention is disabled, skipping attention export.')
    if summary_writer is not None:
      summary_writer.close()
    del model, training_loader, validation_loader, test_loader, loaders_by_split, metrics_by_split, result
    _clear_cuda_cache(hparams)

  summary = {key: _metric_summary(vals) for key, vals in test_metrics.items()}
  split_summary = {
    split_name: {
      key: _metric_summary(vals)
      for key, vals in split_result.items()
    }
    for split_name, split_result in split_metrics.items()
  }
  for split_name in final_splits:
    msg_parts = [
      '{}: {}'.format(
        k,
        'n/a' if split_summary[split_name][k]['mean'] is None else
        '{:.4f} +/- {:.4f}'.format(
          split_summary[split_name][k]['mean'],
          split_summary[split_name][k]['std'],
        ),
      )
      for k in _METRIC_KEYS
    ]
    logging.warning('* Repeated Holdout (k={}) {} results => {}'.format(
      len(seeds), split_name, '; '.join(msg_parts)
    ))

  return {
    'seeds': seeds,
    'results': all_results,
    'summary': summary,
    'split_summary': split_summary,
    'final_eval_splits': list(final_splits),
  }


def fixed_cv_train_eval(hparams, data_name=None):
  """Training and evaluation entry point for fixed 10-fold CV."""
  data_loader = load_data.GraphDataLoaderWrapper(hparams, data_name=data_name)
  split_path = data_loader.get_cv_split_path(ensure_dir=False)
  if not os.path.exists(split_path):
    raise FileNotFoundError(
      'CV split manifest not found: {}. Please run prepare_cv_split.py first.'.format(split_path)
    )

  split_manifest = data_loader.load_cv_split_manifest(split_path)
  fold_count = int(split_manifest['cv_num_folds'])
  final_splits = _final_eval_splits(hparams)
  test_metrics = {key: [] for key in _METRIC_KEYS}
  split_metrics = {
    split_name: {key: [] for key in _METRIC_KEYS}
    for split_name in final_splits
  }
  all_results = []

  fold_indices = list(range(fold_count))
  cv_fold_limit = getattr(hparams, 'cv_fold_limit', None)
  if cv_fold_limit is not None:
    fold_indices = fold_indices[:max(1, int(cv_fold_limit))]

  for fold_idx in fold_indices:
    seed = int(getattr(hparams, 'cv_seed', 1024)) + fold_idx
    test_fold = int(fold_idx)
    val_fold = (test_fold + 1) % fold_count
    logging.warning('* cv fold: {} (train=8 folds, val_fold={}, test_fold={}, seed={})'.format(
      fold_idx, val_fold, test_fold, seed
    ))

    reproducibility.set_seed(seed, cuda_deterministic=(hparams.device == 'cuda'))
    training_loader, validation_loader, test_loader, split_meta = data_loader.get_cv_loaders_from_manifest(
      split_manifest, fold_idx
    )
    loss_config = _configure_fold_loss(hparams, training_loader)
    logging.warning(
      'CV fold {} sizes => train: {}, val: {}, test: {}'.format(
        fold_idx, split_meta['train_size'], split_meta['val_size'], split_meta['test_size']
      )
    )

    summary_writer = None

    model = encoder.MISGLEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
    model, _, best_val_result = train_eval_iter(
      model, training_loader, validation_loader, summary_writer, hparams, dataset_raw=data_loader._dataset_raw
    )

    loaders_by_split = {
      'train': training_loader,
      'val': validation_loader,
      'test': test_loader,
    }
    metrics_by_split = {}
    for split_name in final_splits:
      split_result = evaluate(loaders_by_split[split_name], model, hparams, dataset_name=split_name)
      metrics_by_split[split_name] = _basic_metrics(split_result)
    test_result = metrics_by_split['test']
    all_results.append({
      'fold_idx': int(fold_idx),
      'seed': int(seed),
      'split': split_meta,
      'best_val': best_val_result,
      'loss_config': loss_config,
      'metrics': dict(metrics_by_split['test']),
      'split_metrics': metrics_by_split,
    })
    for key in test_metrics.keys():
      test_metrics[key].append(test_result[key])
    for split_name, split_result in metrics_by_split.items():
      for key in _METRIC_KEYS:
        split_metrics[split_name][key].append(split_result[key])
    logging.warning('CV fold {} selected model metrics => {}'.format(
      fold_idx,
      '; '.join(
        '{} [{}]'.format(split_name, _format_metrics(metrics_by_split[split_name]))
        for split_name in final_splits
      ),
    ))

    _export_predictions_for_splits(loaders_by_split, model, hparams, final_splits, fold_idx=fold_idx)

    if _should_export_attention(hparams):
      out_path = os.path.join(hparams.model_save_path, f'{hparams.timestamp}_cv_fold_{fold_idx}_analyze_attention.xlsx')
      logging.warning(f'[DEBUG] Exporting attention to: {out_path}')
      attention_exporter = _get_lightweight_attention_exporter()
      attention_exporter(
        model,
        test_loader,
        hparams,
        data_loader._dataset_raw,
        out_path,
        sample_frac=_attention_sample_frac(hparams, 'test'),
        split_name='test',
        sample_seed=seed,
        top_k=_attention_top_k(hparams),
      )
      if _should_export_train_attention(hparams):
        attention_exporter(
          model,
          training_loader,
          hparams,
          data_loader._dataset_raw,
          _attention_train_output_path(out_path),
          sample_frac=_attention_sample_frac(hparams, 'train'),
          split_name='train',
          sample_seed=seed,
          top_k=_attention_top_k(hparams),
        )
    else:
      logging.warning('[DEBUG] export_attention is disabled, skipping attention export.')
    if summary_writer is not None:
      summary_writer.close()
    del model, training_loader, validation_loader, test_loader, loaders_by_split, metrics_by_split
    _clear_cuda_cache(hparams)

  summary = {key: _metric_summary(vals) for key, vals in test_metrics.items()}
  split_summary = {
    split_name: {
      key: _metric_summary(vals)
      for key, vals in split_result.items()
    }
    for split_name, split_result in split_metrics.items()
  }
  for split_name in final_splits:
    msg_parts = [
      '{}: {}'.format(
        k,
        'n/a' if split_summary[split_name][k]['mean'] is None else
        '{:.4f} +/- {:.4f}'.format(
          split_summary[split_name][k]['mean'],
          split_summary[split_name][k]['std'],
        ),
      )
      for k in _METRIC_KEYS
    ]
    logging.warning('* Fixed 10-fold CV {} results => {}'.format(split_name, '; '.join(msg_parts)))

  result_path = os.path.join(hparams.model_save_path, f'{hparams.timestamp}_cv_results.json')
  with open(result_path, 'w', encoding='utf-8') as f:
    json.dump({
      'data_name': data_name,
      'split_path': split_path,
      'cv_seed': int(split_manifest['cv_seed']),
      'cv_num_folds': int(split_manifest['cv_num_folds']),
      'cv_val_policy': split_manifest['cv_val_policy'],
      'loss_type': str(getattr(hparams, 'loss_type', 'bce')).strip().lower(),
      'focal_gamma': float(getattr(hparams, 'focal_gamma', 2.0)),
      'label_smoothing': float(getattr(hparams, 'label_smoothing', 0.0)),
      'summary': summary,
      'split_summary': split_summary,
      'final_eval_splits': list(final_splits),
      'fold_results': all_results,
    }, f, indent=2, ensure_ascii=False)
  logging.warning('Saved CV result summary to {}'.format(result_path))

  return {
    'results': all_results,
    'summary': summary,
    'split_summary': split_summary,
    'final_eval_splits': list(final_splits),
    'split_path': split_path,
    'result_path': result_path,
  }


train_eval = fixed_cv_train_eval


def _is_better_val_result(val_result, best_val_result, hparams):
    acc_delta = float(getattr(hparams, 'early_stop_min_delta', 0.0))
    loss_delta = float(getattr(hparams, 'early_stop_loss_delta', 1e-6))
    current_acc = float(val_result['acc'])
    best_acc = float(best_val_result['acc'])
    if current_acc > best_acc + acc_delta:
        return True
    if abs(current_acc - best_acc) <= acc_delta:
        current_loss = float(val_result.get('loss', float('inf')))
        best_loss = float(best_val_result.get('loss', float('inf')))
        return current_loss < best_loss - loss_delta
    return False


def train_eval_iter(model, train_dataset, eval_dataset, writer, hparams, dataset_raw=None):
    """
    Single training loop with validation early stopping.

    - train_dataset is used for parameter updates.
    - eval_dataset is used for validation and early stopping.
    - writer is kept for API compatibility.

    Returns:
      (model, val_accs, best_val_result). The model is restored to the best
      validation checkpoint before returning.
    """
    optimizer = torch.optim.Adam(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=hparams.learning_rate,
      weight_decay=float(getattr(hparams, 'weight_decay', 0.0)),
    )
    device = torch.device(hparams.device)

    best_val_result = {'epoch': 0, 'loss': float('inf'), 'acc': -1e9, 'train_loss': float('inf')}
    best_model_state = None

    val_accs = []

    patience = int(getattr(hparams, 'patience', 50))
    no_improve = 0

    log_interval = 10
    train_eval_interval = int(getattr(hparams, 'train_eval_interval', log_interval))
    enable_train_eval = bool(getattr(hparams, 'enable_train_eval_during_training', True))
    train_eval_max_num_examples = getattr(hparams, 'train_eval_max_num_examples', 100)
    if train_eval_max_num_examples is not None:
      train_eval_max_num_examples = int(train_eval_max_num_examples)
    last_train_acc = None

    for epoch in range(hparams.epoch):
      should_log_epoch = (epoch % log_interval == 0)

      avg_loss = 0.0
      num_batches = 0
      model.train()

      for batch_idx, graph_data in enumerate(train_dataset):
        graph_data = _move_batch_to_device(graph_data, device)
        optimizer.zero_grad()

        ypred_out = model(graph_data)
        loss = get_loss.fused_loss(ypred_out, graph_data[g_key.y], epoch, hparams)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
        optimizer.step()

        avg_loss += loss.item()
        num_batches += 1

      if num_batches == 0:
        raise RuntimeError('Training dataset is empty.')
      avg_loss /= num_batches

      # Evaluate a small training subset for logging.
      should_eval_train = (
        enable_train_eval
        and should_log_epoch
        and train_eval_interval > 0
        and epoch % train_eval_interval == 0
      )
      if should_eval_train:
        train_result = evaluate(train_dataset, model, hparams, max_num_examples=train_eval_max_num_examples)
        last_train_acc = train_result['acc']

      # Validation is used for early stopping and reporting.
      val_result = evaluate(eval_dataset, model, hparams, include_loss=True, loss_epoch=epoch)
      val_accs.append(val_result['acc'])
      if should_log_epoch:
        train_acc_msg = '{:.4f}'.format(last_train_acc) if last_train_acc is not None else 'n/a'
        logging.info(
          'Epoch {} => train loss: {:.4f}, train acc: {}, val loss: {:.4f}, val acc: {:.4f}'.format(
            epoch, avg_loss, train_acc_msg, val_result['loss'], val_result['acc']
          )
        )
        
      # Optionally export GAT1 features every 50 epochs.
      enable_gat_export = bool(getattr(hparams, 'enable_gat_export', False))
      if enable_gat_export and (epoch + 1) % 50 == 0:
          logging.warning(f"Triggering GAT1 feature export at epoch {epoch + 1}")
          from MISGL.utils.export_gat import export_gat1_features
          export_gat1_features(model, eval_dataset, epoch + 1, dataset_raw, split="val")
          
      if _is_better_val_result(val_result, best_val_result, hparams):
        best_val_result.update({
          'acc': val_result['acc'],
          'epoch': epoch,
          'loss': val_result['loss'],
          'train_loss': avg_loss,
        })
        best_model_state = {
          name: value.detach().cpu().clone()
          for name, value in model.state_dict().items()
        }
        if should_log_epoch:
          logging.warning(
            'Best val result: acc {:.4f}, loss {:.4f} @ epoch {}'.format(
              best_val_result['acc'], best_val_result['loss'], best_val_result['epoch']
            )
          )
        no_improve = 0
      else:
        no_improve += 1
        if no_improve >= patience:
          logging.warning('Early stop at epoch {} (patience={})'.format(epoch, patience))
          break

    # Restore best validation checkpoint.
    if best_model_state is not None:
      model.load_state_dict(best_model_state)

    return model, val_accs, {
      'epoch': int(best_val_result['epoch']),
      'acc': float(best_val_result['acc']),
      'loss': float(best_val_result['loss']),
      'train_loss': float(best_val_result['train_loss']),
    }

