# coding=utf-8

import os
import logging
import json
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


_METRIC_KEYS = ('acc', 'prec', 'rec', 'F1')
_FINAL_EVAL_SPLITS = ('train', 'val', 'test')


def _basic_metrics(result):
  return {key: float(result[key]) for key in _METRIC_KEYS}


def _format_metrics(result):
  return ', '.join(f'{key}: {result[key]:.4f}' for key in _METRIC_KEYS)


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


def _get_lightweight_attention_exporter():
  global _LIGHTWEIGHT_ATTENTION_EXPORTER
  if _LIGHTWEIGHT_ATTENTION_EXPORTER is not None:
    return _LIGHTWEIGHT_ATTENTION_EXPORTER

  try:
    from attention_analyzer_impl import export_lightweight_attention_from_model
  except ImportError as impl_error:
    try:
      from attention_analyzer import export_lightweight_attention_from_model
    except ImportError as wrapper_error:
      raise ImportError(
        'Cannot import export_lightweight_attention_from_model from '
        'attention_analyzer_impl or attention_analyzer.'
      ) from wrapper_error
    except AttributeError as wrapper_error:
      raise ImportError(
        'attention_analyzer does not define export_lightweight_attention_from_model.'
      ) from wrapper_error
    else:
      logging.warning(
        'Using export_lightweight_attention_from_model from attention_analyzer '
        'because attention_analyzer_impl import failed: %s',
        impl_error,
      )

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
  训练与评估主入口（Repeated Holdout）。

  - 根据 hparams.holdout_seeds 或 (holdout_runs, cv_seed) 生成多个随机种子
  - 每个 seed 划分 train/val/test，训练模型（在 val 上早停），并在 test 上评估
  - 可选：导出 test set 的 embedding 分析结果（Excel）与 branch-B 注意力（Excel）

  返回：
    dict: {'seeds': [...], 'results': [...], 'summary': {...}}，其中 summary 是多次 holdout 的均值±方差统计
  """
  data_loader = load_data.GraphDataLoaderWrapper(hparams, data_name=data_name)

  # 读取重复留出配置：优先使用 holdout_seeds，否则使用 holdout_runs/cv_seed 生成
  holdout_seeds = getattr(hparams, 'holdout_seeds', None)
  if isinstance(holdout_seeds, list) and len(holdout_seeds) > 0:
    seeds = [int(s) for s in holdout_seeds]
  else:
    holdout_runs = int(getattr(hparams, 'holdout_runs', getattr(hparams, 'fold_num', 10)))
    base_seed = int(getattr(hparams, 'cv_seed', 1024))
    seeds = [base_seed + i for i in range(holdout_runs)]

  final_splits = _final_eval_splits(hparams)
  test_metrics = {'acc': [], 'prec': [], 'rec': [], 'F1': []}
  split_metrics = {
    split_name: {key: [] for key in _METRIC_KEYS}
    for split_name in final_splits
  }
  all_results = []

  for run_idx, seed in enumerate(seeds):
    logging.warning('* holdout run: {} (seed={})'.format(run_idx, seed))

    reproducibility.set_seed(seed, cuda_deterministic=(hparams.device == 'cuda'))

    # 仅返回 train/val/test
    training_loader, validation_loader, test_loader = data_loader.get_holdout_loaders(
      seed=seed, train_frac=0.6, val_frac=0.2, test_frac=0.2
    )

    summary_writer = None

    model = encoder.MISGLEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
    # 训练+早停都用val
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

  summary = {
    key: {
      'mean': float(np.mean(vals)),
      'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    }
    for key, vals in test_metrics.items()
  }
  split_summary = {
    split_name: {
      key: {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
      }
      for key, vals in split_result.items()
    }
    for split_name, split_result in split_metrics.items()
  }
  for split_name in final_splits:
    msg_parts = [
      f'{k}: {split_summary[split_name][k]["mean"]:.4f} +/- {split_summary[split_name][k]["std"]:.4f}'
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
  test_metrics = {'acc': [], 'prec': [], 'rec': [], 'F1': []}
  split_metrics = {
    split_name: {key: [] for key in _METRIC_KEYS}
    for split_name in final_splits
  }
  all_results = []

  for fold_idx in range(fold_count):
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

  summary = {
    key: {
      'mean': float(np.mean(vals)),
      'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    }
    for key, vals in test_metrics.items()
  }
  split_summary = {
    split_name: {
      key: {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
      }
      for key, vals in split_result.items()
    }
    for split_name, split_result in split_metrics.items()
  }
  for split_name in final_splits:
    msg_parts = [
      f'{k}: {split_summary[split_name][k]["mean"]:.4f} +/- {split_summary[split_name][k]["std"]:.4f}'
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
    单次 holdout 下的训练循环（按 epoch 训练 + val 早停）。

    - train_dataset：用于反向传播更新参数
    - eval_dataset：用于评估与 early stopping（val）
    - writer：保留旧接口兼容，当前训练流程不写可视化日志
    
    返回：
      (model, val_accs) 其中 model 会在结束前恢复到 val 最优权重。
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

      # 训练集评估
      should_eval_train = (
        enable_train_eval
        and should_log_epoch
        and train_eval_interval > 0
        and epoch % train_eval_interval == 0
      )
      if should_eval_train:
        train_result = evaluate(train_dataset, model, hparams, max_num_examples=train_eval_max_num_examples)
        last_train_acc = train_result['acc']

      # 验证：用于早停与报告
      val_result = evaluate(eval_dataset, model, hparams, include_loss=True, loss_epoch=epoch)
      val_accs.append(val_result['acc'])
      if should_log_epoch:
        train_acc_msg = '{:.4f}'.format(last_train_acc) if last_train_acc is not None else 'n/a'
        logging.info(
          'Epoch {} => train loss: {:.4f}, train acc: {}, val loss: {:.4f}, val acc: {:.4f}'.format(
            epoch, avg_loss, train_acc_msg, val_result['loss'], val_result['acc']
          )
        )
        
      # 导出 GAT1 特征，每 50 个 epoch 保存一次，第一次保存在第 50 epoch (即 epoch 49)
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

    # 恢复 val 最优权重
    if best_model_state is not None:
      model.load_state_dict(best_model_state)

    return model, val_accs, {
      'epoch': int(best_val_result['epoch']),
      'acc': float(best_val_result['acc']),
      'loss': float(best_val_result['loss']),
      'train_loss': float(best_val_result['train_loss']),
    }


