# coding=utf-8

import os
import time
import logging
import json
import gc
import matplotlib

try:
  import matplotlib.pyplot as plt
except ModuleNotFoundError:
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

import torch
try:
  import tensorboardX
  SummaryWriter = tensorboardX.SummaryWriter
  _figure_to_image = tensorboardX.utils.figure_to_image
except ModuleNotFoundError:
  tensorboardX = None
  from torch.utils.tensorboard import SummaryWriter
  _figure_to_image = None
from MISGL.utils import get_loss
from MISGL.utils import reproducibility
from MISGL.utils.global_variables import *
from MISGL.utils.evaluate import evaluate
from MISGL.utils import load_data
from MISGL.models import encoder
from attention_analyzer import export_lightweight_attention_from_model


def _attention_train_output_path(path):
  base, ext = os.path.splitext(path)
  return f'{base}_train10p{ext}'


_METRIC_KEYS = ('acc', 'prec', 'rec', 'F1')


def _basic_metrics(result):
  return {key: float(result[key]) for key in _METRIC_KEYS}


def _format_metrics(result):
  return ', '.join(f'{key}: {result[key]:.4f}' for key in _METRIC_KEYS)


def _final_eval_splits(hparams):
  raw = getattr(hparams, 'final_eval_splits', ['test'])
  if isinstance(raw, str):
    splits = [s.strip() for s in raw.split(',') if s.strip()]
  elif isinstance(raw, (list, tuple, set)):
    splits = [str(s).strip() for s in raw if str(s).strip()]
  else:
    splits = ['test']

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
    return float(getattr(hparams, 'attention_train_sample_frac', 0.02))
  return float(getattr(hparams, 'attention_sample_frac', 0.1))


def _attention_top_k(hparams):
  return int(getattr(hparams, 'attention_top_k', 20))


def _should_export_train_attention(hparams):
  return bool(getattr(hparams, 'attention_export_train', False))


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

  test_metrics = {'acc': [], 'prec': [], 'rec': [], 'F1': []}
  all_results = []

  for run_idx, seed in enumerate(seeds):
    logging.warning('* holdout run: {} (seed={})'.format(run_idx, seed))

    reproducibility.set_seed(seed, cuda_deterministic=(hparams.device == 'cuda'))

    # 仅返回 train/val/test
    training_loader, validation_loader, test_loader = data_loader.get_holdout_loaders(
      seed=seed, train_frac=0.6, val_frac=0.2, test_frac=0.2
    )

    # 默认关闭 tensorboard 以节省内存
    enable_tensorboard = getattr(hparams, 'enable_tensorboard', False)
    summary_writer = None

    if enable_tensorboard:
      tb_root = getattr(hparams, 'tb_logdir', os.path.join('..', 'result'))
      logdir = os.path.join(tb_root, str(hparams.timestamp) + '/holdout_{}'.format(run_idx))
      if bool(getattr(hparams, 'tb_unique_run_dir', True)) and os.path.exists(logdir):
        logdir = os.path.join(logdir, time.strftime('%Y%m%d-%H%M%S'))
      summary_writer = SummaryWriter(logdir)

    model = encoder.MISGLEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
    # 训练+早停都用val
    train_eval_iter(model, training_loader, validation_loader, summary_writer, hparams, dataset_raw=data_loader._dataset_raw)

    result = evaluate(test_loader, model, hparams, dataset_name="test")
    all_results.append(result)
    for key in test_metrics.keys():
      test_metrics[key].append(result[key])
    logging.warning('Holdout {} test => acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, F1: {:.4f}'.format(
      run_idx, result['acc'], result['prec'], result['rec'], result['F1']
    ))

    if _should_export_attention(hparams):
      out_path = os.path.join(hparams.model_save_path, f'{hparams.timestamp}_holdout_{run_idx}_analyze_attention.xlsx')
      logging.warning(f'[DEBUG] Exporting attention to: {out_path}')
      export_lightweight_attention_from_model(
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
        export_lightweight_attention_from_model(
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
    del model, training_loader, validation_loader, test_loader, result
    _clear_cuda_cache(hparams)

  summary = {
    key: {
      'mean': float(np.mean(vals)),
      'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    }
    for key, vals in test_metrics.items()
  }
  msg_parts = [f'{k}: {summary[k]["mean"]:.4f} ± {summary[k]["std"]:.4f}' for k in ['acc', 'prec', 'rec', 'F1']]
  logging.warning('* Repeated Holdout (k={}) test results => {}'.format(len(seeds), '; '.join(msg_parts)))

  return {'seeds': seeds, 'results': all_results, 'summary': summary}


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

    enable_tensorboard = getattr(hparams, 'enable_tensorboard', False)
    summary_writer = None
    if enable_tensorboard:
      tb_root = getattr(hparams, 'tb_logdir', os.path.join('..', 'result'))
      logdir = os.path.join(tb_root, str(hparams.timestamp) + '/cv_fold_{}'.format(fold_idx))
      if bool(getattr(hparams, 'tb_unique_run_dir', True)) and os.path.exists(logdir):
        logdir = os.path.join(logdir, time.strftime('%Y%m%d-%H%M%S'))
      summary_writer = SummaryWriter(logdir)

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
      export_lightweight_attention_from_model(
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
        export_lightweight_attention_from_model(
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
  msg_parts = [f'{k}: {summary[k]["mean"]:.4f} +/- {summary[k]["std"]:.4f}' for k in ['acc', 'prec', 'rec', 'F1']]
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

  return {'results': all_results, 'summary': summary, 'split_path': split_path, 'result_path': result_path}


train_eval = fixed_cv_train_eval


def train_eval_iter(model, train_dataset, eval_dataset, writer, hparams, dataset_raw=None):
    """
    单次 holdout 下的训练循环（按 epoch 训练 + val 早停）。

    - train_dataset：用于反向传播更新参数
    - eval_dataset：用于评估与 early stopping（val）
    - writer：TensorBoard 记录器（可为 None）
    
    返回：
      (model, val_accs) 其中 model 会在结束前恢复到 val 最优权重。
    """
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hparams.learning_rate)
    device = torch.device(hparams.device)

    best_val_result = {'epoch': 0, 'loss': 0, 'acc': -1e9}
    best_model_state = None

    val_accs = []

    patience = int(getattr(hparams, 'patience', 50))
    no_improve = 0

    writer_batch_idx = list(range(10)) # 默认可视化前10个图
    log_interval = 10
    train_eval_interval = int(getattr(hparams, 'train_eval_interval', 10))
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

        if epoch % 10 == 0 and batch_idx == len(train_dataset) // 2 and writer is not None:
          layer = getattr(model, 'gcn_hpool_layer', None)
          assign_tensor = getattr(layer, 'pool_tensor', None)
          if assign_tensor is not None:
            bs = assign_tensor.size(0)
            safe_idx = [i for i in writer_batch_idx if i < bs]
            if len(safe_idx) > 0:
              log_assignment(assign_tensor, writer, epoch, safe_idx)
              log_graph(graph_data[g_key.adj_mat], graph_data[g_key.node_num], writer, epoch, safe_idx, assign_tensor)

      if num_batches == 0:
        raise RuntimeError('Training dataset is empty.')
      avg_loss /= num_batches
      if writer is not None:
        writer.add_scalar('loss/avg_loss', avg_loss, epoch)

      # 训练集评估
      should_eval_train = train_eval_interval > 0 and epoch % train_eval_interval == 0
      if should_eval_train:
        train_result = evaluate(train_dataset, model, hparams, max_num_examples=100)
        last_train_acc = train_result['acc']
        if writer is not None:
          writer.add_scalar('acc/train_acc', last_train_acc, epoch)

      # 验证：用于早停与报告
      val_result = evaluate(eval_dataset, model, hparams)
      val_accs.append(val_result['acc'])
      if writer is not None:
        writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
      if should_log_epoch:
        logging.info(
          'Epoch {} => loss: {:.4f}, train acc: {:.4f}, val acc: {:.4f}'.format(
            epoch, avg_loss, last_train_acc if last_train_acc is not None else float('nan'), val_result['acc']
          )
        )
        
      # 导出 GAT1 特征，每 50 个 epoch 保存一次，第一次保存在第 50 epoch (即 epoch 49)
      enable_gat_export = bool(getattr(hparams, 'enable_gat_export', False))
      if enable_gat_export and (epoch + 1) % 50 == 0:
          logging.warning(f"Triggering GAT1 feature export at epoch {epoch + 1}")
          from MISGL.utils.export_gat import export_gat1_features
          export_gat1_features(model, eval_dataset, epoch + 1, dataset_raw, split="val")
          
      if val_result['acc'] > best_val_result['acc'] - 1e-7:
        best_val_result.update({'acc': val_result['acc'], 'epoch': epoch, 'loss': avg_loss})
        best_model_state = {
          name: value.detach().cpu().clone()
          for name, value in model.state_dict().items()
        }
        if should_log_epoch:
          logging.warning('Best val result: {:.4f} @ epoch {}'.format(best_val_result['acc'], best_val_result['epoch']))
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
    }


def log_assignment(assign_tensor, writer, epoch, batch_idx):
  """
  将 DiffPool 的 assignment 矩阵（每个子图：node->cluster 的软分配）可视化并写入 TensorBoard。

  assign_tensor: shape 通常为 [batch_size, num_nodes, num_clusters]
  batch_idx: 选取 batch 中要展示的子图索引列表
  """
  plt.switch_backend('agg')
  fig = plt.figure(figsize=(8, 6), dpi=300)

  # has to be smaller than args.batch_size
  for i in range(len(batch_idx)):
      plt.subplot(2, 2, i + 1)
      plt.imshow(assign_tensor.cpu().detach().numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
      cbar = plt.colorbar()
      cbar.solids.set_edgecolor("face")
  plt.tight_layout()
  fig.canvas.draw()
  data = _figure_to_image(fig) if _figure_to_image is not None else _mpl_figure_to_image(fig)
  writer.add_image('assignment', data, epoch)


def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
  """
  可视化原始子图结构与按 cluster 着色后的子图，并写入 TensorBoard。

  - 第 1 张图：直接画子图的邻接结构（带节点标签）
  - 第 2 张图：根据 assignment 的 argmax 得到 cluster label，对节点着色
  """
  plt.switch_backend('agg')
  fig = plt.figure(figsize=(8, 6), dpi=300)

  for i in range(len(batch_idx)):
      ax = plt.subplot(2, 2, i + 1)
      num_nodes = int(batch_num_nodes[batch_idx[i]].item()) if isinstance(batch_num_nodes, torch.Tensor) else int(batch_num_nodes[batch_idx[i]])
      adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().detach().numpy()
      G = nx.from_numpy_array(adj_matrix)
      nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
              edge_color='grey', width=0.5, node_size=300,
              alpha=0.7)
      ax.xaxis.set_visible(False)

  plt.tight_layout()
  fig.canvas.draw()
  data = _figure_to_image(fig) if _figure_to_image is not None else _mpl_figure_to_image(fig)
  writer.add_image('graphs', data, epoch)

  assignment = assign_tensor.cpu().detach().numpy()
  fig = plt.figure(figsize=(8, 6), dpi=300)

  num_clusters = assignment.shape[2]
  all_colors = np.array(range(num_clusters))

  for i in range(len(batch_idx)):
      ax = plt.subplot(2, 2, i + 1)
      num_nodes = int(batch_num_nodes[batch_idx[i]].item()) if isinstance(batch_num_nodes, torch.Tensor) else int(batch_num_nodes[batch_idx[i]])
      adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().detach().numpy()

      label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
      label = label[: batch_num_nodes[batch_idx[i]]]
      node_colors = all_colors[label]

      G = nx.from_numpy_array(adj_matrix)
      nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
              edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
              vmin=0, vmax=num_clusters - 1,
              alpha=0.8)

  plt.tight_layout()
  fig.canvas.draw()
  data = _figure_to_image(fig) if _figure_to_image is not None else _mpl_figure_to_image(fig)
  writer.add_image('graphs_colored', data, epoch)
def _mpl_figure_to_image(fig):

  import numpy as np
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = buf.reshape(h, w, 3)
  return img.transpose(2, 0, 1)
