# coding=utf-8

import os
import time
import logging
import json
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
from attention_analyzer import export_branchB_attention_from_model


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

    model = encoder.GcnHpoolEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
    # 训练+早停都用val
    train_eval_iter(model, training_loader, validation_loader, summary_writer, hparams, dataset_raw=data_loader._dataset_raw)

    result = evaluate(test_loader, model, hparams, dataset_name="test")
    all_results.append(result)
    for key in test_metrics.keys():
      test_metrics[key].append(result[key])
    logging.warning('Holdout {} test => acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, F1: {:.4f}'.format(
      run_idx, result['acc'], result['prec'], result['rec'], result['F1']
    ))

    bb_cfg = getattr(hparams, 'branch_b', None)
    logging.warning(f'[DEBUG] branch_b config: {bb_cfg}')
    if bool(bb_cfg and bb_cfg.get('use', False)):
      out_path = os.path.join(hparams.model_save_path, f'{hparams.timestamp}_holdout_{run_idx}_attention.xlsx')
      logging.warning(f'[DEBUG] Exporting attention to: {out_path}')
      export_branchB_attention_from_model(model, test_loader, hparams, data_loader._dataset_raw, out_path, sample_frac=0.2)
    else:
      logging.warning('[DEBUG] branch_b.use is False or not found, skipping attention export.')
    if summary_writer is not None:
      summary_writer.close()

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
  test_metrics = {'acc': [], 'prec': [], 'rec': [], 'F1': []}
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

    model = encoder.GcnHpoolEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
    model, _, best_val_result = train_eval_iter(
      model, training_loader, validation_loader, summary_writer, hparams, dataset_raw=data_loader._dataset_raw
    )

    result = evaluate(test_loader, model, hparams, dataset_name='test')
    all_results.append({
      'fold_idx': int(fold_idx),
      'seed': int(seed),
      'split': split_meta,
      'best_val': best_val_result,
      'metrics': {
        'acc': float(result['acc']),
        'prec': float(result['prec']),
        'rec': float(result['rec']),
        'F1': float(result['F1']),
      },
    })
    for key in test_metrics.keys():
      test_metrics[key].append(result[key])
    logging.warning('CV fold {} test => acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, F1: {:.4f}'.format(
      fold_idx, result['acc'], result['prec'], result['rec'], result['F1']
    ))

    bb_cfg = getattr(hparams, 'branch_b', None)
    logging.warning(f'[DEBUG] branch_b config: {bb_cfg}')
    if bool(bb_cfg and bb_cfg.get('use', False)):
      out_path = os.path.join(hparams.model_save_path, f'{hparams.timestamp}_cv_fold_{fold_idx}_attention.xlsx')
      logging.warning(f'[DEBUG] Exporting attention to: {out_path}')
      export_branchB_attention_from_model(model, test_loader, hparams, data_loader._dataset_raw, out_path, sample_frac=0.2)
    else:
      logging.warning('[DEBUG] branch_b.use is False or not found, skipping attention export.')
    if summary_writer is not None:
      summary_writer.close()

  summary = {
    key: {
      'mean': float(np.mean(vals)),
      'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    }
    for key, vals in test_metrics.items()
  }
  msg_parts = [f'{k}: {summary[k]["mean"]:.4f} +/- {summary[k]["std"]:.4f}' for k in ['acc', 'prec', 'rec', 'F1']]
  logging.warning('* Fixed 10-fold CV test results => {}'.format('; '.join(msg_parts)))

  result_path = os.path.join(hparams.model_save_path, f'{hparams.timestamp}_cv_results.json')
  with open(result_path, 'w', encoding='utf-8') as f:
    json.dump({
      'data_name': data_name,
      'split_path': split_path,
      'cv_seed': int(split_manifest['cv_seed']),
      'cv_num_folds': int(split_manifest['cv_num_folds']),
      'cv_val_policy': split_manifest['cv_val_policy'],
      'summary': summary,
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
    import copy
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hparams.learning_rate)

    best_val_result = {'epoch': 0, 'loss': 0, 'acc': -1e9}
    best_model_state = None

    train_accs, train_epochs = [], []
    best_val_accs, best_val_epochs, val_accs = [], [], []

    patience = int(getattr(hparams, 'patience', 50))
    no_improve = 0
    
    # 暂时移除projector相关参数和逻辑，因为dataset_raw传递比较复杂，且非核心功能
    writer_batch_idx = list(range(10)) # 默认可视化前10个图

    for epoch in range(hparams.epoch):
      if not epoch % 10:
        logging.info('* Start the {}_th epoch'.format(epoch))

      total_time = 0
      avg_loss = 0.0
      model.train()

      for batch_idx, graph_data in enumerate(train_dataset):
        begin_time = time.time()
        optimizer.zero_grad()

        ypred_out = model(graph_data)
        loss = get_loss.fused_loss(ypred_out, graph_data[g_key.y], epoch, hparams)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
        optimizer.step()

        avg_loss += loss.item()
        elapsed = time.time() - begin_time
        total_time += elapsed

        if epoch % 10 == 0 and batch_idx == len(train_dataset) // 2 and writer is not None:
          layer = getattr(model, 'gcn_hpool_layer', None)
          assign_tensor = getattr(layer, 'pool_tensor', None)
          if assign_tensor is not None:
            bs = assign_tensor.size(0)
            safe_idx = [i for i in writer_batch_idx if i < bs]
            if len(safe_idx) > 0:
              log_assignment(assign_tensor, writer, epoch, safe_idx)
              log_graph(graph_data[g_key.adj_mat], graph_data[g_key.node_num], writer, epoch, safe_idx, assign_tensor)

      avg_loss /= batch_idx + 1
      if writer is not None:
        writer.add_scalar('loss/avg_loss', avg_loss, epoch)

      # 训练集评估
      result = evaluate(train_dataset, model, hparams, max_num_examples=100)
      train_accs.append(result['acc'])
      train_epochs.append(epoch)
      if writer is not None:
        writer.add_scalar('acc/train_acc', result['acc'], epoch)

      # 验证：用于早停与报告
      val_result = evaluate(eval_dataset, model, hparams)
      val_accs.append(val_result['acc'])
      if writer is not None:
        writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
      logging.info(
        'Epoch {} => loss: {:.4f}, train acc: {:.4f}, val acc: {:.4f}'.format(
          epoch, avg_loss, result['acc'], val_result['acc']
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
        best_model_state = copy.deepcopy(model.state_dict())
        logging.warning('Best val result: {:.4f} @ epoch {}'.format(best_val_result['acc'], best_val_result['epoch']))
        no_improve = 0
      else:
        no_improve += 1
        if no_improve >= patience:
          logging.warning('Early stop at epoch {} (patience={})'.format(epoch, patience))
          break

      best_val_epochs.append(best_val_result['epoch'])
      best_val_accs.append(best_val_result['acc'])

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
