# coding=utf-8

import os
import time
import logging
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
import random
from gnn_hpool.utils import get_loss
from gnn_hpool.utils import common_utils
from gnn_hpool.utils.global_variables import *
from gnn_hpool.utils.evaluate import evaluate
from gnn_hpool.utils import load_data
from gnn_hpool.utils.result_analyze_export import export_test_result_analyze_excel
from gnn_hpool.models import gcn_hpool_encoder
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hparams.device == 'cuda':
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    # 仅返回 train/val/test
    training_loader, validation_loader, test_loader = data_loader.get_holdout_loaders(
      seed=seed, train_frac=0.6, val_frac=0.2, test_frac=0.2
    )

    tb_root = getattr(hparams, 'tb_logdir', os.path.join('..', 'result'))
    logdir = os.path.join(tb_root, str(hparams.timestamp) + '/holdout_{}'.format(run_idx))
    if bool(getattr(hparams, 'tb_unique_run_dir', True)) and os.path.exists(logdir):
      logdir = os.path.join(logdir, time.strftime('%Y%m%d-%H%M%S'))
    summary_writer = SummaryWriter(logdir)

    model = gcn_hpool_encoder.GcnHpoolEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
    # 训练+早停都用val
    train_eval_iter(model, training_loader, validation_loader, summary_writer, hparams)

    result = evaluate(test_loader, model, hparams, dataset_name="test")
    all_results.append(result)
    for key in test_metrics.keys():
      test_metrics[key].append(result[key])
    logging.warning('Holdout {} test => acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, F1: {:.4f}'.format(
      run_idx, result['acc'], result['prec'], result['rec'], result['F1']
    ))

    # ra_cfg = getattr(hparams, 'result_analyze', None)
    # if bool(ra_cfg and ra_cfg.get('use', False)) and hasattr(model, 'forward_with_embeddings'):
    #   export_dir = ra_cfg.get('output_dir', getattr(hparams, 'model_save_path', '.'))
    #   sample_seed = int(ra_cfg.get('sample_seed', seed))
    #   sample_frac = float(ra_cfg.get('sample_frac', 0.2))
    #   prefix = f'{hparams.timestamp}_holdout_{run_idx}'
    #   export_test_result_analyze_excel(
    #     model=model,
    #     test_loader=test_loader,
    #     hparams=hparams,
    #     dataset_raw=data_loader._dataset_raw,
    #     output_dir=export_dir,
    #     seed=sample_seed,
    #     sample_frac=sample_frac,
    #     filename_prefix=prefix,
    #     heatmap_filename=f'{prefix}_heatmap.xlsx',
    #   )

    bb_cfg = getattr(hparams, 'branch_b', None)
    if bool(bb_cfg and bb_cfg.get('use', False)):
      out_path = os.path.join(hparams.model_save_path, f'{hparams.timestamp}_holdout_{run_idx}_attention.xlsx')
      export_branchB_attention_from_model(model, test_loader, hparams, data_loader._dataset_raw, out_path, sample_frac=0.2)
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


def train_eval_iter(model, train_dataset, eval_dataset, writer, hparams):
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

    return model, val_accs


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
  """
  将 matplotlib Figure 转成 CHW 格式的 uint8 图像（用于 TensorBoard add_image）。

  当 tensorboardX 不可用时，使用该函数替代 figure_to_image。
  """
  import numpy as np
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = buf.reshape(h, w, 3)
  return img.transpose(2, 0, 1)


def _label_str(v):
  """将二分类标签转成人类可读字符串：1->pos，0->neg。"""
  return 'pos' if int(v) == 1 else 'neg'


def _build_fixed_projector_batch(val_loader, num_graphs):
  """
  从 val_loader 中抽取一个“固定”的小 batch，用于每隔若干 epoch 记录 embedding projector。

  目标：
    - 尽量平衡正负样本（pos/neg 各一半，至少 1 个 pos）
    - 返回的是一个 dict（只包含 Tensor 字段），便于直接喂给 model.forward_with_embeddings
  """
  selected = []
  selected_pos = 0
  selected_neg = 0
  want_pos = max(1, num_graphs // 2)
  want_neg = num_graphs - want_pos

  for batch in val_loader:
    ys = batch[g_key.y].detach().cpu().view(-1).tolist()
    bs = len(ys)
    for i in range(bs):
      y = int(ys[i])
      if y == 1 and selected_pos < want_pos:
        selected.append((batch, i))
        selected_pos += 1
      elif y == 0 and selected_neg < want_neg:
        selected.append((batch, i))
        selected_neg += 1
      if len(selected) >= num_graphs:
        break
    if len(selected) >= num_graphs:
      break

  if len(selected) == 0:
    return None

  out = {}
  keys = list(selected[0][0].keys())
  for k in keys:
    v0 = selected[0][0][k]
    if not torch.is_tensor(v0):
      continue
    pieces = []
    for b, idx in selected:
      pieces.append(b[k][idx].unsqueeze(0))
    out[k] = torch.cat(pieces, dim=0)
  return out


def _map_subgraph_nodes_to_labels(subgraph, node_binary_labels, n_i):
  """
  将 dataset_raw 的全局节点二分类标签映射到“当前子图节点顺序”上。

  - subgraph：networkx 子图结构（节点属性里可能包含 original_id/original_index 等字段）
  - node_binary_labels：全局节点标签数组
  - n_i：当前子图有效节点数（用于截断）
  """
  nodes = list(subgraph.nodes())
  labels = np.zeros(len(nodes), dtype=np.int64)
  total = len(node_binary_labels)
  for j, node_id in enumerate(nodes):
    idx = None
    attr = subgraph.nodes[node_id]
    for key in ('original_id', 'original_index', 'orig_id', 'node_index'):
      if key in attr and attr[key] is not None:
        try:
          idx = int(attr[key])
          break
        except Exception:
          pass
    if idx is None and isinstance(node_id, (int, np.integer)):
      idx = int(node_id)
    if idx is not None and 0 <= idx < total:
      labels[j] = int(node_binary_labels[idx])
    else:
      labels[j] = 0
  return labels[:n_i]


def _get_node_labels_for_batch(dataset_raw, orig_graph_indices, num_list):
  """
  为一个 batch 的多个子图生成节点标签列表。

  返回：
    list[np.ndarray]，长度=子图数；每个元素是该子图的节点标签（长度=对应 n_i）。
  """
  if dataset_raw is None:
    return [np.zeros(n, dtype=np.int64) for n in num_list]
  node_labels_collection = dataset_raw.get('node_binary_labels', None)
  subgraphs = dataset_raw.get('subgraph_structures', None)
  if node_labels_collection is None or subgraphs is None:
    return [np.zeros(n, dtype=np.int64) for n in num_list]

  out = []
  for orig_idx, n_i in zip(orig_graph_indices, num_list):
    if 0 <= orig_idx < len(subgraphs):
      subgraph = subgraphs[orig_idx]
      out.append(_map_subgraph_nodes_to_labels(subgraph, node_labels_collection, n_i))
    else:
      out.append(np.zeros(n_i, dtype=np.int64))
  return out


def _map_subgraph_nodes_to_orig_ids(subgraph, n_i):
  """
  将子图节点映射回原始图节点 id（便于分析与可视化对齐）。

  若无法找到 original_id/original_index 等属性，则回退为 node_id（若为 int）。
  """
  nodes = list(subgraph.nodes())
  out = np.full(len(nodes), -1, dtype=np.int64)
  for j, node_id in enumerate(nodes):
    attr = subgraph.nodes[node_id]
    idx = None
    for key in ('original_id', 'original_index', 'orig_id', 'node_index'):
      if key in attr and attr[key] is not None:
        try:
          idx = int(attr[key])
          break
        except Exception:
          pass
    if idx is None and isinstance(node_id, (int, np.integer)):
      idx = int(node_id)
    out[j] = -1 if idx is None else idx
  return out[:n_i]


def _get_node_orig_ids_for_batch(dataset_raw, orig_graph_indices, num_list):
  """
  为一个 batch 的多个子图生成“原始节点 id”列表。

  返回：
    list[np.ndarray]，长度=子图数；每个元素长度=对应 n_i。
  """
  if dataset_raw is None:
    return [np.full(n, -1, dtype=np.int64) for n in num_list]
  subgraphs = dataset_raw.get('subgraph_structures', None)
  if subgraphs is None:
    return [np.full(n, -1, dtype=np.int64) for n in num_list]

  out = []
  for orig_idx, n_i in zip(orig_graph_indices, num_list):
    if 0 <= orig_idx < len(subgraphs):
      subgraph = subgraphs[orig_idx]
      out.append(_map_subgraph_nodes_to_orig_ids(subgraph, n_i))
    else:
      out.append(np.full(n_i, -1, dtype=np.int64))
  return out


def _log_embedding_projector(writer, model, graph_data, dataset_raw, global_step, max_nodes_per_graph=256):
  """
  将图级/节点级 embedding 写入 TensorBoard Embedding Projector 以便交互式查看。

  - 图级：mean_vec、H1、H2（如果 forward_with_embeddings 返回了对应 key）
  - 节点级：h（每个子图最多记录 max_nodes_per_graph 个节点）
  - metadata：包含 y/pred/prob、orig_graph_idx、subgraph_id，以及节点的 orig_id 与 node_y
  """
  model.eval()
  with torch.no_grad():
    ypred_out, emb = model.forward_with_embeddings(graph_data)
  if emb is None:
    return

  ys = graph_data[g_key.y].detach().cpu().view(-1).tolist()
  logits = ypred_out['ypred_A'] if isinstance(ypred_out, dict) and 'ypred_A' in ypred_out else ypred_out
  probs = torch.sigmoid(logits.view(-1).detach().cpu()).tolist() if logits is not None else [0.0 for _ in ys]
  preds = [1 if float(p) >= 0.5 else 0 for p in probs]

  orig_idx_tensor = graph_data.get(g_key.orig_graph_idx, None)
  if orig_idx_tensor is not None and isinstance(orig_idx_tensor, torch.Tensor):
    orig_graph_indices = [int(i) for i in orig_idx_tensor.detach().cpu().tolist()]
  else:
    orig_graph_indices = list(range(len(ys)))

  subgraph_id_tensor = graph_data.get(g_key.subgraph_id, None)
  if subgraph_id_tensor is not None and isinstance(subgraph_id_tensor, torch.Tensor):
    subgraph_ids = [int(i) for i in subgraph_id_tensor.detach().cpu().tolist()]
  else:
    subgraph_ids = [-1 for _ in ys]

  graph_metadata_header = ['class', 'orig_idx', 'subgraph_id', 'y', 'pred', 'prob']
  graph_metadata = [
    [
      _label_str(ys[i]),
      str(orig_graph_indices[i]),
      str(subgraph_ids[i]),
      str(int(ys[i])),
      str(int(preds[i])),
      f'{float(probs[i]):.4f}',
    ]
    for i in range(len(ys))
  ]

  mean_vec = emb.get('mean_vec', None)
  if mean_vec is not None:
    writer.add_embedding(mean_vec.detach().cpu(), metadata=graph_metadata, metadata_header=graph_metadata_header, global_step=global_step, tag=f'graph_emb/mean_vec/ep_{global_step}')

  H1 = emb.get('graph_emb_H1', None)
  if H1 is not None:
    writer.add_embedding(H1.detach().cpu(), metadata=graph_metadata, metadata_header=graph_metadata_header, global_step=global_step, tag=f'graph_emb/H1/ep_{global_step}')

  H2 = emb.get('graph_emb_H2', None)
  if H2 is not None:
    writer.add_embedding(H2.detach().cpu(), metadata=graph_metadata, metadata_header=graph_metadata_header, global_step=global_step, tag=f'graph_emb/H2/ep_{global_step}')

  h = emb.get('h', None)
  if h is None:
    return

  batch_num_nodes = graph_data[g_key.node_num]
  if isinstance(batch_num_nodes, torch.Tensor):
    num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
  else:
    num_list = [int(n) for n in batch_num_nodes]

  node_labels_per_graph = _get_node_labels_for_batch(dataset_raw, orig_graph_indices, num_list)
  node_orig_ids_per_graph = _get_node_orig_ids_for_batch(dataset_raw, orig_graph_indices, num_list)
  for i, n_i in enumerate(num_list):
    if n_i <= 0:
      continue
    n_use = min(int(n_i), int(max_nodes_per_graph))
    node_emb = h[i, :n_use, :].detach().cpu()
    node_labels = node_labels_per_graph[i][:n_use].tolist()
    node_orig_ids = node_orig_ids_per_graph[i][:n_use].tolist()
    node_metadata_header = ['class', 'orig', 'node_y']
    node_metadata = [[_label_str(node_labels[j]), str(int(node_orig_ids[j])), str(int(node_labels[j]))] for j in range(n_use)]
    tag = f'node_emb/h/graph_{orig_graph_indices[i]}/ep_{global_step}'
    writer.add_embedding(node_emb, metadata=node_metadata, metadata_header=node_metadata_header, global_step=global_step, tag=tag)
