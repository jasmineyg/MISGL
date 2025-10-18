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
import tensorboardX
import random
from gnn_hpool.utils import get_loss
from gnn_hpool.utils import common_utils
from gnn_hpool.utils.global_variables import *
from gnn_hpool.utils.evaluate import evaluate
from gnn_hpool.utils import load_data
from gnn_hpool.models import gcn_hpool_encoder


def train_eval(hparams):
  data_loader = load_data.GraphDataLoaderWrapper(hparams)

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

    summary_writer = tensorboardX.SummaryWriter(
      logdir=os.path.join(hparams.model_save_path, str(hparams.timestamp) + '/holdout_{}'.format(run_idx))
    )

    model = gcn_hpool_encoder.GcnHpoolEncoder(hparams).to(torch.device(hparams.device))
    # 训练+早停都用val
    train_eval_iter(model, training_loader, validation_loader, summary_writer, hparams)

    # 测试：单次评估
    result = evaluate(test_loader, model, hparams, analyze_attention=True, dataset_name="test")
    all_results.append(result)
    for key in test_metrics.keys():
      test_metrics[key].append(result[key])

  # 汇总 mean ± std（不偏估计ddof=1）
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

  # all_vals = []
  # for val_idx in range(hparams.fold_num):
  #   logging.warning('* validation index: {}'.format(val_idx))
  #   training_loader, inner_val_loader, validation_loader = data_loader.get_loader(
  #     val_idx, inner_val_frac=getattr(hparams, 'inner_val_frac', 0.1)
  #   )
  #   summary_writer = tensorboardX.SummaryWriter(
  #     logdir=os.path.join(hparams.model_save_path, str(hparams.timestamp) + '/val_{}'.format(val_idx))
  #   )
  #
  #   model = gcn_hpool_encoder.GcnHpoolEncoder(hparams).to(torch.device(hparams.device))
  #   _, val_accs = train_eval_iter(model, training_loader, inner_val_loader, validation_loader, summary_writer, hparams)
  #   all_vals.append(np.array(val_accs))
  #
  # all_vals = np.vstack(all_vals)
  # all_vals = np.mean(all_vals, axis=0)
  # logging.warning('* all of the validation results: {}'.format(all_vals))
  # logging.warning('* the best validation results & its id: {} @ {}'.format(np.max(all_vals), np.argmax(all_vals)))
  #
  # final_train_loader, final_inner_loader = data_loader.get_full_train_with_inner_loader(
  #   inner_val_frac=getattr(hparams, 'inner_val_frac', 0.1)
  # )
  # final_test_loader = data_loader.get_test_loader()
  # summary_writer = tensorboardX.SummaryWriter(
  #   logdir=os.path.join(hparams.model_save_path, str(hparams.timestamp) + '/final_test')
  # )
  # final_model = gcn_hpool_encoder.GcnHpoolEncoder(hparams).to(torch.device(hparams.device))
  # train_eval_iter(final_model, final_train_loader, final_inner_loader, final_inner_loader, summary_writer, hparams)
  # test_result = evaluate(final_test_loader, final_model, hparams, analyze_attention=True, dataset_name="test")
  # logging.warning('Final test result (acc): {:.4f}'.format(test_result['acc']))

  # function train_eval_iter(model, train_dataset, eval_dataset, writer, hparams)
def train_eval_iter(model, train_dataset, eval_dataset, writer, hparams):
    import copy
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hparams.learning_rate)

    best_val_result = {'epoch': 0, 'loss': 0, 'acc': -1e9}
    best_model_state = None

    train_accs, train_epochs = [], []
    best_val_accs, best_val_epochs, val_accs = [], [], []

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

            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed

            if epoch % 10 == 0 and batch_idx == len(train_dataset) // 2 and writer is not None:
                assign_tensor = model.gcn_hpool_layer.pool_tensor
                bs = assign_tensor.size(0)
                safe_idx = [i for i in writer_batch_idx if i < bs]
                if len(safe_idx) > 0:
                    log_assignment(assign_tensor, writer, epoch, safe_idx)
                    log_graph(graph_data[g_key.adj_mat], graph_data[g_key.node_num], writer, epoch, safe_idx, assign_tensor)

        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if hasattr(hparams, 'branch_b') and hparams.branch_b.get('use', False):
                current_gamma = get_loss.get_gamma(epoch, 
                                                hparams.branch_b.get('gamma_start', 0.3),
                                                hparams.branch_b.get('gamma_end', 0.6),
                                                hparams.branch_b.get('warmup_epochs', 20))
                writer.add_scalar('fusion/gamma', current_gamma, epoch)

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

        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])

    # 恢复 val 最优权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    try:
        matplotlib.style.use('seaborn-v0_8')
    except OSError:
        try:
            matplotlib.style.use('seaborn')
        except OSError:
            matplotlib.style.use('default')

    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, common_utils.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    plt.plot(best_val_epochs, best_val_accs, 'bo')
    plt.legend(['train', 'val'])
    plt.savefig(os.path.join(hparams.model_save_path, str(hparams.timestamp) + '.png'), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs


def log_assignment(assign_tensor, writer, epoch, batch_idx):
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

  data = tensorboardX.utils.figure_to_image(fig)
  writer.add_image('assignment', data, epoch)


def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
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

  data = tensorboardX.utils.figure_to_image(fig)
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

  data = tensorboardX.utils.figure_to_image(fig)
  writer.add_image('graphs_colored', data, epoch)
try:
    import seaborn as sns
except Exception:
    sns = None
