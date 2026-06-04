# coding=utf-8

"""Two-stage coarse-graph relation experiment for MISGL.

Stage 1 gets one fixed embedding for each subgraph, normally z_mil from Branch-B.
Stage 2 treats each subgraph as one coarse node and trains a light relation model
on top of the fixed embeddings and the cached coarse adjacency.
"""

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from MISGL.bin.train_eval import train_eval_iter
from MISGL.models.encoder import MISGLEncoder
from MISGL.utils import hparam
from MISGL.utils import hparams_lib
from MISGL.utils import reproducibility
from MISGL.utils.global_variables import g_key
from MISGL.utils.load_data import GraphDataLoaderWrapper
from MISGL.utils.load_data import GraphDataset


def add_bool_arg(parser, name):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f'--{name}', dest=name, action='store_true', default=None)
    group.add_argument(f'--no_{name}', dest=name, action='store_false')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train relation models on fixed MISGL subgraph embeddings.'
    )
    parser.add_argument('--hparam_path', default='config/b_on.yml')
    parser.add_argument('--data_name', default='ogbn_arxiv')
    parser.add_argument('--processed_data_dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--out_dir', default='result/coarse_relation')

    parser.add_argument('--split_source', choices=['fixed_cv', 'dataset'], default='fixed_cv')
    parser.add_argument('--split_path', default=None)
    parser.add_argument('--fold_idx', type=int, default=None)
    add_bool_arg(parser, 'create_split_if_missing')
    parser.add_argument('--dataset_val_frac', type=float, default=0.2)

    parser.add_argument('--embeddings_path', default=None)
    add_bool_arg(parser, 'force_export_embeddings')
    parser.add_argument('--embedding_key', default='z_mil')
    parser.add_argument('--export_batch_size', type=int, default=None)

    add_bool_arg(parser, 'train_stage1')
    parser.add_argument('--stage1_checkpoint', default=None)
    add_bool_arg(parser, 'stage1_use_position_head')
    parser.add_argument('--stage1_epochs', type=int, default=None)
    parser.add_argument('--stage1_patience', type=int, default=None)
    parser.add_argument('--stage1_lr', type=float, default=None)
    parser.add_argument('--stage1_weight_decay', type=float, default=None)
    parser.add_argument('--save_stage1_checkpoint', default=None)
    add_bool_arg(parser, 'allow_random_stage1')
    add_bool_arg(parser, 'non_strict_checkpoint')

    parser.add_argument('--models', default='mlp,gcn,appnp')
    parser.add_argument('--feature_norm', choices=['none', 'standard', 'l2'], default='standard')
    parser.add_argument('--relation_epochs', type=int, default=500)
    parser.add_argument('--relation_lr', type=float, default=0.001)
    parser.add_argument('--relation_weight_decay', type=float, default=5e-4)
    parser.add_argument('--relation_hidden_dim', type=int, default=256)
    parser.add_argument('--relation_dropout', type=float, default=0.5)
    parser.add_argument('--relation_patience', type=int, default=80)
    parser.add_argument('--selection_metric', choices=['val_roc_auc', 'val_acc', 'val_loss'], default='val_roc_auc')
    parser.add_argument('--appnp_k', type=int, default=10)
    parser.add_argument('--appnp_alpha', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=20)

    add_bool_arg(parser, 'synthetic_smoke')
    parser.add_argument('--smoke_num_nodes', type=int, default=120)
    parser.add_argument('--smoke_feature_dim', type=int, default=32)
    args = parser.parse_args()
    cli_names = cli_option_names(sys.argv[1:])
    apply_yaml_config(args, cli_names)
    apply_missing_bool_defaults(args)
    return args


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


def load_yaml_experiment_config(hparam_path):
    hp = hparam.HParams()
    hp.from_yaml(hparam_path)
    for key in ('coarse_relation_experiment', 'coarse_relation'):
        cfg = getattr(hp, key, None)
        if isinstance(cfg, dict):
            return cfg
    return {}


def apply_yaml_config(args, cli_names):
    cfg = load_yaml_experiment_config(args.hparam_path)
    if not cfg:
        return
    for key, value in cfg.items():
        if not hasattr(args, key):
            logging.warning('Ignoring unknown coarse relation config key: %s', key)
            continue
        if key in cli_names:
            continue
        setattr(args, key, value)


def apply_missing_bool_defaults(args):
    for key in (
        'create_split_if_missing',
        'force_export_embeddings',
        'train_stage1',
        'stage1_use_position_head',
        'allow_random_stage1',
        'non_strict_checkpoint',
        'synthetic_smoke',
    ):
        if getattr(args, key) is None:
            setattr(args, key, False)


def apply_path_templates(args, fold_idx=None):
    resolved_fold_idx = args.fold_idx if fold_idx is None else fold_idx
    if resolved_fold_idx is None:
        resolved_fold_idx = 'all'
    values = {
        'data_name': args.data_name,
        'fold_idx': resolved_fold_idx,
        'seed': args.seed,
    }
    for key in ('out_dir', 'split_path', 'embeddings_path', 'save_stage1_checkpoint'):
        path = getattr(args, key, None)
        if isinstance(path, str) and path:
            setattr(args, key, path.format(**values))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_hparams(args):
    hparams = hparam.HParams()
    hparams.from_yaml(args.hparam_path)
    hparams_lib.apply_defaults(hparams)
    hparams.data_name = args.data_name
    if args.processed_data_dir:
        hparams.processed_data_dir = args.processed_data_dir
    if args.device:
        hparams.device = args.device
    if hparams.device == 'cuda' and not torch.cuda.is_available():
        logging.warning('CUDA requested but unavailable; falling back to CPU.')
        hparams.device = 'cpu'
    return hparams


def clone_hparams(hparams):
    return hparams_lib.copy_hparams(hparams)


def set_position_head_use(hparams, use_position_head):
    cfg = copy.deepcopy(getattr(hparams, 'position_head', {}) or {})
    cfg['use'] = bool(use_position_head)
    hparams.position_head = cfg


def set_hparam_value(hparams, name, value):
    if value is None:
        return
    if name in hparams:
        hparams.set_hparam(name, value)
    else:
        hparams.add_hparam(name, value)


def prepare_data(hparams):
    data_hparams = clone_hparams(hparams)
    set_position_head_use(data_hparams, True)
    data_hparams.preload_data_to_gpu = False
    loader = GraphDataLoaderWrapper(data_hparams, data_name=hparams.data_name)
    attach_labels_to_all_subgraphs(loader)
    if loader.coarse_adj is None:
        raise RuntimeError('Failed to build coarse graph; coarse_adj is None.')
    return loader


def sync_hparams_from_loader(hparams, loader):
    hparams.channel_list = list(loader._hparams.channel_list)
    if hasattr(loader._hparams, 'max_num_nodes'):
        max_num_nodes = int(loader._hparams.max_num_nodes)
        if 'max_num_nodes' in hparams:
            hparams.set_hparam('max_num_nodes', max_num_nodes)
        else:
            hparams.add_hparam('max_num_nodes', max_num_nodes)


def attach_labels_to_all_subgraphs(loader):
    dataset = loader._dataset_raw
    labels = dataset.get('subgraph_labels', None)
    for idx, graph in enumerate(loader._subgraphs):
        graph.graph['orig_idx'] = int(idx)
        graph.graph.setdefault('subgraph_id', int(idx))
        if labels is not None and idx < len(labels):
            graph.graph['label'] = int(labels[idx])


def build_split(loader, args):
    if args.split_source == 'fixed_cv':
        return build_fixed_cv_split(loader, args)
    return build_dataset_split(loader, args)


def build_fixed_cv_split(loader, args):
    split_path = args.split_path or loader.get_cv_split_path(ensure_dir=args.create_split_if_missing)
    if os.path.exists(split_path):
        manifest = loader.load_cv_split_manifest(split_path)
        logging.info('Loaded CV split manifest: %s', split_path)
    else:
        manifest = loader.build_cv_split_manifest()
        if args.create_split_if_missing:
            ensure_parent_dir(split_path)
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            logging.info('Saved CV split manifest: %s', split_path)
        else:
            logging.warning('CV split manifest not found; using an in-memory split: %s', split_path)

    num_folds = int(loader.cv_num_folds)
    test_fold = int(args.fold_idx)
    if test_fold < 0 or test_fold >= num_folds:
        raise IndexError(f'fold_idx out of range: {test_fold}, num_folds={num_folds}')
    val_fold = (test_fold + 1) % num_folds
    train_folds = [i for i in range(num_folds) if i not in (test_fold, val_fold)]

    def fold_indices(fold_id):
        return [int(v) for v in manifest['folds'][int(fold_id)]['sample_indices']]

    train_indices = sorted(idx for fold_id in train_folds for idx in fold_indices(fold_id))
    val_indices = sorted(fold_indices(val_fold))
    test_indices = sorted(fold_indices(test_fold))
    return {
        'source': 'fixed_cv',
        'fold_idx': test_fold,
        'train_folds': train_folds,
        'val_fold': val_fold,
        'test_fold': test_fold,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
    }


def build_dataset_split(loader, args):
    dataset = loader._dataset_raw
    train_pool = np.asarray(dataset['train_test_split']['train_indices'], dtype=np.int64)
    test_indices = np.asarray(dataset['train_test_split']['test_indices'], dtype=np.int64)
    labels = np.asarray(dataset['subgraph_labels'], dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(args.dataset_val_frac),
        random_state=int(args.seed),
    )
    train_rel_idx, val_rel_idx = next(splitter.split(train_pool, labels[train_pool]))
    train_indices = train_pool[train_rel_idx]
    val_indices = train_pool[val_rel_idx]
    return {
        'source': 'dataset',
        'train_indices': sorted(int(v) for v in train_indices.tolist()),
        'val_indices': sorted(int(v) for v in val_indices.tolist()),
        'test_indices': sorted(int(v) for v in test_indices.tolist()),
    }


def indices_to_masks(num_nodes, split, orig_indices=None):
    if orig_indices is None:
        orig_indices = np.arange(num_nodes, dtype=np.int64)
    elif isinstance(orig_indices, torch.Tensor):
        orig_indices = orig_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        orig_indices = np.asarray(orig_indices, dtype=np.int64)
    if orig_indices.shape[0] != num_nodes:
        raise ValueError(f'orig_indices length mismatch: {orig_indices.shape[0]} vs {num_nodes}')

    masks = {}
    for split_name in ('train', 'val', 'test'):
        indices = set(int(v) for v in split[f'{split_name}_indices'])
        mask = np.asarray([int(orig_idx) in indices for orig_idx in orig_indices], dtype=bool)
        masks[split_name] = torch.from_numpy(mask)
    return masks


def make_stage1_hparams(base_hparams, args, use_position_head):
    stage1_hparams = clone_hparams(base_hparams)
    set_position_head_use(stage1_hparams, use_position_head)
    stage1_hparams.preload_data_to_gpu = False
    set_hparam_value(stage1_hparams, 'epoch', args.stage1_epochs)
    set_hparam_value(stage1_hparams, 'patience', args.stage1_patience)
    set_hparam_value(stage1_hparams, 'learning_rate', args.stage1_lr)
    set_hparam_value(stage1_hparams, 'weight_decay', args.stage1_weight_decay)
    return stage1_hparams


def train_or_load_stage1(loader, base_hparams, split, args):
    device = torch.device(base_hparams.device)
    use_position_head = bool(args.stage1_use_position_head)
    stage1_hparams = make_stage1_hparams(base_hparams, args, use_position_head)
    model = MISGLEncoder(stage1_hparams, data_name=base_hparams.data_name).to(device)

    if args.stage1_checkpoint:
        checkpoint = torch.load(args.stage1_checkpoint, map_location=device)
        state_dict = checkpoint
        for key in ('model_state_dict', 'state_dict', 'model'):
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                break
        model.load_state_dict(state_dict, strict=not args.non_strict_checkpoint)
        logging.info('Loaded stage-1 checkpoint: %s', args.stage1_checkpoint)
        return model, stage1_hparams

    if not args.train_stage1:
        if not args.allow_random_stage1:
            raise ValueError(
                'No embeddings_path/checkpoint was provided. Pass --train_stage1, '
                '--stage1_checkpoint, or --allow_random_stage1 for a debugging run.'
            )
        logging.warning('Using randomly initialized stage-1 model; results are for code debugging only.')
        return model, stage1_hparams

    train_loader = build_graph_loader(stage1_hparams, loader, split['train_indices'], shuffle=True)
    val_loader = build_graph_loader(stage1_hparams, loader, split['val_indices'], shuffle=False)
    model, _, best_val = train_eval_iter(
        model,
        train_loader,
        val_loader,
        writer=None,
        hparams=stage1_hparams,
        dataset_raw=loader._dataset_raw,
    )
    logging.info(
        'Stage-1 best val: epoch=%s acc=%.4f loss=%.4f',
        best_val.get('epoch'),
        best_val.get('acc'),
        best_val.get('loss'),
    )
    if args.save_stage1_checkpoint:
        ensure_parent_dir(args.save_stage1_checkpoint)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'hparams': stage1_hparams.values(),
                'best_val': best_val,
            },
            args.save_stage1_checkpoint,
        )
        logging.info('Saved stage-1 checkpoint: %s', args.save_stage1_checkpoint)
    return model, stage1_hparams


def build_graph_loader(hparams, loader, indices, shuffle):
    graph_list = [loader._subgraphs[int(i)] for i in indices]
    dataset = GraphDataset(hparams, graph_list)
    return DataLoader(
        dataset,
        batch_size=int(hparams.batch_size),
        shuffle=bool(shuffle),
        worker_init_fn=reproducibility.worker_init_fn,
    )


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def export_embeddings(model, export_hparams, loader, args):
    export_hparams = clone_hparams(export_hparams)
    export_hparams.preload_data_to_gpu = False
    set_position_head_use(export_hparams, bool(getattr(model, 'use_position_head', False)))
    batch_size = int(args.export_batch_size or export_hparams.batch_size)
    graph_dataset = GraphDataset(export_hparams, loader._subgraphs)
    graph_loader = DataLoader(
        graph_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=reproducibility.worker_init_fn,
    )

    device = torch.device(export_hparams.device)
    model.eval()
    if getattr(model, 'use_position_head', False):
        model.reset_position_memory()
        with torch.inference_mode():
            for batch in graph_loader:
                model.update_position_memory_from_batch(move_batch_to_device(batch, device))

    features = []
    labels = []
    orig_indices = []
    coarse_node_ids = []
    logits = []
    with torch.inference_mode():
        for batch in graph_loader:
            batch = move_batch_to_device(batch, device)
            model_out, emb = model.forward_with_embeddings(batch)
            if args.embedding_key not in emb:
                available = ', '.join(sorted(emb.keys()))
                raise KeyError(f'Embedding key {args.embedding_key!r} not found. Available: {available}')
            features.append(emb[args.embedding_key].detach().cpu())
            labels.append(batch[g_key.y].view(-1).detach().cpu())
            orig_indices.append(batch[g_key.orig_graph_idx].view(-1).detach().cpu())
            if g_key.coarse_node_id in batch:
                coarse_node_ids.append(batch[g_key.coarse_node_id].view(-1).detach().cpu())
            if isinstance(model_out, dict) and 'ypred_A' in model_out:
                logits.append(model_out['ypred_A'].view(-1).detach().cpu())
            elif isinstance(model_out, torch.Tensor):
                logits.append(model_out.view(-1).detach().cpu())

    payload = {
        'features': torch.cat(features, dim=0),
        'labels': torch.cat(labels, dim=0),
        'orig_indices': torch.cat(orig_indices, dim=0),
        'embedding_key': args.embedding_key,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if coarse_node_ids:
        payload['coarse_node_ids'] = torch.cat(coarse_node_ids, dim=0)
    if logits:
        payload['stage1_logits'] = torch.cat(logits, dim=0)
    return payload


def load_or_export_embeddings(loader, base_hparams, split, args):
    if (
        args.embeddings_path
        and os.path.exists(args.embeddings_path)
        and not args.force_export_embeddings
    ):
        payload = torch.load(args.embeddings_path, map_location='cpu')
        logging.info('Loaded embeddings: %s', args.embeddings_path)
        return payload

    model, stage1_hparams = train_or_load_stage1(loader, base_hparams, split, args)
    payload = export_embeddings(model, stage1_hparams, loader, args)
    if args.embeddings_path:
        ensure_parent_dir(args.embeddings_path)
        torch.save(payload, args.embeddings_path)
        logging.info('Saved embeddings: %s', args.embeddings_path)
    return payload


def normalize_features(features, train_mask, mode):
    if mode == 'none':
        return features
    if mode == 'l2':
        return F.normalize(features, p=2, dim=-1)
    train_features = features[train_mask]
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


def scipy_to_torch_sparse(matrix, device):
    matrix = matrix.tocoo().astype(np.float32, copy=False)
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def normalize_adjacency(coarse_adj, mode='symmetric', add_self_loop=True):
    adj = coarse_adj.tocsr().astype(np.float32, copy=True)
    if add_self_loop:
        adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format='csr')
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    if mode == 'row':
        inv_degree = np.zeros_like(degree)
        valid = degree > 0
        inv_degree[valid] = 1.0 / degree[valid]
        return sp.diags(inv_degree, format='csr') @ adj
    inv_sqrt = np.zeros_like(degree)
    valid = degree > 0
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    d_inv = sp.diags(inv_sqrt, format='csr')
    return d_inv @ adj @ d_inv


class RelationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, adj_norm=None):
        del adj_norm
        return self.net(features).view(-1)


class GCNRelation(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, adj_norm):
        h = torch.sparse.mm(adj_norm, features)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = torch.sparse.mm(adj_norm, h)
        return self.fc2(h).view(-1)


class SAGERelation(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, adj_norm):
        neigh = torch.sparse.mm(adj_norm, features)
        h = F.relu(self.fc1(torch.cat([features, neigh], dim=-1)))
        h = self.dropout(h)
        neigh_h = torch.sparse.mm(adj_norm, h)
        return self.fc2(torch.cat([h, neigh_h], dim=-1)).view(-1)


class APPNPRelation(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, k_steps, alpha):
        super().__init__()
        self.predictor = RelationMLP(input_dim, hidden_dim, dropout)
        self.k_steps = int(k_steps)
        self.alpha = float(alpha)

    def forward(self, features, adj_norm):
        logits0 = self.predictor(features, None).view(-1, 1)
        logits = logits0
        for _ in range(self.k_steps):
            logits = (1.0 - self.alpha) * torch.sparse.mm(adj_norm, logits) + self.alpha * logits0
        return logits.view(-1)


def build_relation_model(model_name, input_dim, args):
    name = model_name.strip().lower()
    if name == 'mlp':
        return RelationMLP(input_dim, args.relation_hidden_dim, args.relation_dropout)
    if name == 'gcn':
        return GCNRelation(input_dim, args.relation_hidden_dim, args.relation_dropout)
    if name == 'sage':
        return SAGERelation(input_dim, args.relation_hidden_dim, args.relation_dropout)
    if name == 'appnp':
        return APPNPRelation(
            input_dim,
            args.relation_hidden_dim,
            args.relation_dropout,
            args.appnp_k,
            args.appnp_alpha,
        )
    raise ValueError(f'Unsupported relation model: {model_name}')


def binary_metrics_from_logits(logits, labels, mask):
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if int(mask.sum().item()) == 0:
        return {}
    local_logits = logits[mask]
    local_labels = labels[mask].float()
    loss = F.binary_cross_entropy_with_logits(local_logits, local_labels).item()
    probs = torch.sigmoid(local_logits).detach().cpu().numpy()
    y_true = local_labels.detach().cpu().numpy().astype(np.int64)
    y_pred = (probs > 0.5).astype(np.int64)
    out = {
        'loss': float(loss),
        'acc': float(metrics.accuracy_score(y_true, y_pred)),
        'prec': float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        'rec': float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        'positive_rate': float(y_true.mean()) if y_true.size else 0.0,
        'pred_positive_rate': float(y_pred.mean()) if y_pred.size else 0.0,
        'num_examples': int(y_true.size),
    }
    if np.unique(y_true).size == 2:
        out['roc_auc'] = float(metrics.roc_auc_score(y_true, probs))
        out['average_precision'] = float(metrics.average_precision_score(y_true, probs))
    else:
        out['roc_auc'] = None
        out['average_precision'] = None
    return out


def evaluate_relation_model(model, features, labels, masks, adj_for_model):
    model.eval()
    with torch.inference_mode():
        logits = model(features, adj_for_model)
    result = {}
    for split_name, mask in masks.items():
        metric = binary_metrics_from_logits(logits, labels, mask)
        result[split_name] = metric
    return result, logits.detach().cpu()


def is_better(current_metrics, best_metrics, selection_metric):
    if best_metrics is None:
        return True
    split_name, metric_name = selection_metric.split('_', 1)
    current = current_metrics[split_name].get(metric_name)
    best = best_metrics[split_name].get(metric_name)
    if current is None:
        current = -float('inf')
    if best is None:
        best = -float('inf')
    if metric_name == 'loss':
        return float(current) < float(best) - 1e-8
    return float(current) > float(best) + 1e-8


def train_one_relation_model(model_name, features, labels, masks, adj_tensors, args):
    device = features.device
    model = build_relation_model(model_name, features.size(1), args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.relation_lr),
        weight_decay=float(args.relation_weight_decay),
    )
    train_mask = masks['train'].to(device=device, dtype=torch.bool)
    labels = labels.to(device=device, dtype=torch.float32)
    adj_for_model = adj_tensors['row'] if model_name == 'sage' else adj_tensors['symmetric']

    best_state = None
    best_metrics = None
    best_epoch = -1
    no_improve = 0
    for epoch in range(int(args.relation_epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj_for_model)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        current_metrics, _ = evaluate_relation_model(model, features, labels, masks, adj_for_model)
        if is_better(current_metrics, best_metrics, args.selection_metric):
            best_metrics = current_metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if args.log_interval > 0 and (epoch % args.log_interval == 0 or epoch == args.relation_epochs - 1):
            val_metrics = current_metrics['val']
            logging.info(
                '%s epoch=%d train_loss=%.4f val_acc=%.4f val_auc=%s',
                model_name,
                epoch,
                float(loss.item()),
                float(val_metrics.get('acc', 0.0)),
                format_optional_float(val_metrics.get('roc_auc')),
            )
        if no_improve >= int(args.relation_patience):
            logging.info('%s early stop at epoch=%d', model_name, epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics, final_logits = evaluate_relation_model(model, features, labels, masks, adj_for_model)
    return {
        'model_name': model_name,
        'best_epoch': int(best_epoch),
        'best_metrics': best_metrics,
        'final_metrics': final_metrics,
        'logits': final_logits,
        'state_dict': model.state_dict(),
    }


def format_optional_float(value):
    if value is None:
        return 'n/a'
    return f'{float(value):.4f}'


def save_predictions_csv(path, orig_indices, labels, split_names, model_outputs):
    fieldnames = ['orig_idx', 'label', 'split']
    for model_name in model_outputs:
        fieldnames.append(f'{model_name}_prob')
        fieldnames.append(f'{model_name}_pred')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(len(labels)):
            row = {
                'orig_idx': int(orig_indices[row_idx]),
                'label': int(labels[row_idx]),
                'split': split_names[row_idx],
            }
            for model_name, output in model_outputs.items():
                prob = float(torch.sigmoid(output['logits'][row_idx]).item())
                row[f'{model_name}_prob'] = prob
                row[f'{model_name}_pred'] = int(prob > 0.5)
            writer.writerow(row)


def split_name_array(num_nodes, split, orig_indices=None):
    if orig_indices is None:
        orig_indices = np.arange(num_nodes, dtype=np.int64)
    elif isinstance(orig_indices, torch.Tensor):
        orig_indices = orig_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        orig_indices = np.asarray(orig_indices, dtype=np.int64)
    names = np.full((num_nodes,), 'unused', dtype=object)
    split_sets = {
        name: set(int(idx) for idx in split[f'{name}_indices'])
        for name in ('train', 'val', 'test')
    }
    for row_idx, orig_idx in enumerate(orig_indices):
        for name, index_set in split_sets.items():
            if int(orig_idx) in index_set:
                names[row_idx] = name
                break
    return names


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def run_relation_experiment(
    features,
    labels,
    coarse_adj,
    split,
    args,
    out_dir,
    metadata=None,
    orig_indices=None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    labels = labels.long()
    masks = indices_to_masks(features.size(0), split, orig_indices=orig_indices)
    train_mask = masks['train']
    features = normalize_features(features.float(), train_mask, args.feature_norm).to(device)
    labels_device = labels.to(device)
    masks_device = {key: value.to(device) for key, value in masks.items()}

    adj_symmetric = scipy_to_torch_sparse(
        normalize_adjacency(coarse_adj, mode='symmetric', add_self_loop=True),
        device,
    )
    adj_row = scipy_to_torch_sparse(
        normalize_adjacency(coarse_adj, mode='row', add_self_loop=False),
        device,
    )
    adj_tensors = {
        'symmetric': adj_symmetric,
        'row': adj_row,
    }

    model_names = [name.strip().lower() for name in args.models.split(',') if name.strip()]
    model_outputs = {}
    result_summary = {
        'metadata': metadata or {},
        'split': split,
        'feature_shape': list(features.shape),
        'coarse_graph': {
            'num_nodes': int(coarse_adj.shape[0]),
            'num_edges_directed': int(coarse_adj.nnz),
            'avg_degree_directed': float(coarse_adj.nnz / max(coarse_adj.shape[0], 1)),
        },
        'args': vars(args),
        'models': {},
    }
    for model_name in model_names:
        output = train_one_relation_model(
            model_name,
            features,
            labels_device,
            masks_device,
            adj_tensors,
            args,
        )
        model_outputs[model_name] = output
        result_summary['models'][model_name] = {
            'best_epoch': output['best_epoch'],
            'best_metrics': output['best_metrics'],
            'final_metrics': output['final_metrics'],
        }
        torch.save(
            {
                'state_dict': output['state_dict'],
                'model_name': model_name,
                'input_dim': int(features.size(1)),
                'args': vars(args),
            },
            os.path.join(out_dir, f'{model_name}_relation_model.pt'),
        )

    result_path = os.path.join(out_dir, 'relation_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(result_summary), f, indent=2, ensure_ascii=False)

    if orig_indices is None:
        orig_indices = np.arange(features.size(0), dtype=np.int64)
    elif isinstance(orig_indices, torch.Tensor):
        orig_indices = orig_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        orig_indices = np.asarray(orig_indices, dtype=np.int64)
    split_names = split_name_array(features.size(0), split, orig_indices=orig_indices)
    predictions_path = os.path.join(out_dir, 'relation_predictions.csv')
    save_predictions_csv(predictions_path, orig_indices, labels.cpu().numpy(), split_names, model_outputs)
    logging.info('Saved relation results: %s', result_path)
    logging.info('Saved relation predictions: %s', predictions_path)
    return result_summary


def synthetic_smoke(args):
    rng = np.random.default_rng(args.seed)
    num_nodes = int(args.smoke_num_nodes)
    feature_dim = int(args.smoke_feature_dim)
    labels = torch.from_numpy(rng.integers(0, 2, size=num_nodes).astype(np.int64))
    features = torch.from_numpy(rng.normal(size=(num_nodes, feature_dim)).astype(np.float32))
    rows = []
    cols = []
    for idx in range(num_nodes):
        neigh = rng.choice(num_nodes, size=min(6, num_nodes), replace=False)
        rows.extend([idx] * len(neigh))
        cols.extend(neigh.tolist())
    data = np.ones(len(rows), dtype=np.float32)
    coarse_adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    train_end = int(num_nodes * 0.6)
    val_end = int(num_nodes * 0.8)
    split = {
        'source': 'synthetic_smoke',
        'train_indices': list(range(0, train_end)),
        'val_indices': list(range(train_end, val_end)),
        'test_indices': list(range(val_end, num_nodes)),
    }
    args.relation_epochs = min(int(args.relation_epochs), 5)
    args.relation_patience = min(int(args.relation_patience), 3)
    return run_relation_experiment(
        features,
        labels,
        coarse_adj,
        split,
        args,
        args.out_dir,
        metadata={'synthetic_smoke': True},
        orig_indices=None,
    )


def resolve_fold_indices(loader, args):
    if args.split_source != 'fixed_cv':
        return [None]
    if args.fold_idx is not None:
        return [int(args.fold_idx)]
    return list(range(int(loader.cv_num_folds)))


def make_run_args(args, fold_idx):
    run_args = copy.copy(args)
    if fold_idx is not None:
        run_args.fold_idx = int(fold_idx)
        apply_path_templates(run_args, fold_idx=int(fold_idx))
    else:
        run_args.fold_idx = None
        apply_path_templates(run_args, fold_idx='dataset')
    return run_args


def run_output_dir(args):
    if args.split_source == 'fixed_cv':
        return os.path.join(args.out_dir, args.data_name, f'fold_{args.fold_idx}')
    return os.path.join(args.out_dir, args.data_name, 'dataset_split')


def run_one_split(args, hparams, loader, fold_idx):
    run_args = make_run_args(args, fold_idx)
    out_dir = run_output_dir(run_args)
    split = build_split(loader, run_args)
    logging.info(
        'Split sizes: train=%d val=%d test=%d',
        len(split['train_indices']),
        len(split['val_indices']),
        len(split['test_indices']),
    )
    payload = load_or_export_embeddings(loader, hparams, split, run_args)
    features = payload['features'].float()
    labels = payload['labels'].long()
    if features.size(0) != loader.coarse_adj.shape[0]:
        raise ValueError(
            f'Embedding rows ({features.size(0)}) do not match coarse nodes ({loader.coarse_adj.shape[0]}).'
        )

    if run_args.embeddings_path and not os.path.exists(run_args.embeddings_path):
        ensure_parent_dir(run_args.embeddings_path)
        torch.save(payload, run_args.embeddings_path)

    metadata = {
        'data_name': run_args.data_name,
        'embedding_key': payload.get('embedding_key', run_args.embedding_key),
        'embeddings_path': run_args.embeddings_path,
        'coarse_graph_metadata': loader.coarse_graph_metadata,
    }
    result = run_relation_experiment(
        features,
        labels,
        loader.coarse_adj,
        split,
        run_args,
        out_dir,
        metadata=metadata,
        orig_indices=payload.get('orig_indices'),
    )
    for model_name, model_result in result['models'].items():
        test = model_result['final_metrics']['test']
        logging.info(
            'FINAL %s fold=%s test_acc=%.4f test_auc=%s test_f1=%.4f',
            model_name,
            run_args.fold_idx,
            float(test.get('acc', 0.0)),
            format_optional_float(test.get('roc_auc')),
            float(test.get('f1', 0.0)),
        )
    return {
        'fold_idx': run_args.fold_idx,
        'out_dir': out_dir,
        'result': result,
    }


def aggregate_fold_results(fold_outputs):
    model_values = {}
    for fold_output in fold_outputs:
        fold_idx = fold_output.get('fold_idx')
        models = fold_output['result'].get('models', {})
        for model_name, model_result in models.items():
            test = model_result['final_metrics'].get('test', {})
            bucket = model_values.setdefault(model_name, {})
            for metric_name in ('acc', 'roc_auc', 'average_precision', 'f1', 'prec', 'rec', 'loss'):
                value = test.get(metric_name)
                if value is None:
                    continue
                bucket.setdefault(metric_name, []).append({
                    'fold_idx': fold_idx,
                    'value': float(value),
                })

    summary = {}
    for model_name, metric_values in model_values.items():
        summary[model_name] = {}
        for metric_name, entries in metric_values.items():
            values = np.asarray([entry['value'] for entry in entries], dtype=np.float64)
            summary[model_name][metric_name] = {
                'mean': float(values.mean()) if values.size else None,
                'std': float(values.std(ddof=0)) if values.size else None,
                'num_folds': int(values.size),
                'values': entries,
            }
    return summary


def save_all_folds_summary(args, fold_outputs):
    summary_args = copy.copy(args)
    apply_path_templates(summary_args, fold_idx='all')
    base_out_dir = os.path.join(summary_args.out_dir, summary_args.data_name)
    ensure_parent_dir(os.path.join(base_out_dir, 'dummy'))
    os.makedirs(base_out_dir, exist_ok=True)
    payload = {
        'data_name': summary_args.data_name,
        'fold_indices': [output.get('fold_idx') for output in fold_outputs],
        'summary': aggregate_fold_results(fold_outputs),
        'fold_result_paths': [
            os.path.join(output['out_dir'], 'relation_results.json')
            for output in fold_outputs
        ],
    }
    path = os.path.join(base_out_dir, 'relation_results_all_folds.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)
    logging.info('Saved all-fold summary: %s', path)
    return path


def main():
    args = parse_args()
    setup_logging()
    reproducibility.set_seed(args.seed, cuda_deterministic=False)

    if args.synthetic_smoke:
        apply_path_templates(args, fold_idx=args.fold_idx if args.fold_idx is not None else 'smoke')
        synthetic_smoke(args)
        return

    hparams = load_hparams(args)
    args.device = hparams.device
    loader = prepare_data(hparams)
    sync_hparams_from_loader(hparams, loader)
    fold_indices = resolve_fold_indices(loader, args)
    logging.info('Running folds: %s', fold_indices)
    fold_outputs = []
    for fold_idx in fold_indices:
        logging.info('===== Start fold %s =====', fold_idx)
        fold_outputs.append(run_one_split(args, hparams, loader, fold_idx))
    if len(fold_outputs) > 1:
        save_all_folds_summary(args, fold_outputs)


if __name__ == '__main__':
    main()
