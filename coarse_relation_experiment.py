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
    parser.add_argument('--relation_graph', choices=['coarse', 'zmil_knn'], default='coarse')
    parser.add_argument('--knn_k', type=int, default=16)
    parser.add_argument('--knn_metric', choices=['cosine'], default='cosine')
    parser.add_argument('--knn_weight_mode', choices=['positive_cosine', 'cosine_shift', 'binary'], default='positive_cosine')
    parser.add_argument('--knn_min_similarity', type=float, default=None)
    parser.add_argument('--knn_batch_size', type=int, default=512)
    add_bool_arg(parser, 'knn_symmetrize')
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
    parser.add_argument('--gated_sage_gate_hidden_dim', type=int, default=128)
    parser.add_argument('--edge_gate_hidden_dim', type=int, default=128)
    parser.add_argument('--edge_node_gate_hidden_dim', type=int, default=32)
    parser.add_argument('--edge_aux_loss_weight', type=float, default=0.2)
    add_bool_arg(parser, 'edge_aux_balance')
    parser.add_argument('--edge_residual_init', type=float, default=0.1)
    parser.add_argument('--edge_diagnostic_bins', type=int, default=10)
    parser.add_argument('--comparison_baselines', default='mlp,edge_self_only')
    parser.add_argument('--threshold_metric', choices=['f1', 'acc'], default='f1')
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
        'knn_symmetrize',
        'edge_aux_balance',
        'synthetic_smoke',
    ):
        if getattr(args, key) is None:
            setattr(args, key, key in ('knn_symmetrize', 'edge_aux_balance'))


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


def bounded_weight_feature(values):
    values = torch.as_tensor(values, dtype=torch.float32).clamp_min(0.0)
    transformed = torch.log1p(values)
    if transformed.numel() == 0:
        return transformed
    return transformed / transformed.max().clamp_min(1e-12)


def build_edge_graph_tensors(relation_adj, coarse_reference_adj, raw_features, device):
    relation_coo = relation_adj.tocoo().astype(np.float32, copy=False)
    valid = relation_coo.row != relation_coo.col
    src_np = relation_coo.row[valid].astype(np.int64, copy=False)
    dst_np = relation_coo.col[valid].astype(np.int64, copy=False)
    candidate_weight_np = relation_coo.data[valid].astype(np.float32, copy=False)

    coarse_reference = coarse_reference_adj.tocsr().astype(np.float32, copy=False)
    if src_np.size:
        coarse_weight_np = coarse_reference[src_np, dst_np].A1.astype(np.float32, copy=False)
    else:
        coarse_weight_np = np.empty((0,), dtype=np.float32)

    raw_features_cpu = torch.as_tensor(raw_features, dtype=torch.float32).detach().cpu()
    raw_features_norm = F.normalize(raw_features_cpu, p=2, dim=-1)
    src_cpu = torch.from_numpy(src_np)
    dst_cpu = torch.from_numpy(dst_np)
    cosine_cpu = (
        (raw_features_norm[src_cpu] * raw_features_norm[dst_cpu]).sum(dim=-1)
        if src_np.size
        else torch.empty((0,), dtype=torch.float32)
    )
    candidate_weight_cpu = torch.from_numpy(candidate_weight_np)
    coarse_weight_cpu = torch.from_numpy(coarse_weight_np)
    edge_graph = {
        'src': src_cpu.to(device),
        'dst': dst_cpu.to(device),
        'cosine': cosine_cpu.to(device),
        'candidate_weight': candidate_weight_cpu.to(device),
        'candidate_weight_feature': bounded_weight_feature(candidate_weight_cpu).to(device),
        'coarse_weight': coarse_weight_cpu.to(device),
        'coarse_weight_feature': bounded_weight_feature(coarse_weight_cpu).to(device),
    }
    logging.info(
        'Prepared edge-level graph tensors: directed_edges=%d, coarse_overlap=%d',
        int(src_np.size),
        int(np.count_nonzero(coarse_weight_np)),
    )
    return edge_graph


def build_zmil_knn_adjacency(features, args):
    if isinstance(features, torch.Tensor):
        features_cpu = features.detach().cpu().float()
    else:
        features_cpu = torch.as_tensor(features, dtype=torch.float32)
    num_nodes = int(features_cpu.size(0))
    if num_nodes <= 1:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32), {
            'type': 'zmil_knn',
            'num_nodes': num_nodes,
            'knn_k': 0,
        }

    knn_k = min(int(args.knn_k), num_nodes - 1)
    if knn_k <= 0:
        raise ValueError(f'knn_k must be positive, got {args.knn_k}')
    batch_size = max(int(args.knn_batch_size), 1)
    min_similarity = args.knn_min_similarity
    min_similarity = None if min_similarity is None else float(min_similarity)

    x = F.normalize(features_cpu, p=2, dim=-1)
    rows = []
    cols = []
    data = []
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        sims = x[start:end] @ x.t()
        local_rows = torch.arange(end - start, dtype=torch.long)
        sims[local_rows, torch.arange(start, end, dtype=torch.long)] = -float('inf')
        values, indices = torch.topk(sims, k=knn_k, dim=1, largest=True, sorted=False)
        values_np = values.numpy()
        indices_np = indices.numpy()
        for local_idx in range(end - start):
            src = start + local_idx
            for rank in range(knn_k):
                sim = float(values_np[local_idx, rank])
                if not np.isfinite(sim):
                    continue
                if min_similarity is not None and sim < min_similarity:
                    continue
                if args.knn_weight_mode == 'binary':
                    weight = 1.0
                elif args.knn_weight_mode == 'cosine_shift':
                    weight = 0.5 * (sim + 1.0)
                else:
                    weight = max(sim, 0.0)
                if weight <= 0.0:
                    continue
                rows.append(src)
                cols.append(int(indices_np[local_idx, rank]))
                data.append(weight)

    adj = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    adj.eliminate_zeros()
    if bool(args.knn_symmetrize):
        adj = adj.maximum(adj.T).tocsr()
    metadata = {
        'type': 'zmil_knn',
        'num_nodes': int(num_nodes),
        'num_edges_directed': int(adj.nnz),
        'knn_k': int(knn_k),
        'knn_metric': str(args.knn_metric),
        'knn_weight_mode': str(args.knn_weight_mode),
        'knn_min_similarity': min_similarity,
        'knn_batch_size': int(batch_size),
        'knn_symmetrize': bool(args.knn_symmetrize),
    }
    logging.info(
        'Built z_mil kNN graph: nodes=%d, directed_edges=%d, k=%d, symmetrize=%s',
        num_nodes,
        int(adj.nnz),
        int(knn_k),
        bool(args.knn_symmetrize),
    )
    return adj, metadata


def build_relation_adjacency(base_coarse_adj, features, args):
    relation_graph = str(args.relation_graph).strip().lower()
    if relation_graph == 'coarse':
        metadata = {
            'type': 'coarse',
            'num_nodes': int(base_coarse_adj.shape[0]),
            'num_edges_directed': int(base_coarse_adj.nnz),
        }
        return base_coarse_adj, metadata
    if relation_graph == 'zmil_knn':
        return build_zmil_knn_adjacency(features, args)
    raise ValueError(f'Unsupported relation_graph: {args.relation_graph!r}')


def align_payload_to_coarse_nodes(payload, num_nodes):
    features = payload['features'].detach().cpu().float()
    labels = payload['labels'].detach().cpu().long()
    orig_indices = payload.get('orig_indices')
    if orig_indices is None:
        return features, labels, None
    if isinstance(orig_indices, torch.Tensor):
        orig_indices = orig_indices.detach().cpu().long()
    else:
        orig_indices = torch.as_tensor(orig_indices, dtype=torch.long)
    expected = torch.arange(num_nodes, dtype=torch.long)
    if orig_indices.numel() == num_nodes and torch.equal(orig_indices, expected):
        return features, labels, orig_indices

    if orig_indices.numel() != features.size(0):
        raise ValueError(
            f'orig_indices length ({orig_indices.numel()}) does not match feature rows ({features.size(0)}).'
        )
    aligned_features = features.new_zeros((num_nodes, features.size(1)))
    aligned_labels = labels.new_full((num_nodes,), -1)
    valid = (orig_indices >= 0) & (orig_indices < num_nodes)
    if int(valid.sum().item()) != num_nodes:
        missing_count = num_nodes - int(valid.sum().item())
        raise ValueError(f'Embedding payload does not cover all coarse nodes, missing_count={missing_count}.')
    aligned_features[orig_indices[valid]] = features[valid]
    aligned_labels[orig_indices[valid]] = labels[valid]
    return aligned_features, aligned_labels, expected


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


class GatedSAGERelation(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, gate_hidden_dim):
        super().__init__()
        self.gate1 = nn.Sequential(
            nn.Linear(input_dim * 3, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.fc1 = nn.Linear(input_dim * 2 + 1, hidden_dim)
        self.gate2 = nn.Sequential(
            nn.Linear(hidden_dim * 3, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.fc2 = nn.Linear(hidden_dim * 2 + 1, 1)
        self.dropout = nn.Dropout(dropout)
        self._latest_diagnostics = {}

    @staticmethod
    def _gate_input(self_features, neighbor_features):
        return torch.cat(
            [
                self_features,
                neighbor_features,
                torch.abs(self_features - neighbor_features),
            ],
            dim=-1,
        )

    def forward(self, features, adj_norm):
        neigh = torch.sparse.mm(adj_norm, features)
        gate1 = torch.sigmoid(self.gate1(self._gate_input(features, neigh)))
        h = F.relu(self.fc1(torch.cat([features, gate1 * neigh, gate1], dim=-1)))
        h = self.dropout(h)

        neigh_h = torch.sparse.mm(adj_norm, h)
        gate2 = torch.sigmoid(self.gate2(self._gate_input(h, neigh_h)))
        logits = self.fc2(torch.cat([h, gate2 * neigh_h, gate2], dim=-1)).view(-1)

        self._latest_diagnostics = {
            'gate1': gate1.detach().view(-1).cpu(),
            'gate2': gate2.detach().view(-1).cpu(),
        }
        return logits

    def diagnostics(self):
        return dict(self._latest_diagnostics)


class EdgeSelfOnlyRelation(nn.Module):
    """Self-only ablation with the same self branch as the edge relation models."""

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.self_fc = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self._latest_diagnostics = {}

    def forward(self, features, edge_graph=None):
        del edge_graph
        self_hidden = F.relu(self.self_fc(features))
        logits = self.classifier(self.dropout(self_hidden)).view(-1)
        self_prob = torch.sigmoid(self.classifier(self_hidden).view(-1))
        self._latest_diagnostics = {
            'self_probability': self_prob.detach().cpu(),
            'self_margin': (2.0 * torch.abs(self_prob - 0.5)).detach().cpu(),
        }
        return logits

    def diagnostics(self):
        return dict(self._latest_diagnostics)


class EdgeGateResidualRelation(nn.Module):
    """Filter edges, preserve message magnitude, then conditionally correct nodes."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
        edge_gate_hidden_dim,
        residual_init,
        balance_auxiliary=True,
        node_gate_hidden_dim=32,
        use_node_gate=True,
    ):
        super().__init__()
        self.self_fc = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        edge_feature_dim = input_dim * 4 + 3
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_gate_hidden_dim, 1),
        )
        self.message_fc = nn.Linear(input_dim * 4 + 2, hidden_dim)
        self.node_gate = nn.Sequential(
            nn.Linear(6, node_gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_gate_hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.balance_auxiliary = bool(balance_auxiliary)
        self.use_node_gate = bool(use_node_gate)
        residual_init = min(max(float(residual_init), 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(
            torch.tensor(np.log(residual_init / (1.0 - residual_init)), dtype=torch.float32)
        )
        self._latest_diagnostics = {}
        self._latest_edge_diagnostics = {}
        self._latest_edge_reliability = None
        self._latest_edge_graph = None

    @staticmethod
    def _edge_features(features, edge_graph):
        src = edge_graph['src']
        dst = edge_graph['dst']
        src_features = features[src]
        dst_features = features[dst]
        return torch.cat(
            [
                src_features,
                dst_features,
                torch.abs(src_features - dst_features),
                src_features * dst_features,
                edge_graph['cosine'].view(-1, 1),
                edge_graph['candidate_weight_feature'].view(-1, 1),
                edge_graph['coarse_weight_feature'].view(-1, 1),
            ],
            dim=-1,
        )

    @staticmethod
    def _node_edge_mean(values, src, num_nodes):
        sums = values.new_zeros(num_nodes)
        counts = values.new_zeros(num_nodes)
        sums.index_add_(0, src, values)
        counts.index_add_(0, src, torch.ones_like(values))
        return sums / counts.clamp_min(1.0)

    @staticmethod
    def _node_edge_max(values, src, num_nodes):
        maximum = values.new_zeros(num_nodes)
        if values.numel():
            maximum.scatter_reduce_(0, src, values, reduce='amax', include_self=True)
        return maximum

    @staticmethod
    def _aggregate_neighbors(features, edge_graph, reliability):
        src = edge_graph['src']
        dst = edge_graph['dst']
        candidate_weight = edge_graph['candidate_weight'].clamp_min(0.0)
        effective_weight = reliability * candidate_weight

        weighted_neighbor_sum = features.new_zeros(features.shape)
        weighted_neighbor_sum.index_add_(0, src, effective_weight.view(-1, 1) * features[dst])
        candidate_weight_sum = features.new_zeros(features.size(0))
        candidate_weight_sum.index_add_(0, src, candidate_weight)
        effective_weight_sum = features.new_zeros(features.size(0))
        effective_weight_sum.index_add_(0, src, effective_weight)

        # Divide by the original candidate mass so low reliability shrinks the message.
        neighbor = weighted_neighbor_sum / candidate_weight_sum.clamp_min(1e-12).view(-1, 1)
        effective_weight_ratio = (
            effective_weight_sum / candidate_weight_sum.clamp_min(1e-12)
        )
        return (
            neighbor,
            candidate_weight_sum,
            effective_weight_sum,
            effective_weight_ratio,
            effective_weight,
        )

    def forward(self, features, edge_graph):
        src = edge_graph['src']
        reliability = torch.sigmoid(self.edge_gate(self._edge_features(features, edge_graph))).view(-1)
        (
            neighbor,
            candidate_weight_sum,
            effective_weight_sum,
            effective_weight_ratio,
            effective_weight,
        ) = self._aggregate_neighbors(features, edge_graph, reliability)
        reliability_mean = self._node_edge_mean(reliability, src, features.size(0))
        reliability_max = self._node_edge_max(reliability, src, features.size(0))

        message_input = torch.cat(
            [
                features,
                neighbor,
                torch.abs(features - neighbor),
                features * neighbor,
                candidate_weight_sum.view(-1, 1),
                effective_weight_ratio.view(-1, 1),
            ],
            dim=-1,
        )
        self_hidden = F.relu(self.self_fc(features))
        neighbor_hidden = F.relu(self.self_fc(neighbor))
        self_logits = self.classifier(self_hidden).view(-1)
        neighbor_logits = self.classifier(neighbor_hidden).view(-1)
        self_prob = torch.sigmoid(self_logits)
        neighbor_prob = torch.sigmoid(neighbor_logits)
        self_margin = 2.0 * torch.abs(self_prob - 0.5)
        prediction_agreement = 1.0 - torch.abs(self_prob - neighbor_prob)
        has_candidate = (candidate_weight_sum > 0).float()

        message = torch.tanh(self.message_fc(message_input))
        node_gate_input = torch.stack(
            [
                self_margin,
                reliability_mean,
                reliability_max,
                effective_weight_ratio,
                prediction_agreement,
                has_candidate,
            ],
            dim=-1,
        )
        if self.use_node_gate:
            correction_gate = torch.sigmoid(self.node_gate(node_gate_input)).view(-1)
        else:
            correction_gate = features.new_ones(features.size(0))
        residual_scale = torch.sigmoid(self.residual_logit)
        applied_correction = residual_scale * correction_gate.view(-1, 1) * message
        hidden = self_hidden + applied_correction
        logits = self.classifier(self.dropout(hidden)).view(-1)

        reliability_detached = reliability.detach()
        effective_weight_detached = effective_weight.detach()
        self._latest_diagnostics = {
            'self_probability': self_prob.detach().cpu(),
            'self_margin': self_margin.detach().cpu(),
            'neighbor_probability': neighbor_prob.detach().cpu(),
            'self_neighbor_probability_gap': torch.abs(
                self_prob - neighbor_prob
            ).detach().cpu(),
            'edge_reliability_mean': reliability_mean.detach().cpu(),
            'edge_reliability_max': reliability_max.detach().cpu(),
            'candidate_edge_weight_sum': candidate_weight_sum.detach().cpu(),
            'effective_edge_weight_sum': effective_weight_sum.detach().cpu(),
            'effective_edge_weight_ratio': effective_weight_ratio.detach().cpu(),
            'neighbor_message_norm': neighbor.detach().norm(dim=-1).cpu(),
            'residual_message_norm': message.detach().norm(dim=-1).cpu(),
            'correction_gate': correction_gate.detach().cpu(),
            'applied_correction_norm': applied_correction.detach().norm(dim=-1).cpu(),
            'residual_scale': residual_scale.detach().expand(features.size(0)).cpu(),
        }
        self._latest_edge_diagnostics = {
            key: value.detach().cpu()
            for key, value in edge_graph.items()
            if isinstance(value, torch.Tensor)
        }
        self._latest_edge_diagnostics['reliability'] = reliability_detached.cpu()
        self._latest_edge_diagnostics['effective_weight'] = effective_weight_detached.cpu()
        self._latest_edge_reliability = reliability
        self._latest_edge_graph = edge_graph
        return logits

    def auxiliary_loss(self, labels, train_mask):
        if self._latest_edge_reliability is None or self._latest_edge_graph is None:
            return labels.new_zeros((), dtype=torch.float32)
        src = self._latest_edge_graph['src']
        dst = self._latest_edge_graph['dst']
        supervised = train_mask[src] & train_mask[dst]
        if int(supervised.sum().item()) == 0:
            return self._latest_edge_reliability.sum() * 0.0
        edge_targets = (labels[src] == labels[dst]).float()
        local_targets = edge_targets[supervised]
        sample_weight = None
        if self.balance_auxiliary:
            positive_rate = local_targets.mean()
            if 0.0 < float(positive_rate.item()) < 1.0:
                positive_weight = 0.5 / positive_rate
                negative_weight = 0.5 / (1.0 - positive_rate)
                sample_weight = torch.where(
                    local_targets > 0.5,
                    positive_weight,
                    negative_weight,
                )
        return F.binary_cross_entropy(
            self._latest_edge_reliability[supervised],
            local_targets,
            weight=sample_weight,
        )

    def diagnostics(self):
        return dict(self._latest_diagnostics)

    def edge_diagnostics(self):
        return dict(self._latest_edge_diagnostics)


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
    if name in ('gated_sage', 'position_sage'):
        return GatedSAGERelation(
            input_dim,
            args.relation_hidden_dim,
            args.relation_dropout,
            int(args.gated_sage_gate_hidden_dim),
        )
    if name == 'edge_self_only':
        return EdgeSelfOnlyRelation(
            input_dim,
            args.relation_hidden_dim,
            args.relation_dropout,
        )
    if name in (
        'edge_gate_residual',
        'edge_gate',
        'edge_gate_scaled',
        'edge_gate_node_residual',
    ):
        return EdgeGateResidualRelation(
            input_dim,
            args.relation_hidden_dim,
            args.relation_dropout,
            int(args.edge_gate_hidden_dim),
            float(args.edge_residual_init),
            bool(args.edge_aux_balance),
            int(args.edge_node_gate_hidden_dim),
            name not in ('edge_gate_scaled',),
        )
    if name == 'appnp':
        return APPNPRelation(
            input_dim,
            args.relation_hidden_dim,
            args.relation_dropout,
            args.appnp_k,
            args.appnp_alpha,
        )
    raise ValueError(f'Unsupported relation model: {model_name}')


def binary_metrics_from_logits(logits, labels, mask, threshold=0.5):
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if int(mask.sum().item()) == 0:
        return {}
    local_logits = logits[mask]
    local_labels = labels[mask].float()
    loss = F.binary_cross_entropy_with_logits(local_logits, local_labels).item()
    probs = torch.sigmoid(local_logits).detach().cpu().numpy()
    y_true = local_labels.detach().cpu().numpy().astype(np.int64)
    threshold = float(threshold)
    y_pred = (probs > threshold).astype(np.int64)
    out = {
        'threshold': threshold,
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


def choose_validation_threshold(logits, labels, val_mask, metric_name='f1'):
    val_mask = val_mask.to(device=logits.device, dtype=torch.bool)
    probs = torch.sigmoid(logits[val_mask]).detach().cpu().numpy()
    y_true = labels[val_mask].detach().cpu().numpy().astype(np.int64)
    if probs.size == 0:
        return 0.5
    unique = np.unique(probs)
    if unique.size == 1:
        candidates = np.asarray([0.5], dtype=np.float64)
    else:
        candidates = np.concatenate([
            np.asarray([0.0], dtype=np.float64),
            0.5 * (unique[:-1] + unique[1:]),
            np.asarray([1.0], dtype=np.float64),
        ])

    best_threshold = 0.5
    best_score = -float('inf')
    for threshold in candidates:
        pred = (probs > threshold).astype(np.int64)
        if metric_name == 'acc':
            score = metrics.accuracy_score(y_true, pred)
        else:
            score = metrics.f1_score(y_true, pred, zero_division=0)
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)
        elif abs(float(score) - best_score) <= 1e-12:
            if abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)
    return best_threshold


def evaluate_with_validation_threshold(logits, labels, masks, metric_name):
    threshold = choose_validation_threshold(
        logits,
        labels,
        masks['val'],
        metric_name=metric_name,
    )
    return {
        'selection_split': 'val',
        'selection_metric': str(metric_name),
        'threshold': float(threshold),
        'metrics': {
            split_name: binary_metrics_from_logits(
                logits,
                labels,
                mask,
                threshold=threshold,
            )
            for split_name, mask in masks.items()
        },
    }


def evaluate_relation_model(model, features, labels, masks, adj_for_model):
    model.eval()
    with torch.inference_mode():
        logits = model(features, adj_for_model)
    diagnostics = model.diagnostics() if hasattr(model, 'diagnostics') else {}
    edge_diagnostics = model.edge_diagnostics() if hasattr(model, 'edge_diagnostics') else {}
    result = {}
    for split_name, mask in masks.items():
        metric = binary_metrics_from_logits(logits, labels, mask)
        result[split_name] = metric
    return result, logits.detach().cpu(), diagnostics, edge_diagnostics


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
    model_seed = int(args.seed) + 1009 * int(getattr(args, 'fold_idx', 0) or 0)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    model = build_relation_model(model_name, features.size(1), args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.relation_lr),
        weight_decay=float(args.relation_weight_decay),
    )
    train_mask = masks['train'].to(device=device, dtype=torch.bool)
    labels = labels.to(device=device, dtype=torch.float32)
    row_models = {'sage', 'gated_sage', 'position_sage'}
    edge_models = {
        'edge_self_only',
        'edge_gate_residual',
        'edge_gate',
        'edge_gate_scaled',
        'edge_gate_node_residual',
    }
    if model_name in edge_models:
        adj_for_model = adj_tensors['edge_graph']
    elif model_name in row_models:
        adj_for_model = adj_tensors['row']
    else:
        adj_for_model = adj_tensors['symmetric']

    best_state = None
    best_metrics = None
    best_epoch = -1
    no_improve = 0
    training_history = []
    for epoch in range(int(args.relation_epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj_for_model)
        node_loss = F.binary_cross_entropy_with_logits(logits[train_mask], labels[train_mask])
        edge_aux_loss = (
            model.auxiliary_loss(labels, train_mask)
            if hasattr(model, 'auxiliary_loss')
            else node_loss.new_zeros(())
        )
        loss = node_loss + float(args.edge_aux_loss_weight) * edge_aux_loss
        loss.backward()
        optimizer.step()

        current_metrics, _, _, _ = evaluate_relation_model(
            model, features, labels, masks, adj_for_model
        )
        training_history.append({
            'epoch': int(epoch),
            'total_loss': float(loss.item()),
            'node_loss': float(node_loss.item()),
            'edge_aux_loss': float(edge_aux_loss.item()),
            'val_loss': current_metrics['val'].get('loss'),
            'val_acc': current_metrics['val'].get('acc'),
            'val_roc_auc': current_metrics['val'].get('roc_auc'),
            'val_f1': current_metrics['val'].get('f1'),
        })
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
    final_metrics, final_logits, final_diagnostics, final_edge_diagnostics = evaluate_relation_model(
        model,
        features,
        labels,
        masks,
        adj_for_model,
    )
    threshold_evaluation = evaluate_with_validation_threshold(
        final_logits.to(labels.device),
        labels,
        masks,
        args.threshold_metric,
    )
    return {
        'model_name': model_name,
        'model_seed': model_seed,
        'best_epoch': int(best_epoch),
        'best_metrics': best_metrics,
        'final_metrics': final_metrics,
        'logits': final_logits,
        'diagnostics': final_diagnostics,
        'edge_diagnostics': final_edge_diagnostics,
        'threshold_evaluation': threshold_evaluation,
        'training_history': training_history,
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
        fieldnames.append(f'{model_name}_val_threshold')
        fieldnames.append(f'{model_name}_val_threshold_pred')
        diagnostics = model_outputs[model_name].get('diagnostics', {})
        for diag_name in sorted(diagnostics):
            fieldnames.append(f'{model_name}_{diag_name}')
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
                tuned_threshold = float(output['threshold_evaluation']['threshold'])
                row[f'{model_name}_prob'] = prob
                row[f'{model_name}_pred'] = int(prob > 0.5)
                row[f'{model_name}_val_threshold'] = tuned_threshold
                row[f'{model_name}_val_threshold_pred'] = int(prob > tuned_threshold)
                diagnostics = output.get('diagnostics', {})
                for diag_name, diag_values in diagnostics.items():
                    row[f'{model_name}_{diag_name}'] = float(diag_values[row_idx].item())
            writer.writerow(row)


def write_rows_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_binary_auc(targets, scores):
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    targets = targets[valid]
    scores = scores[valid]
    if targets.size == 0 or np.unique(targets).size < 2:
        return None
    return float(metrics.roc_auc_score(targets, scores))


def safe_average_precision(targets, scores):
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    targets = targets[valid]
    scores = scores[valid]
    if targets.size == 0 or np.unique(targets).size < 2:
        return None
    return float(metrics.average_precision_score(targets, scores))


def edge_group_masks(src, dst, split_names):
    src_split = split_names[src]
    dst_split = split_names[dst]
    return {
        'all': np.ones(src.shape[0], dtype=bool),
        'src_train': src_split == 'train',
        'src_val': src_split == 'val',
        'src_test': src_split == 'test',
        'train_train': (src_split == 'train') & (dst_split == 'train'),
        'val_val': (src_split == 'val') & (dst_split == 'val'),
        'test_test': (src_split == 'test') & (dst_split == 'test'),
        'cross_split': src_split != dst_split,
    }


def summarize_edge_reliability(edge_diagnostics, labels, split_names, bin_count):
    if not edge_diagnostics:
        return {}
    arrays = {
        key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for key, value in edge_diagnostics.items()
    }
    src = arrays['src'].astype(np.int64, copy=False)
    dst = arrays['dst'].astype(np.int64, copy=False)
    reliability = arrays['reliability'].astype(np.float64, copy=False)
    same_label = (labels[src] == labels[dst]).astype(np.int64)
    summary = {'groups': {}, 'bins': []}
    for group_name, group_mask in edge_group_masks(src, dst, split_names).items():
        local_targets = same_label[group_mask]
        local_scores = reliability[group_mask]
        summary['groups'][group_name] = {
            'num_edges': int(group_mask.sum()),
            'same_label_rate': float(local_targets.mean()) if local_targets.size else None,
            'reliability_mean': float(local_scores.mean()) if local_scores.size else None,
            'reliability_same_label_mean': (
                float(local_scores[local_targets == 1].mean())
                if np.any(local_targets == 1) else None
            ),
            'reliability_diff_label_mean': (
                float(local_scores[local_targets == 0].mean())
                if np.any(local_targets == 0) else None
            ),
            'same_label_auc': safe_binary_auc(local_targets, local_scores),
            'same_label_average_precision': safe_average_precision(local_targets, local_scores),
        }

    if reliability.size:
        order = np.argsort(reliability)
        for bin_idx, indices in enumerate(np.array_split(order, max(int(bin_count), 1))):
            if indices.size == 0:
                continue
            summary['bins'].append({
                'bin_idx': int(bin_idx),
                'num_edges': int(indices.size),
                'reliability_min': float(reliability[indices].min()),
                'reliability_max': float(reliability[indices].max()),
                'reliability_mean': float(reliability[indices].mean()),
                'same_label_rate': float(same_label[indices].mean()),
                'candidate_weight_mean': float(arrays['candidate_weight'][indices].mean()),
                'coarse_weight_mean': float(arrays['coarse_weight'][indices].mean()),
                'effective_weight_mean': float(arrays['effective_weight'][indices].mean()),
            })
    return summary


def save_edge_scores_csv(
    path,
    edge_diagnostics,
    orig_indices,
    labels,
    split_names,
):
    arrays = {
        key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for key, value in edge_diagnostics.items()
    }
    src = arrays['src'].astype(np.int64, copy=False)
    dst = arrays['dst'].astype(np.int64, copy=False)
    rows = []
    for edge_idx in range(src.size):
        src_idx = int(src[edge_idx])
        dst_idx = int(dst[edge_idx])
        rows.append({
            'src_orig_idx': int(orig_indices[src_idx]),
            'dst_orig_idx': int(orig_indices[dst_idx]),
            'src_split': split_names[src_idx],
            'dst_split': split_names[dst_idx],
            'src_label': int(labels[src_idx]),
            'dst_label': int(labels[dst_idx]),
            'same_label': int(labels[src_idx] == labels[dst_idx]),
            'edge_aux_supervised': int(
                split_names[src_idx] == 'train' and split_names[dst_idx] == 'train'
            ),
            'cosine': float(arrays['cosine'][edge_idx]),
            'candidate_weight': float(arrays['candidate_weight'][edge_idx]),
            'coarse_weight': float(arrays['coarse_weight'][edge_idx]),
            'reliability': float(arrays['reliability'][edge_idx]),
            'effective_weight': float(arrays['effective_weight'][edge_idx]),
        })
    write_rows_csv(path, rows)


def output_threshold(output, threshold_mode):
    if threshold_mode == 'val_tuned':
        return float(output['threshold_evaluation']['threshold'])
    return 0.5


def build_model_comparison_rows(
    model_outputs,
    labels,
    split_names,
    baseline_model='mlp',
    threshold_mode='fixed_0.5',
):
    if baseline_model not in model_outputs:
        return []
    baseline_probs = torch.sigmoid(model_outputs[baseline_model]['logits']).numpy()
    baseline_threshold = output_threshold(model_outputs[baseline_model], threshold_mode)
    baseline_pred = baseline_probs > baseline_threshold
    rows = []
    for model_name, output in model_outputs.items():
        if model_name == baseline_model:
            continue
        model_probs = torch.sigmoid(output['logits']).numpy()
        model_threshold = output_threshold(output, threshold_mode)
        model_pred = model_probs > model_threshold
        for split_name in ('train', 'val', 'test'):
            split_mask = split_names == split_name
            groups = {
                'both_correct': split_mask & (baseline_pred == labels) & (model_pred == labels),
                'baseline_wrong_model_right': (
                    split_mask & (baseline_pred != labels) & (model_pred == labels)
                ),
                'baseline_right_model_wrong': (
                    split_mask & (baseline_pred == labels) & (model_pred != labels)
                ),
                'both_wrong': split_mask & (baseline_pred != labels) & (model_pred != labels),
            }
            split_count = int(split_mask.sum())
            for group_name, group_mask in groups.items():
                rows.append({
                    'baseline_model': baseline_model,
                    'model': model_name,
                    'threshold_mode': threshold_mode,
                    'baseline_threshold': baseline_threshold,
                    'model_threshold': model_threshold,
                    'split': split_name,
                    'group': group_name,
                    'count': int(group_mask.sum()),
                    'rate_within_split': float(group_mask.sum() / max(split_count, 1)),
                    'baseline_prob_mean': (
                        float(baseline_probs[group_mask].mean()) if np.any(group_mask) else None
                    ),
                    'model_prob_mean': (
                        float(model_probs[group_mask].mean()) if np.any(group_mask) else None
                    ),
                })
    return rows


def build_node_case_rows(
    model_outputs,
    orig_indices,
    labels,
    split_names,
    baseline_model='mlp',
    threshold_mode='fixed_0.5',
):
    if baseline_model not in model_outputs:
        return []
    baseline_probs = torch.sigmoid(model_outputs[baseline_model]['logits']).numpy()
    baseline_threshold = output_threshold(model_outputs[baseline_model], threshold_mode)
    baseline_pred = baseline_probs > baseline_threshold
    rows = []
    for model_name, output in model_outputs.items():
        if model_name == baseline_model:
            continue
        model_probs = torch.sigmoid(output['logits']).numpy()
        model_threshold = output_threshold(output, threshold_mode)
        model_pred = model_probs > model_threshold
        diagnostics = output.get('diagnostics', {})
        for node_idx in range(labels.shape[0]):
            baseline_correct = bool(baseline_pred[node_idx] == labels[node_idx])
            model_correct = bool(model_pred[node_idx] == labels[node_idx])
            if baseline_correct and model_correct:
                group = 'both_correct'
            elif (not baseline_correct) and model_correct:
                group = 'baseline_wrong_model_right'
            elif baseline_correct and (not model_correct):
                group = 'baseline_right_model_wrong'
            else:
                group = 'both_wrong'
            row = {
                'orig_idx': int(orig_indices[node_idx]),
                'label': int(labels[node_idx]),
                'split': split_names[node_idx],
                'baseline_model': baseline_model,
                'model': model_name,
                'threshold_mode': threshold_mode,
                'baseline_threshold': baseline_threshold,
                'model_threshold': model_threshold,
                'group': group,
                'baseline_prob': float(baseline_probs[node_idx]),
                'baseline_pred': int(baseline_pred[node_idx]),
                'model_prob': float(model_probs[node_idx]),
                'model_pred': int(model_pred[node_idx]),
            }
            for diag_name, diag_values in diagnostics.items():
                row[diag_name] = float(diag_values[node_idx].item())
            rows.append(row)
    return rows


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


def summarize_diagnostics_by_split(diagnostics, masks):
    summary = {}
    for diag_name, values in diagnostics.items():
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
        values = values.detach().cpu().float().view(-1)
        diag_summary = {}
        for split_name, mask in masks.items():
            local_mask = mask.detach().cpu().bool().view(-1)
            if local_mask.numel() != values.numel() or int(local_mask.sum().item()) == 0:
                continue
            local_values = values[local_mask]
            diag_summary[split_name] = {
                'mean': float(local_values.mean().item()),
                'std': float(local_values.std(unbiased=False).item()),
                'min': float(local_values.min().item()),
                'max': float(local_values.max().item()),
            }
        summary[diag_name] = diag_summary
    return summary


def run_relation_experiment(
    features,
    labels,
    coarse_adj,
    split,
    args,
    out_dir,
    metadata=None,
    orig_indices=None,
    coarse_reference_adj=None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    labels = labels.long()
    masks = indices_to_masks(features.size(0), split, orig_indices=orig_indices)
    train_mask = masks['train']
    raw_features = features.float()
    if coarse_reference_adj is None:
        coarse_reference_adj = coarse_adj
    edge_graph = build_edge_graph_tensors(
        coarse_adj,
        coarse_reference_adj,
        raw_features,
        device,
    )
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
        'edge_graph': edge_graph,
    }

    model_names = [name.strip().lower() for name in args.models.split(',') if name.strip()]
    model_outputs = {}
    if orig_indices is None:
        orig_indices = np.arange(features.size(0), dtype=np.int64)
    elif isinstance(orig_indices, torch.Tensor):
        orig_indices = orig_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        orig_indices = np.asarray(orig_indices, dtype=np.int64)
    labels_np = labels.cpu().numpy()
    split_names = split_name_array(features.size(0), split, orig_indices=orig_indices)
    result_summary = {
        'metadata': metadata or {},
        'split': split,
        'feature_shape': list(features.shape),
        'relation_graph': {
            'type': str(getattr(args, 'relation_graph', 'coarse')),
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
            'model_seed': output['model_seed'],
            'best_epoch': output['best_epoch'],
            'best_metrics': output['best_metrics'],
            'final_metrics': output['final_metrics'],
            'threshold_evaluation': output['threshold_evaluation'],
            'diagnostics': summarize_diagnostics_by_split(output.get('diagnostics', {}), masks),
        }
        if output.get('edge_diagnostics'):
            edge_summary = summarize_edge_reliability(
                output['edge_diagnostics'],
                labels_np,
                split_names,
                args.edge_diagnostic_bins,
            )
            result_summary['models'][model_name]['edge_reliability'] = edge_summary
            with open(
                os.path.join(out_dir, f'{model_name}_edge_reliability_summary.json'),
                'w',
                encoding='utf-8',
            ) as f:
                json.dump(json_safe(edge_summary), f, indent=2, ensure_ascii=False)
            edge_scores_path = os.path.join(out_dir, f'{model_name}_edge_scores.csv')
            save_edge_scores_csv(
                edge_scores_path,
                output['edge_diagnostics'],
                orig_indices,
                labels_np,
                split_names,
            )
            write_rows_csv(
                os.path.join(out_dir, f'{model_name}_edge_reliability_bins.csv'),
                edge_summary.get('bins', []),
            )
        torch.save(
            {
                'state_dict': output['state_dict'],
                'model_name': model_name,
                'model_seed': output['model_seed'],
                'input_dim': int(features.size(1)),
                'args': vars(args),
            },
            os.path.join(out_dir, f'{model_name}_relation_model.pt'),
        )

    comparison_baselines = [
        value.strip().lower()
        for value in str(args.comparison_baselines).split(',')
        if value.strip()
    ]
    result_summary['model_comparisons'] = {}
    for baseline_model in comparison_baselines:
        if baseline_model not in model_outputs:
            continue
        baseline_payload = {}
        for threshold_mode in ('fixed_0.5', 'val_tuned'):
            comparison_rows = build_model_comparison_rows(
                model_outputs,
                labels_np,
                split_names,
                baseline_model=baseline_model,
                threshold_mode=threshold_mode,
            )
            node_case_rows = build_node_case_rows(
                model_outputs,
                orig_indices,
                labels_np,
                split_names,
                baseline_model=baseline_model,
                threshold_mode=threshold_mode,
            )
            baseline_payload[threshold_mode] = comparison_rows
            mode_suffix = '' if threshold_mode == 'fixed_0.5' else '_val_tuned'
            write_rows_csv(
                os.path.join(
                    out_dir,
                    f'model_comparison{mode_suffix}_vs_{baseline_model}.csv',
                ),
                comparison_rows,
            )
            write_rows_csv(
                os.path.join(
                    out_dir,
                    f'node_cases{mode_suffix}_vs_{baseline_model}.csv',
                ),
                node_case_rows,
            )
        result_summary['model_comparisons'][baseline_model] = baseline_payload

    # Backward-compatible key used by previous all-fold summaries.
    result_summary['model_comparison_vs_mlp'] = (
        result_summary['model_comparisons'].get('mlp', {}).get('fixed_0.5', [])
    )

    threshold_rows = []
    for model_name, output in model_outputs.items():
        for threshold_mode, threshold, split_metrics in (
            ('fixed_0.5', 0.5, output['final_metrics']),
            (
                'val_tuned',
                output['threshold_evaluation']['threshold'],
                output['threshold_evaluation']['metrics'],
            ),
        ):
            for split_name, metric_values in split_metrics.items():
                threshold_rows.append({
                    'model': model_name,
                    'threshold_mode': threshold_mode,
                    'threshold': float(threshold),
                    'selection_metric': (
                        args.threshold_metric if threshold_mode == 'val_tuned' else ''
                    ),
                    'split': split_name,
                    **metric_values,
                })
    write_rows_csv(os.path.join(out_dir, 'threshold_metrics.csv'), threshold_rows)

    training_rows = []
    for model_name, output in model_outputs.items():
        for row in output.get('training_history', []):
            training_rows.append({'model': model_name, **row})
    write_rows_csv(os.path.join(out_dir, 'relation_training_history.csv'), training_rows)

    result_path = os.path.join(out_dir, 'relation_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(result_summary), f, indent=2, ensure_ascii=False)

    predictions_path = os.path.join(out_dir, 'relation_predictions.csv')
    save_predictions_csv(predictions_path, orig_indices, labels_np, split_names, model_outputs)
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
    relation_adj, relation_graph_metadata = build_relation_adjacency(coarse_adj, features, args)
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
        relation_adj,
        split,
        args,
        args.out_dir,
        metadata={'synthetic_smoke': True, 'relation_graph': relation_graph_metadata},
        orig_indices=None,
        coarse_reference_adj=coarse_adj,
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


def dataset_output_dir(args):
    out_dir = os.path.normpath(str(args.out_dir))
    if os.path.basename(out_dir) == str(args.data_name):
        return out_dir
    return os.path.join(out_dir, args.data_name)


def run_output_dir(args):
    graph_name = str(getattr(args, 'relation_graph', 'coarse')).strip().lower()
    base_out_dir = dataset_output_dir(args)
    if args.split_source == 'fixed_cv':
        if graph_name == 'coarse':
            return os.path.join(base_out_dir, f'fold_{args.fold_idx}')
        return os.path.join(base_out_dir, graph_name, f'fold_{args.fold_idx}')
    if graph_name == 'coarse':
        return os.path.join(base_out_dir, 'dataset_split')
    return os.path.join(base_out_dir, graph_name, 'dataset_split')


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
    features, labels, aligned_orig_indices = align_payload_to_coarse_nodes(
        payload,
        num_nodes=int(loader.coarse_adj.shape[0]),
    )
    if features.size(0) != loader.coarse_adj.shape[0]:
        raise ValueError(
            f'Embedding rows ({features.size(0)}) do not match coarse nodes ({loader.coarse_adj.shape[0]}).'
        )

    if run_args.embeddings_path and not os.path.exists(run_args.embeddings_path):
        ensure_parent_dir(run_args.embeddings_path)
        torch.save(payload, run_args.embeddings_path)

    relation_adj, relation_graph_metadata = build_relation_adjacency(loader.coarse_adj, features, run_args)
    metadata = {
        'data_name': run_args.data_name,
        'embedding_key': payload.get('embedding_key', run_args.embedding_key),
        'embeddings_path': run_args.embeddings_path,
        'relation_graph': relation_graph_metadata,
        'coarse_graph_metadata': loader.coarse_graph_metadata,
    }
    result = run_relation_experiment(
        features,
        labels,
        relation_adj,
        split,
        run_args,
        out_dir,
        metadata=metadata,
        orig_indices=aligned_orig_indices,
        coarse_reference_adj=loader.coarse_adj,
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


def aggregate_model_comparisons(fold_outputs):
    grouped = {}
    for fold_output in fold_outputs:
        fold_idx = fold_output.get('fold_idx')
        comparisons = fold_output['result'].get('model_comparisons', {})
        for baseline_model, threshold_payload in comparisons.items():
            for threshold_mode, comparison_rows in threshold_payload.items():
                for row in comparison_rows:
                    key = (
                        baseline_model,
                        threshold_mode,
                        row['model'],
                        row['split'],
                        row['group'],
                    )
                    bucket = grouped.setdefault(key, {'counts': [], 'rates': []})
                    bucket['counts'].append({
                        'fold_idx': fold_idx,
                        'value': int(row['count']),
                    })
                    bucket['rates'].append({
                        'fold_idx': fold_idx,
                        'value': float(row['rate_within_split']),
                    })
    rows = []
    for key, values in sorted(grouped.items()):
        baseline_model, threshold_mode, model_name, split_name, group_name = key
        rates = np.asarray([entry['value'] for entry in values['rates']], dtype=np.float64)
        rows.append({
            'baseline_model': baseline_model,
            'threshold_mode': threshold_mode,
            'model': model_name,
            'split': split_name,
            'group': group_name,
            'count_sum': int(sum(entry['value'] for entry in values['counts'])),
            'rate_mean': float(rates.mean()) if rates.size else None,
            'rate_std': float(rates.std(ddof=0)) if rates.size else None,
            'counts_by_fold': values['counts'],
            'rates_by_fold': values['rates'],
        })
    return rows


def aggregate_threshold_results(fold_outputs):
    grouped = {}
    for fold_output in fold_outputs:
        fold_idx = fold_output.get('fold_idx')
        for model_name, model_result in fold_output['result'].get('models', {}).items():
            fixed_metrics = model_result.get('final_metrics', {}).get('test', {})
            tuned_payload = model_result.get('threshold_evaluation', {})
            tuned_metrics = tuned_payload.get('metrics', {}).get('test', {})
            for threshold_mode, threshold, metric_values in (
                ('fixed_0.5', 0.5, fixed_metrics),
                ('val_tuned', tuned_payload.get('threshold'), tuned_metrics),
            ):
                if threshold is None:
                    continue
                bucket = grouped.setdefault((model_name, threshold_mode), {
                    'threshold': [],
                    'metrics': {},
                })
                bucket['threshold'].append({
                    'fold_idx': fold_idx,
                    'value': float(threshold),
                })
                for metric_name in (
                    'acc',
                    'roc_auc',
                    'average_precision',
                    'f1',
                    'prec',
                    'rec',
                    'loss',
                    'pred_positive_rate',
                ):
                    value = metric_values.get(metric_name)
                    if value is not None:
                        bucket['metrics'].setdefault(metric_name, []).append({
                            'fold_idx': fold_idx,
                            'value': float(value),
                        })

    summary = {}
    for (model_name, threshold_mode), payload in sorted(grouped.items()):
        target = summary.setdefault(model_name, {}).setdefault(threshold_mode, {})
        threshold_values = np.asarray(
            [entry['value'] for entry in payload['threshold']],
            dtype=np.float64,
        )
        target['threshold'] = {
            'mean': float(threshold_values.mean()) if threshold_values.size else None,
            'std': float(threshold_values.std(ddof=0)) if threshold_values.size else None,
            'values': payload['threshold'],
        }
        for metric_name, entries in payload['metrics'].items():
            values = np.asarray([entry['value'] for entry in entries], dtype=np.float64)
            target[metric_name] = {
                'mean': float(values.mean()) if values.size else None,
                'std': float(values.std(ddof=0)) if values.size else None,
                'values': entries,
            }
    return summary


def aggregate_edge_reliability(fold_outputs):
    grouped = {}
    metric_names = (
        'same_label_rate',
        'reliability_mean',
        'reliability_same_label_mean',
        'reliability_diff_label_mean',
        'same_label_auc',
        'same_label_average_precision',
    )
    for fold_output in fold_outputs:
        fold_idx = fold_output.get('fold_idx')
        for model_name, model_result in fold_output['result'].get('models', {}).items():
            groups = model_result.get('edge_reliability', {}).get('groups', {})
            for group_name, group_result in groups.items():
                bucket = grouped.setdefault((model_name, group_name), {})
                for metric_name in metric_names:
                    value = group_result.get(metric_name)
                    if value is not None:
                        bucket.setdefault(metric_name, []).append({
                            'fold_idx': fold_idx,
                            'value': float(value),
                        })
    summary = {}
    for (model_name, group_name), metric_values in sorted(grouped.items()):
        group_summary = summary.setdefault(model_name, {}).setdefault(group_name, {})
        for metric_name, entries in metric_values.items():
            values = np.asarray([entry['value'] for entry in entries], dtype=np.float64)
            group_summary[metric_name] = {
                'mean': float(values.mean()) if values.size else None,
                'std': float(values.std(ddof=0)) if values.size else None,
                'values': entries,
            }
    return summary


def save_all_folds_summary(args, fold_outputs):
    summary_args = copy.copy(args)
    apply_path_templates(summary_args, fold_idx='all')
    graph_name = str(getattr(summary_args, 'relation_graph', 'coarse')).strip().lower()
    dataset_dir = dataset_output_dir(summary_args)
    if graph_name == 'coarse':
        base_out_dir = dataset_dir
    else:
        base_out_dir = os.path.join(dataset_dir, graph_name)
    ensure_parent_dir(os.path.join(base_out_dir, 'dummy'))
    os.makedirs(base_out_dir, exist_ok=True)
    model_comparisons = aggregate_model_comparisons(fold_outputs)
    payload = {
        'data_name': summary_args.data_name,
        'fold_indices': [output.get('fold_idx') for output in fold_outputs],
        'summary': aggregate_fold_results(fold_outputs),
        'threshold_summary': aggregate_threshold_results(fold_outputs),
        'model_comparisons': model_comparisons,
        'model_comparison_vs_mlp': [
            row for row in model_comparisons
            if row['baseline_model'] == 'mlp'
            and row['threshold_mode'] == 'fixed_0.5'
        ],
        'edge_reliability': aggregate_edge_reliability(fold_outputs),
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
