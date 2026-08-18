# coding=utf-8

import argparse
import copy
import json
import logging
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader

from MISGL.bin.train_eval import train_eval_iter
from MISGL.models.encoder import MISGLEncoder
from MISGL.utils import coarse_graph
from MISGL.utils import hparam
from MISGL.utils import hparams_lib
from MISGL.utils import reproducibility
from MISGL.utils.global_variables import g_key
from MISGL.utils.load_data import GraphDataLoaderWrapper
from MISGL.utils.load_data import GraphDataset


class OneLayerCoarseGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.gcn = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_mil, adj_norm):
        propagated = torch.sparse.mm(adj_norm, z_mil)
        z_pos = F.relu(self.gcn(propagated))
        logits = self.classifier(torch.cat([z_mil, z_pos], dim=-1)).view(-1)
        return logits, z_pos


def parse_args():
    parser = argparse.ArgumentParser(description='Lightweight two-stage original-coarse-GCN experiment.')
    parser.add_argument('--hparam_path', default='config/b_on.yml')
    parser.add_argument('--data_name', default=None)
    parser.add_argument('--processed_data_dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--out_dir', default='results/coarse_gcn')
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--all_folds', action='store_true')
    parser.add_argument('--create_split_if_missing', action='store_true')
    parser.add_argument('--top_k', type=int, default=16)
    parser.add_argument('--stage1_epochs', type=int, default=None)
    parser.add_argument('--stage1_patience', type=int, default=None)
    parser.add_argument('--stage2_epochs', type=int, default=300)
    parser.add_argument('--stage2_patience', type=int, default=50)
    parser.add_argument('--stage2_lr', type=float, default=0.001)
    parser.add_argument('--stage2_weight_decay', type=float, default=5e-4)
    parser.add_argument('--stage2_hidden_dim', type=int, default=None)
    parser.add_argument('--stage2_dropout', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--synthetic_smoke', action='store_true')
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def load_hparams(args):
    hparams = hparam.HParams()
    hparams.from_yaml(args.hparam_path)
    hparams_lib.apply_defaults(hparams)
    if args.data_name:
        hparams.data_name = args.data_name
    elif not getattr(hparams, 'data_name', None):
        data_names = getattr(hparams, 'data_name_set', None)
        if isinstance(data_names, (list, tuple)) and data_names:
            hparams.data_name = data_names[0]
        else:
            raise ValueError('Set --data_name or data_name_set in the hparam yaml.')
    if args.processed_data_dir:
        hparams.processed_data_dir = args.processed_data_dir
    if args.device:
        hparams.device = args.device
    if hparams.device == 'cuda' and not torch.cuda.is_available():
        logging.warning('CUDA requested but unavailable; falling back to CPU.')
        hparams.device = 'cpu'
    hparams.preload_data_to_gpu = False
    if args.stage1_epochs is not None:
        hparams.epoch = int(args.stage1_epochs)
    if args.stage1_patience is not None:
        hparams.patience = int(args.stage1_patience)
    return hparams


def sync_hparams_from_loader(hparams, loader):
    hparams.channel_list = list(loader._hparams.channel_list)
    if hasattr(loader._hparams, 'max_num_nodes'):
        value = int(loader._hparams.max_num_nodes)
        if 'max_num_nodes' in hparams:
            hparams.set_hparam('max_num_nodes', value)
        else:
            hparams.add_hparam('max_num_nodes', value)


def ensure_split_manifest(loader, args):
    split_path = loader.get_cv_split_path(ensure_dir=args.create_split_if_missing)
    if os.path.exists(split_path):
        return loader.load_cv_split_manifest(split_path), split_path
    manifest = loader.build_cv_split_manifest()
    if args.create_split_if_missing:
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logging.info('Saved CV split manifest: %s', split_path)
    else:
        logging.warning('CV split manifest not found; using in-memory split only: %s', split_path)
    return manifest, split_path


def make_graph_loader(hparams, graphs, shuffle):
    dataset = GraphDataset(hparams, graphs)
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


def export_z_mil(model, hparams, loader):
    device = torch.device(hparams.device)
    graphs = [loader._subgraphs[int(idx)] for idx in loader.cv_orig_indices]
    export_loader = make_graph_loader(hparams, graphs, shuffle=False)
    features = []
    labels = []
    orig_indices = []
    logits = []
    model.eval()
    with torch.inference_mode():
        for batch in export_loader:
            batch = move_batch_to_device(batch, device)
            model_out, emb = model.forward_with_embeddings(batch)
            if 'z_mil' not in emb:
                raise KeyError('MISGLEncoder.forward_with_embeddings did not return z_mil.')
            features.append(emb['z_mil'].detach().cpu())
            labels.append(batch[g_key.y].view(-1).detach().cpu())
            orig_indices.append(batch[g_key.orig_graph_idx].view(-1).detach().cpu())
            if isinstance(model_out, dict) and 'ypred_A' in model_out:
                logits.append(model_out['ypred_A'].view(-1).detach().cpu())
            elif isinstance(model_out, torch.Tensor):
                logits.append(model_out.view(-1).detach().cpu())

    payload = {
        'z_mil': torch.cat(features, dim=0),
        'labels': torch.cat(labels, dim=0),
        'orig_indices': torch.cat(orig_indices, dim=0),
    }
    if logits:
        payload['stage1_logits'] = torch.cat(logits, dim=0)
    return payload


def active_subgraph_ids(loader):
    ids = []
    for orig_idx in loader.cv_orig_indices:
        graph = loader._subgraphs[int(orig_idx)]
        subgraph_id = graph.graph.get('subgraph_id', orig_idx)
        try:
            subgraph_id = int(subgraph_id)
        except (TypeError, ValueError):
            subgraph_id = int(orig_idx)
        if subgraph_id < 0:
            subgraph_id = int(orig_idx)
        ids.append(subgraph_id)
    return np.asarray(ids, dtype=np.int64)


def build_masks(orig_indices, split_meta):
    if isinstance(orig_indices, torch.Tensor):
        orig_indices = orig_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        orig_indices = np.asarray(orig_indices, dtype=np.int64)
    masks = {}
    for split_name in ('train', 'val', 'test'):
        split_set = set(int(v) for v in split_meta[f'{split_name}_indices'])
        masks[split_name] = torch.from_numpy(
            np.asarray([int(idx) in split_set for idx in orig_indices], dtype=bool)
        )
    return masks


def normalize_for_gcn(coarse_adj):
    adj = coarse_adj.tocsr().astype(np.float32, copy=True)
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format='csr')
    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    inv_degree = np.zeros_like(degree)
    valid = degree > 0
    inv_degree[valid] = 1.0 / degree[valid]
    return (sp.diags(inv_degree, format='csr') @ adj).tocsr()


def scipy_to_torch_sparse(matrix, device):
    coo = matrix.tocoo().astype(np.float32, copy=False)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape), device=device).coalesce()


def metric_from_logits(logits, labels, mask):
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if int(mask.sum().item()) == 0:
        return {'acc': None, 'F1': None, 'prec': None, 'rec': None, 'loss': None}
    local_logits = logits[mask]
    local_labels = labels.to(device=logits.device, dtype=torch.float32)[mask]
    probs = torch.sigmoid(local_logits).detach().cpu().numpy()
    y_true = local_labels.detach().cpu().numpy().astype(np.int64)
    y_pred = (probs > 0.5).astype(np.int64)
    return {
        'acc': float(metrics.accuracy_score(y_true, y_pred)),
        'F1': float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        'prec': float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        'rec': float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        'loss': float(F.binary_cross_entropy_with_logits(local_logits, local_labels).item()),
    }


def evaluate_stage2(model, z_mil, labels, masks, adj_norm):
    model.eval()
    with torch.inference_mode():
        logits, z_pos = model(z_mil, adj_norm)
    return {
        split_name: metric_from_logits(logits, labels, mask)
        for split_name, mask in masks.items()
    }, logits.detach().cpu(), z_pos.detach().cpu()


def is_better(current, best):
    if best is None:
        return True
    cur_acc = current['val']['acc']
    best_acc = best['val']['acc']
    if cur_acc is None:
        return False
    if best_acc is None or cur_acc > best_acc + 1e-8:
        return True
    if abs(cur_acc - best_acc) <= 1e-8:
        return float(current['val']['loss']) < float(best['val']['loss']) - 1e-8
    return False


def train_stage2(z_mil, labels, masks, coarse_adj, args):
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    z_mil = z_mil.to(device=device, dtype=torch.float32).detach()
    labels = labels.to(device=device, dtype=torch.float32)
    masks = {key: value.to(device=device, dtype=torch.bool) for key, value in masks.items()}
    adj_norm = scipy_to_torch_sparse(normalize_for_gcn(coarse_adj), device)
    hidden_dim = int(args.stage2_hidden_dim or z_mil.size(1))
    dropout = float(args.stage2_dropout if args.stage2_dropout is not None else 0.5)
    model = OneLayerCoarseGCN(z_mil.size(1), hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.stage2_lr),
        weight_decay=float(args.stage2_weight_decay),
    )

    train_mask = masks['train']
    best_state = None
    best_metrics = None
    best_epoch = -1
    no_improve = 0
    for epoch in range(int(args.stage2_epochs)):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(z_mil, adj_norm)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        current_metrics, _, _ = evaluate_stage2(model, z_mil, labels, masks, adj_norm)
        if is_better(current_metrics, best_metrics):
            best_metrics = current_metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(args.stage2_patience):
            logging.info('Stage-2 early stop at epoch=%d', epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics, final_logits, z_pos = evaluate_stage2(model, z_mil, labels, masks, adj_norm)
    return {
        'model': model,
        'best_epoch': int(best_epoch),
        'best_metrics': best_metrics,
        'final_metrics': final_metrics,
        'logits': final_logits,
        'z_pos': z_pos,
    }


def train_stage1(hparams, train_loader, val_loader, dataset_raw):
    device = torch.device(hparams.device)
    model = MISGLEncoder(hparams, data_name=hparams.data_name).to(device)
    model, _, best_val = train_eval_iter(
        model,
        train_loader,
        val_loader,
        writer=None,
        hparams=hparams,
        dataset_raw=dataset_raw,
    )
    return model, best_val


def split_summary_metrics(metrics_by_split):
    return {
        split_name: {
            key: metrics_by_split[split_name].get(key)
            for key in ('acc', 'F1', 'prec', 'rec')
        }
        for split_name in ('train', 'val', 'test')
    }


def coarse_cache_file(args, hparams):
    return os.path.join(
        args.out_dir,
        hparams.data_name,
        f'coarse_adj_topk{int(args.top_k)}_noself_sym.npz',
    )


def cache_metadata_matches(metadata, active_ids, top_k):
    cached_ids = np.asarray(metadata.get('active_subgraph_ids', []), dtype=np.int64)
    return (
        cached_ids.shape == active_ids.shape
        and np.array_equal(cached_ids, active_ids)
        and int(metadata.get('num_coarse_nodes', -1)) == int(active_ids.shape[0])
        and metadata.get('top_k') == int(top_k)
        and metadata.get('include_self') is False
        and metadata.get('symmetrize') is True
    )


def load_or_build_coarse_adjacency(args, hparams, loader):
    active_ids = active_subgraph_ids(loader)
    cache_path = coarse_cache_file(args, hparams)
    if os.path.exists(cache_path):
        coarse_adj, coarse_meta = coarse_graph.load_coarse_adjacency(cache_path)
        if cache_metadata_matches(coarse_meta, active_ids, args.top_k):
            logging.info('Loaded cached coarse adjacency: %s', cache_path)
            return coarse_adj, coarse_meta, cache_path
        logging.warning('Ignoring stale coarse adjacency cache: %s', cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logging.info('Building coarse adjacency cache: %s', cache_path)
    coarse_adj, coarse_meta = coarse_graph.build_coarse_adjacency(
        loader.original_graph,
        loader.assignment_matrix,
        active_subgraph_ids=active_ids,
        top_k=int(args.top_k),
        include_self=False,
        symmetrize=True,
    )
    coarse_graph.save_coarse_adjacency(cache_path, coarse_adj, coarse_meta)
    logging.info(
        'Saved coarse adjacency cache: %s nodes=%d directed_edges=%d',
        cache_path,
        int(coarse_adj.shape[0]),
        int(coarse_adj.nnz),
    )
    return coarse_adj, coarse_meta, cache_path


def run_one_fold(args, hparams, loader, split_manifest, fold_idx, coarse_adj, coarse_path):
    fold_dir = os.path.join(args.out_dir, hparams.data_name, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)
    train_loader, val_loader, test_loader, split_meta = loader.get_cv_loaders_from_manifest(split_manifest, fold_idx)
    logging.info(
        'Fold %d sizes: train=%d val=%d test=%d',
        fold_idx,
        split_meta['train_size'],
        split_meta['val_size'],
        split_meta['test_size'],
    )

    model, best_val = train_stage1(hparams, train_loader, val_loader, loader._dataset_raw)
    torch.save(
        {'model_state_dict': model.state_dict(), 'best_val': best_val, 'hparams': hparams.values()},
        os.path.join(fold_dir, 'stage1_branch_b.pt'),
    )

    attention_path = None
    branch_b_cfg = getattr(hparams, 'branch_b', {}) or {}
    if bool(branch_b_cfg.get('use', False)):
        from paper_attention_metrics import export_fold_attention_metrics

        attention_path = os.path.join(fold_dir, 'attention_metrics.json')
        export_fold_attention_metrics(
            model=model,
            test_loader=test_loader,
            hparams=hparams,
            dataset_raw=loader._dataset_raw,
            output_path=attention_path,
            data_name=hparams.data_name,
            cv_seed=int(getattr(hparams, 'cv_seed', args.seed)),
            fold_idx=int(fold_idx),
        )

    payload = export_z_mil(model, hparams, loader)
    z_mil_path = os.path.join(fold_dir, 'z_mil.pt')
    torch.save(payload, z_mil_path)

    masks = build_masks(payload['orig_indices'], split_meta)
    stage2_output = train_stage2(payload['z_mil'], payload['labels'], masks, coarse_adj, args)
    torch.save(
        {
            'state_dict': stage2_output['model'].state_dict(),
            'best_epoch': stage2_output['best_epoch'],
            'input_dim': int(payload['z_mil'].size(1)),
        },
        os.path.join(fold_dir, 'stage2_coarse_gcn.pt'),
    )

    baseline = None
    if 'stage1_logits' in payload:
        baseline = {
            split_name: metric_from_logits(payload['stage1_logits'], payload['labels'], mask)
            for split_name, mask in masks.items()
        }

    result = {
        'data_name': hparams.data_name,
        'fold_idx': int(fold_idx),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'split': split_meta,
        'paths': {
            'z_mil': z_mil_path,
            'coarse_adj': coarse_path,
            'attention_metrics': attention_path,
        },
        'coarse_graph': {
            'num_nodes': int(coarse_adj.shape[0]),
            'num_edges_directed': int(coarse_adj.nnz),
            'top_k': int(args.top_k),
            'normalization': 'stage2_row_normalize_after_self_loop',
        },
        'stage1': {
            'best_val': best_val,
            'baseline_metrics': split_summary_metrics(baseline) if baseline is not None else None,
        },
        'stage2': {
            'best_epoch': stage2_output['best_epoch'],
            'best_metrics': split_summary_metrics(stage2_output['best_metrics']),
            'final_metrics': split_summary_metrics(stage2_output['final_metrics']),
        },
    }
    result_path = os.path.join(fold_dir, 'coarse_gcn_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info('Saved fold result: %s', result_path)
    return result


def aggregate_results(results):
    summary = {}
    for section in ('stage1', 'stage2'):
        summary[section] = {}
        metric_source = 'baseline_metrics' if section == 'stage1' else 'final_metrics'
        for metric_name in ('acc', 'F1', 'prec', 'rec'):
            values = []
            for result in results:
                metrics_block = result[section].get(metric_source)
                if metrics_block is None:
                    continue
                value = metrics_block['test'].get(metric_name)
                if value is not None:
                    values.append(float(value))
            summary[section][metric_name] = {
                'mean': float(np.mean(values)) if values else None,
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'num_folds': len(values),
            }
    return summary


def save_all_folds_summary(args, hparams, results):
    out_dir = os.path.join(args.out_dir, hparams.data_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'coarse_gcn_results_all_folds.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'data_name': hparams.data_name,
                'fold_indices': [int(item['fold_idx']) for item in results],
                'summary': aggregate_results(results),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logging.info('Saved all-fold summary: %s', path)


def synthetic_smoke(args):
    rng = np.random.default_rng(args.seed)
    num_nodes = 24
    feature_dim = 8
    rows = np.arange(num_nodes)
    cols = (rows + 1) % num_nodes
    coarse_adj = sp.csr_matrix(
        (np.ones(num_nodes, dtype=np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes),
    )
    coarse_adj = coarse_adj.maximum(coarse_adj.T).tocsr()
    z_mil = torch.from_numpy(rng.normal(size=(num_nodes, feature_dim)).astype(np.float32))
    labels = torch.from_numpy((z_mil[:, 0].numpy() > 0).astype(np.int64))
    masks = {
        'train': torch.zeros(num_nodes, dtype=torch.bool),
        'val': torch.zeros(num_nodes, dtype=torch.bool),
        'test': torch.zeros(num_nodes, dtype=torch.bool),
    }
    masks['train'][:12] = True
    masks['val'][12:18] = True
    masks['test'][18:] = True
    args.device = 'cpu'
    args.stage2_epochs = min(int(args.stage2_epochs), 3)
    args.stage2_patience = min(int(args.stage2_patience), 2)
    output = train_stage2(z_mil, labels, masks, coarse_adj, args)
    out_dir = os.path.join(args.out_dir, 'synthetic_smoke')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'coarse_gcn_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'stage2': {
                    'best_epoch': output['best_epoch'],
                    'final_metrics': split_summary_metrics(output['final_metrics']),
                }
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logging.info('Saved synthetic smoke result: %s', path)


def main():
    args = parse_args()
    setup_logging()
    reproducibility.set_seed(args.seed, cuda_deterministic=False)
    if args.synthetic_smoke:
        synthetic_smoke(args)
        return

    hparams = load_hparams(args)
    args.device = hparams.device
    loader = GraphDataLoaderWrapper(hparams, data_name=hparams.data_name)
    sync_hparams_from_loader(hparams, loader)
    if loader.original_graph is None:
        raise ValueError('Dataset does not contain original_graph; cannot build original coarse graph.')
    if loader.assignment_matrix is None:
        raise ValueError('Dataset does not contain assignment_matrix; cannot build original coarse graph.')

    split_manifest, split_path = ensure_split_manifest(loader, args)
    logging.info('Using split manifest: %s', split_path)
    coarse_adj, _coarse_meta, coarse_path = load_or_build_coarse_adjacency(args, hparams, loader)
    fold_indices = list(range(int(loader.cv_num_folds))) if args.all_folds else [int(args.fold_idx)]
    results = []
    for fold_idx in fold_indices:
        logging.info('===== Fold %d =====', fold_idx)
        fold_hparams = hparams_lib.copy_hparams(hparams)
        results.append(run_one_fold(args, fold_hparams, loader, split_manifest, fold_idx, coarse_adj, coarse_path))
    if len(results) > 1:
        save_all_folds_summary(args, hparams, results)


if __name__ == '__main__':
    main()
