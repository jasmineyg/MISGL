# coding=utf-8

"""All-fold runner with shared coarse cache and one tensorization per worker."""

import copy
import fcntl
import logging
import os

import torch
from torch.utils.data import DataLoader, Subset

import run_coarse_gcn_paper as core
from MISGL.utils import reproducibility
from MISGL.utils.global_variables import g_key
from MISGL.utils.load_data import GraphDataLoaderWrapper, GraphDataset


_original_load_or_build = core.load_or_build_coarse_adjacency


def _load_or_build_shared(args, hparams, loader):
    experiment_root = os.path.dirname(os.path.abspath(hparams.cv_split_dir))
    shared_root = os.path.join(experiment_root, 'shared_coarse')
    lock_root = os.path.join(shared_root, '.locks')
    os.makedirs(lock_root, exist_ok=True)
    safe_name = str(hparams.data_name).replace('/', '__').replace('\\', '__')
    lock_path = os.path.join(lock_root, safe_name + '.lock')
    cache_args = copy.copy(args)
    cache_args.out_dir = shared_root
    with open(lock_path, 'a', encoding='utf-8') as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _original_load_or_build(cache_args, hparams, loader)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _cached_build_loaders(self, train_idx, val_idx, test_idx):
    if not hasattr(self, '_paper_full_dataset'):
        logging.info('Tensorizing all %d CV graphs once for reuse across folds.', len(self.cv_graphs))
        self._paper_full_dataset = GraphDataset(self._hparams, self.cv_graphs)
        logging.info('Finished reusable full-dataset tensorization.')
    dataset = self._paper_full_dataset

    def make(indices, shuffle):
        return DataLoader(
            Subset(dataset, list(indices)),
            batch_size=int(self._hparams.batch_size),
            shuffle=bool(shuffle),
            worker_init_fn=reproducibility.worker_init_fn,
        )

    return make(train_idx, True), make(val_idx, False), make(test_idx, False)


def _export_z_mil_cached(model, hparams, loader):
    dataset = getattr(loader, '_paper_full_dataset', None)
    if dataset is None:
        dataset = GraphDataset(loader._hparams, loader.cv_graphs)
        loader._paper_full_dataset = dataset
    export_loader = DataLoader(
        dataset,
        batch_size=int(hparams.batch_size),
        shuffle=False,
        worker_init_fn=reproducibility.worker_init_fn,
    )
    device = torch.device(hparams.device)
    features, labels, orig_indices, logits = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in export_loader:
            batch = core.move_batch_to_device(batch, device)
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


core.load_or_build_coarse_adjacency = _load_or_build_shared
core.export_z_mil = _export_z_mil_cached
GraphDataLoaderWrapper._build_loaders_from_indices = _cached_build_loaders

from run_coarse_gcn_paper_allfolds import main


if __name__ == '__main__':
    main()

