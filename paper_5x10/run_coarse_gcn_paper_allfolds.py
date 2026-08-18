# coding=utf-8

"""All-fold entry point that resumes completed folds without reloading the dataset."""

import json
import logging
import os

import torch

import run_coarse_gcn_paper as core
from MISGL.utils import hparams_lib
from MISGL.utils import reproducibility
from MISGL.utils.load_data import GraphDataLoaderWrapper


def _fold_result_path(args, hparams, fold_idx):
    return os.path.join(args.out_dir, hparams.data_name, f'fold_{fold_idx}', 'coarse_gcn_results.json')


def _load_completed_fold(args, hparams, fold_idx):
    path = _fold_result_path(args, hparams, fold_idx)
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            result = json.load(handle)
        if result['stage1']['baseline_metrics']['test'].get('acc') is None:
            return None
        if result['stage2']['final_metrics']['test'].get('acc') is None:
            return None
        branch_b = getattr(hparams, 'branch_b', {}) or {}
        if bool(branch_b.get('use', False)):
            attention_path = os.path.join(os.path.dirname(path), 'attention_metrics.json')
            with open(attention_path, 'r', encoding='utf-8') as handle:
                attention = json.load(handle)
            if 'positive_bags' not in attention:
                return None
        return result
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def main():
    args = core.parse_args()
    core.setup_logging()
    reproducibility.set_seed(args.seed, cuda_deterministic=False)
    if args.synthetic_smoke:
        core.synthetic_smoke(args)
        return
    hparams = core.load_hparams(args)
    args.device = hparams.device
    loader = GraphDataLoaderWrapper(hparams, data_name=hparams.data_name)
    core.sync_hparams_from_loader(hparams, loader)
    if loader.original_graph is None or loader.assignment_matrix is None:
        raise ValueError('Dataset must contain original_graph and assignment_matrix.')
    split_manifest, split_path = core.ensure_split_manifest(loader, args)
    logging.info('Using split manifest: %s', split_path)
    coarse_adj, _coarse_meta, coarse_path = core.load_or_build_coarse_adjacency(args, hparams, loader)
    fold_indices = list(range(int(loader.cv_num_folds))) if args.all_folds else [int(args.fold_idx)]
    results = []
    for fold_idx in fold_indices:
        completed = _load_completed_fold(args, hparams, fold_idx)
        if completed is not None:
            logging.info('Resume: fold %d is already complete.', fold_idx)
            results.append(completed)
            continue
        logging.info('===== Fold %d =====', fold_idx)
        fold_hparams = hparams_lib.copy_hparams(hparams)
        results.append(core.run_one_fold(
            args, fold_hparams, loader, split_manifest, fold_idx, coarse_adj, coarse_path,
        ))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(results) > 1:
        core.save_all_folds_summary(args, hparams, results)


if __name__ == '__main__':
    main()

