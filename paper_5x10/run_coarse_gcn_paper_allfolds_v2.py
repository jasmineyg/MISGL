# coding=utf-8

"""All-fold resume runner with a per-dataset shared coarse-adjacency cache."""

import copy
import fcntl
import os

import run_coarse_gcn_paper as core


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


core.load_or_build_coarse_adjacency = _load_or_build_shared

from run_coarse_gcn_paper_allfolds import main


if __name__ == '__main__':
    main()

