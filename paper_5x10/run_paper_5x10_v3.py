# coding=utf-8

"""Final efficient runner: one data load per 10 folds and shared coarse caches."""

import os
import subprocess
import sys
import time

import run_paper_5x10 as base


def run_group(group, gpu, args):
    dataset = group['dataset']
    family = group['family']
    repeat_idx = group['repeat_idx']
    seed = group['seed']
    out_dir, _ = base.result_path(args.root, family, dataset, repeat_idx, 0)
    missing_before = []
    for fold_idx in range(10):
        _, result = base.result_path(args.root, family, dataset, repeat_idx, fold_idx)
        if not base.result_is_complete(result, require_attention=(family == 'mil')):
            missing_before.append(fold_idx)
    if not missing_before:
        return []

    log_dir = os.path.join(args.root, 'logs', f'gpu_{gpu}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset['key']}_{family}_repeat{repeat_idx}_seed{seed}.log")
    command = [
        sys.executable, os.path.join(base.WORK, 'run_coarse_gcn_paper_allfolds_v2.py'),
        '--hparam_path', group['config'], '--data_name', dataset['name'],
        '--processed_data_dir', dataset['data_dir'], '--device', 'cuda', '--out_dir', out_dir,
        '--all_folds', '--create_split_if_missing',
        '--stage2_epochs', str(args.stage2_epochs), '--stage2_patience', str(args.stage2_patience),
        '--seed', str(seed),
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f'\n[{time.strftime("%F %T")}] START folds={missing_before} command={command!r}\n')
        log.flush()
        proc = subprocess.run(command, cwd=base.WORK, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f'[{time.strftime("%F %T")}] END rc={proc.returncode}\n')

    failures = []
    for fold_idx in range(10):
        _, result = base.result_path(args.root, family, dataset, repeat_idx, fold_idx)
        if not base.result_is_complete(result, require_attention=(family == 'mil')):
            failures.append({'fold': fold_idx, 'returncode': int(proc.returncode), 'log': log_path})
    return failures


if __name__ == '__main__':
    base.run_group = run_group
    base.main()

