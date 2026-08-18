# coding=utf-8

"""Run the paper's 5 repeats x 10 folds matrix with resume support."""

import argparse
import copy
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import yaml


WORK = '/data/yg/Subgraph-MIL/diffpool2'
DATA_ROOT = '/data/yg/Subgraph-MIL/Data/processed_data'
DEFAULT_ROOT = os.path.join(WORK, 'results', 'paper_5x10_20260813')
SEEDS = [1024, 2048, 4096, 8192, 16384]

DATASETS = [
    dict(key='products', paper='ogbn_products', name='ogbn_products', data_dir=DATA_ROOT, kind='real'),
    dict(key='products_oracle', paper='products_oracle', name='ogbn_products/ogbn_products_semantic_oracle_striped', data_dir=f'{DATA_ROOT}/semantic_oracle_striped', kind='real'),
    dict(key='products_perturb50', paper='products_扰动50', name='ogbn_products/ogbn_products_metis_perturbed_50', data_dir=f'{DATA_ROOT}/partition_variants', kind='real'),
    dict(key='products_random', paper='products_random', name='ogbn_products/ogbn_products_random_constrained', data_dir=f'{DATA_ROOT}/partition_variants', kind='real'),
    dict(key='reddit', paper='reddit', name='reddit', data_dir=DATA_ROOT, kind='real'),
    dict(key='reddit_oracle', paper='reddit_oracle', name='reddit/reddit_semantic_oracle_striped', data_dir=f'{DATA_ROOT}/semantic_oracle_striped', kind='real'),
    dict(key='reddit_perturb50', paper='reddit_扰动50', name='reddit/reddit_metis_perturbed_50', data_dir=f'{DATA_ROOT}/partition_variants', kind='real'),
    dict(key='reddit_random', paper='reddit_random', name='reddit/reddit_random_constrained', data_dir=f'{DATA_ROOT}/partition_variants', kind='real'),
    dict(key='arxiv', paper='ogbn_arxiv', name='ogbn_arxiv', data_dir=DATA_ROOT, kind='real'),
    dict(key='arxiv_oracle', paper='arxiv_oracle', name='ogbn_arxiv/ogbn_arxiv_semantic_oracle_striped', data_dir=f'{DATA_ROOT}/semantic_oracle_striped', kind='real'),
    dict(key='arxiv_perturb50', paper='arxiv_扰动50', name='ogbn_arxiv/ogbn_arxiv_metis_perturbed_50', data_dir=f'{DATA_ROOT}/partition_variants', kind='real'),
    dict(key='arxiv_random', paper='arxiv_random', name='ogbn_arxiv/ogbn_arxiv_random_constrained', data_dir=f'{DATA_ROOT}/partition_variants', kind='real'),
    dict(key='syn1', paper='syn1', name='synthetic_milinst_mil_strong_pos_random_v2', data_dir=f'{DATA_ROOT}/synthetic_mil_consistent/synthetic_milinst_mil_strong_pos_random_v2', kind='synthetic'),
    dict(key='syn2', paper='syn2', name='synthetic_milinst_mil_weak_pos_strong_v2', data_dir=f'{DATA_ROOT}/synthetic_mil_consistent/synthetic_milinst_mil_weak_pos_strong_v2', kind='synthetic'),
    dict(key='syn3', paper='syn3', name='synthetic_milinst_both_useful_v2', data_dir=f'{DATA_ROOT}/synthetic_mil_consistent/synthetic_milinst_both_useful_v2', kind='synthetic'),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in SEEDS))
    parser.add_argument('--stage2_epochs', type=int, default=200)
    parser.add_argument('--stage2_patience', type=int, default=35)
    return parser.parse_args()


def config_source(kind, family):
    if kind == 'synthetic':
        filename = 'config_synthetic_mean_pool.yml' if family == 'mean' else 'config_synthetic_mil_head.yml'
    else:
        filename = 'b_off.yml' if family == 'mean' else 'b_on.yml'
    return os.path.join(WORK, 'config', filename)


def write_config(root, kind, family, seed):
    with open(config_source(kind, family), 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg = copy.deepcopy(cfg)
    cfg['cv_seed'] = int(seed)
    cfg['cv_num_folds'] = 10
    cfg['cv_val_policy'] = 'adjacent'
    cfg['cv_use_all_samples'] = True
    cfg['cv_split_dir'] = os.path.join(root, 'splits')
    cfg['export_attention'] = False
    cfg['analyze_attention'] = False
    cfg['enable_tensorboard'] = False
    cfg['preload_data_to_gpu'] = False
    path = os.path.join(root, 'configs', f'{kind}_{family}_seed{seed}.yml')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
    return path


def result_path(root, family, dataset, repeat_idx, fold_idx):
    out_dir = os.path.join(root, family, dataset['key'], f'repeat_{repeat_idx}')
    return out_dir, os.path.join(out_dir, dataset['name'], f'fold_{fold_idx}', 'coarse_gcn_results.json')


def result_is_complete(path, require_attention):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        for section, block in (('stage1', 'baseline_metrics'), ('stage2', 'final_metrics')):
            if payload[section][block]['test'].get('acc') is None:
                return False
        if require_attention:
            attention_path = os.path.join(os.path.dirname(path), 'attention_metrics.json')
            with open(attention_path, 'r', encoding='utf-8') as handle:
                attention = json.load(handle)
            if 'positive_bags' not in attention:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_groups(root, seeds):
    configs = {}
    for kind in ('real', 'synthetic'):
        for family in ('mean', 'mil'):
            for seed in seeds:
                configs[(kind, family, seed)] = write_config(root, kind, family, seed)
    groups = []
    for dataset in DATASETS:
        for repeat_idx, seed in enumerate(seeds, start=1):
            for family in ('mean', 'mil'):
                groups.append({
                    'dataset': dataset, 'repeat_idx': repeat_idx, 'seed': seed, 'family': family,
                    'config': configs[(dataset['kind'], family, seed)],
                })
    return groups


def run_group(group, gpu, args):
    dataset = group['dataset']
    family = group['family']
    repeat_idx = group['repeat_idx']
    seed = group['seed']
    out_dir, _ = result_path(args.root, family, dataset, repeat_idx, 0)
    log_dir = os.path.join(args.root, 'logs', f'gpu_{gpu}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset['key']}_{family}_repeat{repeat_idx}_seed{seed}.log")
    failures = []
    for fold_idx in range(10):
        _, expected = result_path(args.root, family, dataset, repeat_idx, fold_idx)
        if result_is_complete(expected, require_attention=(family == 'mil')):
            continue
        command = [
            sys.executable, os.path.join(WORK, 'run_coarse_gcn_paper.py'),
            '--hparam_path', group['config'], '--data_name', dataset['name'],
            '--processed_data_dir', dataset['data_dir'], '--device', 'cuda', '--out_dir', out_dir,
            '--fold_idx', str(fold_idx), '--create_split_if_missing',
            '--stage2_epochs', str(args.stage2_epochs), '--stage2_patience', str(args.stage2_patience),
            '--seed', str(seed),
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f'\n[{time.strftime("%F %T")}] START fold={fold_idx} command={command!r}\n')
            log.flush()
            proc = subprocess.run(command, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)
            log.write(f'[{time.strftime("%F %T")}] END fold={fold_idx} rc={proc.returncode}\n')
        if proc.returncode != 0 or not result_is_complete(expected, require_attention=(family == 'mil')):
            failures.append({'fold': fold_idx, 'returncode': int(proc.returncode), 'log': log_path})
    return failures


def worker(gpu, task_queue, args, outcomes, lock):
    while True:
        try:
            group = task_queue.get_nowait()
        except queue.Empty:
            return
        label = f"{group['dataset']['key']}/{group['family']}/r{group['repeat_idx']}"
        print(f'[{time.strftime("%F %T")}] GPU {gpu} START {label}', flush=True)
        try:
            failures = run_group(group, gpu, args)
            outcome = {'label': label, 'gpu': gpu, 'failures': failures}
        except Exception as exc:
            outcome = {'label': label, 'gpu': gpu, 'failures': [{'error': repr(exc), 'traceback': traceback.format_exc()}]}
        with lock:
            outcomes.append(outcome)
            with open(os.path.join(args.root, 'orchestrator_status.json'), 'w', encoding='utf-8') as handle:
                json.dump({'updated_at': time.strftime('%F %T'), 'outcomes': outcomes}, handle, indent=2, ensure_ascii=False)
        print(f'[{time.strftime("%F %T")}] GPU {gpu} END {label} failures={len(outcome["failures"])}', flush=True)
        task_queue.task_done()


def main():
    args = parse_args()
    args.root = os.path.abspath(args.root)
    gpus = [item.strip() for item in args.gpus.split(',') if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    if len(seeds) != 5:
        raise ValueError(f'Exactly five repeat seeds are required, got {seeds}')
    os.makedirs(args.root, exist_ok=True)
    free_gb = shutil.disk_usage(args.root).free / (1024 ** 3)
    if free_gb < 40:
        raise RuntimeError(f'At least 40 GiB free space is required; only {free_gb:.1f} GiB remains.')
    groups = build_groups(args.root, seeds)
    manifest = {
        'protocol': '5_repeats_x_10_folds_grouped_stratified_cv_8_1_1',
        'seeds': seeds, 'folds': 10,
        'models': {'mean/stage1': 'GAT+mean pool', 'mil/stage1': 'MIL-HEAD', 'mean/stage2': 'POS-HEAD', 'mil/stage2': 'MISGL'},
        'datasets': DATASETS, 'created_at': time.strftime('%F %T'),
    }
    with open(os.path.join(args.root, 'experiment_manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    task_queue = queue.Queue()
    for group in groups:
        task_queue.put(group)
    outcomes = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(gpu, task_queue, args, outcomes, lock)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [outcome for outcome in outcomes if outcome['failures']]
    final = {
        'completed_at': time.strftime('%F %T'), 'group_count': len(groups),
        'completed_group_count': len(outcomes), 'failed_group_count': len(failures), 'failures': failures,
    }
    with open(os.path.join(args.root, 'orchestrator_final.json'), 'w', encoding='utf-8') as handle:
        json.dump(final, handle, indent=2, ensure_ascii=False)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

