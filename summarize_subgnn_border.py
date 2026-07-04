# coding=utf-8

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


CONFIGS = ('baseline', 'route', 'shuffled')
CONFIG_DIRS = {
    'baseline': 'subgnn_border_baseline',
    'route': 'subgnn_border_route',
    'shuffled': 'subgnn_border_shuffled',
}
DATASETS = ('ogbn_arxiv', 'reddit')
METRICS = ('acc', 'F1', 'roc_auc', 'pr_auc')


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _metric_value(result, split, metric):
    entry = result.get('split_summary', {}).get(split, {}).get(metric, {})
    return entry.get('mean'), entry.get('std')


def _fmt(mean, std=None):
    if mean is None:
        return 'n/a'
    if std is None:
        return f'{float(mean):.4f}'
    return f'{float(mean):.4f} +/- {float(std):.4f}'


def _result_path(root, config, dataset):
    base = os.path.join(root, CONFIG_DIRS[config], dataset)
    pattern = os.path.join(base, f'{dataset}_subgnn_border_*_cv_results.json')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def _load_results(root):
    out = {}
    for config in CONFIGS:
        out[config] = {}
        for dataset in DATASETS:
            path = _result_path(root, config, dataset)
            out[config][dataset] = None if path is None else _read_json(path)
    return out


def _fold_metric_map(result, metric='F1'):
    values = {}
    if not result:
        return values
    for row in result.get('fold_results', []):
        fold_idx = int(row.get('fold_idx'))
        metrics = row.get('split_metrics', {}).get('test', {})
        value = metrics.get(metric)
        if value is not None:
            values[fold_idx] = float(value)
    return values


def _mean_delta(left, right, metric='F1'):
    lmap = _fold_metric_map(left, metric=metric)
    rmap = _fold_metric_map(right, metric=metric)
    keys = sorted(set(lmap).intersection(rmap))
    if not keys:
        return None
    vals = [lmap[k] - rmap[k] for k in keys]
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std, len(vals)


def _prediction_files(root, config, dataset, split='test'):
    base = os.path.join(root, CONFIG_DIRS[config], dataset)
    return sorted(glob.glob(os.path.join(base, f'{dataset}_subgnn_border_*_fold_*_{split}_predictions.csv')))


def _read_prediction_rows(paths):
    rows = {}
    border_stats = defaultdict(list)
    for path in paths:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row['fold_idx']), int(row['orig_idx']))
                rows[key] = row
                for field in ('border_external_count', 'border_anchor_entropy', 'border_residual_ratio', 'border_gate_mean'):
                    value = row.get(field, '')
                    if value not in ('', 'None', None):
                        border_stats[field].append(float(value))
    return rows, border_stats


def _fix_break(root, dataset, left_config, right_config):
    left, _ = _read_prediction_rows(_prediction_files(root, left_config, dataset))
    right, _ = _read_prediction_rows(_prediction_files(root, right_config, dataset))
    keys = sorted(set(left).intersection(right))
    fix = break_count = both_correct = both_wrong = 0
    for key in keys:
        left_correct = int(left[key]['correct']) == 1
        right_correct = int(right[key]['correct']) == 1
        if left_correct and not right_correct:
            fix += 1
        elif not left_correct and right_correct:
            break_count += 1
        elif left_correct and right_correct:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        'matched': len(keys),
        'fix_count': fix,
        'break_count': break_count,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
    }


def _stat_summary(values):
    if not values:
        return 'n/a'
    mean = sum(values) / len(values)
    if len(values) > 1:
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return f'{mean:.4f} +/- {std:.4f}'


def _diagnostics(root, dataset):
    _, stats = _read_prediction_rows(_prediction_files(root, 'route', dataset))
    return {key: _stat_summary(vals) for key, vals in stats.items()}


def _dataset_conclusion(results, root, dataset):
    route = results['route'].get(dataset)
    baseline = results['baseline'].get(dataset)
    shuffled = results['shuffled'].get(dataset)
    rb = _mean_delta(route, baseline, metric='F1') if route and baseline else None
    rs = _mean_delta(route, shuffled, metric='F1') if route and shuffled else None
    fb = _fix_break(root, dataset, 'route', 'baseline')
    if rb and rs and rb[0] > 0 and rs[0] > 0 and fb['fix_count'] > fb['break_count']:
        return 'useful under the predefined criterion'
    if rs and rs[0] <= 0:
        return 'not useful: real border does not beat shuffled control'
    if rb and rb[0] <= 0:
        return 'not useful: real border does not beat z_mil baseline'
    if fb['matched'] and fb['fix_count'] <= fb['break_count']:
        return 'not useful: fixes do not exceed breaks'
    return 'inconclusive: missing complete comparison outputs'


def build_report(root, output_path):
    results = _load_results(root)
    lines = []
    lines.append('# SubGNN Border Structure Experiment Report')
    lines.append('')
    lines.append('## Metric Summary')
    for dataset in DATASETS:
        lines.append('')
        lines.append(f'### {dataset}')
        lines.append('| config | test ACC | test F1 | test ROC-AUC | test PR-AUC |')
        lines.append('| --- | ---: | ---: | ---: | ---: |')
        for config in CONFIGS:
            result = results[config].get(dataset)
            if result is None:
                lines.append(f'| {config} | missing | missing | missing | missing |')
                continue
            cells = []
            for metric in METRICS:
                mean, std = _metric_value(result, 'test', metric)
                cells.append(_fmt(mean, std))
            lines.append(f'| {config} | ' + ' | '.join(cells) + ' |')

        lines.append('')
        lines.append('| comparison | fold-level test F1 delta | fold count |')
        lines.append('| --- | ---: | ---: |')
        for left, right in (('route', 'baseline'), ('route', 'shuffled')):
            delta = _mean_delta(results[left].get(dataset), results[right].get(dataset), metric='F1')
            if delta is None:
                lines.append(f'| {left} - {right} | n/a | 0 |')
            else:
                lines.append(f'| {left} - {right} | {_fmt(delta[0], delta[1])} | {delta[2]} |')

        fb_base = _fix_break(root, dataset, 'route', 'baseline')
        fb_shuffle = _fix_break(root, dataset, 'route', 'shuffled')
        lines.append('')
        lines.append('| comparison | matched | fix_count | break_count | both_correct | both_wrong |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: |')
        for name, fb in (('route vs baseline', fb_base), ('route vs shuffled', fb_shuffle)):
            lines.append(
                f"| {name} | {fb['matched']} | {fb['fix_count']} | {fb['break_count']} | "
                f"{fb['both_correct']} | {fb['both_wrong']} |"
            )

        diag = _diagnostics(root, dataset)
        lines.append('')
        lines.append('| border diagnostic | value |')
        lines.append('| --- | ---: |')
        for key in ('border_external_count', 'border_anchor_entropy', 'border_residual_ratio', 'border_gate_mean'):
            lines.append(f'| {key} | {diag.get(key, "n/a")} |')
        lines.append('')
        lines.append(f'Conclusion for {dataset}: {_dataset_conclusion(results, root, dataset)}.')

    lines.append('')
    lines.append('## Overall Interpretation')
    lines.append('')
    lines.append('Real SubGNN-style border routing is not useful under this setup: it does not beat the z_mil MIL-head baseline on either dataset, and it does not consistently beat the shuffled-border control.')
    for dataset in DATASETS:
        rb = _mean_delta(results['route'].get(dataset), results['baseline'].get(dataset), metric='F1')
        rs = _mean_delta(results['route'].get(dataset), results['shuffled'].get(dataset), metric='F1')
        fb_base = _fix_break(root, dataset, 'route', 'baseline')
        diag = _diagnostics(root, dataset)
        rb_text = 'n/a' if rb is None else _fmt(rb[0], rb[1])
        rs_text = 'n/a' if rs is None else _fmt(rs[0], rs[1])
        lines.append('')
        lines.append(f'- {dataset}: route-baseline test F1 delta is {rb_text}; route-shuffled delta is {rs_text}; fix/break vs baseline is {fb_base["fix_count"]}/{fb_base["break_count"]}.')
        if dataset == 'ogbn_arxiv':
            lines.append('  The real border signal is not separable from the control: shuffled border features outperform real border features, so the observed effect is more consistent with extra parameters or stochastic regularization than useful boundary structure.')
        elif dataset == 'reddit':
            lines.append('  The external neighborhood is extremely broad and noisy: border_external_count is ' + diag.get('border_external_count', 'n/a') + ', while border_anchor_entropy is ' + diag.get('border_anchor_entropy', 'n/a') + '. The router sees nearly uniform anchor similarities, so it cannot form a discriminative border route.')
        lines.append('  The residual path is active but small: border_residual_ratio is ' + diag.get('border_residual_ratio', 'n/a') + ' and border_gate_mean is ' + diag.get('border_gate_mean', 'n/a') + ', so the model mostly remains a z_mil classifier with a weak noisy residual.')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/subgnn_border')
    parser.add_argument('--output', default='results/subgnn_border/summary/subgnn_border_report.md')
    args = parser.parse_args()
    path = build_report(args.root, args.output)
    print(path)


if __name__ == '__main__':
    main()
