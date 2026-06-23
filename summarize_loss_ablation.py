# coding=utf-8

import argparse
import csv
import json
import os

import numpy as np


LOSSES = ('bce', 'focal', 'weighted_bce')
DATASETS = ('ogbn_arxiv', 'reddit')
METRICS = ('acc', 'F1', 'prec', 'rec', 'balanced_acc', 'roc_auc', 'pr_auc')


def _load_result(root, loss_name, dataset_name):
    path = os.path.join(
        root,
        loss_name,
        dataset_name,
        '{}_loss_ablation_{}_cv_results.json'.format(dataset_name, loss_name),
    )
    with open(path, 'r', encoding='utf-8') as handle:
        return path, json.load(handle)


def _fold_values(result, metric):
    return np.asarray(
        [float(fold['metrics'][metric]) for fold in result['fold_results']],
        dtype=np.float64,
    )


def _paired_bootstrap_ci(candidate, baseline, seed=20260623, samples=20000):
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _classification(mean_delta, ci):
    if mean_delta > 0.0 and ci[0] > 0.0:
        return 'stable_gain'
    if mean_delta > 0.0:
        return 'possible_gain'
    return 'no_acc_gain'


def summarize(root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    loaded = {}
    source_files = {}
    for dataset in DATASETS:
        loaded[dataset] = {}
        source_files[dataset] = {}
        for loss_name in LOSSES:
            path, result = _load_result(root, loss_name, dataset)
            loaded[dataset][loss_name] = result
            source_files[dataset][loss_name] = path

    summary = {
        'root': os.path.abspath(root),
        'bootstrap_seed': 20260623,
        'bootstrap_samples': 20000,
        'datasets': {},
        'source_files': source_files,
    }
    csv_rows = []
    markdown = [
        '# GAT → MIL-Head Loss Ablation',
        '',
        'Primary criterion: test ACC at threshold 0.5; fixed 10-fold paired comparison.',
        '',
    ]

    for dataset in DATASETS:
        dataset_summary = {'losses': {}, 'comparisons_vs_bce': {}}
        markdown.extend([
            '## {}'.format(dataset),
            '',
            '| Loss | ACC | F1 | Precision | Recall | Balanced ACC | ROC-AUC | PR-AUC |',
            '|---|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for loss_name in LOSSES:
            result = loaded[dataset][loss_name]
            loss_metrics = {}
            row = {'dataset': dataset, 'loss': loss_name}
            for metric in METRICS:
                values = _fold_values(result, metric)
                metric_summary = {
                    'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)),
                }
                loss_metrics[metric] = metric_summary
                row['{}_mean'.format(metric)] = metric_summary['mean']
                row['{}_std'.format(metric)] = metric_summary['std']
            dataset_summary['losses'][loss_name] = loss_metrics
            csv_rows.append(row)
            markdown.append(
                '| {loss} | {acc:.4f} ± {acc_std:.4f} | {f1:.4f} ± {f1_std:.4f} | '
                '{prec:.4f} | {rec:.4f} | {balanced:.4f} | {roc:.4f} | {pr:.4f} |'.format(
                    loss=loss_name,
                    acc=loss_metrics['acc']['mean'],
                    acc_std=loss_metrics['acc']['std'],
                    f1=loss_metrics['F1']['mean'],
                    f1_std=loss_metrics['F1']['std'],
                    prec=loss_metrics['prec']['mean'],
                    rec=loss_metrics['rec']['mean'],
                    balanced=loss_metrics['balanced_acc']['mean'],
                    roc=loss_metrics['roc_auc']['mean'],
                    pr=loss_metrics['pr_auc']['mean'],
                )
            )

        baselines = {
            metric: _fold_values(loaded[dataset]['bce'], metric)
            for metric in ('acc', 'F1')
        }
        markdown.extend(['', 'Paired test deltas versus BCE:', ''])
        for loss_name in ('focal', 'weighted_bce'):
            comparison = {}
            for metric in ('acc', 'F1'):
                candidate = _fold_values(loaded[dataset][loss_name], metric)
                baseline = baselines[metric]
                delta = candidate - baseline
                ci = _paired_bootstrap_ci(candidate, baseline)
                metric_comparison = {
                    'mean_delta': float(delta.mean()),
                    'std_delta': float(delta.std(ddof=1)),
                    'bootstrap_95_ci': ci,
                    'wins': int(np.sum(delta > 1e-12)),
                    'ties': int(np.sum(np.abs(delta) <= 1e-12)),
                    'losses': int(np.sum(delta < -1e-12)),
                    'fold_deltas': [float(x) for x in delta],
                }
                if metric == 'acc':
                    metric_comparison['classification'] = _classification(
                        float(delta.mean()), ci
                    )
                comparison[metric] = metric_comparison
            dataset_summary['comparisons_vs_bce'][loss_name] = comparison
            acc_comparison = comparison['acc']
            f1_comparison = comparison['F1']
            markdown.append(
                '- {}: delta_ACC={:+.4f}, 95% CI [{:+.4f}, {:+.4f}], '
                'W/T/L={}/{}/{}, {}; delta_F1={:+.4f}, 95% CI '
                '[{:+.4f}, {:+.4f}], W/T/L={}/{}/{}.'.format(
                    loss_name,
                    acc_comparison['mean_delta'],
                    acc_comparison['bootstrap_95_ci'][0],
                    acc_comparison['bootstrap_95_ci'][1],
                    acc_comparison['wins'],
                    acc_comparison['ties'],
                    acc_comparison['losses'],
                    acc_comparison['classification'],
                    f1_comparison['mean_delta'],
                    f1_comparison['bootstrap_95_ci'][0],
                    f1_comparison['bootstrap_95_ci'][1],
                    f1_comparison['wins'],
                    f1_comparison['ties'],
                    f1_comparison['losses'],
                )
            )
        summary['datasets'][dataset] = dataset_summary
        markdown.append('')

    json_path = os.path.join(output_dir, 'loss_ablation_summary.json')
    csv_path = os.path.join(output_dir, 'loss_ablation_metrics.csv')
    markdown_path = os.path.join(output_dir, 'LOSS_ABLATION_REPORT.md')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    with open(markdown_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(markdown) + '\n')
    return json_path, csv_path, markdown_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        default='results/loss_ablation/gat_mil_head_20260623',
    )
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(args.root, 'summary')
    for path in summarize(args.root, output_dir):
        print(path)


if __name__ == '__main__':
    main()
