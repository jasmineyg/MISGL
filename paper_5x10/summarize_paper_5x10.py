# coding=utf-8

"""Validate and summarize all 50 test folds for paper sections 5 and 6."""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_paper_5x10 import DATASETS, DEFAULT_ROOT, SEEDS, result_path


MODEL_SOURCES = {
    'GAT+mean pool': ('mean', 'stage1', 'baseline_metrics'),
    'MIL-HEAD': ('mil', 'stage1', 'baseline_metrics'),
    'POS-HEAD': ('mean', 'stage2', 'final_metrics'),
    'MISGL': ('mil', 'stage2', 'final_metrics'),
}
METRICS = ('acc', 'F1', 'prec', 'rec')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=DEFAULT_ROOT)
    return parser.parse_args()


def stats(values):
    return {
        'mean': float(np.mean(values)) if values else None,
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
        'num': len(values),
    }


def fmt(value):
    if value['mean'] is None:
        return 'MISSING'
    return f"{100.0 * value['mean']:.2f} ± {100.0 * value['std']:.2f}"


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    summary_dir = os.path.join(root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    fold_rows, missing, attention_rows = [], [], []
    aggregate = {}

    for dataset in DATASETS:
        aggregate[dataset['key']] = {}
        for model, (family, section, block) in MODEL_SOURCES.items():
            values = {metric: [] for metric in METRICS}
            for repeat_idx, seed in enumerate(SEEDS, start=1):
                for fold_idx in range(10):
                    _, path = result_path(root, family, dataset, repeat_idx, fold_idx)
                    try:
                        with open(path, 'r', encoding='utf-8') as handle:
                            payload = json.load(handle)
                        test_metrics = payload[section][block]['test']
                        row = {'dataset_key': dataset['key'], 'paper_name': dataset['paper'], 'model': model,
                               'repeat': repeat_idx, 'seed': seed, 'fold': fold_idx}
                        for metric in METRICS:
                            value = test_metrics.get(metric)
                            row[metric] = value
                            if value is not None:
                                values[metric].append(float(value))
                        fold_rows.append(row)
                    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        missing.append({'dataset': dataset['key'], 'model': model, 'repeat': repeat_idx,
                                        'fold': fold_idx, 'path': path, 'error': repr(exc)})
            aggregate[dataset['key']][model] = {metric: stats(values[metric]) for metric in METRICS}

        for repeat_idx, seed in enumerate(SEEDS, start=1):
            for fold_idx in range(10):
                _, result_file = result_path(root, 'mil', dataset, repeat_idx, fold_idx)
                path = os.path.join(os.path.dirname(result_file), 'attention_metrics.json')
                try:
                    with open(path, 'r', encoding='utf-8') as handle:
                        payload = json.load(handle)
                    for bag in payload['positive_bags']:
                        attention_rows.append({'dataset_key': dataset['key'], 'paper_name': dataset['paper'],
                                               'repeat': repeat_idx, 'seed': seed, 'fold': fold_idx, **bag})
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    missing.append({'dataset': dataset['key'], 'model': 'MIL-attention', 'repeat': repeat_idx,
                                    'fold': fold_idx, 'path': path, 'error': repr(exc)})

    write_csv(os.path.join(summary_dir, 'all_fold_metrics.csv'),
              ['dataset_key', 'paper_name', 'model', 'repeat', 'seed', 'fold', *METRICS], fold_rows)
    attention_fields = [
        'dataset_key', 'paper_name', 'repeat', 'seed', 'fold', 'orig_graph_idx', 'subgraph_id', 'num_nodes',
        'positive_instance_count', 'positive_instance_prevalence', 'positive_attention_mass',
        'positive_attention_enrichment', 'attention_ranking_auc', 'y_prob', 'correct',
    ]
    write_csv(os.path.join(summary_dir, 'attention_enrichment_distribution.csv'), attention_fields, attention_rows)

    attention_summary = []
    for dataset in DATASETS:
        rows = [row for row in attention_rows if row['dataset_key'] == dataset['key']]
        correct = [row for row in rows if row['correct']]
        wrong = [row for row in rows if not row['correct']]
        auc = [float(row['attention_ranking_auc']) for row in rows if row['attention_ranking_auc'] is not None]
        enrichment = [float(row['positive_attention_enrichment']) for row in rows if row['positive_attention_enrichment'] is not None]
        mass = [float(row['positive_attention_mass']) for row in rows]
        correct_e = [float(row['positive_attention_enrichment']) for row in correct if row['positive_attention_enrichment'] is not None]
        wrong_e = [float(row['positive_attention_enrichment']) for row in wrong if row['positive_attention_enrichment'] is not None]
        attention_summary.append({
            'dataset_key': dataset['key'], 'paper_name': dataset['paper'], 'positive_bag_observations': len(rows),
            'ranking_auc_mean': stats(auc)['mean'], 'ranking_auc_std': stats(auc)['std'],
            'positive_attention_mass_mean': stats(mass)['mean'], 'positive_attention_mass_std': stats(mass)['std'],
            'enrichment_mean': stats(enrichment)['mean'], 'enrichment_std': stats(enrichment)['std'],
            'correct_enrichment_mean': stats(correct_e)['mean'], 'correct_enrichment_std': stats(correct_e)['std'],
            'correct_n': len(correct_e), 'incorrect_enrichment_mean': stats(wrong_e)['mean'],
            'incorrect_enrichment_std': stats(wrong_e)['std'], 'incorrect_n': len(wrong_e),
        })
    write_csv(os.path.join(summary_dir, 'attention_summary.csv'), list(attention_summary[0].keys()), attention_summary)

    table_lines = ['| Dataset | GAT+mean pool | MIL-HEAD | POS-HEAD | MISGL |',
                   '|---|---:|---:|---:|---:|']
    for dataset in DATASETS:
        cells = [dataset['paper']] + [fmt(aggregate[dataset['key']][model]['acc']) for model in MODEL_SOURCES]
        table_lines.append('| ' + ' | '.join(cells) + ' |')
    with open(os.path.join(summary_dir, 'paper_acc_table.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(table_lines) + '\n')

    values_correct, values_wrong, positions_correct, positions_wrong, labels = [], [], [], [], []
    for idx, dataset in enumerate(DATASETS):
        rows = [row for row in attention_rows if row['dataset_key'] == dataset['key']]
        c = [float(row['positive_attention_enrichment']) for row in rows
             if row['correct'] and row['positive_attention_enrichment'] is not None]
        w = [float(row['positive_attention_enrichment']) for row in rows
             if not row['correct'] and row['positive_attention_enrichment'] is not None]
        if c:
            values_correct.append(c); positions_correct.append(idx * 3.0 - 0.45)
        if w:
            values_wrong.append(w); positions_wrong.append(idx * 3.0 + 0.45)
        labels.append(dataset['paper'])
    fig, ax = plt.subplots(figsize=(19, 8))
    if values_correct:
        bp = ax.boxplot(values_correct, positions=positions_correct, widths=0.7, patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set_facecolor('#4C78A8')
    if values_wrong:
        bp = ax.boxplot(values_wrong, positions=positions_wrong, widths=0.7, patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set_facecolor('#E45756')
    ax.set_xticks([idx * 3.0 for idx in range(len(DATASETS))])
    ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_ylabel('Positive Attention Enrichment')
    ax.set_title('Correct vs. Incorrect Positive Bags (5×10 CV)')
    ax.grid(axis='y', alpha=0.25)
    ax.plot([], [], color='#4C78A8', linewidth=8, label='Correct positive bag')
    ax.plot([], [], color='#E45756', linewidth=8, label='Incorrect positive bag')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, 'attention_enrichment_boxplot.png'), dpi=220)
    plt.close(fig)

    completeness = {
        'expected_metric_observations': len(DATASETS) * len(MODEL_SOURCES) * 50,
        'actual_metric_observations': len(fold_rows),
        'expected_attention_fold_files': len(DATASETS) * 50,
        'missing_count': len(missing), 'missing': missing,
    }
    with open(os.path.join(summary_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump({'aggregate': aggregate, 'attention': attention_summary, 'completeness': completeness},
                  handle, indent=2, ensure_ascii=False)
    with open(os.path.join(summary_dir, 'completeness.json'), 'w', encoding='utf-8') as handle:
        json.dump(completeness, handle, indent=2, ensure_ascii=False)
    print(json.dumps(completeness, ensure_ascii=False))
    if missing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

