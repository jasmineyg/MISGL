# coding=utf-8
import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from MISGL.models.encoder import MISGLEncoder
from MISGL.utils.global_variables import g_key
from MISGL.utils.hparam import HParams
from MISGL.utils.load_data import GraphDataLoaderWrapper


def _init_stats():
    return {
        'pos_weights': [],
        'neg_weights': [],
        'pos_bag_top1_hits': [],
        'pos_bag_top3_hits': [],
        'pos_bag_top5_hits': [],
        'correct_pos_bag_count': 0,
        'correct_neg_bag_count': 0,
        'wrong_bag_pos_node_counts': [],
    }


def _write_summary_sheet(writer, stats):
    try:
        summary_data = {
            'Metric': [
                'Average Positive Node Weight',
                'Average Negative Node Weight',
                'Top-1 Hit Probability (Positive Bags)',
                'Top-3 Hit Probability (Positive Bags)',
                'Top-5 Hit Probability (Positive Bags)',
                'Correctly Classified Positive Bags',
                'Correctly Classified Negative Bags',
                'Wrongly Classified Bags - Positive Node Counts',
            ],
            'Value': [
                np.mean(stats['pos_weights']) if stats['pos_weights'] else 0.0,
                np.mean(stats['neg_weights']) if stats['neg_weights'] else 0.0,
                np.mean(stats['pos_bag_top1_hits']) if stats['pos_bag_top1_hits'] else 0.0,
                np.mean(stats['pos_bag_top3_hits']) if stats['pos_bag_top3_hits'] else 0.0,
                np.mean(stats['pos_bag_top5_hits']) if stats['pos_bag_top5_hits'] else 0.0,
                stats['correct_pos_bag_count'],
                stats['correct_neg_bag_count'],
                str(stats['wrong_bag_pos_node_counts']),
            ],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    except Exception as exc:
        logging.error('Error writing summary sheet: %s', exc)
        pd.DataFrame({'Error': [str(exc)]}).to_excel(writer, sheet_name='Summary_Error', index=False)


def _reorder_sheets_to_front(writer, sheet_name='Summary'):
    try:
        book = writer.book
        if sheet_name in book.sheetnames:
            sheets = book._sheets
            target_sheet = book[sheet_name]
            sheets.remove(target_sheet)
            sheets.insert(0, target_sheet)
    except Exception as exc:
        logging.error('Error reordering sheets: %s', exc)


def _sanitize_sheet_name(name):
    sheet_name = str(name)
    for char in (':', '\\', '/', '?', '*', '[', ']'):
        sheet_name = sheet_name.replace(char, '_')
    sheet_name = sheet_name.strip() or 'sheet'
    return sheet_name[:31]


def _unique_sheet_name(base_name, used_names):
    base = _sanitize_sheet_name(base_name)
    if base not in used_names:
        used_names.add(base)
        return base

    suffix = 1
    while True:
        suffix_text = f'_{suffix}'
        candidate = f'{base[:31 - len(suffix_text)]}{suffix_text}'
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _append_suffix_to_filename(path, suffix):
    base, ext = os.path.splitext(path)
    return f'{base}{suffix}{ext}'


def _ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _build_export_loader(loader):
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        worker_init_fn=loader.worker_init_fn,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=False,
        timeout=loader.timeout,
    )


def _resolve_sample_positions(total_count, sample_frac, sample_seed):
    if total_count <= 0 or sample_frac <= 0:
        return set()
    if sample_frac >= 1.0:
        return set(range(total_count))

    sample_count = int(round(float(total_count) * float(sample_frac)))
    sample_count = max(1, min(total_count, sample_count))
    rng = random.Random(0 if sample_seed is None else int(sample_seed))
    return set(rng.sample(list(range(total_count)), sample_count))


def _extract_batch_preds_and_labels(logits, labels, batch_size):
    if logits is None or labels is None:
        return [None] * batch_size, [None] * batch_size

    if logits.dim() == 2 and logits.size(1) == 2:
        preds = logits.argmax(dim=1).detach().cpu().tolist()
    elif logits.dim() == 2 and logits.size(1) == 1:
        preds = (torch.sigmoid(logits) > 0.5).long().view(-1).detach().cpu().tolist()
    elif logits.dim() == 1:
        preds = (torch.sigmoid(logits) > 0.5).long().detach().cpu().tolist()
    else:
        logging.warning('Unexpected logits shape: %s', tuple(logits.shape))
        preds = [0] * batch_size

    return preds, labels.detach().cpu().tolist()


def _extract_weights_per_graph(branch_b_out, num_list):
    a_pad = branch_b_out.get('a_pad', None)
    a_flat = branch_b_out.get('a', None)
    weights_list = []

    if a_pad is not None:
        a_pad_np = a_pad.detach().cpu().numpy()
        for idx, num_nodes in enumerate(num_list):
            weights_list.append(a_pad_np[idx, :num_nodes])
        return weights_list

    if a_flat is not None:
        a_flat_np = a_flat.detach().cpu().numpy()
        cursor = 0
        for num_nodes in num_list:
            weights_list.append(a_flat_np[cursor: cursor + num_nodes])
            cursor += num_nodes
        return weights_list

    return None


def _map_subgraph_nodes_to_labels(subgraph, node_binary_labels, num_nodes):
    nodes = list(subgraph.nodes())
    labels = np.zeros(len(nodes), dtype=np.int64)
    total = len(node_binary_labels)

    for idx, node_id in enumerate(nodes):
        orig_idx = None
        if isinstance(node_id, (int, np.integer)):
            orig_idx = int(node_id)
        else:
            attr = subgraph.nodes[node_id]
            for key in ('original_index', 'orig_id', 'node_index', 'original_id'):
                if key in attr and attr[key] is not None:
                    try:
                        orig_idx = int(attr[key])
                        break
                    except Exception:
                        pass

        if orig_idx is not None and 0 <= orig_idx < total:
            labels[idx] = int(node_binary_labels[orig_idx])

    return labels[:num_nodes]


def _resolve_node_binary_labels(dataset_raw, orig_idx, num_nodes):
    if dataset_raw is None:
        return np.zeros(num_nodes, dtype=np.int64)

    node_labels_collection = dataset_raw.get('node_binary_labels', None)
    subgraphs = dataset_raw.get('subgraph_structures', None)
    if node_labels_collection is None:
        return np.zeros(num_nodes, dtype=np.int64)

    if subgraphs is not None and 0 <= orig_idx < len(subgraphs):
        return _map_subgraph_nodes_to_labels(subgraphs[orig_idx], node_labels_collection, num_nodes)

    if 0 <= orig_idx < len(node_labels_collection):
        labels = np.asarray(node_labels_collection[orig_idx], dtype=np.int64)
        return labels[:num_nodes]

    return np.zeros(num_nodes, dtype=np.int64)


def _export_branchB_attention(
    model,
    loader,
    dataset_raw,
    output_path,
    sample_frac=1.0,
    split_name='test',
    sample_seed=None,
):
    export_loader = _build_export_loader(loader)
    total_graphs = len(export_loader.dataset)
    selected_positions = _resolve_sample_positions(total_graphs, sample_frac, sample_seed)
    stats = _init_stats()

    _ensure_parent_dir(output_path)
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    sheet_name_used = set()

    with torch.no_grad():
        global_pos = 0
        for graph_data in export_loader:
            out = model(graph_data)

            if isinstance(out, dict) and 'ypred_A' in out:
                logits = out['ypred_A']
            elif isinstance(out, dict) and 'ypred' in out:
                logits = out['ypred']
            else:
                logits = None

            labels = graph_data[g_key.y] if g_key.y in graph_data else None
            batch_num_nodes = graph_data[g_key.node_num]
            orig_idx_tensor = graph_data[g_key.orig_graph_idx]

            if isinstance(batch_num_nodes, torch.Tensor):
                num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
            else:
                num_list = [int(n) for n in batch_num_nodes]

            if isinstance(orig_idx_tensor, torch.Tensor):
                orig_indices = [int(i) for i in orig_idx_tensor.detach().cpu().tolist()]
            else:
                orig_indices = [int(i) for i in orig_idx_tensor]

            batch_preds, batch_labels = _extract_batch_preds_and_labels(logits, labels, len(num_list))

            if not isinstance(out, dict) or out.get('branch_b') is None:
                logging.warning('Split %s: current batch has no branch_b output, skipped.', split_name)
                global_pos += len(num_list)
                continue

            weights_list = _extract_weights_per_graph(out['branch_b'], num_list)
            if weights_list is None:
                logging.warning('Split %s: branch_b attention weights not found, skipped batch.', split_name)
                global_pos += len(num_list)
                continue

            for idx, num_nodes in enumerate(num_list):
                current_pos = global_pos
                global_pos += 1
                if current_pos not in selected_positions or num_nodes <= 0:
                    continue

                weights = weights_list[idx]
                orig_idx = orig_indices[idx]
                pred = batch_preds[idx]
                label = batch_labels[idx]
                node_bin_labels = _resolve_node_binary_labels(dataset_raw, orig_idx, num_nodes)

                pos_mask = node_bin_labels == 1
                neg_mask = node_bin_labels == 0
                if pos_mask.any():
                    stats['pos_weights'].extend(weights[pos_mask].tolist())
                if neg_mask.any():
                    stats['neg_weights'].extend(weights[neg_mask].tolist())

                is_correct = False
                if pred is not None and label is not None:
                    if pred == label:
                        is_correct = True
                        if label == 1:
                            stats['correct_pos_bag_count'] += 1
                        else:
                            stats['correct_neg_bag_count'] += 1
                    else:
                        stats['wrong_bag_pos_node_counts'].append(int(np.sum(node_bin_labels)))
                else:
                    logging.warning(
                        'Split %s graph %s: pred=%s, label=%s, skip classification stats.',
                        split_name,
                        orig_idx,
                        pred,
                        label,
                    )

                df = pd.DataFrame({
                    'weight': weights,
                    'node_binary_label': node_bin_labels,
                }).sort_values(by='weight', ascending=False).reset_index(drop=True)

                if label == 1:
                    stats['pos_bag_top1_hits'].append(1 if df.head(1)['node_binary_label'].sum() > 0 else 0)
                    stats['pos_bag_top3_hits'].append(1 if df.head(3)['node_binary_label'].sum() > 0 else 0)
                    stats['pos_bag_top5_hits'].append(1 if df.head(5)['node_binary_label'].sum() > 0 else 0)

                correct_flag = 1 if is_correct else 0
                sheet_name = _unique_sheet_name(f'{orig_idx}_{correct_flag}', sheet_name_used)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    _write_summary_sheet(writer, stats)
    _reorder_sheets_to_front(writer, 'Summary')

    if not sheet_name_used and not stats['pos_weights']:
        logging.warning('No attention maps were exported for %s split to %s.', split_name, output_path)
        pd.DataFrame({'info': ['No attention data exported']}).to_excel(writer, sheet_name='No_Data', index=False)

    writer.close()
    logging.info(
        'Exported branch B attention for %s split to %s (selected %d / %d graphs).',
        split_name,
        output_path,
        len(selected_positions),
        total_graphs,
    )


def export_branchB_attention_from_model(
    model,
    loader,
    hparams,
    dataset_raw,
    output_path,
    sample_frac=1.0,
    split_name='test',
    sample_seed=None,
):
    model.eval()
    if not hasattr(hparams, 'branch_b') or not hparams.branch_b.get('use', False):
        raise RuntimeError('branch_b.use is disabled, cannot export attention.')

    if dataset_raw is not None and dataset_raw.get('node_binary_labels', None) is None:
        logging.warning("node_binary_labels is missing in dataset_raw; exporting zero labels as fallback.")

    _export_branchB_attention(
        model=model,
        loader=loader,
        dataset_raw=dataset_raw,
        output_path=output_path,
        sample_frac=sample_frac,
        split_name=split_name,
        sample_seed=sample_seed,
    )


def export_branchB_attention_to_excel(hparams, output_path, seed=None):
    data_loader = GraphDataLoaderWrapper(hparams)

    if seed is None:
        holdout_seeds = getattr(hparams, 'holdout_seeds', None)
        if isinstance(holdout_seeds, list) and len(holdout_seeds) > 0:
            seed = int(holdout_seeds[0])
        else:
            seed = int(getattr(hparams, 'cv_seed', 1024))

    training_loader, _, test_loader = data_loader.get_holdout_loaders(
        seed=seed,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
    )

    model = MISGLEncoder(hparams).to(torch.device(hparams.device))
    model.eval()

    export_branchB_attention_from_model(
        model,
        test_loader,
        hparams,
        data_loader._dataset_raw,
        output_path,
        sample_frac=1.0,
        split_name='test',
        sample_seed=seed,
    )

    train_output_path = _append_suffix_to_filename(output_path, '_train10p')
    export_branchB_attention_from_model(
        model,
        training_loader,
        hparams,
        data_loader._dataset_raw,
        train_output_path,
        sample_frac=0.1,
        split_name='train',
        sample_seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description='Export Branch B attention to Excel.')
    parser.add_argument(
        '--hparam_path',
        type=str,
        default='./config/hparams_testdb.yml',
        help='Path to the hparams yaml file.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Holdout seed. Defaults to the first holdout seed or cv_seed.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output Excel path. Train 10%% sample export is written to a sibling *_train10p.xlsx file.',
    )
    args = parser.parse_args()

    hparams = HParams()
    hparams.from_yaml(args.hparam_path)

    data_name = getattr(hparams, 'data_name', None)
    if data_name is None or str(data_name).strip() == '':
        data_name_set = getattr(hparams, 'data_name_set', None)
        if isinstance(data_name_set, list) and len(data_name_set) > 0:
            data_name = str(data_name_set[0]).strip()
        else:
            raise RuntimeError('Dataset is not specified; please provide data_name or data_name_set in yaml.')
    hparams.data_name = data_name

    base_ts = getattr(hparams, 'timestamp', None)
    base_ts = str(base_ts).strip() if base_ts is not None else ''
    if base_ts == '':
        base_ts = 'run'
    hparams.timestamp = f'{data_name}_{base_ts}'

    base_save_path = getattr(hparams, 'model_save_path', None)
    if base_save_path:
        hparams.model_save_path = os.path.join(base_save_path, data_name)
    else:
        hparams.model_save_path = os.path.join('results', data_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.cuda_visible_devices

    if args.output is None:
        out_dir = getattr(hparams, 'model_save_path', '.')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f'{hparams.timestamp}_attention.xlsx')
    else:
        output_path = args.output

    export_branchB_attention_to_excel(hparams, output_path, seed=args.seed)


__all__ = [
    'export_branchB_attention_from_model',
    'export_branchB_attention_to_excel',
    'main',
]
