# coding=utf-8
import os
import argparse
import logging
import numpy as np
import torch
import pandas as pd
import random

from gnn_hpool.utils.hparam import HParams
from gnn_hpool.utils.global_variables import g_key
from gnn_hpool.utils.load_data import GraphDataLoaderWrapper
from gnn_hpool.models.gcn_hpool_encoder import GcnHpoolEncoder


def export_branchB_attention_to_excel(hparams, output_path, seed=None):
    # 构造数据加载器（使用 Holdout 划分）
    data_loader = GraphDataLoaderWrapper(hparams)

    if seed is None:
        holdout_seeds = getattr(hparams, 'holdout_seeds', None)
        if isinstance(holdout_seeds, list) and len(holdout_seeds) > 0:
            seed = int(holdout_seeds[0])
        else:
            seed = int(getattr(hparams, 'cv_seed', 1024))

    _, _, test_loader = data_loader.get_holdout_loaders(
        seed=seed, train_frac=0.6, val_frac=0.2, test_frac=0.2
    )

    model = GcnHpoolEncoder(hparams).to(torch.device(hparams.device))
    model.eval()

    if not hasattr(hparams, 'branch_b') or not hparams.branch_b.get('use', False):
        raise RuntimeError("branch_b.use 未开启，无法导出注意力 a。请在配置中启用分支B。")

    # 从原始数据集读取节点二分类标签
    dataset_raw = data_loader._dataset_raw
    node_labels_collection = dataset_raw.get('node_binary_labels', None)
    if node_labels_collection is None:
        logging.warning("未在数据集 pkl 中找到 key='node_binary_labels'，将以全0占位。")

    # 写 Excel（每个sheet一个graph）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    with torch.no_grad():
        sheet_name_used = set()
        for _, graph_data in enumerate(test_loader):
            # 前向调用，获得分支B的对齐注意力（按 [B, M] 填充）
            out = model(graph_data)
            if not isinstance(out, dict) or 'branch_b' not in out or out['branch_b'] is None:
                logging.warning("当前批次没有分支B输出，已跳过。")
                continue

            b_out = out['branch_b']
            a_pad = b_out.get('a_pad', None)        # [B, M]
            mask_valid = b_out.get('mask_valid', None)  # [B, M]，未使用，仅说明有效位置
            if a_pad is None:
                logging.warning("未找到 a_pad（分支B注意力对齐向量），已跳过。")
                continue

            # 取每个样本的真实节点数与原始图索引
            batch_num_nodes = graph_data[g_key.node_num]
            orig_idx_tensor = graph_data[g_key.orig_graph_idx]

            # 转到 CPU 处理
            a_pad_np = a_pad.detach().cpu().numpy()
            if isinstance(batch_num_nodes, torch.Tensor):
                num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
            else:
                num_list = [int(n) for n in batch_num_nodes]

            if isinstance(orig_idx_tensor, torch.Tensor):
                orig_indices = [int(i) for i in orig_idx_tensor.detach().cpu().tolist()]
            else:
                orig_indices = [int(i) for i in orig_idx_tensor]

            # 逐图写入
            for i, n_i in enumerate(num_list):
                if n_i <= 0:
                    continue
                weights = a_pad_np[i, :n_i]
                orig_idx = orig_indices[i]

                if node_labels_collection is not None and 0 <= orig_idx < len(node_labels_collection):
                    node_bin_labels = np.asarray(node_labels_collection[orig_idx])
                    node_bin_labels = node_bin_labels[:n_i]
                else:
                    node_bin_labels = np.zeros(n_i, dtype=np.int64)

                df = pd.DataFrame({
                    'weight': weights,
                    'node_binary_label': node_bin_labels
                })
                # 按权重降序排列并重置索引
                df = df.sort_values(by='weight', ascending=False).reset_index(drop=True)

                # 每个sheet一个graph；避免重名
                base_name = f'graph_{orig_idx}'
                sheet_name = base_name
                suffix = 1
                while sheet_name in sheet_name_used:
                    sheet_name = f'{base_name}_{suffix}'
                    suffix += 1
                sheet_name_used.add(sheet_name)

                df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()
    print(f'Exported branch B attention for test graphs to {output_path}')


def export_branchB_attention_from_model(model, test_loader, hparams, dataset_raw, output_path, sample_frac=0.2):
    model.eval()
    if not hasattr(hparams, 'branch_b') or not hparams.branch_b.get('use', False):
        raise RuntimeError("branch_b.use 未开启，无法导出注意力 a。")

    node_labels_collection = dataset_raw.get('node_binary_labels', None)
    subgraphs = dataset_raw.get('subgraph_structures', None)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 采样逻辑
    total_graphs = len(test_loader.dataset)
    num_sample = int(total_graphs * sample_frac)
    if num_sample < 1 and total_graphs > 0: num_sample = 1
    sampled_indices = set(random.sample(range(total_graphs), num_sample))
    
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    with torch.no_grad():
        sheet_name_used = set()
        current_idx = 0
        for _, graph_data in enumerate(test_loader):
            out = model(graph_data)
            if not isinstance(out, dict) or 'branch_b' not in out or out['branch_b'] is None:
                # 即使没有输出，也要增加计数以保持对齐（假设 batch size 是确定的）
                # 但这里更安全的是根据 node_num 的长度来增加
                batch_num_nodes = graph_data['node_num']
                bs = len(batch_num_nodes) if not isinstance(batch_num_nodes, torch.Tensor) else batch_num_nodes.size(0)
                current_idx += bs
                continue

            b_out = out['branch_b']
            a_pad = b_out.get('a_pad', None)
            if a_pad is None:
                batch_num_nodes = graph_data['node_num']
                bs = len(batch_num_nodes) if not isinstance(batch_num_nodes, torch.Tensor) else batch_num_nodes.size(0)
                current_idx += bs
                continue

            batch_num_nodes = graph_data['node_num']
            orig_idx_tensor = graph_data['orig_graph_idx']

            a_pad_np = a_pad.detach().cpu().numpy()
            if isinstance(batch_num_nodes, torch.Tensor):
                num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
            else:
                num_list = [int(n) for n in batch_num_nodes]
            if isinstance(orig_idx_tensor, torch.Tensor):
                orig_indices = [int(i) for i in orig_idx_tensor.detach().cpu().tolist()]
            else:
                orig_indices = [int(i) for i in orig_idx_tensor]

            for i, n_i in enumerate(num_list):
                if current_idx not in sampled_indices:
                    current_idx += 1
                    continue
                
                if n_i <= 0:
                    current_idx += 1
                    continue
                weights = a_pad_np[i, :n_i]
                orig_idx = orig_indices[i]

                # 修复：按子图节点映射到原图索引再取标签
                if node_labels_collection is not None and subgraphs is not None and 0 <= orig_idx < len(subgraphs):
                    subgraph = subgraphs[orig_idx]
                    node_bin_labels = _map_subgraph_nodes_to_labels(subgraph, node_labels_collection, n_i)
                else:
                    node_bin_labels = np.zeros(n_i, dtype=np.int64)

                # 导出仅包含权重与节点二分类标签
                df = pd.DataFrame({
                    'weight': weights,
                    'node_binary_label': node_bin_labels
                })
                df = df.sort_values(by='weight', ascending=False).reset_index(drop=True)

                base_name = f'graph_{orig_idx}'
                sheet_name = base_name
                suffix = 1
                while sheet_name in sheet_name_used:
                    sheet_name = f'{base_name}_{suffix}'
                    suffix += 1
                sheet_name_used.add(sheet_name)

                df.to_excel(writer, sheet_name=sheet_name, index=False)
                current_idx += 1

    writer.close()

def _map_subgraph_nodes_to_labels(subgraph, node_binary_labels, n_i):
    nodes = list(subgraph.nodes())
    labels = np.zeros(len(nodes), dtype=np.int64)
    total = len(node_binary_labels)
    for j, node_id in enumerate(nodes):
        idx = None
        # 优先：节点ID就是原图索引（常见情况）
        if isinstance(node_id, (int, np.integer)):
            idx = int(node_id)
        else:
            # 回退：从节点属性取原图索引
            attr = subgraph.nodes[node_id]
            for key in ('original_index', 'orig_id', 'node_index'):
                if key in attr and attr[key] is not None:
                    try:
                        idx = int(attr[key])
                        break
                    except Exception:
                        pass
        if idx is not None and 0 <= idx < total:
            labels[j] = int(node_binary_labels[idx])
        else:
            labels[j] = 0  # 无法映射时置0
    return labels[:n_i]


def main():
    parser = argparse.ArgumentParser(description='Export Branch B attention (a) to Excel.')
    parser.add_argument('--hparam_path', type=str, default='./config/hparams_testdb.yml',
                        help='配置文件路径（.yml）。')
    parser.add_argument('--seed', type=int, default=None,
                        help='Holdout 随机种子（默认取配置中的第一个 holdout_seeds 或 cv_seed）。')
    parser.add_argument('--output', type=str, default=None,
                        help='导出 Excel 文件路径（默认写到 model_save_path/timestamp_branchB_attention.xlsx）。')
    args = parser.parse_args()

    # 读取配置
    hparams = HParams()
    hparams.from_yaml(args.hparam_path)

    data_name = getattr(hparams, 'data_name', None)
    if data_name is None or str(data_name).strip() == '':
        data_name_set = getattr(hparams, 'data_name_set', None)
        if isinstance(data_name_set, list) and len(data_name_set) > 0:
            data_name = str(data_name_set[0]).strip()
        else:
            raise RuntimeError('未指定数据集：请在yml中提供 data_name_set。')
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

    # 设备与可见 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.cuda_visible_devices

    # 输出文件默认路径
    if args.output is None:
        out_dir = getattr(hparams, 'model_save_path', '.')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f'{hparams.timestamp}_attention.xlsx')
    else:
        output_path = args.output

    export_branchB_attention_to_excel(hparams, output_path, seed=args.seed)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
