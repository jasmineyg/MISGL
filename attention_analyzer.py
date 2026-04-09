# coding=utf-8
import os
import argparse
import logging
import numpy as np
import torch
import pandas as pd
import random

from MISGL.utils.hparam import HParams
from MISGL.utils.global_variables import g_key
from MISGL.utils.load_data import GraphDataLoaderWrapper
from MISGL.models.encoder import GcnHpoolEncoder


def _write_summary_sheet(writer, stats):
    """
    将统计信息写入 'Summary' 工作表
    """
    try:
        # 计算平均权重
        avg_pos_weight = np.mean(stats['pos_weights']) if stats['pos_weights'] else 0.0
        avg_neg_weight = np.mean(stats['neg_weights']) if stats['neg_weights'] else 0.0
        avg_pos_bag_top1_hit_rate = np.mean(stats['pos_bag_top1_hits']) if stats['pos_bag_top1_hits'] else 0.0
        avg_pos_bag_top3_hit_rate = np.mean(stats['pos_bag_top3_hits']) if stats['pos_bag_top3_hits'] else 0.0
        avg_pos_bag_top5_hit_rate = np.mean(stats['pos_bag_top5_hits']) if stats['pos_bag_top5_hits'] else 0.0
        
        # 构建 Summary DataFrame
        summary_data = {
            'Metric': [
                'Average Positive Node Weight',
                'Average Negative Node Weight',
                'Top-1 Hit Probability (Positive Bags)',
                'Top-3 Hit Probability (Positive Bags)',
                'Top-5 Hit Probability (Positive Bags)',
                'Correctly Classified Positive Bags',
                'Correctly Classified Negative Bags',
                'Wrongly Classified Bags - Positive Node Counts'
            ],
            'Value': [
                avg_pos_weight,
                avg_neg_weight,
                avg_pos_bag_top1_hit_rate,
                avg_pos_bag_top3_hit_rate,
                avg_pos_bag_top5_hit_rate,
                stats['correct_pos_bag_count'],
                stats['correct_neg_bag_count'],
                str(stats['wrong_bag_pos_node_counts']) # 转为字符串以存储在单元格中
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        
        # 将 Summary sheet 插入到第一个位置
        # openpyxl 的 writer.book.create_sheet 可以指定 index
        # 但 pandas 的 to_excel 默认是在末尾追加
        # 所以先写入，然后通过 openpyxl 调整 sheet 顺序
        
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # 也可以添加更详细的错误分类包的正节点数列表（如果列表很长，放在单独的列可能更好）
        # 这里为了简单，如果需要详细展开，可以额外加列
        if stats['wrong_bag_pos_node_counts']:
            df_wrong_details = pd.DataFrame({
                'Wrong_Bag_Index': range(len(stats['wrong_bag_pos_node_counts'])),
                'Positive_Node_Count': stats['wrong_bag_pos_node_counts']
            })
            pass
            
    except Exception as e:
        logging.error(f"Error writing summary sheet: {e}")
        # 写入错误信息以防万一
        pd.DataFrame({'Error': [str(e)]}).to_excel(writer, sheet_name='Summary_Error', index=False)

def _reorder_sheets_to_front(writer, sheet_name='Summary'):
    """
    将指定的 sheet 移动到第一个位置
    """
    try:
        book = writer.book
        if sheet_name in book.sheetnames:
            sheets = book._sheets
            target_sheet = book[sheet_name]
            sheets.remove(target_sheet)
            sheets.insert(0, target_sheet)
    except Exception as e:
        logging.error(f"Error reordering sheets: {e}")


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

    # 统计容器
    stats = {
        'pos_weights': [],
        'neg_weights': [],
        'pos_bag_top1_hits': [],
        'pos_bag_top3_hits': [],
        'pos_bag_top5_hits': [],
        'correct_pos_bag_count': 0,
        'correct_neg_bag_count': 0,
        'wrong_bag_pos_node_counts': []
    }

    current_idx = 0

    with torch.no_grad():
        sheet_name_used = set()
        for _, graph_data in enumerate(test_loader):
            # 前向调用
            out = model(graph_data)
            
            # 获取预测结果和真实标签
            if 'ypred_A' in out:
                logits = out['ypred_A']
            elif isinstance(out, dict) and 'ypred' in out: # 兼容旧接口
                logits = out['ypred']
            else:
                # 无法获取预测结果，无法判断正确性
                logits = None
            
            # 获取 Ground Truth
            if g_key.y in graph_data:
                labels = graph_data[g_key.y]
            else:
                labels = None

            if not isinstance(out, dict) or 'branch_b' not in out or out['branch_b'] is None:
                logging.warning("当前批次没有分支B输出，已跳过。")
                # 更新索引计数
                batch_num_nodes = graph_data[g_key.node_num]
                bs = len(batch_num_nodes) if not isinstance(batch_num_nodes, torch.Tensor) else batch_num_nodes.size(0)
                current_idx += bs
                continue

            b_out = out['branch_b']
            a_pad = b_out.get('a_pad', None)        # [B, M]
            a_flat = b_out.get('a', None)           # [Sum(N)]
            
            # 取每个样本的真实节点数与原始图索引
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
            
            # 处理 labels 和 preds
            batch_preds = []
            batch_labels = []
            if logits is not None and labels is not None:
                # 获取预测类别：对于二分类，通常是 argmax(logits) 或者 threshold=0.5
                # 这里假设 logits 是 [B, 2] 或者 [B, 1]
                if logits.dim() == 2 and logits.size(1) == 2:
                     preds = logits.argmax(dim=1).detach().cpu().tolist()
                elif logits.dim() == 2 and logits.size(1) == 1:
                     preds = (torch.sigmoid(logits) > 0.5).long().view(-1).detach().cpu().tolist()
                elif logits.dim() == 1:
                     preds = (torch.sigmoid(logits) > 0.5).long().detach().cpu().tolist()
                else:
                     logging.warning(f"Unexpected logits shape: {logits.shape}")
                     preds = [0] * len(num_list) # Fallback

                batch_labels = labels.detach().cpu().tolist()
                batch_preds = preds
            else:
                batch_preds = [None] * len(num_list)
                batch_labels = [None] * len(num_list)

            # 准备 weights_list
            weights_list = []
            if a_pad is not None:
                a_pad_np = a_pad.detach().cpu().numpy()
                for i, n_i in enumerate(num_list):
                    weights_list.append(a_pad_np[i, :n_i])
            elif a_flat is not None:
                a_flat_np = a_flat.detach().cpu().numpy()
                curr = 0
                for n_i in num_list:
                    weights_list.append(a_flat_np[curr : curr+n_i])
                    curr += n_i
            else:
                logging.warning("未找到 a_pad 或 a（分支B注意力），已跳过。")
                current_idx += len(num_list)
                continue

            # 逐图处理
            for i, n_i in enumerate(num_list):
                if n_i <= 0:
                    current_idx += 1
                    continue
                
                weights = weights_list[i]
                orig_idx = orig_indices[i]
                pred = batch_preds[i]
                label = batch_labels[i]
                
                # 获取节点标签
                if node_labels_collection is not None and 0 <= orig_idx < len(node_labels_collection):
                    node_bin_labels = np.asarray(node_labels_collection[orig_idx])
                    node_bin_labels = node_bin_labels[:n_i]
                else:
                    node_bin_labels = np.zeros(n_i, dtype=np.int64)

                # --- 统计逻辑 (针对所有图) ---
                # 1. 权重统计
                pos_mask = (node_bin_labels == 1)
                neg_mask = (node_bin_labels == 0)
                if pos_mask.any():
                    stats['pos_weights'].extend(weights[pos_mask].tolist())
                if neg_mask.any():
                    stats['neg_weights'].extend(weights[neg_mask].tolist())
                
                # 2. Bag 分类统计
                is_correct = False
                if pred is not None and label is not None:
                    if pred == label:
                        is_correct = True
                        if label == 1:
                            stats['correct_pos_bag_count'] += 1
                        else:
                            stats['correct_neg_bag_count'] += 1
                    else:
                        # 分类错误
                        is_correct = False
                        # 统计正节点数量
                        pos_node_count = np.sum(node_bin_labels)
                        stats['wrong_bag_pos_node_counts'].append(int(pos_node_count))
                
                # --- 导出逻辑 (全部导出) ---
                df = pd.DataFrame({
                    'weight': weights,
                    'node_binary_label': node_bin_labels
                })
                # 按权重降序排列并重置索引
                df = df.sort_values(by='weight', ascending=False).reset_index(drop=True)

                if label == 1:
                    # Top-1 Hit
                    top1_labels = df.head(1)['node_binary_label']
                    hit1 = 1 if top1_labels.sum() > 0 else 0
                    stats['pos_bag_top1_hits'].append(hit1)
                    
                    # Top-3 Hit
                    top3_labels = df.head(3)['node_binary_label']
                    hit3 = 1 if top3_labels.sum() > 0 else 0
                    stats['pos_bag_top3_hits'].append(hit3)

                    # Top-5 Hit
                    top5_labels = df.head(5)['node_binary_label']
                    hit5 = 1 if top5_labels.sum() > 0 else 0
                    stats['pos_bag_top5_hits'].append(hit5)

                # 命名规则: {id}_{correctness}
                correct_flag = 1 if is_correct else 0
                base_name = f'{orig_idx}_{correct_flag}'
                sheet_name = base_name
                suffix = 1
                while sheet_name in sheet_name_used:
                    sheet_name = f'{base_name}_{suffix}'
                    suffix += 1
                sheet_name_used.add(sheet_name)

                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                current_idx += 1

    # 生成总结 Sheet
    _write_summary_sheet(writer, stats)
    
    # 调整 Summary 到第一个位置
    _reorder_sheets_to_front(writer, 'Summary')

    if len(sheet_name_used) == 0 and len(stats['pos_weights']) == 0:
        logging.warning(f"No attention maps were exported to {output_path}. Saving an empty sheet.")
        pd.DataFrame({'info': ['No attention data exported']}).to_excel(writer, sheet_name='No_Data', index=False)

    writer.close()
    print(f'Exported branch B attention for test graphs to {output_path}')


def export_branchB_attention_from_model(model, test_loader, hparams, dataset_raw, output_path, sample_frac=0.2):
    model.eval()
    if not hasattr(hparams, 'branch_b') or not hparams.branch_b.get('use', False):
        raise RuntimeError("branch_b.use 未开启，无法导出注意力 a。")

    node_labels_collection = dataset_raw.get('node_binary_labels', None)
    subgraphs = dataset_raw.get('subgraph_structures', None)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 统计容器
    stats = {
        'pos_weights': [],
        'neg_weights': [],
        'pos_bag_top1_hits': [],
        'pos_bag_top3_hits': [],
        'pos_bag_top5_hits': [],
        'correct_pos_bag_count': 0,
        'correct_neg_bag_count': 0,
        'wrong_bag_pos_node_counts': []
    }

    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    with torch.no_grad():
        sheet_name_used = set()
        current_idx = 0
        for _, graph_data in enumerate(test_loader):
            out = model(graph_data)
            
            # 获取预测结果和真实标签
            if 'ypred_A' in out:
                logits = out['ypred_A']
            elif isinstance(out, dict) and 'ypred' in out:
                logits = out['ypred']
            else:
                logits = None
            
            # 获取 Ground Truth
            if g_key.y in graph_data:
                labels = graph_data[g_key.y]
            else:
                labels = None
                
            if not isinstance(out, dict) or 'branch_b' not in out or out['branch_b'] is None:
                # 即使没有输出，也要增加计数以保持对齐
                batch_num_nodes = graph_data['node_num']
                bs = len(batch_num_nodes) if not isinstance(batch_num_nodes, torch.Tensor) else batch_num_nodes.size(0)
                current_idx += bs
                continue

            b_out = out['branch_b']
            a_pad = b_out.get('a_pad', None)
            a_flat = b_out.get('a', None)

            batch_num_nodes = graph_data['node_num']
            orig_idx_tensor = graph_data['orig_graph_idx']

            if isinstance(batch_num_nodes, torch.Tensor):
                num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
            else:
                num_list = [int(n) for n in batch_num_nodes]
            if isinstance(orig_idx_tensor, torch.Tensor):
                orig_indices = [int(i) for i in orig_idx_tensor.detach().cpu().tolist()]
            else:
                orig_indices = [int(i) for i in orig_idx_tensor]
            
            # 处理 labels 和 preds
            batch_preds = []
            batch_labels = []
            if logits is not None and labels is not None:
                # 获取预测类别：对于二分类，通常是 argmax(logits) 或者 threshold=0.5
                # 这里假设 logits 是 [B, 2] 或者 [B, 1]
                if logits.dim() == 2 and logits.size(1) == 2:
                     preds = logits.argmax(dim=1).detach().cpu().tolist()
                elif logits.dim() == 2 and logits.size(1) == 1:
                     preds = (torch.sigmoid(logits) > 0.5).long().view(-1).detach().cpu().tolist()
                elif logits.dim() == 1:
                     preds = (torch.sigmoid(logits) > 0.5).long().detach().cpu().tolist()
                else:
                     logging.warning(f"Unexpected logits shape: {logits.shape}")
                     preds = [0] * len(num_list) # Fallback

                batch_labels = labels.detach().cpu().tolist()
                batch_preds = preds
            else:
                batch_preds = [None] * len(num_list)
                batch_labels = [None] * len(num_list)

            weights_list = []
            if a_pad is not None:
                a_pad_np = a_pad.detach().cpu().numpy()
                for i, n_i in enumerate(num_list):
                    weights_list.append(a_pad_np[i, :n_i])
            elif a_flat is not None:
                a_flat_np = a_flat.detach().cpu().numpy()
                curr = 0
                for n_i in num_list:
                    weights_list.append(a_flat_np[curr : curr+n_i])
                    curr += n_i
            else:
                current_idx += len(num_list)
                continue

            for i, n_i in enumerate(num_list):
                if n_i <= 0:
                    current_idx += 1
                    continue
                
                weights = weights_list[i]
                orig_idx = orig_indices[i]
                pred = batch_preds[i]
                label = batch_labels[i]

                # 修复：按子图节点映射到原图索引再取标签
                if node_labels_collection is not None and subgraphs is not None and 0 <= orig_idx < len(subgraphs):
                    subgraph = subgraphs[orig_idx]
                    node_bin_labels = _map_subgraph_nodes_to_labels(subgraph, node_labels_collection, n_i)
                else:
                    node_bin_labels = np.zeros(n_i, dtype=np.int64)

                # --- 统计逻辑 (针对所有图) ---
                # 1. 权重统计
                pos_mask = (node_bin_labels == 1)
                neg_mask = (node_bin_labels == 0)
                if pos_mask.any():
                    stats['pos_weights'].extend(weights[pos_mask].tolist())
                if neg_mask.any():
                    stats['neg_weights'].extend(weights[neg_mask].tolist())
                
                # 2. Bag 分类统计
                is_correct = False
                if pred is not None and label is not None:
                    if pred == label:
                        is_correct = True
                        if label == 1:
                            stats['correct_pos_bag_count'] += 1
                        else:
                            stats['correct_neg_bag_count'] += 1
                    else:
                        is_correct = False
                        # 统计错误分类包中的正节点数量
                        # 注意：这里的“错误分类”是指：
                        #   1. 真实为正(1) -> 预测为负(0) (False Negative)
                        #   2. 真实为负(0) -> 预测为正(1) (False Positive)
                        # 我们统计的是该包内真实标签为1的节点数
                        pos_node_count = np.sum(node_bin_labels)
                        stats['wrong_bag_pos_node_counts'].append(int(pos_node_count))
                else:
                    logging.warning(f"Graph {orig_idx}: pred={pred}, label={label}, skipping classification stats.")
                
                # --- 导出逻辑 (全部导出) ---
                # 导出仅包含权重与节点二分类标签
                df = pd.DataFrame({
                    'weight': weights,
                    'node_binary_label': node_bin_labels
                })
                df = df.sort_values(by='weight', ascending=False).reset_index(drop=True)

                if label == 1:
                    # Top-1 Hit
                    top1_labels = df.head(1)['node_binary_label']
                    hit1 = 1 if top1_labels.sum() > 0 else 0
                    stats['pos_bag_top1_hits'].append(hit1)
                    
                    # Top-3 Hit
                    top3_labels = df.head(3)['node_binary_label']
                    hit3 = 1 if top3_labels.sum() > 0 else 0
                    stats['pos_bag_top3_hits'].append(hit3)

                    # Top-5 Hit
                    top5_labels = df.head(5)['node_binary_label']
                    hit5 = 1 if top5_labels.sum() > 0 else 0
                    stats['pos_bag_top5_hits'].append(hit5)

                correct_flag = 1 if is_correct else 0
                base_name = f'{orig_idx}_{correct_flag}'
                sheet_name = base_name
                suffix = 1
                while sheet_name in sheet_name_used:
                    sheet_name = f'{base_name}_{suffix}'
                    suffix += 1
                sheet_name_used.add(sheet_name)

                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                current_idx += 1

    # 生成总结 Sheet
    _write_summary_sheet(writer, stats)
    
    # 调整 Summary 到第一个位置
    _reorder_sheets_to_front(writer, 'Summary')

    if len(sheet_name_used) == 0 and len(stats['pos_weights']) == 0:
        logging.warning(f"No attention maps were exported to {output_path} (no graphs selected or available). Saving an empty sheet.")
        pd.DataFrame({'info': ['No attention data exported']}).to_excel(writer, sheet_name='No_Data', index=False)

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
