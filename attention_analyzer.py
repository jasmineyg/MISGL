import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import pickle


def load_real_node_labels(hparams, graph_idx):
    """
    从原始数据加载真实的节点标签
    
    Args:
        hparams: 超参数配置
        graph_idx: 图索引
    
    Returns:
        node_labels: numpy数组，形状为[num_nodes]，值为0或1
    """
    try:
        # 构建数据集路径
        processed_data_dir = getattr(hparams, 'processed_data_dir', '/data/yg/Subgraph-MIL/Data/processed_data')
        data_name = getattr(hparams, 'data_name', None)
        if not data_name:
            print(f"缺少data_name参数，无法加载真实节点标签")
            return None
            
        dataset_path = os.path.join(processed_data_dir, f'{data_name}_processed.pkl')
        
        # 读取pickle数据
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # 获取测试集索引
        test_indices = dataset['train_test_split']['test_indices']
        
        # 检查图索引是否有效
        if graph_idx >= len(test_indices):
            print(f"图索引 {graph_idx} 超出测试集范围 (测试集大小: {len(test_indices)})")
            return None
            
        # 获取实际的图索引
        actual_graph_idx = test_indices[graph_idx]
        
        # 方法1：尝试使用subgraph_structures中的节点信息
        if 'subgraph_structures' in dataset:
            subgraphs = dataset['subgraph_structures']
            if actual_graph_idx < len(subgraphs):
                graph = subgraphs[actual_graph_idx]
                nodelist = list(graph.nodes())
                
                # 检查是否有node_binary_labels
                if 'node_binary_labels' in dataset:
                    node_binary_labels = dataset['node_binary_labels']
                    node_labels = []
                    
                    for node_id in nodelist:
                        if node_id < len(node_binary_labels):
                            node_labels.append(node_binary_labels[node_id])
                        else:
                            node_labels.append(0)  # 默认为0
                    
                    result = np.array(node_labels, dtype=int)
                    print(f"图 {graph_idx}: 从subgraph_structures+node_binary_labels加载了 {len(result)} 个节点标签，正类节点数: {np.sum(result)}")
                    return result
        
        # 方法2：如果方法1失败，尝试使用subgraph_assignment
        if 'subgraph_assignment' in dataset and 'node_binary_labels' in dataset:
            subgraph_assignment = dataset['subgraph_assignment']
            node_binary_labels = dataset['node_binary_labels']
            
            if actual_graph_idx < len(subgraph_assignment):
                subgraph_nodes = subgraph_assignment[actual_graph_idx]
                
                # 检查subgraph_nodes的类型
                if isinstance(subgraph_nodes, int):
                    print(f"警告：subgraph_assignment[{actual_graph_idx}] 是单个整数 {subgraph_nodes}，而不是节点列表")
                    return None
                
                # 提取这些节点的二分类标签
                node_labels = []
                for node_id in subgraph_nodes:
                    if node_id < len(node_binary_labels):
                        node_labels.append(node_binary_labels[node_id])
                    else:
                        print(f"警告：节点ID {node_id} 超出node_binary_labels范围")
                        node_labels.append(0)  # 默认为0
                
                result = np.array(node_labels, dtype=int)
                print(f"图 {graph_idx}: 从subgraph_assignment+node_binary_labels加载了 {len(result)} 个节点标签，正类节点数: {np.sum(result)}")
                return result
        
        print(f"无法加载图 {graph_idx} 的节点标签：缺少必要的数据字段")
        return None
        
    except Exception as e:
        print(f"加载节点标签失败 (图索引: {graph_idx}): {e}")
        return None


def analyze_attention_and_hit_at_k(model, dataset, hparams, output_path='attention_analysis.xlsx', k=3, r_percent=10):
    """
    分析分支B的注意力权重并计算Hit@k指标
    
    Args:
        model: 训练好的模型
        dataset: 测试数据集
        hparams: 超参数配置
        output_path: Excel输出路径
        k: Hit@k中的k值，默认为3
    
    Returns:
        dict: 包含分析结果的字典
    """
    # 导入全局变量
    from gnn_hpool.utils.global_variables import g_key
    
    model.eval()
    
    # 检查模型是否有分支B
    if not (hasattr(hparams, 'branch_b') and hparams.branch_b.get('use', False)):
        print("模型未启用分支B，无法进行注意力分析")
        return None
    
    all_graph_data = []
    positive_graphs_data = []
    
    print("开始提取注意力权重...")
    
    with torch.no_grad():
        global_graph_idx = 0

        for batch_idx, data in enumerate(dataset):
            print(f"处理批次 {batch_idx}")
            for key, value in data.items():
                data[key] = value.to(hparams.device)

            out = model(data)
            if not (isinstance(out, dict) and 'branch_b' in out and out['branch_b'] is not None):
                print(f"批次 {batch_idx} 没有分支B输出")
                global_graph_idx += len(data[g_key.y])
                continue

            branch_b = out['branch_b']
            a_pad = branch_b.get('a_pad', None)           # [B,M]
            mask_valid = branch_b.get('mask_valid', None) # [B,M] bool
            graph_labels = data[g_key.y].view(-1)         # [B]
            node_num = data[g_key.node_num].view(-1)      # [B]

            if a_pad is None or mask_valid is None:
                print("模型未返回 a_pad/mask_valid，无法稳定对齐；请应用模型端补丁")
                continue

            B, M = a_pad.size()
            print(f"处理批次 {batch_idx}: {B} 个图, 每图填充长度 M={M}")

            for i in range(B):
                current_graph_idx = global_graph_idx + i
                graph_label_i = int(graph_labels[i].item())  # 显式定义该图的标签
                n_i = int(node_num[i].item())
                if n_i <= 0:
                    print(f"  图 {current_graph_idx}: 真实节点数为 0，跳过")
                    continue

                valid_mask_i = mask_valid[i]          # [M] bool
                a_i_real = a_pad[i][valid_mask_i]     # [n_i]
                print(f"  处理图 {current_graph_idx}: 真实节点数 n_i={n_i}, 标签(真实): {graph_label_i}")

                # 载入该图的真实节点标签（若不可用仍导出注意力）
                node_labels = load_real_node_labels(hparams, current_graph_idx)
                node_labels_arr = None
                if node_labels is not None:
                    node_labels = np.asarray(node_labels)
                    if len(node_labels) == M:
                        node_labels_arr = node_labels[valid_mask_i.cpu().numpy()]  # 对齐到 n_i
                    elif len(node_labels) == n_i:
                        node_labels_arr = node_labels
                    else:
                        print(f"  图 {current_graph_idx}: node_y长度不匹配 ({len(node_labels)} vs {n_i})，按 n_i 裁剪")
                        node_labels_arr = node_labels[:n_i]

                # 计算 Hit 指标（仅当存在标签且为正图）
                if node_labels_arr is not None:
                    positive_nodes = int(np.sum(node_labels_arr))
                    if graph_label_i == 1 and positive_nodes > 0:
                        top_k = min(k, n_i)
                        top_k_idx = torch.topk(a_i_real, top_k, largest=True).indices.cpu().numpy()
                        hit_k = int(np.any(node_labels_arr[top_k_idx] == 1))

                        r_count = max(1, int(np.ceil(r_percent / 100.0 * n_i)))
                        top_r_idx = torch.topk(a_i_real, r_count, largest=True).indices.cpu().numpy()
                        hit_r = int(np.any(node_labels_arr[top_r_idx] == 1))

                else:
                    print(f"  图 {current_graph_idx}: 节点标签不可用，仅导出注意力")

                # 组装Excel节点数据（确保使用英文键 attention_weight）
                node_data = []
                for node_id in range(n_i):
                    cls = int(node_labels_arr[node_id]) if node_labels_arr is not None else -1
                    node_data.append({
                        'node_id': node_id,
                        'attention_weight': float(a_i_real[node_id].item()),
                        'node_class': cls
                    })

                # 排序并写入 graph_info（统一使用 graph_label_i）
                node_data.sort(key=lambda x: x['attention_weight'], reverse=True)
                graph_info = {
                    'graph_idx': current_graph_idx,
                    'graph_label': graph_label_i,   # ← 修正
                    'num_nodes': n_i,
                    'node_data': node_data,
                }
                all_graph_data.append(graph_info)
                if graph_label_i == 1:             # ← 修正
                    positive_graphs_data.append(graph_info)

            global_graph_idx += B
    
    print(f"共处理 {len(all_graph_data)} 个图，其中 {len(positive_graphs_data)} 个正图")
    
    # 生成Excel文件
    print("生成Excel文件...")
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for graph_info in all_graph_data:
                # 创建DataFrame
                df_data = []
                for node in node_data:
                    df_data.append([
                        node['node_id'],
                        node['attention_weight'],
                        node['node_class']
                    ])
                df = pd.DataFrame(df_data, columns=['节点编号', '注意力权重', '节点类别'])
                
                # 下游DataFrame组装保持：
                # df_data = [[node['node_id'], node['attention_weight'], node['node_class']] for node in node_data]
                # df = pd.DataFrame(df_data, columns=['节点编号', '注意力权重', '节点类别'])
                
                # 写入sheet，sheet名称包含图索引和标签
                sheet_name = f"Graph_{graph_info['graph_idx']}_Label_{graph_info['graph_label']}"
                # 限制sheet名称长度
                if len(sheet_name) > 31:
                    sheet_name = f"G{graph_info['graph_idx']}_L{graph_info['graph_label']}"
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Excel文件已保存: {output_path}")
    except Exception as e:
        print(f"保存Excel文件失败: {e}")
        return None
    
    # 计算Hit@k指标
    print(f"计算Hit@{k}指标...")
    hit_count = 0
    total_positive_graphs = len(positive_graphs_data)
    
    if total_positive_graphs == 0:
        hit_at_k = 0.0
    else:
        for graph_info in positive_graphs_data:
            # 获取top-k节点
            top_k_nodes = graph_info['node_data'][:min(k, len(graph_info['node_data']))]
            
            # 检查是否有正类节点
            has_positive = any(node['node_class'] == 1 for node in top_k_nodes)
            
            if has_positive:
                hit_count += 1
        
        hit_at_k = hit_count / total_positive_graphs
    
    # 打印结果
    print(f"\n=== Hit@{k} 结果 ===")
    print(f"Hit@{k}: {hit_at_k:.4f} ({hit_count}/{total_positive_graphs})")
    
    results = {
        'excel_path': output_path,
        'hit_at_k': hit_at_k,
        'hit_count': hit_count,
        'total_graphs': len(all_graph_data),
        'positive_graphs': len(positive_graphs_data)
    }
    
    return results