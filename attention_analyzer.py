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


def generate_node_labels(graph_label, num_nodes, seed=42):
    """
    临时生成节点标签的函数（备用方案）
    
    Args:
        graph_label: 图级标签 (0或1)
        num_nodes: 节点数量
        seed: 随机种子
    
    Returns:
        node_labels: numpy数组，形状为[num_nodes]，值为0或1
    """
    np.random.seed(seed)
    
    if graph_label == 0:
        # 负图：所有节点都是负类
        return np.zeros(num_nodes, dtype=int)
    else:
        # 正图：随机生成一些正类节点
        # 确保至少有一个正类节点
        node_labels = np.zeros(num_nodes, dtype=int)
        num_positive = max(1, min(num_nodes, int(num_nodes * 0.3)))  # 30%的节点为正类，但不超过总节点数
        # 防止num_positive超过num_nodes导致np.random.choice出错
        if num_positive > num_nodes:
            num_positive = num_nodes
        positive_indices = np.random.choice(num_nodes, num_positive, replace=False)
        node_labels[positive_indices] = 1
        return node_labels


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

            a = out['branch_b']['a'].view(-1).clamp(1e-6, 1-1e-6)  # [B*M]
            graph_labels = data[g_key.y].view(-1)                  # [B]
            x = data[g_key.x]                                      # [B, M, D]

            B = graph_labels.size(0)
            if a.numel() % B != 0:
                print(f"错误：注意力长度 {a.numel()} 不能被 batch_size {B} 整除")
            M = a.numel() // B

            # 构造“batch”分组（等价 PyG 的 data.batch）
            batch_vec = torch.arange(B, device=a.device).repeat_interleave(M)  # [B*M]

            # 用特征非零判断真实节点（避免依赖 node_num）
            real_mask = (x.abs().sum(dim=-1) > 0)       # [B, M]
            real_mask_flat = real_mask.view(-1)         # [B*M]

            print(f"批次 {batch_idx}: {B} 个图, 总节点块长度: {a.numel()}, 每图块大小 M: {M}")

            for i in range(B):
                current_graph_idx = global_graph_idx + i
                graph_label = int(graph_labels[i].item())

                # 取出第 i 图所有节点的注意力（固定块），再过滤真实节点
                a_i = a[i*M:(i+1)*M]                         # [M]
                real_i = real_mask[i]                        # [M]
                a_i_real = a_i[real_i]                       # [n_i]
                n_i = int(a_i_real.numel())

                print(f"  处理图 {current_graph_idx}: 真实节点数 n_i={n_i}, 标签(真实): {graph_label}")

                # 加载该图的真实节点标签 node_y（来自数据集）
                node_labels = load_real_node_labels(hparams, current_graph_idx)
                if node_labels is None:
                    print(f"  图 {current_graph_idx}: 无法加载真实节点标签，跳过真实Hit统计")
                    continue
                if len(node_labels) != n_i:
                    print(f"  图 {current_graph_idx}: node_y长度不匹配 ({len(node_labels)} vs {n_i})，按较小值对齐")
                    n_i = min(n_i, len(node_labels))
                    a_i_real = a_i_real[:n_i]
                    node_labels = np.array(node_labels, dtype=int)[:n_i]
                else:
                    node_labels = np.array(node_labels, dtype=int)

                positive_nodes = int(np.sum(node_labels))
                if graph_label == 1 and positive_nodes > 0:
                    # 只在正图且至少1个正节点时统计
                    # Hit@k
                    top_k = min(k, n_i)
                    top_k_idx = torch.topk(a_i_real, top_k, largest=True).indices.cpu().numpy()
                    hit_k = int(np.any(node_labels[top_k_idx] == 1))

                    # Hit@r%
                    r_count = max(1, int(np.ceil(r_percent / 100.0 * n_i)))
                    top_r_idx = torch.topk(a_i_real, r_count, largest=True).indices.cpu().numpy()
                    hit_r = int(np.any(node_labels[top_r_idx] == 1))

                    # 记录到列表（供最终汇总）
                    # ... existing code ...
                else:
                    print(f"  图 {current_graph_idx}: 跳过Hit统计（非正图或无正节点，正节点数={positive_nodes}）")

                # 组装Excel节点数据（仅真实节点）
                node_data = []
                for node_id in range(n_i):
                    node_data.append({
                        'node_id': node_id,
                        'attention_weight': float(a_i_real[node_id].item()),
                        'node_class': int(node_labels[node_id])
                    })
                node_data.sort(key=lambda x: x['attention_weight'], reverse=True)

                graph_info = {
                    'graph_idx': current_graph_idx,
                    'graph_label': graph_label,
                    'num_nodes': n_i,
                    'node_data': node_data
                }
                all_graph_data.append(graph_info)
                if graph_label == 1:
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
                for node in graph_info['node_data']:
                    df_data.append([
                        node['node_id'],
                        node['attention_weight'],
                        node['node_class']
                    ])
                
                df = pd.DataFrame(df_data, columns=['节点编号', '注意力权重', '节点类别'])
                
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