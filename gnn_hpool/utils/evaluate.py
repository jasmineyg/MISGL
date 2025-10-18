# coding=utf-8

import torch
import numpy as np
import sklearn.metrics as metrics

from gnn_hpool.utils.global_variables import *


def evaluate(dataset, model, hparams, max_num_examples=None, dataset_name="", analyze_attention=False):
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            for key, value in data.items():
                data[key] = value.to(hparams.device)
            
            out = model(data)
            
            # 处理两种输出格式：字典格式（有分支B）和张量格式（无分支B）
            if isinstance(out, dict) and 'ypred_A' in out:
                # 字典格式：有分支B的情况
                logits_A = out['ypred_A']  # [B] 或 [B,1]
                
                use_b = ('branch_b' in out) and (out['branch_b'] is not None) \
                        and ('y_B' in out['branch_b']) \
                        and getattr(hparams, 'branch_b', None) and hparams.branch_b.get('use', False)

                if use_b:
                    # 评估：在概率域融合
                    p_A = torch.sigmoid(logits_A).view(-1)      # [B]
                    p_B = out['branch_b']['y_B'].view(-1)       # [B]
                    gamma_end = hparams.branch_b.get('gamma_end', 0.6)
                    p = gamma_end * p_B + (1 - gamma_end) * p_A  # [B]
                else:
                    p = torch.sigmoid(logits_A).view(-1)  # [B]
            else:
                # 张量格式：无分支B的情况（兼容旧版本）
                logits_A = out  # [B] 或 [B,1]
                p = torch.sigmoid(logits_A).view(-1)  # [B]

            pred = (p > 0.5).long().cpu().numpy()
            y = data[g_key.y].view(-1).cpu().numpy()
            
            preds.append(pred)
            labels.append(y)

            if max_num_examples is not None:
                if (batch_idx + 1) * len(pred) > max_num_examples:
                    break

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    result = {
        'prec': metrics.precision_score(labels, preds, average='binary'),
        'rec': metrics.recall_score(labels, preds, average='binary'),
        'acc': metrics.accuracy_score(labels, preds),
        'F1': metrics.f1_score(labels, preds, average='binary')
    }
    
    # 添加数据集名称标识
    prefix = f"[{dataset_name}]" if dataset_name else ""
    # print(f'{prefix}  acc: {result["acc"]:.4f}, prec: {result["prec"]:.4f}, rec: {result["rec"]:.4f}, F1: {result["F1"]:.4f}')
    
    # 如果需要进行注意力分析且是测试集
    if analyze_attention and dataset_name == "test":
        print("\n开始注意力分析...")
        try:
            # 添加项目根目录到系统路径
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from attention_analyzer import analyze_attention_and_hit_at_k
            
            # 执行注意力分析，k=3
            analysis_results = analyze_attention_and_hit_at_k(
                model=model,
                dataset=dataset,
                hparams=hparams,
                output_path='attention_analysis_results.xlsx',
                k=3
            )
            
            if analysis_results:
                print(f"\n=== 注意力分析完成 ===")
                print(f"Excel文件路径: {analysis_results['excel_path']}")
                print(f"Hit@3: {analysis_results['hit_at_k']:.4f}")
                print(f"命中数量: {analysis_results['hit_count']}/{analysis_results['positive_graphs']}")
                print(f"总图数量: {analysis_results['total_graphs']}")
            else:
                print("注意力分析失败")
                
        except Exception as e:
            print(f"注意力分析出错: {e}")
    
    return result
