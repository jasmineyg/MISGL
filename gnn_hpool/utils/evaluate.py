# coding=utf-8

import torch
import numpy as np
import sklearn.metrics as metrics

from gnn_hpool.utils.global_variables import *


def evaluate(dataset, model, hparams, max_num_examples=None, dataset_name=""):
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
    
    return result
