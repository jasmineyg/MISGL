# coding=utf-8

import torch
import numpy as np
import sklearn.metrics as metrics

from MISGL.utils.global_variables import *


def evaluate(dataset, model, hparams, max_num_examples=None, dataset_name=""):
    model.eval()
    preds, labels = [], []
    device = torch.device(hparams.device)

    def _needs_device_move(value):
        if not isinstance(value, torch.Tensor):
            return False
        if value.device.type != device.type:
            return True
        return device.index is not None and value.device.index != device.index
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            batch = {
                key: value.to(device, non_blocking=True)
                if _needs_device_move(value) else value
                for key, value in data.items()
            }
            
            out = model(batch)
            
            # 只使用分支A的分类头输出，不做融合
            if isinstance(out, dict) and 'ypred_A' in out:
                logits_A = out['ypred_A']  # [B] 或 [B,1]
                if isinstance(logits_A, torch.Tensor):
                    p = torch.sigmoid(logits_A).view(-1)  # [B]
                else:
                    # Fallback if logits_A is not a tensor for some reason
                    p = torch.zeros(len(batch[g_key.y]), device=device)
            else:
                logits_A = out  # [B] 或 [B,1]
                p = torch.sigmoid(logits_A).view(-1)  # [B]

            pred = (p > 0.5).long().cpu().numpy()
            y = batch[g_key.y].view(-1).cpu().numpy()
            
            preds.append(pred)
            labels.append(y)

            if max_num_examples is not None:
                if (batch_idx + 1) * len(pred) > max_num_examples:
                    break

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    result = {
        'prec': metrics.precision_score(labels, preds, average='binary', zero_division=0),
        'rec': metrics.recall_score(labels, preds, average='binary', zero_division=0),
        'acc': metrics.accuracy_score(labels, preds),
        'F1': metrics.f1_score(labels, preds, average='binary', zero_division=0)
    }
    
    # 添加数据集名称标识
    prefix = f"[{dataset_name}]" if dataset_name else ""
    # print(f'{prefix}  acc: {result["acc"]:.4f}, prec: {result["prec"]:.4f}, rec: {result["rec"]:.4f}, F1: {result["F1"]:.4f}')
    
    nested = {
        'A': dict(result),
        'B': dict(result),
        'AB': dict(result)
    }
    return {**nested, **result}
