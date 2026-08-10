# coding=utf-8

import torch
import numpy as np
import sklearn.metrics as metrics
import torch.nn.functional as F

from MISGL.utils.global_variables import *


def evaluate(dataset, model, hparams, max_num_examples=None, dataset_name="", include_loss=False, loss_epoch=0):
    del loss_epoch
    model.eval()
    preds, labels, probabilities = [], [], []
    loss_sum = 0.0
    loss_count = 0
    device = torch.device(hparams.device)

    def _needs_device_move(value):
        if not isinstance(value, torch.Tensor):
            return False
        if value.device.type != device.type:
            return True
        return device.index is not None and value.device.index != device.index

    with torch.inference_mode():
        for batch_idx, data in enumerate(dataset):
            batch = {
                key: value.to(device, non_blocking=True)
                if _needs_device_move(value) else value
                for key, value in data.items()
            }

            out = model(batch)
            if isinstance(out, dict) and 'ypred_A' in out:
                logits_A = out['ypred_A']
                if isinstance(logits_A, torch.Tensor):
                    p = torch.sigmoid(logits_A).view(-1)
                else:
                    p = torch.zeros(len(batch[g_key.y]), device=device)
            else:
                logits_A = out
                p = torch.sigmoid(logits_A).view(-1)

            if include_loss and isinstance(logits_A, torch.Tensor):
                batch_targets = batch[g_key.y].view(-1).float()
                batch_loss = F.binary_cross_entropy_with_logits(logits_A.view(-1), batch_targets)
                batch_size = int(batch_targets.size(0))
                loss_sum += float(batch_loss.item()) * batch_size
                loss_count += batch_size

            pred = (p > 0.5).long().cpu().numpy()
            y = batch[g_key.y].view(-1).cpu().numpy()

            preds.append(pred)
            labels.append(y)
            probabilities.append(p.cpu().numpy())

            if max_num_examples is not None:
                if (batch_idx + 1) * len(pred) > max_num_examples:
                    break

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    probabilities = np.concatenate(probabilities, axis=0)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    has_both_classes = np.unique(labels).size == 2

    result = {
        'prec': metrics.precision_score(labels, preds, average='binary', zero_division=0),
        'rec': metrics.recall_score(labels, preds, average='binary', zero_division=0),
        'acc': metrics.accuracy_score(labels, preds),
        'F1': metrics.f1_score(labels, preds, average='binary', zero_division=0),
        'balanced_acc': metrics.balanced_accuracy_score(labels, preds),
        'roc_auc': metrics.roc_auc_score(labels, probabilities) if has_both_classes else None,
        'pr_auc': metrics.average_precision_score(labels, probabilities) if has_both_classes else None,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    if include_loss:
        result['loss'] = loss_sum / max(loss_count, 1)

    nested = {
        'A': dict(result),
        'B': dict(result),
        'AB': dict(result),
    }
    return {**nested, **result}
