# coding=utf-8

import torch
import numpy as np
import sklearn.metrics as metrics
import torch.nn.functional as F

from MISGL.utils.global_variables import *


def evaluate(dataset, model, hparams, max_num_examples=None, dataset_name="", include_loss=False, loss_epoch=0):
    del loss_epoch
    model.eval()
    preds, labels = [], []
    loss_sum = 0.0
    loss_count = 0
    device = torch.device(hparams.device)

    def _needs_device_move(value):
        if not isinstance(value, torch.Tensor):
            return False
        if value.device.type != device.type:
            return True
        return device.index is not None and value.device.index != device.index

    def _move_batch_to_device(data):
        return {
            key: value.to(device, non_blocking=True) if _needs_device_move(value) else value
            for key, value in data.items()
        }

    def _reached_max_examples(batch_idx, batch_size):
        if max_num_examples is None:
            return False
        return (batch_idx + 1) * batch_size > int(max_num_examples)

    has_position_memory = (
        bool(getattr(model, 'use_position_head', False))
        and hasattr(model, 'snapshot_position_memory')
        and hasattr(model, 'restore_position_memory')
        and hasattr(model, 'reset_position_memory')
        and hasattr(model, 'update_position_memory_from_batch')
    )
    position_memory_snapshot = None
    if has_position_memory:
        position_memory_snapshot = model.snapshot_position_memory()
        model.reset_position_memory()
        with torch.inference_mode():
            for batch_idx, data in enumerate(dataset):
                batch = _move_batch_to_device(data)
                model.update_position_memory_from_batch(batch)
                batch_size = int(batch[g_key.y].view(-1).size(0))
                if _reached_max_examples(batch_idx, batch_size):
                    break

    try:
        with torch.inference_mode():
            for batch_idx, data in enumerate(dataset):
                batch = _move_batch_to_device(data)

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

                if _reached_max_examples(batch_idx, len(pred)):
                    break
    finally:
        if has_position_memory:
            model.restore_position_memory(position_memory_snapshot)

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    result = {
        'prec': metrics.precision_score(labels, preds, average='binary', zero_division=0),
        'rec': metrics.recall_score(labels, preds, average='binary', zero_division=0),
        'acc': metrics.accuracy_score(labels, preds),
        'F1': metrics.f1_score(labels, preds, average='binary', zero_division=0)
    }
    if include_loss:
        result['loss'] = loss_sum / max(loss_count, 1)

    nested = {
        'A': dict(result),
        'B': dict(result),
        'AB': dict(result)
    }
    return {**nested, **result}
