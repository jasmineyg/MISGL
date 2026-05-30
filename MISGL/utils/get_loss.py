# coding=utf-8

import torch
import torch.nn.functional as F


def cross_entropy(prediction, reference):
    return F.cross_entropy(prediction, reference, reduction='mean')


def get_gamma(epoch, gamma_start=0.3, gamma_end=0.6, warmup_epochs=20):
    return gamma_start if epoch < warmup_epochs else gamma_end


def bce_from_probs(probs, targets):
    probs = torch.clamp(probs.view(-1), min=1e-8, max=1 - 1e-8)
    targets = targets.view(-1).float()
    return -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs)).mean()


def _smooth_binary_targets(targets, hparams):
    targets = targets.view(-1).float()
    smoothing = float(getattr(hparams, 'label_smoothing', 0.0))
    if smoothing < 0.0 or smoothing >= 1.0:
        raise ValueError('label_smoothing must be in [0.0, 1.0).')
    if smoothing == 0.0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def _attention_shape_loss_config(hparams):
    branch_b_cfg = getattr(hparams, 'branch_b', None)
    if not isinstance(branch_b_cfg, dict) or not bool(branch_b_cfg.get('use', False)):
        return False, 0.0, 1e-8

    enabled = bool(branch_b_cfg.get('attention_shape_loss_enabled', True))
    weight = float(branch_b_cfg.get('attention_shape_loss_weight', 0.0))
    eps = float(branch_b_cfg.get('attention_shape_loss_eps', 1e-8))
    if weight < 0.0:
        raise ValueError('branch_b.attention_shape_loss_weight must be non-negative.')
    if eps <= 0.0:
        raise ValueError('branch_b.attention_shape_loss_eps must be positive.')
    return enabled, weight, eps


def mil_attention_shape_loss(attention, batch, targets, eps=1e-8):
    attention = attention.view(-1)
    batch = batch.to(device=attention.device, dtype=torch.long).view(-1)
    if attention.size(0) != batch.size(0):
        raise ValueError('attention and batch must have the same first dimension.')
    if attention.numel() == 0:
        return attention.new_zeros(())

    targets = targets.to(device=attention.device).view(-1).float()
    bag_count = int(batch.max().item()) + 1
    if targets.numel() < bag_count:
        raise ValueError('targets must contain at least one label for each attention bag.')

    lengths = torch.bincount(batch, minlength=bag_count).to(dtype=attention.dtype)
    valid_bags = lengths > 1.0
    if not torch.any(valid_bags):
        return attention.new_zeros(())

    safe_attention = attention.clamp_min(eps)
    sum_per_bag = attention.new_zeros((bag_count,))
    sum_per_bag.scatter_add_(0, batch, safe_attention)
    probs = safe_attention / sum_per_bag[batch].clamp_min(eps)

    entropy_terms = -(probs * torch.log(probs.clamp_min(eps)))
    entropy = attention.new_zeros((bag_count,))
    entropy.scatter_add_(0, batch, entropy_terms)
    normalized_entropy = entropy / torch.log(lengths.clamp_min(2.0)).clamp_min(eps)
    normalized_entropy = normalized_entropy.clamp(min=0.0, max=1.0)

    positive_bag = targets[:bag_count] > 0.5
    bag_loss = torch.where(positive_bag, normalized_entropy, 1.0 - normalized_entropy)
    return bag_loss[valid_bags].mean()


def fused_loss(model_output, targets, epoch, hparams):
    if isinstance(model_output, dict) and 'ypred_A' in model_output:
        logits_A = model_output['ypred_A']
    else:
        logits_A = model_output
    smoothed_targets = _smooth_binary_targets(targets, hparams)
    loss = F.binary_cross_entropy_with_logits(logits_A.view(-1), smoothed_targets)

    enabled, weight, eps = _attention_shape_loss_config(hparams)
    if not enabled or weight == 0.0 or not isinstance(model_output, dict):
        return loss

    branch_b = model_output.get('branch_b')
    if not isinstance(branch_b, dict):
        return loss

    attention = branch_b.get('a')
    batch = branch_b.get('batch')
    if not isinstance(attention, torch.Tensor) or not isinstance(batch, torch.Tensor):
        return loss

    return loss + weight * mil_attention_shape_loss(attention, batch, targets, eps=eps)
