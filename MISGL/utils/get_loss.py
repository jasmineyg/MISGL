# coding=utf-8

import torch
import torch.nn.functional as F


SUPPORTED_LOSS_TYPES = ('bce', 'focal', 'weighted_bce')


def _smooth_binary_targets(targets, hparams):
    targets = targets.view(-1).float()
    smoothing = float(getattr(hparams, 'label_smoothing', 0.0))
    if smoothing < 0.0 or smoothing >= 1.0:
        raise ValueError('label_smoothing must be in [0.0, 1.0).')
    if smoothing == 0.0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def _loss_type(hparams):
    loss_type = str(getattr(hparams, 'loss_type', 'bce')).strip().lower()
    if loss_type not in SUPPORTED_LOSS_TYPES:
        raise ValueError(
            'Unsupported loss_type {!r}; expected one of {}.'.format(
                loss_type, ', '.join(SUPPORTED_LOSS_TYPES)
            )
        )
    return loss_type


def binary_classification_loss(logits, targets, hparams):
    logits = logits.view(-1)
    targets = _smooth_binary_targets(targets, hparams).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    if logits.numel() != targets.numel():
        raise ValueError('logits and targets must contain the same number of values.')

    loss_type = _loss_type(hparams)
    if loss_type == 'bce':
        return F.binary_cross_entropy_with_logits(logits, targets)

    if loss_type == 'focal':
        gamma = float(getattr(hparams, 'focal_gamma', 2.0))
        if gamma < 0.0:
            raise ValueError('focal_gamma must be non-negative.')
        per_example_bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none',
        )
        focal_factor = torch.pow(-torch.expm1(-per_example_bce), gamma)
        return (focal_factor * per_example_bce).mean()

    pos_weight = getattr(hparams, 'loss_pos_weight', None)
    if pos_weight is None:
        raise ValueError(
            'weighted_bce requires loss_pos_weight computed from the current training fold.'
        )
    pos_weight = float(pos_weight)
    if pos_weight <= 0.0:
        raise ValueError('loss_pos_weight must be positive.')
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=logits.new_tensor(pos_weight),
    )


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
    del epoch
    if isinstance(model_output, dict) and 'ypred_A' in model_output:
        logits_A = model_output['ypred_A']
    else:
        logits_A = model_output
    loss = binary_classification_loss(logits_A, targets, hparams)

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
