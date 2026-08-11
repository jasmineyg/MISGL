"""Losses for binary subgraph classification."""

from typing import Optional

import torch
import torch.nn.functional as F

from MISGL.config import Config
from MISGL.models.encoder import ModelOutput


def binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """Compute the configured binary classification loss."""
    logits = logits.reshape(-1)
    targets = targets.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have equal shape, got {logits.shape} and {targets.shape}"
        )

    smoothing = config.training.label_smoothing
    if smoothing:
        targets = targets * (1.0 - smoothing) + 0.5 * smoothing

    loss_name = config.training.loss
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets)

    if loss_name == "focal":
        per_sample = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability_error = -torch.expm1(-per_sample)
        return (probability_error.pow(config.training.focal_gamma) * per_sample).mean()

    if loss_name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("weighted_bce requires the training-fold positive-class weight")
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=logits.new_tensor(pos_weight),
        )

    raise ValueError(f"unsupported loss: {loss_name}")


def attention_shape_loss(
    attention: torch.Tensor,
    bag_index: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Encourage concentrated positive-bag and diffuse negative-bag attention."""
    attention = attention.reshape(-1)
    bag_index = bag_index.to(device=attention.device, dtype=torch.long).reshape(-1)
    if attention.shape != bag_index.shape:
        raise ValueError("attention and bag_index must have equal shape")
    if attention.numel() == 0:
        raise ValueError("attention_shape_loss received no nodes")

    bag_count = int(bag_index.max().item()) + 1
    targets = targets.to(device=attention.device, dtype=attention.dtype).reshape(-1)
    if targets.numel() != bag_count:
        raise ValueError(
            f"targets must contain one label per bag, got {targets.numel()} for {bag_count} bags"
        )

    lengths = torch.bincount(bag_index, minlength=bag_count).to(attention.dtype)
    valid_bags = lengths > 1
    if not torch.any(valid_bags):
        raise ValueError("attention_shape_loss requires at least one multi-node bag")

    probabilities = attention.clamp_min(eps)
    probability_sum = attention.new_zeros(bag_count)
    probability_sum.scatter_add_(0, bag_index, probabilities)
    probabilities = probabilities / probability_sum[bag_index]

    entropy_terms = -(probabilities * probabilities.log())
    entropy = attention.new_zeros(bag_count)
    entropy.scatter_add_(0, bag_index, entropy_terms)
    normalized_entropy = entropy / lengths.clamp_min(2).log()
    normalized_entropy = normalized_entropy.clamp(0.0, 1.0)

    positive_bags = targets > 0.5
    per_bag = torch.where(positive_bags, normalized_entropy, 1.0 - normalized_entropy)
    return per_bag[valid_bags].mean()


def model_loss(
    output: ModelOutput,
    targets: torch.Tensor,
    config: Config,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """Compute classification and optional MIL attention-shape loss."""
    loss = binary_loss(output.logits, targets, config, pos_weight=pos_weight)
    weight = config.mil_head.attention_loss_weight
    if not config.mil_head.enabled or weight == 0.0:
        return loss
    if output.attention is None or output.bag_index is None:
        raise ValueError("MIL attention loss is enabled but the model returned no attention")
    return loss + weight * attention_shape_loss(
        output.attention,
        output.bag_index,
        targets,
        config.mil_head.attention_loss_eps,
    )
