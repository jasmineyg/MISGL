# coding=utf-8

import torch
import torch.nn as nn


class _GatedAttentionScorer(nn.Module):
    """Gated-attention scorer for node-level MIL weights."""

    def __init__(self, in_dim, attn_hidden=128, gate_hidden=None):
        super().__init__()
        del gate_hidden
        self.V = nn.Linear(in_dim, attn_hidden)
        self.U = nn.Linear(in_dim, attn_hidden)
        self.w = nn.Linear(attn_hidden, 1, bias=False)

    def forward(self, phi):
        t1 = torch.tanh(self.V(phi))
        t2 = torch.sigmoid(self.U(phi))
        gated_features = t1 * t2
        return self.w(gated_features).squeeze(-1)


class MILBranchB(nn.Module):
    """Gated-attention MIL head that aggregates node embeddings into bag embeddings."""

    def __init__(
        self,
        node_dim,
        attn_hidden=128,
        gate_hidden=None,
        num_classes=2,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        attn_dropout=0.0,
        structural_dim=0,
    ):
        del num_classes, num_layers, num_heads, mlp_ratio, dropout, attn_dropout
        super().__init__()
        self.node_dim = int(node_dim)
        self.structural_dim = int(structural_dim)
        self.scorer = _GatedAttentionScorer(
            in_dim=self.node_dim + self.structural_dim,
            attn_hidden=attn_hidden,
            gate_hidden=gate_hidden,
        )

    def graph_softmax(self, scores, batch, tau=1.0, eps=1e-9):
        if scores.dim() != 1:
            raise ValueError(f'Expected scores to have shape [N], got {tuple(scores.shape)}')
        if batch.dim() != 1:
            raise ValueError(f'Expected batch to have shape [N], got {tuple(batch.shape)}')
        if scores.size(0) != batch.size(0):
            raise ValueError('scores and batch must have the same first dimension.')

        batch = batch.long()
        bag_count = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if bag_count == 0:
            return torch.zeros_like(scores)

        scaled_scores = scores / tau
        max_per_bag = scores.new_full((bag_count,), -torch.inf)
        if not hasattr(max_per_bag, 'scatter_reduce_'):
            weights = torch.zeros_like(scores)
            for bag_idx in range(bag_count):
                node_mask = batch == bag_idx
                if not torch.any(node_mask):
                    continue
                weights[node_mask] = torch.softmax(scaled_scores[node_mask], dim=0)
            return weights.clamp(1e-6, 1.0 - 1e-6).clamp_min(eps)

        max_per_bag.scatter_reduce_(0, batch, scaled_scores, reduce='amax', include_self=True)

        exp_scores = torch.exp(scaled_scores - max_per_bag[batch])
        sum_per_bag = scores.new_zeros((bag_count,))
        sum_per_bag.scatter_add_(0, batch, exp_scores)
        weights = exp_scores / sum_per_bag[batch].clamp_min(eps)
        return weights.clamp(1e-6, 1.0 - 1e-6).clamp_min(eps)

    def _pad_attention(self, attention, batch):
        batch = batch.long()
        bag_count = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        lengths = torch.bincount(batch, minlength=bag_count)
        max_nodes = int(lengths.max().item()) if lengths.numel() > 0 else 0

        a_pad = attention.new_zeros((bag_count, max_nodes))
        mask_valid = torch.zeros((bag_count, max_nodes), dtype=torch.bool, device=attention.device)

        for bag_idx in range(bag_count):
            bag_attention = attention[batch == bag_idx]
            length = bag_attention.size(0)
            if length == 0:
                continue
            a_pad[bag_idx, :length] = bag_attention
            mask_valid[bag_idx, :length] = True

        return a_pad, mask_valid

    def forward(
        self,
        h,
        batch,
        eps=None,
        y=None,
        k=5,
        return_padded_attention=False,
        structural_features=None,
    ):
        del y, k
        if h.dim() != 2:
            raise ValueError(f'Expected h to have shape [N, D], got {tuple(h.shape)}')
        if batch.dim() != 1:
            raise ValueError(f'Expected batch to have shape [N], got {tuple(batch.shape)}')
        if h.size(0) != batch.size(0):
            raise ValueError('h and batch must have the same first dimension.')
        if h.size(0) == 0:
            raise ValueError('MILBranchB received an empty batch of nodes.')

        if eps is None:
            eps = 1e-9

        batch = batch.long()
        score_input = h
        if self.structural_dim > 0:
            if structural_features is None:
                raise ValueError('structural_features is required when structural_dim > 0.')
            if structural_features.dim() != 2:
                raise ValueError(
                    f'Expected structural_features to have shape [N, G], got {tuple(structural_features.shape)}'
                )
            if structural_features.size(0) != h.size(0):
                raise ValueError('structural_features and h must have the same first dimension.')
            if structural_features.size(1) != self.structural_dim:
                raise ValueError(
                    f'Expected structural_features dim {self.structural_dim}, got {structural_features.size(1)}.'
                )
            score_input = torch.cat([h, structural_features.to(device=h.device, dtype=h.dtype)], dim=-1)

        scores = self.scorer(score_input).clamp(min=-12.0, max=12.0)
        attention = self.graph_softmax(scores, batch, tau=1.0, eps=eps)

        weighted_nodes = attention.unsqueeze(-1) * h
        bag_count = int(batch.max().item()) + 1
        z_B = h.new_zeros((bag_count, h.size(1)))
        z_B.index_add_(0, batch, weighted_nodes)

        output = {
            'z_B': z_B,
            'a': attention,
        }
        if return_padded_attention:
            a_pad, mask_valid = self._pad_attention(attention, batch)
            output['a_pad'] = a_pad
            output['mask_valid'] = mask_valid
        return output
