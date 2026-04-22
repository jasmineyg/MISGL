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
    ):
        del num_classes, num_layers, num_heads, mlp_ratio, dropout, attn_dropout
        super().__init__()
        self.scorer = _GatedAttentionScorer(
            in_dim=node_dim,
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
        weights = torch.zeros_like(scores)
        bag_count = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        for bag_idx in range(bag_count):
            node_mask = batch == bag_idx
            if not torch.any(node_mask):
                continue
            bag_scores = scores[node_mask] / tau
            bag_weights = torch.softmax(bag_scores, dim=0)
            weights[node_mask] = bag_weights
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

    def forward(self, h, batch, eps=None, y=None, k=5):
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
        scores = self.scorer(h).clamp(min=-12.0, max=12.0)
        attention = self.graph_softmax(scores, batch, tau=1.0, eps=eps)

        weighted_nodes = attention.unsqueeze(-1) * h
        bag_count = int(batch.max().item()) + 1
        z_B = h.new_zeros((bag_count, h.size(1)))
        z_B.index_add_(0, batch, weighted_nodes)

        a_pad, mask_valid = self._pad_attention(attention, batch)
        return {
            'z_B': z_B,
            'a': attention,
            'a_pad': a_pad,
            'mask_valid': mask_valid,
        }
