# coding=utf-8

import torch
import torch.nn as nn


class _LegacyGatedAttentionScorer(nn.Module):
    """gated-attention 打分器 s_i"""
    def __init__(self, in_dim, attn_hidden=128, gate_hidden=None):
        super().__init__()
        gate_hidden = attn_hidden if gate_hidden is None else gate_hidden
        self.V = nn.Linear(in_dim, attn_hidden)
        self.U = nn.Linear(in_dim, attn_hidden)
        self.w = nn.Linear(attn_hidden, 1, bias=False)
        # self.u = nn.Linear(gate_hidden, 1, bias=False)

    def forward(self, phi):  # [N, 4d]
        t1 = torch.tanh(self.V(phi))
        t2 = torch.sigmoid(self.U(phi))
        gated_features = t1 * t2
        return self.w(gated_features).squeeze(-1)  # s: [N]


class _LegacyMILBranchB(nn.Module):
    """
    分支B 特征 -> 打分 -> 返回图级预测与可解释中间量
    """
    def __init__(self, node_dim, attn_hidden=128, gate_hidden=None, num_classes=2):
        super().__init__()
        self.scorer = _LegacyGatedAttentionScorer(in_dim=node_dim, attn_hidden=attn_hidden, gate_hidden=gate_hidden)
        # 节点分类头，用于 bind_loss
        # self.node_classifier = nn.Linear(node_dim, num_classes)

    def graph_softmax(self, s, batch, tau=1.0, eps=1e-9):
        s = s / tau
        m = scatter_max(s, batch, dim=0)[0][batch]  # [N]
        exp = torch.exp(s - m)
        Z = scatter_add(exp, batch, dim=0)[batch] + eps  # [N]
        return (exp / Z).clamp(1e-6, 1 - 1e-6)  # [N]

    def forward(self, h, batch, eps=None, y=None, k=5):
        s = self.scorer(h)
        s = s.clamp(min=-12.0, max=12.0)

        if eps is None:
            eps = 1e-9
        a_attn = self.graph_softmax(s, batch, tau=1.0, eps=eps)
        a_attn = torch.clamp(a_attn, min=1e-6, max=1.0 - 1e-6)

        z_B = scatter_add(a_attn.unsqueeze(-1) * h, batch, dim=0)

        return {'z_B': z_B, 'a': a_attn} # , 'bind_loss': bind_loss}


class TransformerMILBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        mlp_hidden_dim = max(dim, int(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_path = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, valid_mask):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=~valid_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.drop_path(attn_out)
        x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x + self.mlp(self.norm2(x))
        x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x, attn_weights


class MILBranchB(nn.Module):
    """Lightweight TransMIL-style branch B with a learnable class token."""

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
        del attn_hidden, gate_hidden, num_classes
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError(
                f'node_dim ({node_dim}) must be divisible by num_heads ({num_heads}).'
            )

        self.node_dim = int(node_dim)
        self.num_layers = max(1, int(num_layers))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.node_dim))
        self.layers = nn.ModuleList([
            TransformerMILBlock(
                dim=self.node_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.node_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _pack_batch(self, h, batch):
        if h.dim() != 2:
            raise ValueError(f'Expected h to have shape [N, D], got {tuple(h.shape)}')
        if batch.dim() != 1:
            raise ValueError(f'Expected batch to have shape [N], got {tuple(batch.shape)}')
        if h.size(0) != batch.size(0):
            raise ValueError('h and batch must have the same first dimension.')
        if h.size(0) == 0:
            raise ValueError('MILBranchB received an empty batch of nodes.')

        batch = batch.long()
        bag_count = int(batch.max().item()) + 1
        lengths = torch.bincount(batch, minlength=bag_count)
        max_nodes = int(lengths.max().item())
        x_pad = h.new_zeros((bag_count, max_nodes, h.size(1)))
        mask_valid = torch.zeros((bag_count, max_nodes), dtype=torch.bool, device=h.device)

        cursor = 0
        for bag_idx, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            next_cursor = cursor + length
            x_pad[bag_idx, :length] = h[cursor:next_cursor]
            mask_valid[bag_idx, :length] = True
            cursor = next_cursor

        return x_pad, mask_valid, lengths

    def _extract_cls_attention(self, attn_weights, mask_valid, eps):
        cls_to_nodes = attn_weights[:, :, 0, 1:].mean(dim=1)
        cls_to_nodes = cls_to_nodes.masked_fill(~mask_valid, 0.0)
        denom = cls_to_nodes.sum(dim=1, keepdim=True).clamp_min(eps)
        return cls_to_nodes / denom

    def _flatten_valid_attention(self, a_pad, lengths):
        chunks = []
        for bag_idx, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            chunks.append(a_pad[bag_idx, :length])
        return torch.cat(chunks, dim=0) if chunks else a_pad.new_zeros((0,))

    def forward(self, h, batch, eps=None, y=None, k=5):
        del y, k
        if eps is None:
            eps = 1e-9

        x_pad, mask_valid, lengths = self._pack_batch(h, batch)
        batch_size = x_pad.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x_pad], dim=1)

        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=h.device)
        valid_with_cls = torch.cat([cls_mask, mask_valid], dim=1)

        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, valid_with_cls)

        x = self.final_norm(x)
        x = x * valid_with_cls.unsqueeze(-1).to(dtype=x.dtype)

        z_B = x[:, 0, :]
        a_pad = self._extract_cls_attention(attn_weights, mask_valid, eps=eps)
        a = self._flatten_valid_attention(a_pad, lengths)

        return {
            'z_B': z_B,
            'a': a,
            'a_pad': a_pad,
            'mask_valid': mask_valid,
        }
