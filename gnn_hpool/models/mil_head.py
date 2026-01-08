# coding=utf-8

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add, scatter_max


class GatedAttentionScorer(nn.Module):
    """Step3: gated-attention 打分器 s_i"""
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.U = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)
        self.u = nn.Linear(hidden, 1, bias=False)

    def forward(self, phi):  # [N, 4d]
        t1 = torch.tanh(self.V(phi))
        t2 = torch.sigmoid(self.U(phi))
        return (self.w(t1) * self.u(t2)).squeeze(-1)  # s: [N]


class MILBranchB(nn.Module):
    """
    分支B顶层：Step1 均值上下文 -> Step2 节点-图匹配特征 -> Step3 打分 -> Step4 双读出+门控
    forward 返回图级预测与可解释中间量（a/s/c/g 等）
    """
    def __init__(self, node_dim, attn_hidden=128):
        super().__init__()
        self.scorer = GatedAttentionScorer(in_dim=3 * node_dim, hidden=attn_hidden)

    def graph_softmax(self, s, batch, tau=1.0, eps=1e-9):
        s = s / tau
        m = scatter_max(s, batch, dim=0)[0][batch]  # [N]
        exp = torch.exp(s - m)
        Z = scatter_add(exp, batch, dim=0)[batch] + eps  # [N]
        return (exp / Z).clamp(1e-6, 1 - 1e-6)  # [N]


    def forward(self, h, edge_index, batch, c_override=None):
        c = scatter_mean(h, batch, dim=0) if c_override is None else c_override
        c_i = c[batch]
        phi = torch.cat([h, h * c_i, (h - c_i).abs()], dim=-1)

        s = self.scorer(phi)
        s = s.clamp(min=-12.0, max=12.0)

        a_attn = self.graph_softmax(s, batch, tau=1.0, eps=1e-9)
        a_attn = torch.clamp(a_attn, min=1e-6, max=1.0 - 1e-6)

        z_B = scatter_add(a_attn.unsqueeze(-1) * h, batch, dim=0)

        return {'z_B': z_B, 'a': a_attn}