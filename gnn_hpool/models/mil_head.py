# coding=utf-8

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add


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
    def __init__(self, node_dim, attn_hidden=128, gate_hidden=64):
        super().__init__()
        self.scorer = GatedAttentionScorer(in_dim=4 * node_dim, hidden=attn_hidden)
        self.gate = nn.Sequential(  # g = σ(MLP_g(c))
            nn.Linear(node_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1)
        )

    def forward(self, h, edge_index, batch, c_override=None):
        # Step1: graph context c [B,d]
        c = scatter_mean(h, batch, dim=0) if c_override is None else c_override
        
        # Step2: phi [N,4d]
        c_i = c[batch]
        phi = torch.cat([h, c_i, h*c_i, (h-c_i).abs()], dim=-1)
        
        # Step3: scores -> probs
        s = self.scorer(phi)                      # [N]
        s = s.clamp(min=-12.0, max=12.0)          # 防止极端值
        a = torch.sigmoid(s)                      # [N]
        a = torch.clamp(a, min=1e-6, max=1.0 - 1e-6)  # 改为非就地操作
        
        # Step4: dual readout
        # Noisy-OR per graph
        log1m = torch.log1p(-a)
        sumlog = scatter_add(log1m, batch, dim=0)   # [B]
        y_sparse = 1.0 - torch.exp(sumlog)          # [B]
        y_sparse = torch.clamp(y_sparse, min=1e-6, max=1.0 - 1e-6)
        
        # Mean per graph
        y_dense = scatter_mean(a, batch, dim=0)
        y_dense = torch.clamp(y_dense, min=1e-6, max=1.0 - 1e-6)
        
        # Graph gate
        g = torch.sigmoid(self.gate(c)).squeeze(-1) # [B]
        g = torch.clamp(g, min=1e-4, max=1.0 - 1e-4)
        y_B = (1 - g) * y_sparse + g * y_dense      # [B] 图级概率
        y_B = torch.clamp(y_B, min=1e-6, max=1.0 - 1e-6)
        
        return {'y_B': y_B, 'a': a, 'gate': g}