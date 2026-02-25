# coding=utf-8

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add, scatter_max


class GatedAttentionScorer(nn.Module):
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


class MILBranchB(nn.Module):
    """
    分支B 特征 -> 打分 -> 返回图级预测与可解释中间量
    """
    def __init__(self, node_dim, attn_hidden=128, gate_hidden=None, num_classes=2):
        super().__init__()
        self.scorer = GatedAttentionScorer(in_dim=node_dim, attn_hidden=attn_hidden, gate_hidden=gate_hidden)
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
        
        # --- Bind Loss Calculation ---
        # bind_loss = torch.tensor(0.0, device=h.device)
        # if y is not None:
        #     # 1. 计算节点 logits [N, K]
        #     node_logits = self.node_classifier(h)
            
        #     # 2. 准备标签 [N]
        #     # y 通常是 [B] 或 [B, 1]，先转为 long
        #     if y.dim() > 1:
        #         y = y.squeeze()
        #     y_long = y.long()
        #     y_nodes = y_long[batch] # 广播到节点
            
        #     # 3. 计算交叉熵 [N]
        #     ce_each = nn.functional.cross_entropy(node_logits, y_nodes, reduction='none')
            
        #     # 4. 正 bag 过滤 (y!=0)
        #     # 假设 y=0 是负类/背景
        #     is_pos_bag = (y_long > 0) # [B]
        #     is_pos_node = is_pos_bag[batch] # [N]

        #     if is_pos_node.any():
        #         # Per-bag Top-K binding (only for positive bags): avoid cross-bag coupling.
        #         pos_bag_ids = torch.unique(batch[is_pos_node])
        #         per_bag_losses = []
        #         for b in pos_bag_ids.tolist():
        #             mask_b = (batch == b)
        #             attn_b = a_attn[mask_b]
        #             ce_b = ce_each[mask_b]

        #             if attn_b.numel() == 0:
        #                 continue
        #             curr_k = min(k, int(attn_b.numel()))
        #             topv, topi = torch.topk(attn_b, k=curr_k, largest=True, sorted=False)

        #             # Re-normalize within selected nodes
        #             topv = topv / (topv.sum() + 1e-8)
        #             per_bag_losses.append((topv * ce_b[topi]).sum())

        #         if len(per_bag_losses) > 0:
        #             bind_loss = torch.stack(per_bag_losses).mean()

        return {'z_B': z_B, 'a': a_attn} # , 'bind_loss': bind_loss}
