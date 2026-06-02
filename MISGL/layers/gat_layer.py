# coding=utf-8

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter


class ResidualGATLayer(Module):
  def __init__(self, in_dim, out_dim, hparams=None, heads=4, attn_dropout=0.1, feat_dropout=0.1, alpha=0.2, concat=True, residual=True):
    super(ResidualGATLayer, self).__init__()
    self._hparams = hparams
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.heads = heads
    self.alpha = alpha
    self.concat = concat
    self.residual = residual
    per_head = out_dim if not concat else max(1, out_dim // heads)
    self.w = Parameter(torch.empty(heads, in_dim, per_head))
    self.attn_src = Parameter(torch.empty(heads, per_head))
    self.attn_dst = Parameter(torch.empty(heads, per_head))
    self.leaky_relu = torch.nn.LeakyReLU(alpha)
    self.attn_drop = torch.nn.Dropout(attn_dropout)
    self.feat_drop = torch.nn.Dropout(feat_dropout)
    need_proj = residual and (in_dim != (per_head if not concat else heads * per_head))
    self.res_proj = torch.nn.Linear(in_dim, (per_head if not concat else heads * per_head)) if need_proj else None
    need_out = (per_head if not concat else heads * per_head) != out_dim
    self.out_proj = torch.nn.Linear((per_head if not concat else heads * per_head), out_dim) if need_out else None
    self.last_attention_summary = None
    self.reset_parameters()

  def reset_parameters(self):
    gain = torch.nn.init.calculate_gain('leaky_relu', self.alpha)
    for h in range(self.heads):
      torch.nn.init.xavier_uniform_(self.w[h], gain=gain)
    torch.nn.init.xavier_uniform_(self.attn_src, gain=gain)
    torch.nn.init.xavier_uniform_(self.attn_dst, gain=gain)
    if self.res_proj is not None:
      torch.nn.init.xavier_uniform_(self.res_proj.weight, gain=gain)
      if self.res_proj.bias is not None:
        torch.nn.init.constant_(self.res_proj.bias, 0.0)
    if self.out_proj is not None:
      torch.nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
      if self.out_proj.bias is not None:
        torch.nn.init.constant_(self.out_proj.bias, 0.0)

  def forward(self, x, adj, mask=None):
    device = x.device
    x = self.feat_drop(x)
    B, N, Fin = x.size()
    Wh = torch.einsum('bni,hio->bhno', x, self.w)
    src_scores = (Wh * self.attn_src.view(1, self.heads, 1, -1)).sum(dim=-1)
    dst_scores = (Wh * self.attn_dst.view(1, self.heads, 1, -1)).sum(dim=-1)
    e = self.leaky_relu(src_scores.unsqueeze(3) + dst_scores.unsqueeze(2))
    eye = torch.eye(N, device=device, dtype=adj.dtype).unsqueeze(0).expand(B, N, N)
    adj_eff = adj + eye
    attn_mask = adj_eff != 0
    e = e.masked_fill(attn_mask.unsqueeze(1) == 0, float('-inf'))
    alpha = torch.softmax(e, dim=3)
    alpha_summary = alpha.mean(dim=(1, 2))
    if mask is not None:
      alpha_summary = alpha_summary * mask.squeeze(-1)
    self.last_attention_summary = alpha_summary.detach()
    alpha = self.attn_drop(alpha)
    z = torch.einsum('bhij,bhjo->bhio', alpha, Wh)
    if self.concat:
      z = z.reshape(B, N, -1)
    else:
      z = z.mean(dim=1)
    res = x if self.res_proj is None else self.res_proj(x)
    out = z + res if self.residual else z
    if self.out_proj is not None:
      out = self.out_proj(out)
    if mask is not None:
      out = out * mask
    return out

