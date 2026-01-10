# coding=utf-8

import math
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
    self.a_src = Parameter(torch.empty(heads, per_head, 1))
    self.a_dst = Parameter(torch.empty(heads, per_head, 1))
    self.leakyrelu = torch.nn.LeakyReLU(alpha)
    self.attn_drop = torch.nn.Dropout(attn_dropout)
    self.feat_drop = torch.nn.Dropout(feat_dropout)
    need_proj = residual and (in_dim != (per_head if not concat else heads * per_head))
    self.res_proj = torch.nn.Linear(in_dim, (per_head if not concat else heads * per_head)) if need_proj else None
    need_out = (per_head if not concat else heads * per_head) != out_dim
    self.out_proj = torch.nn.Linear((per_head if not concat else heads * per_head), out_dim) if need_out else None
    self.reset_parameters()

  def reset_parameters(self):
    gain = torch.nn.init.calculate_gain('relu')
    for h in range(self.heads):
      torch.nn.init.xavier_uniform_(self.w[h], gain=gain)
      torch.nn.init.xavier_uniform_(self.a_src[h], gain=gain)
      torch.nn.init.xavier_uniform_(self.a_dst[h], gain=gain)
    if self.res_proj is not None:
      torch.nn.init.xavier_uniform_(self.res_proj.weight, gain=gain)
      if self.res_proj.bias is not None:
        torch.nn.init.constant_(self.res_proj.bias, 0.0)
    if self.out_proj is not None:
      torch.nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
      if self.out_proj.bias is not None:
        torch.nn.init.constant_(self.out_proj.bias, 0.0)

  def forward(self, x, adj, mask=None):
    device = getattr(self._hparams, 'device', x.device) if self._hparams is not None else x.device
    x = self.feat_drop(x)
    B, N, Fin = x.size()
    Wh = torch.einsum('bni,hio->bhno', x, self.w)
    src = torch.einsum('bhno,hoc->bhnc', Wh, self.a_src).squeeze(-1)
    dst = torch.einsum('bhno,hoc->bhnc', Wh, self.a_dst).squeeze(-1)
    e = self.leakyrelu(src.unsqueeze(3) + dst.unsqueeze(2))
    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
    adj_eff = adj + eye
    e = e.masked_fill(adj_eff.unsqueeze(1) == 0, float('-inf'))
    alpha = torch.softmax(e, dim=3)
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

