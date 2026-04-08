# coding=utf-8

import torch
import torch.nn as nn


class GraphormerSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).'
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias, valid_mask):
        batch_size, num_nodes, _ = x.size()

        qkv = self.qkv(x).view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_bias is not None:
            logits = logits + attn_bias

        key_mask = valid_mask[:, None, None, :]
        logits = logits.masked_fill(~key_mask, -1e9)

        attn = torch.softmax(logits, dim=-1)
        attn = attn * valid_mask[:, None, :, None].to(dtype=attn.dtype)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class GraphormerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dropout=0.1, dropout=0.1, ffn_mult=4):
        super().__init__()
        ffn_dim = hidden_dim * ffn_mult

        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.attn = GraphormerSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias, valid_mask):
        x = x + self.attn(self.ln_attn(x), attn_bias, valid_mask)
        x = x + self.ffn(self.ln_ffn(x))
        return x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)


class GraphormerNodeEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        attn_dropout=0.1,
        ffn_mult=4,
        spatial_pos_max=32,
        degree_max=32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spatial_pos_max = max(1, int(spatial_pos_max))
        self.degree_max = max(1, int(degree_max))
        self.unreachable_bucket = self.spatial_pos_max + 1

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.degree_encoder = nn.Embedding(self.degree_max + 1, hidden_dim)
        self.spatial_pos_encoder = nn.Embedding(self.spatial_pos_max + 2, num_heads)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            GraphormerEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_mult=ffn_mult,
            )
            for _ in range(max(1, int(num_layers)))
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj, batch_num_nodes):
        valid_mask = self._build_valid_mask(adj.size(0), adj.size(1), batch_num_nodes, x.device)
        degree_ids = self._compute_degree_ids(adj, valid_mask)
        spatial_ids = self._compute_spatial_pos_ids(adj, valid_mask)

        h = self.input_proj(x) + self.degree_encoder(degree_ids)
        h = self.input_dropout(h)
        h = h * valid_mask.unsqueeze(-1).to(dtype=h.dtype)

        attn_bias = self.spatial_pos_encoder(spatial_ids).permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            h = layer(h, attn_bias, valid_mask)

        h = self.final_ln(h)
        h = h * valid_mask.unsqueeze(-1).to(dtype=h.dtype)
        return h

    def _build_valid_mask(self, batch_size, max_nodes, batch_num_nodes, device):
        if isinstance(batch_num_nodes, torch.Tensor):
            num_nodes = batch_num_nodes.view(batch_size).long().to(device=device)
        else:
            num_nodes = torch.tensor(
                [int(n) for n in batch_num_nodes],
                dtype=torch.long,
                device=device,
            )
        arange = torch.arange(max_nodes, device=device).unsqueeze(0)
        return arange < num_nodes.unsqueeze(1)

    def _compute_degree_ids(self, adj, valid_mask):
        edge_mask = (adj > 0) & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        degree = edge_mask.sum(dim=-1).long()
        degree = torch.clamp(degree, min=0, max=self.degree_max)
        degree = degree.masked_fill(~valid_mask, 0)
        return degree

    def _compute_spatial_pos_ids(self, adj, valid_mask):
        batch_size, max_nodes, _ = adj.size()
        device = adj.device
        inf = float(max_nodes + self.spatial_pos_max + 1)

        valid_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        edge_mask = (adj > 0) & valid_pair

        dist = torch.full((batch_size, max_nodes, max_nodes), inf, device=device, dtype=torch.float32)
        dist = dist.masked_fill(edge_mask, 1.0)

        diag = torch.arange(max_nodes, device=device)
        dist[:, diag, diag] = 0.0
        dist = dist.masked_fill(~valid_pair, inf)

        for k in range(max_nodes):
            via_k = dist[:, :, k].unsqueeze(-1) + dist[:, k, :].unsqueeze(1)
            dist = torch.minimum(dist, via_k)

        spatial_ids = torch.clamp(dist, min=0.0, max=float(self.spatial_pos_max)).long()
        unreachable_mask = (~valid_pair) | torch.isinf(dist) | (dist >= inf)
        spatial_ids = spatial_ids.masked_fill(unreachable_mask, self.unreachable_bucket)
        return spatial_ids
