# coding=utf-8

import math

import torch
import torch.nn as nn


class SubGNNBorderRouter(nn.Module):
    """Encode SubGNN border anchors with BiLSTM and fuse them into z_mil."""

    def __init__(
        self,
        input_dim,
        num_anchors,
        node_feature_dim,
        anchor_embed_dim=32,
        anchor_encoder_hidden_dim=None,
        anchor_encoder_layers=1,
        anchor_encoder_dropout=0.0,
        anchor_walk_aggregator='sum',
        gate_hidden_dim=None,
        dropout=0.1,
        residual_init=0.1,
        softmax_temperature=1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_anchors = int(num_anchors)
        self.node_feature_dim = int(node_feature_dim)
        self.anchor_embed_dim = int(anchor_embed_dim)
        self.anchor_encoder_hidden_dim = int(anchor_encoder_hidden_dim or anchor_embed_dim)
        self.anchor_encoder_layers = int(anchor_encoder_layers)
        self.anchor_walk_aggregator = str(anchor_walk_aggregator)
        self.softmax_temperature = max(float(softmax_temperature), 1e-6)
        if self.num_anchors <= 0:
            raise ValueError('SubGNNBorderRouter requires num_anchors > 0.')
        if self.node_feature_dim <= 0:
            raise ValueError('SubGNNBorderRouter requires node_feature_dim > 0.')

        lstm_dropout = float(anchor_encoder_dropout) if self.anchor_encoder_layers > 1 else 0.0
        self.anchor_lstm = nn.LSTM(
            input_size=self.node_feature_dim,
            hidden_size=self.anchor_encoder_hidden_dim,
            num_layers=self.anchor_encoder_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.anchor_lstm_head = nn.Linear(self.anchor_encoder_hidden_dim * 2, self.anchor_embed_dim)

        gate_hidden_dim = int(gate_hidden_dim if gate_hidden_dim is not None else self.input_dim)
        self.border_proj = nn.Sequential(
            nn.Linear(self.anchor_embed_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.anchor_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _encode_anchor_walks(self, anchor_walk_features, dtype, device):
        if anchor_walk_features.dim() == 5:
            # DataLoader stacks the same global anchor tensor for every sample; keep one copy.
            anchor_walk_features = anchor_walk_features[0]
        if anchor_walk_features.dim() != 4:
            raise ValueError(
                'Expected anchor_walk_features shape [A, W, L, F] or [B, A, W, L, F], '
                f'got {tuple(anchor_walk_features.shape)}'
            )
        if anchor_walk_features.size(0) != self.num_anchors:
            raise ValueError(f'Expected {self.num_anchors} anchors, got {anchor_walk_features.size(0)}.')
        if anchor_walk_features.size(-1) != self.node_feature_dim:
            raise ValueError(
                f'Expected anchor node feature dim {self.node_feature_dim}, got {anchor_walk_features.size(-1)}.'
            )

        features = anchor_walk_features.to(device=device, dtype=dtype)
        n_anchors, n_walks, walk_len, feat_dim = features.shape
        walk_input = features.reshape(n_anchors * n_walks, walk_len, feat_dim)
        lstm_out, _ = self.anchor_lstm(walk_input)
        if self.anchor_walk_aggregator == 'last':
            walk_hidden = lstm_out[:, -1, :]
        elif self.anchor_walk_aggregator == 'mean':
            walk_hidden = lstm_out.mean(dim=1)
        elif self.anchor_walk_aggregator == 'sum':
            walk_hidden = lstm_out.sum(dim=1)
        else:
            raise ValueError(f'Unsupported anchor_walk_aggregator: {self.anchor_walk_aggregator}')
        walk_embed = self.anchor_lstm_head(walk_hidden)
        # SubGNN sums the encoded random-walk representations for each structure anchor.
        return walk_embed.view(n_anchors, n_walks, self.anchor_embed_dim).sum(dim=1)

    def forward(self, z_mil, border_anchor_sim, anchor_walk_features):
        if z_mil.dim() != 2:
            raise ValueError(f'Expected z_mil to have shape [B, D], got {tuple(z_mil.shape)}')
        if border_anchor_sim.dim() != 2:
            raise ValueError(
                f'Expected border_anchor_sim to have shape [B, A], got {tuple(border_anchor_sim.shape)}'
            )
        if border_anchor_sim.size(0) != z_mil.size(0):
            raise ValueError('border_anchor_sim and z_mil must have the same batch size.')
        if border_anchor_sim.size(1) != self.num_anchors:
            raise ValueError(
                f'Expected {self.num_anchors} border anchors, got {border_anchor_sim.size(1)}.'
            )

        sim = border_anchor_sim.to(device=z_mil.device, dtype=z_mil.dtype)
        anchor_weights = torch.softmax(sim / self.softmax_temperature, dim=-1)
        anchor_embeds = self._encode_anchor_walks(anchor_walk_features, dtype=z_mil.dtype, device=z_mil.device)
        border_embed = anchor_weights @ anchor_embeds
        z_border = self.border_proj(border_embed)
        border_gate = self.gate(torch.cat([z_mil, z_border], dim=-1))
        residual = self.residual_scale * border_gate * z_border
        z_fused = z_mil + residual

        entropy = -(anchor_weights * anchor_weights.clamp_min(1e-12).log()).sum(dim=-1)
        if self.num_anchors > 1:
            entropy = entropy / math.log(float(self.num_anchors))
        residual_ratio = residual.norm(dim=-1) / z_mil.norm(dim=-1).clamp_min(1e-12)

        return {
            'z_fused': z_fused,
            'z_border': z_border,
            'border_gate': border_gate,
            'border_anchor_weights': anchor_weights,
            'border_anchor_entropy': entropy,
            'border_anchor_embeds': anchor_embeds,
            'border_residual': residual,
            'border_residual_ratio': residual_ratio,
        }
