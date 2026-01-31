# pgnn_position_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class PGNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dist_trainable=True):
        super().__init__()
        self.dist_trainable = dist_trainable
        if dist_trainable:
            self.dist_compute = Nonlinear(1, hidden_dim, 1)
        self.linear_hidden = nn.Linear(input_dim * 2, hidden_dim)
        self.linear_out_position = nn.Linear(hidden_dim, 1)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze(-1)

        subset_features = feature[dists_argmax.reshape(-1), :]
        subset_features = subset_features.reshape(dists_argmax.size(0), dists_argmax.size(1), feature.size(1))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).expand(-1, dists_max.size(1), -1)
        messages = torch.cat([messages, self_feature], dim=-1)

        messages = F.relu(self.linear_hidden(messages))      # [N, M, hidden]
        out_position = self.linear_out_position(messages).squeeze(-1)  # [N, M]
        out_structure = messages.mean(dim=1)                 # [N, hidden]
        return out_position, out_structure

class PositionConcatClassifier(nn.Module):
    def __init__(self, h_dim, num_classes, p_hidden_dim=32, p_out_dim=32, dist_trainable=True):
        super().__init__()
        self.pos_layer = PGNNLayer(h_dim, p_hidden_dim, dist_trainable=dist_trainable)
        self.p_proj = nn.Linear(-1, -1)  # 占位，首次 forward 时重建
        self.p_out_dim = p_out_dim
        self.classifier = nn.Linear(h_dim + p_out_dim, num_classes)

    def forward(self, h, dists_max, dists_argmax):
        p, _ = self.pos_layer(h, dists_max, dists_argmax)    # p: [N, M]
        if self.p_proj.in_features != p.size(1):
            self.p_proj = nn.Linear(p.size(1), self.p_out_dim).to(p.device)
        p = self.p_proj(p)                                   # [N, p_out_dim]
        z = torch.cat([h, p], dim=-1)
        return self.classifier(z)