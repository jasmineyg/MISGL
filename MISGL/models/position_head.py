# coding=utf-8

import torch
import torch.nn as nn


class ResidualGCNPositionHead(nn.Module):
    """One-layer residual GCN over the coarse subgraph graph."""

    def __init__(
        self,
        node_dim,
        num_layers=1,
        dropout=0.1,
        row_normalize=True,
        residual_init=0.1,
    ):
        super().__init__()
        if int(num_layers) != 1:
            raise ValueError('ResidualGCNPositionHead currently supports num_layers=1 only.')

        self.node_dim = int(node_dim)
        self.row_normalize = bool(row_normalize)
        self.neighbor_proj = nn.Linear(self.node_dim, self.node_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=float(dropout))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))
        self.register_buffer('coarse_memory', torch.zeros((0, self.node_dim), dtype=torch.float32))

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.neighbor_proj.weight)
        if self.neighbor_proj.bias is not None:
            torch.nn.init.constant_(self.neighbor_proj.bias, 0.0)

    def reset_memory(self, num_nodes=0, device=None, dtype=None):
        memory_device = self.coarse_memory.device if device is None else device
        memory_dtype = self.coarse_memory.dtype if dtype is None else dtype
        self.coarse_memory = torch.zeros(
            (int(num_nodes), self.node_dim),
            device=memory_device,
            dtype=memory_dtype,
        )

    def snapshot_memory(self):
        return self.coarse_memory.detach().clone()

    def restore_memory(self, snapshot):
        if snapshot is None:
            self.reset_memory()
        else:
            self.coarse_memory = snapshot.detach().clone()

    def _ensure_memory(self, num_nodes, device, dtype):
        num_nodes = int(num_nodes)
        needs_resize = self.coarse_memory.size(0) != num_nodes or self.coarse_memory.size(1) != self.node_dim
        needs_device = self.coarse_memory.device != device
        needs_dtype = self.coarse_memory.dtype != dtype
        if needs_resize or needs_device or needs_dtype:
            self.reset_memory(num_nodes=num_nodes, device=device, dtype=dtype)

    def update_memory(self, coarse_node_id, center_h, coarse_node_num=None):
        if coarse_node_id is None:
            return
        coarse_node_id = coarse_node_id.to(device=center_h.device, dtype=torch.long).view(-1)
        if coarse_node_id.numel() == 0:
            return

        valid = coarse_node_id >= 0
        if not torch.any(valid):
            return

        if coarse_node_num is not None:
            if isinstance(coarse_node_num, torch.Tensor):
                num_nodes = int(coarse_node_num.max().item())
            else:
                num_nodes = int(coarse_node_num)
        else:
            num_nodes = int(coarse_node_id[valid].max().item()) + 1
        self._ensure_memory(num_nodes, center_h.device, center_h.dtype)
        self.coarse_memory[coarse_node_id[valid]] = center_h.detach()[valid]

    def _replace_current_batch_neighbors(self, neighbor_index, neighbor_features, center_h, coarse_node_id, valid_mask):
        coarse_node_id = coarse_node_id.to(device=center_h.device, dtype=torch.long).view(-1)
        for batch_pos in range(center_h.size(0)):
            node_id = coarse_node_id[batch_pos]
            if node_id < 0:
                continue
            match_mask = (neighbor_index == node_id) & valid_mask
            if torch.any(match_mask):
                neighbor_features[match_mask] = center_h[batch_pos]
        return neighbor_features

    def forward(self, center_h, coarse_node_id, neighbor_index, neighbor_weight, coarse_node_num=None):
        if center_h.dim() != 2:
            raise ValueError(f'Expected center_h to have shape [B, D], got {tuple(center_h.shape)}')
        if center_h.size(1) != self.node_dim:
            raise ValueError(f'Expected center_h dim {self.node_dim}, got {center_h.size(1)}')

        coarse_node_id = coarse_node_id.to(device=center_h.device, dtype=torch.long).view(-1)
        neighbor_index = neighbor_index.to(device=center_h.device, dtype=torch.long)
        neighbor_weight = neighbor_weight.to(device=center_h.device, dtype=center_h.dtype)
        if neighbor_index.dim() != 2:
            raise ValueError(f'Expected neighbor_index to have shape [B, K], got {tuple(neighbor_index.shape)}')
        if neighbor_weight.shape != neighbor_index.shape:
            raise ValueError('neighbor_weight must have the same shape as neighbor_index.')
        if coarse_node_id.size(0) != center_h.size(0) or neighbor_index.size(0) != center_h.size(0):
            raise ValueError('coarse ids, neighbor tensors, and center_h must share batch size.')

        if coarse_node_num is not None:
            if isinstance(coarse_node_num, torch.Tensor):
                num_nodes = int(coarse_node_num.max().item())
            else:
                num_nodes = int(coarse_node_num)
        else:
            max_ids = [coarse_node_id[coarse_node_id >= 0]]
            if torch.any(neighbor_index >= 0):
                max_ids.append(neighbor_index[neighbor_index >= 0])
            max_id_tensor = torch.cat(max_ids) if max_ids else center_h.new_zeros((0,), dtype=torch.long)
            num_nodes = int(max_id_tensor.max().item()) + 1 if max_id_tensor.numel() > 0 else 0

        self._ensure_memory(num_nodes, center_h.device, center_h.dtype)
        self.update_memory(coarse_node_id, center_h, coarse_node_num=num_nodes)

        valid_mask = (neighbor_index >= 0) & (neighbor_weight > 0)
        safe_neighbor_index = neighbor_index.clamp_min(0)
        if self.coarse_memory.size(0) == 0:
            neighbor_features = center_h.new_zeros((*neighbor_index.shape, center_h.size(1)))
        else:
            safe_neighbor_index = safe_neighbor_index.clamp_max(self.coarse_memory.size(0) - 1)
            neighbor_features = self.coarse_memory[safe_neighbor_index]
        neighbor_features = neighbor_features * valid_mask.unsqueeze(-1).to(dtype=center_h.dtype)
        neighbor_features = self._replace_current_batch_neighbors(
            neighbor_index,
            neighbor_features,
            center_h,
            coarse_node_id,
            valid_mask,
        )

        weights = neighbor_weight * valid_mask.to(dtype=center_h.dtype)
        if self.row_normalize:
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        aggregated = torch.sum(weights.unsqueeze(-1) * neighbor_features, dim=1)
        message = self.dropout(self.activation(self.neighbor_proj(aggregated)))
        return center_h + self.residual_scale.to(dtype=center_h.dtype) * message
