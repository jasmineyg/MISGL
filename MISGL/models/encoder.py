# coding=utf-8

import torch
import torch.nn as nn

from MISGL.layers.gat_layer import ResidualGATLayer
from MISGL.models.mil_head import MILBranchB
from MISGL.utils.global_variables import g_key
from MISGL.utils import hparams_lib


class MISGLEncoder(nn.Module):
    """Simple encoder: raw node features -> 1-layer GAT -> pooled graph embedding -> classifier."""

    def __init__(self, hparams, data_name=None):
        super(MISGLEncoder, self).__init__()

        self._hparams = hparams_lib.copy_hparams(hparams)
        self.data_name = data_name if data_name is not None else getattr(self._hparams, 'data_name', None)
        self._device = torch.device(self._hparams.device)

        bb_cfg = getattr(self._hparams, 'branch_b', None)
        self.use_branch_b = bool(bb_cfg and bb_cfg.get('use', False))
        self.use_coarse_graph = bool(getattr(self._hparams, 'use_coarse_graph', False))

        in_dim = int(self._hparams.channel_list[0])
        hidden_dim = int(self._hparams.channel_list[1])
        classifier_hidden_dim = int(self._hparams.channel_list[-2])
        classifier_out_dim = int(self._hparams.channel_list[-1])
        dropout = float(getattr(self._hparams, 'dropout', 0.3))
        negative_slope = float(getattr(self._hparams, 'leaky_relu_alpha', 0.2))
        gat_heads = int(getattr(self._hparams, 'gat_heads', 4))
        gat_attn_dp = float(getattr(self._hparams, 'gat_attn_dropout', dropout))
        gat_feat_dp = float(getattr(self._hparams, 'gat_feat_dropout', dropout))
        gat_alpha = float(getattr(self._hparams, 'gat_alpha', 0.2))
        gat_concat = bool(getattr(self._hparams, 'gat_concat', True))
        gat_residual = bool(getattr(self._hparams, 'gat_residual', True))

        self.gat_layer = ResidualGATLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            hparams=self._hparams,
            heads=gat_heads,
            attn_dropout=gat_attn_dp,
            feat_dropout=gat_feat_dp,
            alpha=gat_alpha,
            concat=gat_concat,
            residual=gat_residual,
        )

        # 分类器 两层MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden_dim, classifier_out_dim),
        )

        if self.use_branch_b:
            bb_attn_hidden = int(bb_cfg.get('attn_hidden', 128))
            bb_gate_hidden = int(bb_cfg.get('gate_hidden', bb_attn_hidden))
            self.branch_b_use_structural_features = bool(
                bb_cfg.get('use_structural_features', bb_cfg.get('structural_features', False))
            )
            self.branch_b_structure_undirected = bool(bb_cfg.get('structural_undirected', True))
            self.branch_b_structural_feature_names = (
                'degree_norm',
                'log_degree_norm',
                'avg_neighbor_degree_norm',
                '2_hop_walk_log_norm',
            ) if self.branch_b_use_structural_features else ()
            self.branch_b_head = MILBranchB(
                node_dim=hidden_dim,
                attn_hidden=bb_attn_hidden,
                gate_hidden=bb_gate_hidden,
                structural_dim=len(self.branch_b_structural_feature_names),
            )
        else:
            self.branch_b_use_structural_features = False
            self.branch_b_structure_undirected = True
            self.branch_b_structural_feature_names = ()
            self.branch_b_head = None

        self.reset_parameters()

    def reset_parameters(self):
        self.gat_layer.reset_parameters()
        gain = torch.nn.init.calculate_gain('leaky_relu', float(getattr(self._hparams, 'leaky_relu_alpha', 0.2)))
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, graph_input):
        ypred, _ = self._encode(graph_input, return_embeddings=False)
        return ypred

    def forward_with_embeddings(self, graph_input):
        return self._encode(graph_input, return_embeddings=True)

    def _encode(self, graph_input, return_embeddings=False):
        x = graph_input[g_key.x]
        adj = graph_input[g_key.adj_mat]
        batch_num_nodes = graph_input[g_key.node_num]

        max_nodes = x.size(1)
        mask = self.construct_mask(max_nodes, batch_num_nodes, x.device)
        h = self.gat_layer(x, adj, mask)
        h1 = self._masked_mean_pool(h, batch_num_nodes)
        classifier_input = h1
        branch_b_out = None

        if self.use_branch_b:
            h_flat, batch_index = self._flatten_valid_nodes(h, batch_num_nodes)
            structural_flat = None
            if self.branch_b_use_structural_features:
                structural_features = self._compute_branch_b_structural_features(
                    adj,
                    batch_num_nodes,
                    dtype=h.dtype,
                )
                structural_flat, _ = self._flatten_valid_nodes(structural_features, batch_num_nodes)
            branch_b_out = (
                self.branch_b_head(
                    h_flat,
                    batch_index,
                    return_padded_attention=return_embeddings,
                    structural_features=structural_flat,
                )
                if h_flat.size(0) > 0 else None
            )
            if branch_b_out is not None:
                classifier_input = branch_b_out['z_B']

        ypred = self.classifier(classifier_input)

        # Keep compatibility with existing analysis/export scripts.
        self.current_x2 = h
        self.current_h1 = h1
        self.current_graph_emb_classifier = classifier_input

        if self.use_branch_b:
            model_out = {'ypred_A': ypred, 'branch_b': branch_b_out}
        else:
            model_out = ypred

        if not return_embeddings:
            return model_out, None

        emb = {
            'h': h,
            'mean_vec': h1,
            'graph_emb_H1': h1,
            'graph_emb_classifier': classifier_input,
            'graph_emb': classifier_input,
        }
        if branch_b_out is not None:
            emb['z_B'] = branch_b_out['z_B']
        return model_out, emb

    def _compute_branch_b_structural_features(self, adj, batch_num_nodes, dtype=None):
        if adj.dim() != 3:
            raise ValueError(f'Expected adj to have shape [B, N, N], got {tuple(adj.shape)}')

        batch_size, max_nodes, _ = adj.size()
        device = adj.device
        out_dtype = dtype if dtype is not None else torch.float32
        lengths = self._lengths_tensor(batch_num_nodes, device=device)
        valid_mask = torch.arange(max_nodes, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        valid_pair_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)

        adj_bool = adj != 0
        if self.branch_b_structure_undirected:
            adj_bool = adj_bool | (adj.transpose(1, 2) != 0)
        eye = torch.eye(max_nodes, device=device, dtype=torch.bool).unsqueeze(0)
        adj_bool = adj_bool & valid_pair_mask & (~eye)
        adj_float = adj_bool.to(dtype=out_dtype)

        graph_size = lengths.to(dtype=out_dtype).clamp_min(1.0)
        max_degree = (graph_size - 1.0).clamp_min(1.0)
        degree = adj_float.sum(dim=-1)
        degree_norm = degree / max_degree.view(batch_size, 1)
        log_degree_norm = torch.log1p(degree) / torch.log1p(max_degree).view(batch_size, 1)

        neighbor_degree_sum = torch.bmm(adj_float, degree.unsqueeze(-1)).squeeze(-1)
        avg_neighbor_degree = neighbor_degree_sum / degree.clamp_min(1.0)
        avg_neighbor_degree_norm = avg_neighbor_degree / max_degree.view(batch_size, 1)

        two_hop_walk_count = neighbor_degree_sum
        two_hop_denom = torch.pow(max_degree, 2).clamp_min(1.0)
        two_hop_walk_log_norm = (
            torch.log1p(two_hop_walk_count) / torch.log1p(two_hop_denom).view(batch_size, 1)
        )

        structural_features = torch.stack(
            [
                degree_norm,
                log_degree_norm,
                avg_neighbor_degree_norm,
                two_hop_walk_log_norm,
            ],
            dim=-1,
        )
        return structural_features * valid_mask.unsqueeze(-1).to(dtype=out_dtype)

    def _flatten_valid_nodes(self, node_embeddings, batch_num_nodes):
        if isinstance(batch_num_nodes, torch.Tensor):
            lengths = batch_num_nodes.to(device=node_embeddings.device, dtype=torch.long).view(-1)
        else:
            lengths = torch.tensor(
                [int(n) for n in batch_num_nodes],
                dtype=torch.long,
                device=node_embeddings.device,
            )

        batch_size, max_nodes, hidden_dim = node_embeddings.size()
        valid_mask = torch.arange(max_nodes, device=node_embeddings.device).unsqueeze(0) < lengths.unsqueeze(1)
        if torch.any(valid_mask):
            batch_index = torch.arange(batch_size, device=node_embeddings.device).repeat_interleave(lengths.clamp_min(0))
            return node_embeddings[valid_mask], batch_index

        return (
            node_embeddings.new_zeros((0, hidden_dim)),
            torch.zeros((0,), dtype=torch.long, device=node_embeddings.device),
        )

    def _masked_mean_pool(self, node_embeddings, batch_num_nodes):
        if isinstance(batch_num_nodes, torch.Tensor):
            num_list = batch_num_nodes.view(-1).float().to(device=node_embeddings.device)
        else:
            num_list = torch.tensor(
                [float(int(n)) for n in batch_num_nodes],
                device=node_embeddings.device,
            )

        sum_vec = node_embeddings.sum(dim=1)
        denom = torch.clamp(num_list, min=1.0).unsqueeze(1)
        return sum_vec / denom

    def construct_mask(self, max_nodes, batch_num_nodes, device=None):
        mask_device = self._device if device is None else device
        if isinstance(batch_num_nodes, torch.Tensor):
            lengths = batch_num_nodes.to(device=mask_device, dtype=torch.long).view(-1)
        else:
            lengths = torch.tensor(
                [int(n) for n in batch_num_nodes],
                dtype=torch.long,
                device=mask_device,
            )

        mask = torch.arange(max_nodes, device=mask_device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask.to(dtype=torch.float32).unsqueeze(2)

    def _lengths_tensor(self, batch_num_nodes, device):
        if isinstance(batch_num_nodes, torch.Tensor):
            return batch_num_nodes.to(device=device, dtype=torch.long).view(-1)
        return torch.tensor(
            [int(n) for n in batch_num_nodes],
            dtype=torch.long,
            device=device,
        )
