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

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden_dim, classifier_out_dim),
        )

        if self.use_branch_b:
            bb_attn_hidden = int(bb_cfg.get('attn_hidden', 128))
            bb_gate_hidden = int(bb_cfg.get('gate_hidden', bb_attn_hidden))
            self.branch_b_head = MILBranchB(
                node_dim=hidden_dim,
                attn_hidden=bb_attn_hidden,
                gate_hidden=bb_gate_hidden,
            )
        else:
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
            branch_b_out = self.branch_b_head(h_flat, batch_index) if h_flat.size(0) > 0 else None
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

    def _flatten_valid_nodes(self, node_embeddings, batch_num_nodes):
        if isinstance(batch_num_nodes, torch.Tensor):
            num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
        else:
            num_list = [int(n) for n in batch_num_nodes]

        chunks = []
        batch_chunks = []
        for bag_idx, num_nodes in enumerate(num_list):
            if num_nodes <= 0:
                continue
            chunks.append(node_embeddings[bag_idx, :num_nodes, :])
            batch_chunks.append(
                torch.full(
                    (num_nodes,),
                    bag_idx,
                    dtype=torch.long,
                    device=node_embeddings.device,
                )
            )

        if chunks:
            return torch.cat(chunks, dim=0), torch.cat(batch_chunks, dim=0)

        hidden_dim = node_embeddings.size(-1)
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
            num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
        else:
            num_list = [int(n) for n in batch_num_nodes]

        batch_size = len(num_list)
        out_tensor = torch.zeros(batch_size, max_nodes, device=mask_device)
        for i, n in enumerate(num_list):
            out_tensor[i, :n] = 1.0
        return out_tensor.unsqueeze(2)

