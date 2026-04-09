# coding=utf-8

import os
import pickle
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from MISGL.utils.global_variables import g_key
from MISGL.utils import hparams_lib
from MISGL.layers import gcn_layer
from MISGL.models.mil_head import MILBranchB
from MISGL.layers.graphormer_layer import GraphormerNodeEncoder


class GcnHpoolEncoder(nn.Module):
    """
    GCN/HPool Encoder Refactored.
    Logic Flow:
    1. Feature Projection -> x1 # 线形层
    2. Backbone (Graphormer) -> x2
    3. Pooling: x1 -> h1, x2 -> h2
    4. Coarse Graph: h1 -> Coarse GCN -> h4
    5. MIL Branch: x2 -> MIL Head -> h3
    6. Classifier: [h1, h2, h4, h3] -> ypred
    """

    def __init__(self, hparams, data_name=None):
        super(GcnHpoolEncoder, self).__init__()

        self._hparams = hparams_lib.copy_hparams(hparams)
        self.data_name = data_name if data_name is not None else getattr(self._hparams, 'data_name', None)
        self._device = torch.device(self._hparams.device)

        bb_cfg = getattr(self._hparams, 'branch_b', None)
        self.use_branch_b = bool(bb_cfg and bb_cfg.get('use', False))
        self.use_coarse_graph = bool(getattr(self._hparams, 'use_coarse_graph', False))

        in_dim = self._hparams.channel_list[0]
        hidden_dim = self._hparams.channel_list[1]

        graphormer_heads = getattr(self._hparams, "graphormer_heads", getattr(self._hparams, "gat_heads", 4))
        graphormer_attn_dp = getattr(
            self._hparams,
            "graphormer_attn_dropout",
            getattr(self._hparams, "gat_attn_dropout", getattr(self._hparams, "dropout", 0.3))
        )
        graphormer_dp = getattr(
            self._hparams,
            "graphormer_dropout",
            getattr(self._hparams, "dropout", 0.3)
        )
        graphormer_num_layers = getattr(self._hparams, "graphormer_num_layers", 1)
        graphormer_ffn_mult = getattr(self._hparams, "graphormer_ffn_mult", 4)
        graphormer_spatial_pos_max = getattr(
            self._hparams,
            "graphormer_spatial_pos_max",
            getattr(self._hparams, "max_num_nodes", 32)
        )
        graphormer_degree_max = getattr(
            self._hparams,
            "graphormer_degree_max",
            getattr(self._hparams, "max_num_nodes", 32)
        )

        self.graphormer_encoder = GraphormerNodeEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=graphormer_heads,
            num_layers=graphormer_num_layers,
            dropout=graphormer_dp,
            attn_dropout=graphormer_attn_dp,
            ffn_mult=graphormer_ffn_mult,
            spatial_pos_max=graphormer_spatial_pos_max,
            degree_max=graphormer_degree_max,
        )

        if self.use_branch_b:
            attn_hidden = bb_cfg.get('attn_hidden', 128)
            gate_hidden = bb_cfg.get('gate_hidden', attn_hidden)
            self.mil_head = MILBranchB(hidden_dim, attn_hidden=attn_hidden, gate_hidden=gate_hidden)

        if self.use_coarse_graph:
            self._init_coarse_graph()
            num_nodes = self._hatA_full.size(0)
            self.coarse_gcn_dim = hidden_dim
            self.register_buffer('coarse_node_features', torch.zeros(num_nodes, self.coarse_gcn_dim))
            self.cg_conv1 = nn.Linear(self.coarse_gcn_dim, self.coarse_gcn_dim)
            self.cg_ln1 = nn.LayerNorm(self.coarse_gcn_dim)
            self.cg_conv2 = nn.Linear(self.coarse_gcn_dim, self.coarse_gcn_dim)
            self.cg_ln2 = nn.LayerNorm(self.coarse_gcn_dim)
            self.cg_dropout = nn.Dropout(0.5)

        classifier_input_dim = hidden_dim
        if self.use_coarse_graph:
            classifier_input_dim += self.coarse_gcn_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self._hparams.channel_list[-2]),
            nn.LeakyReLU(negative_slope=float(getattr(self._hparams, 'leaky_relu_alpha', 0.2))),
            nn.Dropout(p=getattr(self._hparams, "dropout", 0.3)),
            nn.Linear(self._hparams.channel_list[-2], self._hparams.channel_list[-1])
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters (Xavier for GCNs)."""
        for m in self.modules():
            if isinstance(m, gcn_layer.GraphConvolution):
                torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, graph_input):
        """
        Forward pass.
        Returns dictionary containing 'ypred' and optional auxiliary outputs.
        """
        x = graph_input[g_key.x]
        adj = graph_input[g_key.adj_mat]
        batch_num_nodes = graph_input[g_key.node_num]
        
        x2 = self.graphormer_encoder(x, adj, batch_num_nodes)

        max_nodes = adj.size(1)
        mask = self.construct_mask(max_nodes, batch_num_nodes)
        if mask is not None:
            x2 = x2 * mask

        aux_out = {}
        h2 = self._masked_mean_pool(x2, mask, batch_num_nodes)

        h4 = None
        if self.use_coarse_graph:
            subgraph_ids = graph_input[g_key.subgraph_id]
            if isinstance(subgraph_ids, torch.Tensor):
                ids = subgraph_ids.long()
            else:
                ids = torch.tensor([int(i) for i in subgraph_ids], dtype=torch.long, device=self._device)
            if ids.device != self._device:
                ids = ids.to(self._device)
            X_full = self.coarse_node_features.detach().clone()
            X_full[ids] = h2
            self.coarse_node_features.data[ids] = h2.detach()
            adj = self._hatA_full
            h_gcn = self.cg_conv1(X_full)
            h_gcn = torch.matmul(adj, h_gcn)
            h_gcn = self.cg_ln1(h_gcn + X_full)
            h_gcn = F.relu(h_gcn)
            h_gcn = self.cg_dropout(h_gcn)
            X_l1 = h_gcn
            h_gcn = self.cg_conv2(X_l1)
            h_gcn = torch.matmul(adj, h_gcn)
            h_gcn = self.cg_ln2(h_gcn + X_l1)
            h_gcn = F.relu(h_gcn)
            h_gcn = self.cg_dropout(h_gcn)
            h4 = h_gcn[ids]

        h3 = None
        if self.use_branch_b:
            h_flat, batch_vec = self._flatten_batch(x2, batch_num_nodes)
            mil_out = self.mil_head(h_flat, batch_vec)
            h3 = mil_out['z_B']
            aux_out['branch_b'] = mil_out

        concat_list = [h3 if self.use_branch_b and h3 is not None else h2]
        if self.use_coarse_graph and h4 is not None:
            concat_list.append(h4)
            
        final_out = torch.cat(concat_list, dim=1)
        ypred = self.classifier(final_out)
                
        self.current_x2 = x2
        if self.use_branch_b:
             return {'ypred_A': ypred, 'branch_b': aux_out.get('branch_b')}
        else:
             return ypred

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _flatten_batch(self, h, batch_num_nodes):
        """Flattens [B, N, D] tensor to [Sum(N), D] and creates batch index vector."""
        if isinstance(batch_num_nodes, torch.Tensor):
            num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
        else:
            num_list = [int(n) for n in batch_num_nodes]
            
        B = h.size(0)
        chunks = [h[i, :num_list[i], :] for i in range(B)]
        h_flat = torch.cat(chunks, dim=0)
        
        batch_vec = torch.cat([
            torch.full((num_list[i],), i, device=self._device, dtype=torch.long)
            for i in range(B)
        ], dim=0)
        
        return h_flat, batch_vec

    def _masked_mean_pool(self, node_embeddings, embedding_mask, batch_num_nodes):
        if embedding_mask is not None:
            node_embeddings = node_embeddings * embedding_mask
            
        if isinstance(batch_num_nodes, torch.Tensor):
            num_list = batch_num_nodes.view(-1).float().to(device=node_embeddings.device)
        else:
            num_list = torch.tensor([float(int(n)) for n in batch_num_nodes], device=node_embeddings.device)
            
        sum_vec = node_embeddings.sum(dim=1)
        denom = torch.clamp(num_list, min=1.0).unsqueeze(1)
        return sum_vec / denom

    def construct_mask(self, max_nodes, batch_num_nodes):
        if isinstance(batch_num_nodes, torch.Tensor):
            num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
        else:
            num_list = [int(n) for n in batch_num_nodes]
            
        packed_masks = [torch.ones(n, device=self._device) for n in num_list]
        batch_size = len(num_list)
        out_tensor = torch.zeros(batch_size, max_nodes, device=self._device)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :num_list[i]] = mask
        return out_tensor.unsqueeze(2)

    # --------------------------------------------------------------------------
    # Coarse Graph Initialization & Computation Logic
    # --------------------------------------------------------------------------

    def _init_coarse_graph(self):
        """Reads processed data and pre-computes coarse graph adjacency and PGNN distances."""
        data_dir = getattr(self._hparams, 'processed_data_dir')
        data_name = self.data_name
        dataset_path = os.path.join(data_dir, f'{data_name}_processed.pkl')
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            
        # Use CPU for heavy matrix operations to save GPU memory
        cpu_device = torch.device('cpu')
        
        G = dataset['original_graph']
        nodelist = list(G.nodes())
        A_np = nx.to_numpy_array(G, nodelist=nodelist, dtype=float)
        # Keep A_full on CPU
        _A_full = torch.tensor(A_np, dtype=torch.float32, device=cpu_device)
        
        S_np = dataset['assignment_matrix']
        # Keep S_full on CPU
        _S_full = torch.tensor(S_np, dtype=torch.float32, device=cpu_device)
        
        # Perform matrix multiplication on CPU
        n_vec = torch.clamp(_S_full.sum(dim=0), min=1.0)
        Ac = torch.matmul(_S_full.transpose(0, 1), torch.matmul(_A_full, _S_full))
        
        # Changed to GCN Style Normalization: E_ij / sqrt(N_i * N_j)
        # Old was Density Normalization: E_ij / (N_i * N_j)
        denom = torch.sqrt(torch.ger(n_vec, n_vec))
        _hatA_full = Ac / denom
        
        # Sparsification (still on CPU)
        top_k = getattr(self._hparams, 'coarse_graph_topk', 0)
        if top_k > 0:
            values, indices = torch.topk(_hatA_full, k=min(top_k, _hatA_full.size(1)), dim=1)
            mask = torch.zeros_like(_hatA_full)
            mask.scatter_(1, indices, 1.0)
            mask = (mask + mask.t() > 0).float()
            # Binarize edges: Set all retained edges to 1.0 (unweighted) per user request
            _hatA_full = mask
        
        # Optimize: Convert to Sparse Tensor to save GPU memory
        # Dense matrix O(N^2) -> Sparse matrix O(E)
        _hatA_full = _hatA_full.to_sparse()
            
        # Move final result to GPU
        self._hatA_full = _hatA_full.to(self._device)
        
        # Clean up CPU tensors
        del _A_full
        del _S_full
        del Ac
        del denom
        import gc
        gc.collect()
