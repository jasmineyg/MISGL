# coding=utf-8

import os
import pickle
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_hpool.utils.global_variables import g_key
from gnn_hpool.utils import hparams_lib
from gnn_hpool.utils.coarse_graph_analyze import analyze_and_export, default_coarsegraph_analyze_out_xlsx
from gnn_hpool.layers import gcn_layer
from gnn_hpool.models.mil_head import MILBranchB
from gnn_hpool.layers.gin_layer import GINLayer
from gnn_hpool.layers.gat_layer import ResidualGATLayer
from gnn_hpool.utils.pgnn_precompute import build_pgnn_inputs
from pgnn_position_head import PositionConcatClassifier


class GcnHpoolEncoder(nn.Module):
    """
    GCN/HPool Encoder Refactored.
    
    Logic Flow:
    1. Common: GAT Layer -> h1
    2. Case 1 (Branch B=False, Coarse=False): h1 -> Mean Pooling -> Classifier
    3. Case 2 (Branch B=True,  Coarse=False): h1 -> MIL Head -> h2 -> Classifier
    4. Case 3 (Branch B=True,  Coarse=True ): h1 -> MIL Head -> h2 -> Coarse Graph GCNs -> h3 -> Classifier
    
    Note: Branch A is currently disabled/removed for clarity.
    """

    def __init__(self, hparams, data_name=None):
        super(GcnHpoolEncoder, self).__init__()

        self._hparams = hparams_lib.copy_hparams(hparams)
        self.data_name = data_name if data_name is not None else getattr(self._hparams, 'data_name', None)
        self._device = torch.device(self._hparams.device)
        self._layer_norms = nn.ModuleDict()

        # --- Configuration ---
        bb_cfg = getattr(self._hparams, 'branch_b', None)
        self.use_branch_b = bool(bb_cfg and bb_cfg.get('use', False))
        self.use_coarse_graph = bool(getattr(self._hparams, 'use_coarse_graph', False))
        
        # Dimensions
        in_dim = self._hparams.channel_list[0]
        hidden_dim = self._hparams.channel_list[1] # Main hidden dimension (h1, h2, h3)
        
        # --- 1. Backbone (Proj + GAT) ---
        proj_dim = getattr(self._hparams, 'feat_proj_dim', 1024)
        self.feat_proj = nn.Linear(in_dim, proj_dim)

        gat_heads = getattr(self._hparams, "gat_heads", 4)
        gat_attn_dp = getattr(self._hparams, "gat_attn_dropout", getattr(self._hparams, "dropout", 0.3))
        gat_feat_dp = getattr(self._hparams, "gat_feat_dropout", getattr(self._hparams, "dropout", 0.3))
        
        self.gat_layer = ResidualGATLayer(
            in_dim=proj_dim,
            out_dim=hidden_dim,
            hparams=self._hparams,
            heads=gat_heads,
            attn_dropout=gat_attn_dp,
            feat_dropout=gat_feat_dp,
            alpha=getattr(self._hparams, "gat_alpha", 0.2),
            concat=getattr(self._hparams, "gat_concat", True),
            residual=getattr(self._hparams, "gat_residual", True)
        )
        self.dropout_gat = nn.Dropout(p=getattr(self._hparams, "dropout", 0.3))

        # self.gin_layer = GINLayer(
        #     in_dim=proj_dim,
        #     out_dim=hidden_dim,
        #     eps=getattr(self._hparams, "gin_eps", 0.0),
        #     train_eps=getattr(self._hparams, "gin_train_eps", True)
        # )
        # self.dropout_gin = nn.Dropout(p=getattr(self._hparams, "dropout", 0.3))

        # --- 2. Branch B (MIL Head) ---
        if self.use_branch_b:
            attn_hidden = bb_cfg.get('attn_hidden', 128)
            gate_hidden = bb_cfg.get('gate_hidden', attn_hidden)
            self.mil_head = MILBranchB(hidden_dim, attn_hidden=attn_hidden, gate_hidden=gate_hidden)

        # --- 3. Coarse Graph ---
        self._coarse_graph_analyze = {'counter': 0, 'seen': set()}
        self._coarse_graph_analyze_full_done = False
        
        if self.use_coarse_graph:
            self._init_coarse_graph()
            self.coarse_gcn1 = gcn_layer.GraphConvolution(hidden_dim, hidden_dim, self._hparams)
            self.coarse_gcn2 = gcn_layer.GraphConvolution(hidden_dim, hidden_dim, self._hparams)
            self.dropout_coarse = nn.Dropout(p=getattr(self._hparams, "dropout", 0.3))

        # --- 4. Classifier ---
        # Input dimension is consistently 'hidden_dim' across all flows
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, self._hparams.channel_list[-2]),
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
        
        # --- Step 1: Backbone (h1) ---
        x_proj = self.feat_proj(x)
        h1 = F.relu(self.gat_layer(x_proj, adj))
        h1 = self.apply_ln(h1)
        h1 = self.dropout_gat(h1)

        # h1 = F.relu(self.gin_layer(x, adj))
        # h1 = self.apply_ln(h1)
        # h1 = self.dropout_gin(h1)
        
        # Prepare Mask [B, N, 1]
        max_nodes = adj.size(1)
        mask = self.construct_mask(max_nodes, batch_num_nodes)
        if mask is not None:
            h1 = h1 * mask

        aux_out = {} # Store auxiliary outputs (like attention weights)

        # --- Logic Branching ---
        
        if not self.use_branch_b:
            # Case 1: Branch B Closed -> Mean Pooling
            h_out = self._masked_mean_pool(h1, mask, batch_num_nodes)
            # Note: Coarse graph is ignored if Branch B is closed, based on user requirements.
            
        else:
            # Branch B Open
            # Prepare flattened input for MIL Head
            h_flat, batch_vec, num_list = self._flatten_batch(h1, batch_num_nodes)
            
            # Run MIL Head
            mil_out = self.mil_head(h_flat, batch_vec)
            h2 = mil_out['z_B'] # [B, hidden_dim]
            aux_out['branch_b'] = mil_out
            
            if not self.use_coarse_graph:
                # Case 2: Branch B Open, Coarse Closed -> Use h2
                h_out = h2
            else:
                # Case 3: Branch B Open, Coarse Open -> h2 -> Coarse GCN -> h3
                subgraph_ids = graph_input[g_key.subgraph_id]
                h3 = self._run_coarse_graph_layers(h2, subgraph_ids)
                h_out = h3

        # --- Step 4: Classifier ---
        ypred = self.classifier(h_out)
        
        # Return format consistent with expectations (can return dict or just ypred)
        # Previous implementation returned dict if Branch B was used.
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
        
        return h_flat, batch_vec, num_list

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

    def _run_coarse_graph_layers(self, h2, subgraph_ids):
        """Calculates Coarse Graph and runs 2-layer GCN."""
        # h2 serves as H0 for Coarse Graph
        hatA, H0, inv_idx = self._compute_coarse_graph(h2, subgraph_ids)
        
        hatA = hatA.unsqueeze(0) # [1, B, B]
        H0 = H0.unsqueeze(0)     # [1, B, D]
        
        # Layer 1
        H1 = F.relu(self.coarse_gcn1(H0, hatA))
        H1 = self.apply_ln(H1)
        H1 = self.dropout_coarse(H1)
        
        # Layer 2 (with Residual to H0)
        H2 = F.relu(self.coarse_gcn2(H1, hatA))
        H2 = self.apply_ln(H2) + H0
        
        # Restore order
        H2 = H2.squeeze(0)
        h3 = H2[inv_idx, :]
        return h3

    def apply_ln(self, x):
        dim = int(x.size(-1))
        key = str(dim)
        if key not in self._layer_norms:
            self._layer_norms[key] = torch.nn.LayerNorm(dim, elementwise_affine=True)
        
        ln = self._layer_norms[key]
        if ln.weight.device != x.device:
            ln = ln.to(device=x.device)
            self._layer_norms[key] = ln
        return ln(x)

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
    # Coarse Graph Initialization & Computation Logic (Preserved)
    # --------------------------------------------------------------------------

    def _init_coarse_graph(self):
        """Reads processed data and pre-computes coarse graph adjacency."""
        data_dir = getattr(self._hparams, 'processed_data_dir')
        data_name = self.data_name
        dataset_path = os.path.join(data_dir, f'{data_name}_processed.pkl')
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            
        G = dataset['original_graph']
        nodelist = list(G.nodes())
        A_np = nx.to_numpy_array(G, nodelist=nodelist, dtype=float)
        self._A_full = torch.tensor(A_np, dtype=torch.float32, device=self._device)
        
        S_np = dataset['assignment_matrix']
        self._S_full = torch.tensor(S_np, dtype=torch.float32, device=self._device)
        
        n_vec = torch.clamp(self._S_full.sum(dim=0), min=1.0)
        Ac = torch.matmul(self._S_full.transpose(0, 1), torch.matmul(self._A_full, self._S_full))
        denom = torch.ger(n_vec, n_vec)
        self._hatA_full = Ac / denom
        
        # Sparsification
        top_k = getattr(self._hparams, 'coarse_graph_topk', 0)
        if top_k > 0:
            values, indices = torch.topk(self._hatA_full, k=min(top_k, self._hatA_full.size(1)), dim=1)
            mask = torch.zeros_like(self._hatA_full)
            mask.scatter_(1, indices, 1.0)
            mask = (mask + mask.t() > 0).float()
            self._hatA_full = self._hatA_full * mask
            
        self._maybe_analyze_full_coarse_graph()

    def _compute_coarse_graph(self, mean_vec, subgraph_id_tensor):
        """Reorders pre-computed HatA based on current batch's subgraph IDs."""
        ids = [int(i) for i in subgraph_id_tensor.detach().cpu().tolist()] if isinstance(subgraph_id_tensor, torch.Tensor) else [int(i) for i in subgraph_id_tensor]
        
        # Sort to match pre-computed global order
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        cols_sorted = [ids[i] for i in order]
        
        hatA = self._hatA_full[cols_sorted][:, cols_sorted]
        H0 = mean_vec[order, :]
        
        # Inverse permutation to restore batch order
        inv_order = [0] * len(order)
        for pos, orig_i in enumerate(order):
            inv_order[orig_i] = pos
        inv_idx = torch.tensor(inv_order, dtype=torch.long, device=self._device)
        
        return hatA, H0, inv_idx

    def _maybe_analyze_full_coarse_graph(self):
        if self._coarse_graph_analyze_full_done:
            return
        mode = str(getattr(self._hparams, 'coarse_graph_analyze_mode', 'full')).strip().lower()
        if mode not in ('full', 'both'):
            return
            
        threshold = float(getattr(self._hparams, 'coarse_graph_analyze_threshold', 0.0))
        out_xlsx = default_coarsegraph_analyze_out_xlsx(self._hparams, data_name=self.data_name)
        
        try:
            n = int(self._hatA_full.size(0))
        except Exception:
            return
            
        node_ids = list(range(n))
        data_name = str(self.data_name or 'data').strip()
        graph_name = f'FULL_{data_name}'
        
        saved_path = analyze_and_export(
            graphs=[{'graph_name': graph_name, 'hatA': self._hatA_full, 'node_ids': node_ids, 'print_summary': True}],
            out_xlsx=out_xlsx,
            threshold=threshold,
        )
        print(f'[CoarseGraph] Export xlsx => {saved_path}')
        self._coarse_graph_analyze_full_done = True