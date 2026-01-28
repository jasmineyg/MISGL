# coding=utf-8

import os
import pickle

import networkx as nx
import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F

from gnn_hpool.utils.global_variables import g_key
from gnn_hpool.utils import hparams_lib
from gnn_hpool.utils.coarse_graph_analyze import analyze_and_export, default_coarsegraph_analyze_out_xlsx
from gnn_hpool.models.gcn_hpool_submodel import GcnHpoolSubmodel
from gnn_hpool.layers import gcn_layer
from gnn_hpool.models.mil_head import MILBranchB
from gnn_hpool.layers.gat_layer import ResidualGATLayer


class GcnHpoolEncoder(Module):
  """GCN/HPool 编码器：支持 Branch-A(Hpool) 与 Branch-B(MIL) 的图级预测。"""

  def __init__(self, hparams, data_name=None):
    """初始化模型、构建网络结构，并按需加载 coarse graph 的缓存数据。"""
    super(GcnHpoolEncoder, self).__init__()

    self._hparams = hparams_lib.copy_hparams(hparams)
    self.data_name = data_name if data_name is not None else getattr(self._hparams, 'data_name', None)
    self._layer_norms = torch.nn.ModuleDict()
    self.build_graph()
    self.reset_parameters()

    self._device = torch.device(self._hparams.device)
    self._coarse_graph_cache = None
    self._coarse_graph_analyze = {'counter': 0, 'seen': set()}
    self._coarse_graph_analyze_full_done = False
    if bool(getattr(self, '_use_coarse_graph', False)):
      self._init_coarse_graph()

  def reset_parameters(self):
    """对 GraphConvolution 层进行参数初始化（Xavier + bias=0）。"""
    for m in self.modules():
      if isinstance(m, gcn_layer.GraphConvolution):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
          torch.nn.init.constant_(m.bias, 0.0)

  def build_graph(self):
    """搭建网络结构：入口 GAT、HPool 子模块、可选 coarse-graph GCN、以及最终 MLP 分类头。"""
    heads = getattr(self._hparams, "gat_heads", 4)
    attn_dp = getattr(self._hparams, "gat_attn_dropout", getattr(self._hparams, "dropout", 0.3))
    feat_dp = getattr(self._hparams, "gat_feat_dropout", getattr(self._hparams, "dropout", 0.3))
    alpha = getattr(self._hparams, "gat_alpha", 0.2)
    concat = getattr(self._hparams, "gat_concat", True)
    residual = getattr(self._hparams, "gat_residual", True)

    self.entry_conv_A_1 = ResidualGATLayer(
      in_dim=self._hparams.channel_list[0],
      out_dim=self._hparams.channel_list[2],
      hparams=self._hparams,
      heads=heads,
      attn_dropout=attn_dp,
      feat_dropout=feat_dp,
      alpha=alpha,
      concat=concat,
      residual=residual
    )
    dp = getattr(self._hparams, "dropout", 0.3)
    self.dropout_entry_A = torch.nn.Dropout(p=dp)

    self.entry_conv_B_1 = ResidualGATLayer(
      in_dim=self._hparams.channel_list[0],
      out_dim=self._hparams.channel_list[2],
      hparams=self._hparams,
      heads=heads,
      attn_dropout=attn_dp,
      feat_dropout=feat_dp,
      alpha=alpha,
      concat=concat,
      residual=residual
    )
    self.dropout_entry_B = torch.nn.Dropout(p=dp)

    self.gcn_hpool_layer = GcnHpoolSubmodel(
      self._hparams.channel_list[2], self._hparams.channel_list[3], self._hparams.channel_list[4],
      self._hparams.node_list[0], self._hparams.node_list[1], self._hparams.node_list[2],
      self._hparams
    )

    input_dim_A = (
      self._hparams.channel_list[2] +
      2 * self._hparams.channel_list[3] + self._hparams.channel_list[4]
    )

    ba_cfg = getattr(self._hparams, 'branch_a', None)
    self._use_branch_a = True if ba_cfg is None else bool(ba_cfg.get('use', True))

    bb_cfg = getattr(self._hparams, 'branch_b', None)
    self._use_branch_b = bool(bb_cfg and bb_cfg.get('use', False))
    extra_dim_B = 0
    if self._use_branch_b:
        node_dim_B = self._hparams.channel_list[2]
        attn_hidden = bb_cfg.get('attn_hidden', 128)
        gate_hidden = bb_cfg.get('gate_hidden', attn_hidden)
        self.mil_branch_b = MILBranchB(node_dim_B, attn_hidden=attn_hidden, gate_hidden=gate_hidden)
        extra_dim_B = node_dim_B

    mean_pool_dim = self._hparams.channel_list[2]
    self._use_coarse_graph = bool(getattr(self._hparams, 'use_coarse_graph', False)) and (not self._use_branch_a) and (not self._use_branch_b)
    if self._use_coarse_graph:
      self.coarse_gcn1 = gcn_layer.GraphConvolution(mean_pool_dim, mean_pool_dim, self._hparams)
      self.coarse_gcn2 = gcn_layer.GraphConvolution(mean_pool_dim, mean_pool_dim, self._hparams)
      self.dropout_coarse = torch.nn.Dropout(p=dp)

    if (not self._use_branch_a) and (not self._use_branch_b):
      pred_input_dim = mean_pool_dim
    elif self._use_branch_a and self._use_branch_b:
      pred_input_dim = input_dim_A + extra_dim_B
    elif self._use_branch_a:
      pred_input_dim = input_dim_A
    else:
      pred_input_dim = extra_dim_B

    self.pred_model = torch.nn.Sequential(
      torch.nn.Linear(pred_input_dim, self._hparams.channel_list[-2]),
      torch.nn.LeakyReLU(negative_slope=float(getattr(self._hparams, 'leaky_relu_alpha', 0.2))),
      torch.nn.Dropout(p=dp),
      torch.nn.Linear(self._hparams.channel_list[-2], self._hparams.channel_list[-1])
    )

  def _masked_mean_pool(self, node_embeddings, embedding_mask, batch_num_nodes):
    """按图做 mean pooling（可选 mask），并用 batch_num_nodes 做归一化。"""
    if embedding_mask is not None:
      node_embeddings = node_embeddings * embedding_mask
    if isinstance(batch_num_nodes, torch.Tensor):
      num_list = batch_num_nodes.view(-1).float().to(device=node_embeddings.device)
    else:
      num_list = torch.tensor([float(int(n)) for n in batch_num_nodes], device=node_embeddings.device)
    sum_vec = node_embeddings.sum(dim=1)
    denom = torch.clamp(num_list, min=1.0).unsqueeze(1)
    return sum_vec / denom

  def _init_coarse_graph(self):
    """从 processed 数据中读取原图与 assignment matrix，预计算 coarse 图的归一化邻接矩阵。"""
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
    
    # 稀疏化逻辑：只保留每个节点权重最大的 top_k 条边
    top_k = getattr(self._hparams, 'coarse_graph_topk', 0)
    if top_k > 0:
        values, indices = torch.topk(self._hatA_full, k=min(top_k, self._hatA_full.size(1)), dim=1)
        mask = torch.zeros_like(self._hatA_full)
        mask.scatter_(1, indices, 1.0)
        # 对称化掩码：只要单向是 Top-K，就保留双向连接 (OR 逻辑)
        mask = (mask + mask.t() > 0).float()
        self._hatA_full = self._hatA_full * mask
        
    self._maybe_analyze_full_coarse_graph()

  def _compute_coarse_graph(self, mean_vec, subgraph_id_tensor):
    """按 subgraph_id 选取/重排 coarse 图子矩阵，并返回用于恢复原顺序的索引。"""
    ids = [int(i) for i in subgraph_id_tensor.detach().cpu().tolist()] if isinstance(subgraph_id_tensor, torch.Tensor) else [int(i) for i in subgraph_id_tensor]
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    cols_sorted = [ids[i] for i in order]
    hatA = self._hatA_full[cols_sorted][:, cols_sorted]
    H0 = mean_vec[order, :]
    inv_order = [0] * len(order)
    for pos, orig_i in enumerate(order):
      inv_order[orig_i] = pos
    inv_idx = torch.tensor(inv_order, dtype=torch.long, device=self._device)
    return hatA, H0, inv_idx

  def _maybe_analyze_full_coarse_graph(self):
    if not bool(getattr(self, '_use_coarse_graph', False)):
      return
    if bool(getattr(self, '_coarse_graph_analyze_full_done', False)):
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

  def forward(self, graph_input):
    """前向推理：支持 A 分支(HPool)与 B 分支(MIL)组合，或走 coarse-graph mean pooling 路径。"""

    node_feature = graph_input[g_key.x]
    adjacency_mat = graph_input[g_key.adj_mat]
    batch_num_nodes = graph_input[g_key.node_num]

    # input mask
    max_num_nodes = adjacency_mat.size()[1]
    embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)

    use_a = bool(getattr(self, '_use_branch_a', True))
    use_b = bool(getattr(self, '_use_branch_b', False))

    output = None
    if use_a:
        embedding_single_A = F.relu(self.entry_conv_A_1(node_feature, adjacency_mat))
        embedding_single_A = self.apply_ln(embedding_single_A)
        embedding_single_A = self.dropout_entry_A(embedding_single_A)
        embedding_tensor_A = embedding_single_A
        if embedding_mask is not None:
            embedding_tensor_A = embedding_tensor_A * embedding_mask
        output_1, _ = torch.max(embedding_tensor_A, dim=1)

        output_2, _, _, _ = self.gcn_hpool_layer(
            embedding_tensor_A, node_feature, adjacency_mat, embedding_mask
        )
        output = torch.cat([output_1, output_2], dim=1)

    if (not use_a) and (not use_b):
        embedding_single = F.relu(self.entry_conv_A_1(node_feature, adjacency_mat))
        embedding_single = self.apply_ln(embedding_single)
        embedding_single = self.dropout_entry_A(embedding_single)
        mean_vec = self._masked_mean_pool(embedding_single, embedding_mask, batch_num_nodes)
        if bool(getattr(self, '_use_coarse_graph', False)):
            subgraph_id_tensor = graph_input[g_key.subgraph_id]
            hatA, H0, inv_idx = self._compute_coarse_graph(mean_vec, subgraph_id_tensor)
            # self._maybe_analyze_coarse_graph(hatA, subgraph_id_tensor)
            hatA = hatA.unsqueeze(0)
            H0 = H0.unsqueeze(0)
            H1 = F.relu(self.coarse_gcn1(H0, hatA))
            H1 = self.apply_ln(H1)
            H1 = self.dropout_coarse(H1)
            H2 = F.relu(self.coarse_gcn2(H1, hatA))
            H2 = self.apply_ln(H2) + H0  # 残差连接
            H2 = H2.squeeze(0)
            updated_H2 = H2[inv_idx, :]
            ypred = self.pred_model(updated_H2)
        else:
            ypred = self.pred_model(mean_vec)
        return ypred

    if use_b:
        embedding_single_B = F.relu(self.entry_conv_B_1(node_feature, adjacency_mat))
        embedding_single_B = self.apply_ln(embedding_single_B)
        embedding_single_B = self.dropout_entry_B(embedding_single_B)
        embedding_tensor_B = embedding_single_B
        if embedding_mask is not None:
            embedding_tensor_B = embedding_tensor_B * embedding_mask

        B = embedding_tensor_B.size(0)
        if isinstance(batch_num_nodes, torch.Tensor):
            num_list = [int(n) for n in batch_num_nodes.detach().cpu().tolist()]
        else:
            num_list = [int(n) for n in batch_num_nodes]
        chunks = [embedding_tensor_B[i, :num_list[i], :] for i in range(B)]
        h_flat = torch.cat(chunks, dim=0)
        batch_vec = torch.cat([
            torch.full((num_list[i],), i, device=self._device, dtype=torch.long)
            for i in range(B)
        ], dim=0)
        b_out = self.mil_branch_b(h_flat, batch_vec)

        M = embedding_tensor_B.size(1)
        if embedding_mask is not None:
            mask_valid = embedding_mask.squeeze(2).bool()  # [B,M]
        else:
            mask_valid = torch.zeros(B, M, dtype=torch.bool, device=self._device)
            for i in range(B):
                mask_valid[i, :num_list[i]] = True

        a = b_out['a']  # [sum_i n_i]
        a_pad = torch.zeros(B, M, device=self._device)
        for i in range(B):
            idx_i = (batch_vec == i).nonzero(as_tuple=True)[0]
            n_i = num_list[i]
            if idx_i.numel() > 0 and n_i > 0:
                a_pad[i, :n_i] = a[idx_i][:n_i]
        a_pad = torch.clamp(a_pad, min=1e-6, max=1.0 - 1e-6)
        b_out['a_pad'] = a_pad
        b_out['mask_valid'] = mask_valid

        # [Debug Helper] 将特征转为 numpy 方便调试查看数值
        # if 'z_B' in b_out:
        #     b_out['z_B_np'] = b_out['z_B'].detach().cpu().numpy()
        # if 'output' in locals() and isinstance(output, torch.Tensor):
        #     b_out['feat_A_np'] = output.detach().cpu().numpy()

        feat_in = b_out['z_B'] if not use_a else torch.cat([output, b_out['z_B']], dim=1)
        if isinstance(feat_in, torch.Tensor) and feat_in.size(1) != b_out['z_B'].size(1):
            b_out['feat_cat_np'] = feat_in.detach().cpu().numpy()
        ypred = self.pred_model(feat_in)
        return {'ypred_A': ypred, 'branch_b': b_out}

    ypred = self.pred_model(output)
    return ypred

  def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
    """三层 GCN 前向：拼接每一层输出作为节点级 embedding（可选 mask）。"""
    out_all = []

    layer_out_1 = F.relu(conv_first(x, adj))
    layer_out_1 = self.apply_ln(layer_out_1)
    out_all.append(layer_out_1)

    layer_out_2 = F.relu(conv_block(layer_out_1, adj))
    layer_out_2 = self.apply_ln(layer_out_2)
    out_all.append(layer_out_2)

    layer_out_3 = conv_last(layer_out_2, adj)
    out_all.append(layer_out_3)
    out_all = torch.cat(out_all, dim=2)
    if embedding_mask is not None:
      out_all = out_all * embedding_mask

    return out_all

  def apply_ln(self, x):
      """按特征维缓存 LayerNorm，并保证与输入张量在同一 device。"""
      dim = int(x.size(-1))
      key = str(dim)
      if key in self._layer_norms:
          ln = self._layer_norms[key]
      else:
          ln = torch.nn.LayerNorm(dim, elementwise_affine=True)
          self._layer_norms[key] = ln
      if ln.weight.device != x.device:
          ln = ln.to(device=x.device)
          self._layer_norms[key] = ln
      return ln(x)

  def construct_mask(self, max_nodes, batch_num_nodes):
      """构建 [B, max_nodes, 1] 的节点有效性 mask（每个图前 num_nodes 为 1）。"""
      # For each num_nodes in batch_num_nodes, the first num_nodes entries are 1's.
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
