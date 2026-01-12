import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F
from gnn_hpool.utils.global_variables import g_key
from gnn_hpool.utils import hparams_lib
from gnn_hpool.layers.gat_layer import ResidualGATLayer
from gnn_hpool.layers.gcn_layer import GraphConvolution
import os
import pickle
import networkx as nx

class GATMeanEncoder(Module):
  def __init__(self, hparams):
    super(GATMeanEncoder, self).__init__()
    self._hparams = hparams_lib.copy_hparams(hparams)
    self.build_graph()
    self._device = torch.device(self._hparams.device)
    self.use_coarse = bool(getattr(self._hparams, 'use_coarse_graph', True))
    if self.use_coarse:
      data_dir = getattr(self._hparams, 'processed_data_dir')
      data_name = getattr(self._hparams, 'data_name')
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
      self.coarse_graph_cache = None

  def build_graph(self):
    heads = getattr(self._hparams, "gat_heads", 4)
    attn_dp = getattr(self._hparams, "gat_attn_dropout", getattr(self._hparams, "dropout", 0.5))
    feat_dp = getattr(self._hparams, "gat_feat_dropout", getattr(self._hparams, "dropout", 0.5))
    alpha = getattr(self._hparams, "gat_alpha", 0.2)
    concat = getattr(self._hparams, "gat_concat", True)
    residual = getattr(self._hparams, "gat_residual", True)
    self.entry_conv = ResidualGATLayer(
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
    dp = getattr(self._hparams, "dropout", 0.5)
    self.dropout_entry = torch.nn.Dropout(p=dp)
    self.pred_model = torch.nn.Sequential(
      torch.nn.Linear(self._hparams.channel_list[2], self._hparams.channel_list[-2]),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=dp),
      torch.nn.Linear(self._hparams.channel_list[-2], self._hparams.channel_list[-1])
    )
    if bool(getattr(self._hparams, 'use_coarse_graph', True)):
      self.coarse_gcn1 = GraphConvolution(self._hparams.channel_list[2], self._hparams.channel_list[2], self._hparams)
      self.coarse_gcn2 = GraphConvolution(self._hparams.channel_list[2], self._hparams.channel_list[2], self._hparams)
      self.dropout_coarse = torch.nn.Dropout(p=dp)

  def forward(self, graph_input):
    node_feature = graph_input[g_key.x]
    adjacency_mat = graph_input[g_key.adj_mat]
    batch_num_nodes = graph_input[g_key.node_num]
    subgraph_id_tensor = graph_input[g_key.subgraph_id]
    max_num_nodes = adjacency_mat.size()[1]
    embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
    embedding_single = F.relu(self.entry_conv(node_feature, adjacency_mat))
    embedding_single = self.apply_bn(embedding_single)
    embedding_single = self.dropout_entry(embedding_single)
    if embedding_mask is not None:
      embedding_single = embedding_single * embedding_mask
    if isinstance(batch_num_nodes, torch.Tensor):
      num_list = batch_num_nodes.view(-1).float()
    else:
      num_list = torch.tensor([float(int(n)) for n in batch_num_nodes], device=self._device)
    sum_vec = embedding_single.sum(dim=1)
    denom = torch.clamp(num_list, min=1.0).unsqueeze(1)
    mean_vec = sum_vec / denom
    if self.use_coarse:
      self._compute_coarse_graph(mean_vec, subgraph_id_tensor)
      hatA = self.coarse_graph_cache['hat_adj'].unsqueeze(0)
      H0 = self.coarse_graph_cache['features'].unsqueeze(0)
      H1 = F.relu(self.coarse_gcn1(H0, hatA))
      H1 = self.dropout_coarse(H1)
      H2 = F.relu(self.coarse_gcn2(H1, hatA)).squeeze(0)
      inv_idx = self.coarse_graph_cache['inv_order']
      updated_mean = H2[inv_idx, :]
      ypred = self.pred_model(updated_mean)
    else:
      ypred = self.pred_model(mean_vec)
    return ypred

  def apply_bn(self, x):
    bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self._device)
    return bn_module(x)

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


  def _compute_coarse_graph(self, mean_vec, subgraph_id_tensor):
    ids = [int(i) for i in subgraph_id_tensor.detach().cpu().tolist()] if isinstance(subgraph_id_tensor, torch.Tensor) else [int(i) for i in subgraph_id_tensor]
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    cols_sorted = [ids[i] for i in order]
    batch_idx_order = order
    hatA = self._hatA_full[cols_sorted][:, cols_sorted]
    H0 = mean_vec[batch_idx_order, :]
    inv_order = [0] * len(order)
    for pos, orig_i in enumerate(order):
      inv_order[orig_i] = pos
    inv_idx = torch.tensor(inv_order, dtype=torch.long, device=self._device)
    self.coarse_graph_cache = {
      'subgraph_ids': torch.tensor(cols_sorted, device=self._device),
      'hat_adj': hatA,
      'features': H0,
      'order': torch.tensor(order, dtype=torch.long, device=self._device),
      'inv_order': inv_idx
    }
