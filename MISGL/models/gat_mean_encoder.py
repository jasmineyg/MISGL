import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F
from MISGL.utils.global_variables import g_key
from MISGL.utils import hparams_lib
from MISGL.utils.coarse_graph_analyze import analyze_and_export, default_coarsegraph_analyze_out_xlsx
from MISGL.layers.gat_layer import ResidualGATLayer
from MISGL.layers.gcn_layer import GraphConvolution
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
    self._coarse_graph_analyze = {'counter': 0, 'seen': set()}
    self._coarse_graph_analyze_full_done = False
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
      self._maybe_analyze_full_coarse_graph()

  def _maybe_analyze_full_coarse_graph(self):
    if not bool(getattr(self, 'use_coarse', False)):
      return
    if bool(getattr(self, '_coarse_graph_analyze_full_done', False)):
      return
    mode = str(getattr(self._hparams, 'coarse_graph_analyze_mode', 'full')).strip().lower()
    if mode not in ('full', 'both'):
      return
    threshold = float(getattr(self._hparams, 'coarse_graph_analyze_threshold', 0.0))
    out_xlsx = default_coarsegraph_analyze_out_xlsx(self._hparams)
    n = int(self._hatA_full.size(0))
    node_ids = list(range(n))
    data_name = str(getattr(self._hparams, 'data_name', 'data')).strip() or 'data'
    graph_name = f'FULL_{data_name}'
    saved_path = analyze_and_export(
      graphs=[{'graph_name': graph_name, 'hatA': self._hatA_full, 'node_ids': node_ids, 'print_summary': True}],
      out_xlsx=out_xlsx,
      threshold=threshold,
    )
    print(f'[CoarseGraph] Export xlsx => {saved_path}')
    self._coarse_graph_analyze_full_done = True

  def build_graph(self):
    heads = getattr(self._hparams, "gat_heads", 4)
    attn_dp = getattr(self._hparams, "gat_attn_dropout", getattr(self._hparams, "dropout", 0.5))
    feat_dp = getattr(self._hparams, "gat_feat_dropout", getattr(self._hparams, "dropout", 0.5))
    alpha = getattr(self._hparams, "gat_alpha", 0.2)
    concat = getattr(self._hparams, "gat_concat", True)
    residual = getattr(self._hparams, "gat_residual", True)
    self.entry_conv1 = ResidualGATLayer(
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
    self.entry_conv2 = ResidualGATLayer(
      in_dim=self._hparams.channel_list[2],
      out_dim=self._hparams.channel_list[2],
      hparams=self._hparams,
      heads=heads,
      attn_dropout=attn_dp,
      feat_dropout=feat_dp,
      alpha=alpha,
      concat=concat,
      residual=residual
    )
    self.bn1 = torch.nn.BatchNorm1d(self._hparams.channel_list[2])
    self.bn2 = torch.nn.BatchNorm1d(self._hparams.channel_list[2])
    dp = getattr(self._hparams, "dropout", 0.5)
    self.dropout_entry = torch.nn.Dropout(p=dp)
    self.pred_model = torch.nn.Sequential(
      torch.nn.Linear(self._hparams.channel_list[2], self._hparams.channel_list[-2]),
      torch.nn.LeakyReLU(negative_slope=float(getattr(self._hparams, 'leaky_relu_alpha', 0.2))),
      torch.nn.Dropout(p=dp),
      torch.nn.Linear(self._hparams.channel_list[-2], self._hparams.channel_list[-1])
    )
    if bool(getattr(self._hparams, 'use_coarse_graph', True)):
      self.coarse_gcn1 = GraphConvolution(self._hparams.channel_list[2], self._hparams.channel_list[2], self._hparams)
      self.coarse_gcn2 = GraphConvolution(self._hparams.channel_list[2], self._hparams.channel_list[2], self._hparams)
      self.dropout_coarse = torch.nn.Dropout(p=dp)

  def forward(self, graph_input):
    ypred, _ = self._encode(graph_input, return_embeddings=False)
    return ypred

  def forward_with_embeddings(self, graph_input):
    return self._encode(graph_input, return_embeddings=True)

  def _encode(self, graph_input, return_embeddings=False):
    node_feature = graph_input[g_key.x]
    adjacency_mat = graph_input[g_key.adj_mat]
    batch_num_nodes = graph_input[g_key.node_num]
    subgraph_id_tensor = graph_input[g_key.subgraph_id]
    max_num_nodes = adjacency_mat.size()[1]
    embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)

    h = self.entry_conv1(node_feature, adjacency_mat)
    h = self.apply_bn(h, mask=embedding_mask, bn_module=self.bn1)
    h = F.relu(h)
    h = self.dropout_entry(h)

    h = self.entry_conv2(h, adjacency_mat)
    h = self.apply_bn(h, mask=embedding_mask, bn_module=self.bn2)
    h = F.relu(h)
    h = self.dropout_entry(h)

    embedding_single = h
    if embedding_mask is not None:
      embedding_single = embedding_single * embedding_mask
    if isinstance(batch_num_nodes, torch.Tensor):
      num_list = batch_num_nodes.view(-1).float()
    else:
      num_list = torch.tensor([float(int(n)) for n in batch_num_nodes], device=self._device)
    sum_vec = embedding_single.sum(dim=1)
    denom = torch.clamp(num_list, min=1.0).unsqueeze(1)
    mean_vec = sum_vec / denom

    embeddings = None
    if self.use_coarse:
      self._compute_coarse_graph(mean_vec, subgraph_id_tensor)
      mode = str(getattr(self._hparams, 'coarse_graph_analyze_mode', 'full')).strip().lower()
      state = getattr(self, '_coarse_graph_analyze', {'counter': 0, 'seen': set()})
      max_graphs = getattr(self._hparams, 'coarse_graph_analyze_max_graphs', None)
      threshold = float(getattr(self._hparams, 'coarse_graph_analyze_threshold', 0.0))
      node_ids = self.coarse_graph_cache['subgraph_ids'].detach().cpu().tolist()
      sig = tuple(int(x) for x in node_ids)
      seen = state.get('seen', set())
      if mode in ('batch', 'both') and sig not in seen:
        if max_graphs is not None and int(state.get('counter', 0)) >= int(max_graphs):
          pass
        else:
          seen.add(sig)
          state['seen'] = seen
          state['counter'] = int(state.get('counter', 0)) + 1
          k = int(state['counter'])
          if len(node_ids) > 0:
            graph_name = f'coarse_{k}_B{len(node_ids)}_{int(node_ids[0])}_{int(node_ids[-1])}'
          else:
            graph_name = f'coarse_{k}_B0'
          out_xlsx = default_coarsegraph_analyze_out_xlsx(self._hparams)
          analyze_and_export(
            graphs=[{'graph_name': graph_name, 'hatA': self.coarse_graph_cache['hat_adj'], 'node_ids': [int(x) for x in node_ids], 'print_summary': True}],
            out_xlsx=out_xlsx,
            threshold=threshold,
          )
      self._coarse_graph_analyze = state
      hatA = self.coarse_graph_cache['hat_adj'].unsqueeze(0)
      H0 = self.coarse_graph_cache['features'].unsqueeze(0)
      H1 = F.relu(self.coarse_gcn1(H0, hatA))
      H1 = self.dropout_coarse(H1)
      H2 = F.relu(self.coarse_gcn2(H1, hatA)).squeeze(0)
      inv_idx = self.coarse_graph_cache['inv_order']
      updated_H1 = H1.squeeze(0)[inv_idx, :]
      updated_H2 = H2[inv_idx, :]
      ypred = self.pred_model(updated_H2)
      if return_embeddings:
        embeddings = {
          'h': h,
          'mean_vec': mean_vec,
          'graph_emb_H1': updated_H1,
          'graph_emb_H2': updated_H2,
          'graph_emb': updated_H2,
        }
    else:
      ypred = self.pred_model(mean_vec)
      if return_embeddings:
        embeddings = {
          'h': h,
          'mean_vec': mean_vec,
          'graph_emb': mean_vec,
        }

    return ypred, embeddings

  def apply_bn(self, x, mask=None, bn_module=None):
    bn_layer = bn_module if bn_module is not None else self.bn1

    if mask is None:
        x = x.transpose(1, 2)
        x = bn_layer(x)
        x = x.transpose(1, 2)
        return x
    
    # x: [B, N, C], mask: [B, N, 1]
    mask_bool = mask.squeeze(-1).bool()
    valid_x = x[mask_bool] # [Total_Valid, C]
    
    if valid_x.size(0) > 0:
        valid_x = bn_layer(valid_x)
        
    out = torch.zeros_like(x)
    out[mask_bool] = valid_x
    return out

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
