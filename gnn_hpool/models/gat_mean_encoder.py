import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F
from gnn_hpool.utils.global_variables import g_key
from gnn_hpool.utils import hparams_lib
from gnn_hpool.layers.gat_layer import ResidualGATLayer

class GATMeanEncoder(Module):
  def __init__(self, hparams):
    super(GATMeanEncoder, self).__init__()
    self._hparams = hparams_lib.copy_hparams(hparams)
    self.build_graph()
    self._device = torch.device(self._hparams.device)

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

  def forward(self, graph_input):
    node_feature = graph_input[g_key.x]
    adjacency_mat = graph_input[g_key.adj_mat]
    batch_num_nodes = graph_input[g_key.node_num]
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
