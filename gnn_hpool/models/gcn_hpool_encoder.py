# coding=utf-8

import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F

from gnn_hpool.utils.global_variables import g_key
from gnn_hpool.utils import hparams_lib
from gnn_hpool.models.gcn_hpool_submodel import GcnHpoolSubmodel
from gnn_hpool.layers import gcn_layer


class GcnHpoolEncoder(Module):

  def __init__(self, hparams):
    super(GcnHpoolEncoder, self).__init__()

    self._hparams = hparams_lib.copy_hparams(hparams)
    self.build_graph()
    self.reset_parameters()

    self._device = torch.device(self._hparams.device)

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, gcn_layer.GraphConvolution):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
          torch.nn.init.constant_(m.bias, 0.0)

  def build_graph(self):

    # entry GCN 改为单层（输出维度对齐 channel_list[2]）
    self.entry_conv = gcn_layer.GraphConvolution(
      in_features=self._hparams.channel_list[0],
      out_features=self._hparams.channel_list[2],
      hparams=self._hparams,
    )

    # 子模块保持不变，但入参特征维度改为单层输出（不再 3x 拼接）
    self.gcn_hpool_layer = GcnHpoolSubmodel(
      self._hparams.channel_list[2], self._hparams.channel_list[3], self._hparams.channel_list[4],
      self._hparams.node_list[0], self._hparams.node_list[1], self._hparams.node_list[2],
      self._hparams
    )

    # 预测层输入维度：入口一路 + 子模块一路（不再乘 3）
    input_dim = (
      self._hparams.channel_list[2] +
      2 * self._hparams.channel_list[3] + self._hparams.channel_list[4]
    )
    self.pred_model = torch.nn.Sequential(
      torch.nn.Linear(input_dim, self._hparams.channel_list[-2]),
      torch.nn.ReLU(),
      torch.nn.Linear(self._hparams.channel_list[-2], self._hparams.channel_list[-1])
    )

  def forward(self, graph_input):

    node_feature = graph_input[g_key.x]
    adjacency_mat = graph_input[g_key.adj_mat]
    batch_num_nodes = graph_input[g_key.node_num]

    # input mask
    max_num_nodes = adjacency_mat.size()[1]
    embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)

    # entry embedding gcn（单层输出，不再三份拼接）
    embedding_single = F.relu(self.entry_conv(node_feature, adjacency_mat))
    embedding_single = self.apply_bn(embedding_single)
    embedding_tensor_1 = embedding_single
    if embedding_mask is not None:
        embedding_tensor_1 = embedding_tensor_1 * embedding_mask
    output_1, _ = torch.max(embedding_tensor_1, dim=1)

    # 子模块调用保持原逻辑（多层GCN在子模块中执行）
    output_2, _, _, _ = self.gcn_hpool_layer(
        embedding_tensor_1, node_feature, adjacency_mat, embedding_mask
    )

    output = torch.cat([output_1, output_2], dim=1)
    ypred = self.pred_model(output)
    return ypred

  def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
    out_all = []

    layer_out_1 = F.relu(conv_first(x, adj))
    layer_out_1 = self.apply_bn(layer_out_1)
    out_all.append(layer_out_1)

    layer_out_2 = F.relu(conv_block(layer_out_1, adj))
    layer_out_2 = self.apply_bn(layer_out_2)
    out_all.append(layer_out_2)

    layer_out_3 = conv_last(layer_out_2, adj)
    out_all.append(layer_out_3)
    out_all = torch.cat(out_all, dim=2)
    if embedding_mask is not None:
      out_all = out_all * embedding_mask

    return out_all

  def apply_bn(self, x):
      ''' Batch normalization of 3D tensor x
      '''
      bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self._device)
      return bn_module(x)

  def construct_mask(self, max_nodes, batch_num_nodes):
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
