# coding=utf-8

import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F

from gnn_hpool.utils.global_variables import g_key
from gnn_hpool.utils import hparams_lib
from gnn_hpool.models.gcn_hpool_submodel import GcnHpoolSubmodel
from gnn_hpool.layers import gcn_layer
from gnn_hpool.models.mil_head import MILBranchB
from gnn_hpool.layers.gat_layer import ResidualGATLayer


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

    heads = getattr(self._hparams, "gat_heads", 4)
    attn_dp = getattr(self._hparams, "gat_attn_dropout", getattr(self._hparams, "dropout", 0.5))
    feat_dp = getattr(self._hparams, "gat_feat_dropout", getattr(self._hparams, "dropout", 0.5))
    alpha = getattr(self._hparams, "gat_alpha", 0.2)
    concat = getattr(self._hparams, "gat_concat", True)
    residual = getattr(self._hparams, "gat_residual", True)

    self.entry_conv_A = ResidualGATLayer(
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
    self.dropout_entry_A = torch.nn.Dropout(p=dp)

    self.entry_conv_B = ResidualGATLayer(
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

    bb_cfg = getattr(self._hparams, 'branch_b', None)
    self._use_branch_b = bool(bb_cfg and bb_cfg.get('use', False))
    self._branch_b_only = bool(bb_cfg and bb_cfg.get('only', False))
    extra_dim_B = 0
    if self._use_branch_b:
        node_dim_B = self._hparams.channel_list[2]
        attn_hidden = bb_cfg.get('attn_hidden', 128)
        self.mil_branch_b = MILBranchB(node_dim_B, attn_hidden)
        extra_dim_B = node_dim_B

    pred_input_dim = extra_dim_B if (self._use_branch_b and self._branch_b_only) else (input_dim_A + extra_dim_B)
    self.pred_model = torch.nn.Sequential(
      torch.nn.Linear(pred_input_dim, self._hparams.channel_list[-2]),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=dp),
      torch.nn.Linear(self._hparams.channel_list[-2], self._hparams.channel_list[-1])
    )

  def forward(self, graph_input):

    node_feature = graph_input[g_key.x]
    adjacency_mat = graph_input[g_key.adj_mat]
    batch_num_nodes = graph_input[g_key.node_num]

    # input mask
    max_num_nodes = adjacency_mat.size()[1]
    embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)

    only_branch_b = getattr(self, '_use_branch_b', False) and getattr(self, '_branch_b_only', False)

    if not only_branch_b:
        embedding_single_A = F.relu(self.entry_conv_A(node_feature, adjacency_mat))
        embedding_single_A = self.apply_bn(embedding_single_A)
        embedding_single_A = self.dropout_entry_A(embedding_single_A)
        embedding_tensor_A = embedding_single_A
        if embedding_mask is not None:
            embedding_tensor_A = embedding_tensor_A * embedding_mask
        output_1, _ = torch.max(embedding_tensor_A, dim=1)

        output_2, _, _, node_embed = self.gcn_hpool_layer(
            embedding_tensor_A, node_feature, adjacency_mat, embedding_mask
        )
        output = torch.cat([output_1, output_2], dim=1)

    if getattr(self, '_use_branch_b', False):
        embedding_single_B = F.relu(self.entry_conv_B(node_feature, adjacency_mat))
        embedding_single_B = self.apply_bn(embedding_single_B)
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
        b_out = self.mil_branch_b(h_flat, None, batch_vec)

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

        if getattr(self, '_branch_b_only', False):
            ypred = self.pred_model(b_out['z_B'])
        else:
            ypred = self.pred_model(torch.cat([output, b_out['z_B']], dim=1))
        return {'ypred_A': ypred, 'branch_b': b_out}

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
