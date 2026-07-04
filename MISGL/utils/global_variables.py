# coding=utf-8


class GKey(object):

  def __init__(self):
    self.adj_mat = 'adj_mat'
    self.x = 'x'
    self.y = 'y'
    self.node_num = 'node_num'
    self.orig_graph_idx = 'orig_graph_idx'
    self.subgraph_id = 'subgraph_id'
    self.structural_features = 'structural_features'
    self.border_anchor_sim = 'border_anchor_sim'
    self.border_anchor_walk_features = 'border_anchor_walk_features'
    self.border_external_count = 'border_external_count'


g_key = GKey()
EPS = 1e-30
writer_batch_idx = [0, 3, 6, 9]
