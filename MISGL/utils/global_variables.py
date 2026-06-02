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
    self.coarse_node_id = 'coarse_node_id'
    self.coarse_node_num = 'coarse_node_num'
    self.coarse_neighbor_index = 'coarse_neighbor_index'
    self.coarse_neighbor_weight = 'coarse_neighbor_weight'


g_key = GKey()
EPS = 1e-30
writer_batch_idx = [0, 3, 6, 9]
