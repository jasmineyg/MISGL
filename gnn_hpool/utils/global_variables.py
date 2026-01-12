# coding=utf-8


class GKey(object):

  def __init__(self):
    self.adj_mat = 'adj_mat'
    self.x = 'x'
    self.y = 'y'
    self.node_num = 'node_num'
    self.orig_graph_idx = 'orig_graph_idx'
    self.subgraph_id = 'subgraph_id'


g_key = GKey()
EPS = 1e-30
writer_batch_idx = [0, 3, 6, 9]
