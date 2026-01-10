# coding=utf-8

import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import random
import logging
import pickle
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader

from gnn_hpool.utils import hparams_lib
from gnn_hpool.utils.global_variables import *


# follow a discussion here: https://github.com/RexYing/diffpool/issues/17

class GraphDataset(Dataset):

  def __init__(self, hparams, graph_list):
    self._hparams = hparams_lib.copy_hparams(hparams)
    self._device = torch.device(self._hparams.device)
    self.graph_list = []
    self.processed_graph_list = self.preprocess_graph(graph_list)

  def preprocess_graph(self, graph_list):
    processed_graph_list = []
    for graph in graph_list:
      graph_tmp_dict = {}
      # 使用固定节点顺序以保证邻接矩阵与特征对齐
      nodelist = list(graph.nodes())
      adj = nx.to_numpy_array(graph, nodelist=nodelist, dtype=np.float32)
      feature_dim = self._hparams.channel_list[0]
      node_tmp_feature = np.zeros((self._hparams.max_num_nodes, feature_dim), dtype=np.float32)
      for idx, node_id in enumerate(nodelist):
        node_feat = graph.nodes[node_id].get('features')
        if node_feat is None:
          raise ValueError('节点缺少 features 属性，无法构建输入特征')
        node_tmp_feature[idx, :feature_dim] = np.asarray(node_feat, dtype=np.float32)
      num_nodes = adj.shape[0]
      graph_tmp_dict[g_key.x] = torch.tensor(node_tmp_feature, dtype=torch.float32).to(self._device)
      graph_tmp_dict[g_key.y] = torch.tensor(int(graph.graph.get('label', 0)), dtype=torch.long).to(self._device)
      graph_tmp_dict[g_key.node_num] = torch.tensor(num_nodes, dtype=torch.int16).to(self._device)
      graph_tmp_dict[g_key.adj_mat] = torch.zeros(self._hparams.max_num_nodes, self._hparams.max_num_nodes).to(self._device)
      graph_tmp_dict[g_key.adj_mat][:num_nodes, :num_nodes] = torch.tensor(adj, dtype=torch.float32).to(self._device)
      orig_idx = graph.graph.get('orig_idx', -1)
      try:
        orig_idx = int(orig_idx)
      except (TypeError, ValueError):
        orig_idx = -1
      graph_tmp_dict[g_key.orig_graph_idx] = torch.tensor(orig_idx, dtype=torch.long).to(self._device)
      processed_graph_list.append(graph_tmp_dict)
    return processed_graph_list

  def __len__(self):
    return len(self.processed_graph_list)

  def __getitem__(self, idx):
    return self.processed_graph_list[idx]


class GraphDataLoaderWrapper(object):

  def __init__(self, hparams):
    self._hparams = hparams_lib.copy_hparams(hparams)

    processed_data_dir = getattr(self._hparams, 'processed_data_dir', '/data/yg/Subgraph-MIL/Data/processed_data')
    data_name = getattr(self._hparams, 'data_name', None)
    if not data_name:
      raise ValueError('缺少 data_name 参数，请在配置中设置 data_name')
    dataset_path = os.path.join(processed_data_dir, f'{data_name}_processed.pkl')

    use_synth = bool(getattr(self._hparams, 'synthetic', False))
    if (not use_synth) and os.path.exists(dataset_path):
      with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    else:
      rng = np.random.RandomState(getattr(self._hparams, 'cv_seed', 1024))
      num_graphs = int(getattr(self._hparams, 'synthetic_num_graphs', 200))
      max_nodes = int(getattr(self._hparams, 'max_num_nodes', 20))
      feat_dim = int(self._hparams.channel_list[0])
      graphs = []
      for i in range(num_graphs):
        n = rng.randint(5, max_nodes)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u in range(n):
          for v in range(u + 1, n):
            if rng.rand() < 0.1:
              G.add_edge(u, v)
        for u in G.nodes():
          vec = rng.randn(feat_dim).astype(np.float32)
          G.nodes[u]['features'] = vec
        label = int(rng.rand() < 0.5)
        G.graph['label'] = label
        G.graph['group_id'] = i // max(1, (num_graphs // 10))
        graphs.append(G)
      split = int(num_graphs * 0.8)
      dataset = {
        'subgraph_structures': graphs,
        'train_test_split': {
          'train_indices': list(range(split)),
          'test_indices': list(range(split, num_graphs))
        },
        'feature_dimension': feat_dim,
        'dataset_metadata': {
          'feature_dim': feat_dim,
          'max_num_nodes': max_nodes
        }
      }

    subgraphs = dataset['subgraph_structures']
    train_indices = dataset['train_test_split']['train_indices']
    test_indices = dataset['train_test_split']['test_indices']

    feature_dim = int(dataset.get('feature_dimension', dataset['dataset_metadata']['feature_dim']))
    self._hparams.channel_list[0] = feature_dim
    max_num_nodes = int(dataset['dataset_metadata'].get('max_num_nodes', max(len(g.nodes()) for g in subgraphs)))
    self._hparams.max_num_nodes = max_num_nodes

    self.train_graphs = [subgraphs[i] for i in train_indices]
    self.test_graphs = [subgraphs[i] for i in test_indices]

    self._subgraphs = subgraphs
    self._dataset_raw = dataset
    self.all_indices = list(train_indices) + list(test_indices)
    self.all_graphs = []
    for orig_idx in self.all_indices:
      graph = subgraphs[orig_idx]
      graph.graph['orig_idx'] = int(orig_idx)
      self.all_graphs.append(graph)
    self.all_labels = np.array([int(g.graph.get('label', 0)) for g in self.all_graphs])
    self.all_groups = self._resolve_group_ids(dataset, subgraphs, self.all_indices)

    # 使用配置文件中的fold_num（原K折，不再用于Holdout，但保留）
    self.fold_num = getattr(self._hparams, 'fold_num', 5)
    self.train_count = len(self.train_graphs)
    self.val_size = max(1, self.train_count // self.fold_num)

    # 预生成分层K折索引（不重叠、类别均衡）
    self._train_labels = np.array([int(g.graph.get('label', 0)) for g in self.train_graphs])
    skf = StratifiedKFold(
        n_splits=self.fold_num,
        shuffle=True,
        random_state=getattr(self._hparams, 'cv_seed', 1024)
    )
    self.folds = [(tr_idx, val_idx) for tr_idx, val_idx in skf.split(np.arange(self.train_count), self._train_labels)]

  def get_loader(self, val_idx, inner_val_frac=None):
    # 分层K折：严格不重叠
    train_idx, val_idx_arr = self.folds[val_idx]
    train_graphs = [self.train_graphs[i] for i in train_idx]
    val_graphs = [self.train_graphs[i] for i in val_idx_arr]
  
    logging.info('\n * the length of training sets is {}; \n * the length of validation sets is {}'
                 .format(len(train_graphs), len(val_graphs)))
  
    # 可选：在训练集内再切 inner-val（分层）
    inner_loader = None
    if inner_val_frac is None:
        inner_val_frac = getattr(self._hparams, 'inner_val_frac', 0.1)
    if inner_val_frac and inner_val_frac > 0.0:
        labels_tr = np.array([int(g.graph.get('label', 0)) for g in train_graphs])
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=inner_val_frac,
            random_state=getattr(self._hparams, 'cv_seed', 1024)
        )
        main_idx, inner_idx = next(sss.split(np.arange(len(train_graphs)), labels_tr))
        inner_graphs = [train_graphs[i] for i in inner_idx]
        train_graphs = [train_graphs[i] for i in main_idx]
        inner_set = GraphDataset(self._hparams, inner_graphs)
        inner_loader = DataLoader(inner_set, batch_size=self._hparams.batch_size, shuffle=False)
  
    training_set = GraphDataset(self._hparams, train_graphs)
    validation_set = GraphDataset(self._hparams, val_graphs)
  
    training_loader = DataLoader(training_set, batch_size=self._hparams.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=self._hparams.batch_size, shuffle=False)
  
    return training_loader, inner_loader, validation_loader

  def get_full_train_with_inner_loader(self, inner_val_frac=None):
      if inner_val_frac is None:
          inner_val_frac = getattr(self._hparams, 'inner_val_frac', 0.1)
      train_graphs = list(self.train_graphs)
      labels_tr = np.array([int(g.graph.get('label', 0)) for g in train_graphs])
  
      sss = StratifiedShuffleSplit(
          n_splits=1,
          test_size=inner_val_frac,
          random_state=getattr(self._hparams, 'cv_seed', 1024)
      )
      main_idx, inner_idx = next(sss.split(np.arange(len(train_graphs)), labels_tr))
      main_graphs = [train_graphs[i] for i in main_idx]
      inner_graphs = [train_graphs[i] for i in inner_idx]
  
      training_set = GraphDataset(self._hparams, main_graphs)
      inner_set = GraphDataset(self._hparams, inner_graphs)
  
      training_loader = DataLoader(training_set, batch_size=self._hparams.batch_size, shuffle=True)
      inner_loader = DataLoader(inner_set, batch_size=self._hparams.batch_size, shuffle=False)
      return training_loader, inner_loader
  
  def get_full_train_loader(self):
    training_set = GraphDataset(self._hparams, self.train_graphs)
    return DataLoader(training_set, batch_size=self._hparams.batch_size, shuffle=True)

  def get_test_loader(self):
    test_set = GraphDataset(self._hparams, self.test_graphs)
    return DataLoader(test_set, batch_size=self._hparams.batch_size, shuffle=False)

  def _resolve_group_ids(self, dataset, subgraphs, indices):
    # 优先从dataset字典尝试取组ID数组（必须与subgraphs一一对齐）
    possible_keys = [
        'group_ids', 'groups', 'subject_ids', 'patient_ids',
        'case_ids', 'slice_groups', 'slide_ids', 'group_idx'
    ]
    for key in possible_keys:
      if key in dataset:
        arr = dataset[key]
        if isinstance(arr, (list, np.ndarray)) and len(arr) == len(subgraphs):
          return [arr[i] for i in indices]

    # 次选：从每个Graph的graph属性尝试取组ID；若无则退化为每图独立组
    group_keys_in_graph = [
        'group_id', 'group', 'subject_id', 'patient_id',
        'case_id', 'slice_group', 'slide_id'
    ]
    resolved = []
    for i in indices:
      G = subgraphs[i]
      gid = None
      for k in group_keys_in_graph:
        if k in G.graph:
          gid = G.graph.get(k)
          break
      if gid is None:
        gid = i
      resolved.append(gid)
    return resolved

  def get_holdout_loaders(self, seed=None, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    """
    重复随机留出：分组分层的6:2:2划分
    - 组：若能解析到，则按组划分；否则每图视作独立组
    - 分层：在“组”层面按多数标签进行分层抽样
    返回：training_loader, validation_loader, test_loader
    """
    # 校验比例
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
      raise ValueError(f'划分比例之和应为1.0，当前={total}')
    if seed is None:
      seed = getattr(self._hparams, 'cv_seed', 1024)

    labels = self.all_labels
    groups = np.array(self.all_groups)
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    num_groups = len(unique_groups)

    # 计算每组标签（多数表决；若组内标签一致则直接使用）
    group_labels_list = [[] for _ in range(num_groups)]
    for sample_idx, g_idx in enumerate(group_inverse):
      group_labels_list[g_idx].append(labels[sample_idx])
    group_labels = np.array([
      int(np.round(np.mean(lst))) if len(set(lst)) > 1 else lst[0]
      for lst in group_labels_list
    ])

    # Step1: 组层面划分 test vs 其余（分层随机）
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    group_indices = np.arange(num_groups)
    train_val_group_idx, test_group_idx = next(sss_test.split(group_indices, group_labels))

    # Step2: 在train_val内部再划分 train vs val（相对比例）
    relative_val_frac = val_frac / (train_frac + val_frac)  # 0.2/0.8 = 0.25
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed + 1)
    tr_group_idx, val_group_idx = next(sss_val.split(train_val_group_idx, group_labels[train_val_group_idx]))

    # 映射到样本索引
    train_groups = set(unique_groups[train_val_group_idx[tr_group_idx]])
    val_groups   = set(unique_groups[train_val_group_idx[val_group_idx]])
    test_groups  = set(unique_groups[test_group_idx])

    train_idx = [i for i, g in enumerate(groups) if g in train_groups]
    val_idx   = [i for i, g in enumerate(groups) if g in val_groups]
    test_idx  = [i for i, g in enumerate(groups) if g in test_groups]

    train_graphs = [self.all_graphs[i] for i in train_idx]
    val_graphs   = [self.all_graphs[i] for i in val_idx]
    test_graphs  = [self.all_graphs[i] for i in test_idx]

    logging.info(f'Holdout split sizes: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}')

    # 构建 DataLoader
    training_set   = GraphDataset(self._hparams, train_graphs)
    validation_set = GraphDataset(self._hparams, val_graphs)
    test_set       = GraphDataset(self._hparams, test_graphs)

    training_loader   = DataLoader(training_set, batch_size=self._hparams.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=self._hparams.batch_size, shuffle=False)
    test_loader       = DataLoader(test_set, batch_size=self._hparams.batch_size, shuffle=False)

    return training_loader, validation_loader, test_loader


def read_graphfile(datadir, dataname, max_nodes=None):
  ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
      graph index starts with 1 in file
  Returns:
      List of networkx objects with graph and node labels
  '''
  prefix = os.path.join(datadir, dataname, dataname)
  filename_graph_indic = prefix + '_graph_indicator.txt'
  # index of graphs that a given node belongs to
  graph_indic = {}
  with open(filename_graph_indic) as f:
    i = 1
    for line in f:
      line = line.strip("\n")
      graph_indic[i] = int(line)
      i += 1

  filename_nodes = prefix + '_node_labels.txt'
  node_labels = []
  try:
    with open(filename_nodes) as f:
      for line in f:
        line = line.strip("\n")
        node_labels += [int(line) - 1]
    num_unique_node_labels = max(node_labels) + 1
  except IOError:
    print('No node labels')

  filename_node_attrs = prefix + '_node_attributes.txt'
  node_attrs = []
  try:
    with open(filename_node_attrs) as f:
      for line in f:
        line = line.strip("\s\n")
        attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
        node_attrs.append(np.array(attrs))
  except IOError:
    print('No node attributes')

  label_has_zero = False
  filename_graphs = prefix + '_graph_labels.txt'
  graph_labels = []

  # assume that all graph labels appear in the dataset
  # (set of labels don't have to be consecutive)
  label_vals = []
  with open(filename_graphs) as f:
    for line in f:
      line = line.strip("\n")
      val = int(line)
      # if val == 0:
      #    label_has_zero = True
      if val not in label_vals:
        label_vals.append(val)
      graph_labels.append(val)
  # graph_labels = np.array(graph_labels)
  label_map_to_int = {val: i for i, val in enumerate(label_vals)}
  graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
  # if label_has_zero:
  #    graph_labels += 1

  filename_adj = prefix + '_A.txt'
  adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
  index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
  num_edges = 0
  with open(filename_adj) as f:
    for line in f:
      line = line.strip("\n").split(",")
      e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
      adj_list[graph_indic[e0]].append((e0, e1))
      index_graph[graph_indic[e0]] += [e0, e1]
      num_edges += 1
  for k in index_graph.keys():
    index_graph[k] = [u - 1 for u in set(index_graph[k])]

  graphs = []
  for i in range(1, 1 + len(adj_list)):
    # indexed from 1 here
    G = nx.from_edgelist(adj_list[i])
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
      continue

    # add features and labels
    G.graph['label'] = graph_labels[i - 1]
    for u in G.nodes():
      if len(node_labels) > 0:
        node_label_one_hot = [0] * num_unique_node_labels
        node_label = node_labels[u - 1]
        node_label_one_hot[node_label] = 1
        # 兼容 NetworkX 3.x：使用 G.nodes
        G.nodes[u]['label'] = node_label_one_hot
      if len(node_attrs) > 0:
        G.nodes[u]['feat'] = node_attrs[u - 1]
    if len(node_attrs) > 0:
      G.graph['feat_dim'] = node_attrs[0].shape[0]

    # relabeling（统一使用 NetworkX 2/3 通用写法）
    mapping = {}
    it = 0
    for n in G.nodes:
        mapping[n] = it
        it += 1

    # indexed from 0
    graphs.append(nx.relabel_nodes(G, mapping))
  return graphs
