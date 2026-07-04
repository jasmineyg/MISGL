# coding=utf-8

import networkx as nx
import numpy as np
import os
import logging
import pickle
import json
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader

from MISGL.utils import hparams_lib
from MISGL.utils.global_variables import *
from MISGL.utils import reproducibility
from MISGL.utils import subgnn_border


# follow a discussion here: https://github.com/RexYing/diffpool/issues/17

class GraphDataset(Dataset):

  def __init__(self, hparams, graph_list, subgnn_border_anchor_walk_features=None):
    self._hparams = hparams_lib.copy_hparams(hparams)
    preload_to_gpu = bool(getattr(self._hparams, 'preload_data_to_gpu', True))
    target_device = self._hparams.device if preload_to_gpu else 'cpu'
    self._device = torch.device(target_device)
    bb_cfg = getattr(self._hparams, 'branch_b', None)
    self._use_structural_features = bool(
      bb_cfg and bb_cfg.get('use', False) and bb_cfg.get('use_structural_features', bb_cfg.get('structural_features', False))
    )
    self._structure_undirected = bool(bb_cfg.get('structural_undirected', True)) if bb_cfg else True
    self._structural_feature_dim = 7
    self._use_subgnn_border = subgnn_border.subgnn_border_enabled(self._hparams)
    border_cfg = getattr(self._hparams, 'subgnn_border', {})
    self._subgnn_border_num_anchors = int(border_cfg.get('num_anchors', 16)) if isinstance(border_cfg, dict) else 16
    self._subgnn_border_anchor_walk_features = None
    if self._use_subgnn_border and subgnn_border_anchor_walk_features is not None:
      self._subgnn_border_anchor_walk_features = torch.tensor(
        np.asarray(subgnn_border_anchor_walk_features, dtype=np.float32), dtype=torch.float32
      ).to(self._device)
    self.graph_list = []
    self.processed_graph_list = self.preprocess_graph(graph_list)

  def preprocess_graph(self, graph_list):
    processed_graph_list = []
    for graph in graph_list:
      graph_tmp_dict = {}
      # 浣跨敤鍥哄畾鑺傜偣椤哄簭浠ヤ繚璇侀偦鎺ョ煩闃典笌鐗瑰緛瀵归綈
      nodelist = list(graph.nodes())
      adj = nx.to_numpy_array(graph, nodelist=nodelist, dtype=np.float32)
      feature_dim = self._hparams.channel_list[0]
      node_tmp_feature = np.zeros((self._hparams.max_num_nodes, feature_dim), dtype=np.float32)
      for idx, node_id in enumerate(nodelist):
        node_feat = graph.nodes[node_id].get('features')
        if node_feat is None:
          raise ValueError('鑺傜偣缂哄皯 features 灞炴€э紝鏃犳硶鏋勫缓杈撳叆鐗瑰緛')
        node_tmp_feature[idx, :feature_dim] = np.asarray(node_feat, dtype=np.float32)
      num_nodes = adj.shape[0]
      graph_tmp_dict[g_key.x] = torch.tensor(node_tmp_feature, dtype=torch.float32).to(self._device)
      graph_tmp_dict[g_key.y] = torch.tensor(int(graph.graph.get('label', 0)), dtype=torch.long).to(self._device)
      graph_tmp_dict[g_key.node_num] = torch.tensor(num_nodes, dtype=torch.int16).to(self._device)
      graph_tmp_dict[g_key.adj_mat] = torch.zeros(self._hparams.max_num_nodes, self._hparams.max_num_nodes).to(self._device)
      graph_tmp_dict[g_key.adj_mat][:num_nodes, :num_nodes] = torch.tensor(adj, dtype=torch.float32).to(self._device)
      if self._use_structural_features:
        structural_features = np.zeros((self._hparams.max_num_nodes, self._structural_feature_dim), dtype=np.float32)
        if num_nodes > 0:
          structural_features[:num_nodes, :] = self._compute_structural_features_np(adj)
        graph_tmp_dict[g_key.structural_features] = torch.tensor(structural_features, dtype=torch.float32).to(self._device)
      if self._use_subgnn_border:
        border_anchor_sim = graph.graph.get('border_anchor_sim', None)
        if border_anchor_sim is None:
          raise ValueError('subgnn_border is enabled but graph is missing border_anchor_sim.')
        border_anchor_sim = np.asarray(border_anchor_sim, dtype=np.float32)
        if border_anchor_sim.shape != (self._subgnn_border_num_anchors,):
          raise ValueError(
            'Expected border_anchor_sim shape ({},), got {}'.format(
              self._subgnn_border_num_anchors, border_anchor_sim.shape
            )
          )
        graph_tmp_dict[g_key.border_anchor_sim] = torch.tensor(border_anchor_sim, dtype=torch.float32).to(self._device)
        if self._subgnn_border_anchor_walk_features is None:
          raise ValueError('subgnn_border is enabled but shared border_anchor_walk_features are missing.')
        graph_tmp_dict[g_key.border_anchor_walk_features] = self._subgnn_border_anchor_walk_features
        graph_tmp_dict[g_key.border_external_count] = torch.tensor(
          float(graph.graph.get('border_external_count', 0.0)), dtype=torch.float32
        ).to(self._device)
      orig_idx = graph.graph.get('orig_idx', -1)
      try:
        orig_idx = int(orig_idx)
      except (TypeError, ValueError):
        orig_idx = -1
      graph_tmp_dict[g_key.orig_graph_idx] = torch.tensor(orig_idx, dtype=torch.long).to(self._device)
      subgraph_id = graph.graph.get('subgraph_id', -1)
      try:
        subgraph_id = int(subgraph_id)
      except (TypeError, ValueError):
        subgraph_id = -1
      graph_tmp_dict[g_key.subgraph_id] = torch.tensor(subgraph_id, dtype=torch.long).to(self._device)
      processed_graph_list.append(graph_tmp_dict)
    return processed_graph_list

  def _compute_structural_features_np(self, adj):
    adj_bool = adj != 0
    if self._structure_undirected:
      adj_bool = np.logical_or(adj_bool, adj_bool.T)
    np.fill_diagonal(adj_bool, False)
    adj_float = adj_bool.astype(np.float32, copy=False)

    num_nodes = adj_float.shape[0]
    max_degree = max(float(num_nodes - 1), 1.0)
    degree = adj_float.sum(axis=-1)
    degree_norm = degree / max_degree
    log_degree_norm = np.log1p(degree) / np.log1p(max_degree)

    neighbor_degree_sum = adj_float @ degree
    avg_neighbor_degree = neighbor_degree_sum / np.maximum(degree, 1.0)
    avg_neighbor_degree_norm = avg_neighbor_degree / max_degree

    two_hop_walk_count = neighbor_degree_sum
    two_hop_denom = max(max_degree ** 2, 1.0)
    two_hop_walk_log_norm = np.log1p(two_hop_walk_count) / np.log1p(two_hop_denom)

    two_path_count = adj_float @ adj_float
    closed_wedge_count = (two_path_count * adj_float).sum(axis=-1)
    triangle_count = closed_wedge_count / 2.0
    max_triangle_count = max(max_degree * (max_degree - 1.0) / 2.0, 1.0)
    triangle_count_log_norm = np.log1p(triangle_count) / np.log1p(max_triangle_count)

    possible_wedge_count = degree * (degree - 1.0)
    clustering_coeff = np.divide(
      closed_wedge_count,
      np.maximum(possible_wedge_count, 1.0),
      out=np.zeros_like(degree),
      where=possible_wedge_count > 0,
    )

    core_number_norm = self._compute_core_number_norm_np(adj_float, max_degree)
    return np.stack(
      [
        degree_norm,
        log_degree_norm,
        avg_neighbor_degree_norm,
        two_hop_walk_log_norm,
        triangle_count_log_norm,
        clustering_coeff,
        core_number_norm,
      ],
      axis=-1,
    ).astype(np.float32, copy=False)

  @staticmethod
  def _compute_core_number_norm_np(adj_float, max_degree):
    num_nodes = adj_float.shape[0]
    if num_nodes == 0:
      return np.zeros((0,), dtype=np.float32)

    remaining = np.ones(num_nodes, dtype=bool)
    working_degree = adj_float.sum(axis=-1).astype(np.float32, copy=True)
    local_core = np.zeros((num_nodes,), dtype=np.float32)
    running_core = 0.0

    for _ in range(num_nodes):
      masked_degree = np.where(remaining, working_degree, np.inf)
      node_idx = int(np.argmin(masked_degree))
      running_core = max(running_core, float(masked_degree[node_idx]))
      local_core[node_idx] = running_core
      remaining[node_idx] = False
      working_degree = np.maximum(working_degree - adj_float[:, node_idx], 0.0)

    return local_core / max_degree

  def __len__(self):
    return len(self.processed_graph_list)

  def __getitem__(self, idx):
    return self.processed_graph_list[idx]


class GraphDataLoaderWrapper(object):

  def __init__(self, hparams, data_name=None):
    self._hparams = hparams_lib.copy_hparams(hparams)
    self.data_name = data_name if data_name is not None else getattr(self._hparams, 'data_name', None)

    processed_data_dir = getattr(self._hparams, 'processed_data_dir', '/data/yg/Subgraph-MIL/Data/processed_data')
    if data_name is None:
      data_name = self.data_name
    dataset_path = os.path.join(processed_data_dir, f'{data_name}_processed.pkl')
    logging.warning(f'[DEBUG] Attempting to load dataset from: {dataset_path}')
    if not os.path.exists(dataset_path):
        logging.error(f'[ERROR] Dataset file not found: {dataset_path}')
        logging.error(f'[ERROR] Current working directory: {os.getcwd()}')
        # 灏濊瘯鍦ㄥ綋鍓嶇洰褰曟垨鐩稿鐩綍鏌ユ壘
        alt_path = os.path.join('processed_data', f'{data_name}_processed.pkl')
        if os.path.exists(alt_path):
             logging.warning(f'[DEBUG] Found dataset at alternative path: {alt_path}')
             dataset_path = alt_path
        else:
             raise FileNotFoundError(
               'Dataset file not found for {}. Checked: {!r} and {!r}. '
               'Update processed_data_dir in the yaml or pass --processed_data_dir.'
               .format(data_name, dataset_path, alt_path)
             )
    
    with open(dataset_path, 'rb') as f:
      dataset = pickle.load(f)

    subgraphs = dataset['subgraph_structures']
    train_indices = dataset['train_test_split']['train_indices']
    test_indices = dataset['train_test_split']['test_indices']

    feature_dim = int(dataset.get('feature_dimension', dataset['dataset_metadata']['feature_dim']))
    self._hparams.channel_list[0] = feature_dim
    max_num_nodes = int(dataset['dataset_metadata'].get('max_num_nodes', max(len(g.nodes()) for g in subgraphs)))
    self._set_or_add_hparam('max_num_nodes', max_num_nodes)

    self.train_graphs = [subgraphs[i] for i in train_indices]
    self.test_graphs = [subgraphs[i] for i in test_indices]

    self._subgraphs = subgraphs
    self._dataset_raw = dataset
    self.original_graph = dataset.get('original_graph', None)
    self.assignment_matrix = dataset.get('assignment_matrix', None)
    self.all_indices = list(train_indices) + list(test_indices)
    self.all_graphs = []
    subgraph_labels = dataset.get('subgraph_labels', None)
    for orig_idx in self.all_indices:
      graph = subgraphs[orig_idx]
      graph.graph['orig_idx'] = int(orig_idx)
      if subgraph_labels is not None and 0 <= int(orig_idx) < len(subgraph_labels):
        try:
          graph.graph['label'] = int(subgraph_labels[int(orig_idx)])
        except Exception:
          pass
      self.all_graphs.append(graph)
    self.all_labels = np.array([int(g.graph.get('label', 0)) for g in self.all_graphs])
    self._subgnn_border_info = None
    self._subgnn_border_anchor_walk_features = None
    if subgnn_border.subgnn_border_enabled(self._hparams):
      self._attach_subgnn_border_features(dataset, subgraphs)
    self.all_groups = self._resolve_group_ids(dataset, subgraphs, self.all_indices)

    self.cv_seed = int(getattr(self._hparams, 'cv_seed', 1024))
    self.cv_num_folds = int(getattr(self._hparams, 'cv_num_folds', getattr(self._hparams, 'fold_num', 10)))
    self.cv_val_policy = str(getattr(self._hparams, 'cv_val_policy', 'adjacent'))
    self.cv_use_all_samples = bool(getattr(self._hparams, 'cv_use_all_samples', True))

    if self.cv_use_all_samples:
      self.cv_graphs = list(self.all_graphs)
      self.cv_labels = np.array(self.all_labels)
      self.cv_groups = np.array(self.all_groups, dtype=object)
      self.cv_orig_indices = list(self.all_indices)
    else:
      self.cv_graphs = list(self.train_graphs)
      self.cv_labels = np.array([int(g.graph.get('label', 0)) for g in self.train_graphs])
      self.cv_groups = np.array(self._resolve_group_ids(dataset, subgraphs, train_indices), dtype=object)
      self.cv_orig_indices = list(train_indices)
    self._cv_index_by_orig_idx = {int(orig_idx): idx for idx, orig_idx in enumerate(self.cv_orig_indices)}
    self.cv_folds = None
    self.cv_build_info = None

  def _attach_subgnn_border_features(self, dataset, subgraphs):
    border_info = subgnn_border.build_subgnn_border_features(
      dataset,
      self._hparams,
      data_name=self.data_name,
    )
    if border_info is None:
      return
    features = border_info['features']
    external_counts = border_info['external_counts']
    self._subgnn_border_anchor_walk_features = border_info.get('anchor_walk_features')
    if features.shape[0] != len(subgraphs):
      raise ValueError(
        'SubGNN border feature row count mismatch: {} vs {}'.format(features.shape[0], len(subgraphs))
      )
    for idx, graph in enumerate(subgraphs):
      graph.graph['border_anchor_sim'] = features[idx].astype(np.float32, copy=False)
      graph.graph['border_external_count'] = float(external_counts[idx])
    diagnostics_path = subgnn_border.maybe_write_subgnn_border_diagnostics(
      self._hparams,
      self.data_name,
      border_info.get('diagnostics'),
    )
    self._subgnn_border_info = {
      'diagnostics': border_info.get('diagnostics'),
      'diagnostics_path': diagnostics_path,
      'anchor_walk_features_shape': None if self._subgnn_border_anchor_walk_features is None else list(self._subgnn_border_anchor_walk_features.shape),
    }
  def _set_or_add_hparam(self, name, value):
    if name in self._hparams:
      self._hparams.set_hparam(name, value)
    else:
      self._hparams.add_hparam(name, value)

  def _resolve_group_ids(self, dataset, subgraphs, indices):
    # 浼樺厛浠巇ataset瀛楀吀灏濊瘯鍙栫粍ID鏁扮粍锛堝繀椤讳笌subgraphs涓€涓€瀵归綈锛?
    possible_keys = [
        'group_ids', 'groups', 'subject_ids', 'patient_ids',
        'case_ids', 'slice_groups', 'slide_ids', 'group_idx'
    ]
    for key in possible_keys:
      if key in dataset:
        arr = dataset[key]
        if isinstance(arr, (list, np.ndarray)) and len(arr) == len(subgraphs):
          return [arr[i] for i in indices]

    # 娆￠€夛細浠庢瘡涓狦raph鐨刧raph灞炴€у皾璇曞彇缁処D锛涜嫢鏃犲垯閫€鍖栦负姣忓浘鐙珛缁?
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
    閲嶅闅忔満鐣欏嚭锛氬垎缁勫垎灞傜殑6:2:2鍒掑垎
    - 缁勶細鑻ヨ兘瑙ｆ瀽鍒帮紝鍒欐寜缁勫垝鍒嗭紱鍚﹀垯姣忓浘瑙嗕綔鐙珛缁?
    - 鍒嗗眰锛氬湪鈥滅粍鈥濆眰闈㈡寜澶氭暟鏍囩杩涜鍒嗗眰鎶芥牱
    杩斿洖锛歵raining_loader, validation_loader, test_loader
    """
    # 鏍￠獙姣斾緥
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
      raise ValueError(f'鍒掑垎姣斾緥涔嬪拰搴斾负1.0锛屽綋鍓?{total}')
    if seed is None:
      seed = getattr(self._hparams, 'cv_seed', 1024)

    labels = self.all_labels
    groups = np.array(self.all_groups)
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    num_groups = len(unique_groups)

    # 璁＄畻姣忕粍鏍囩锛堝鏁拌〃鍐筹紱鑻ョ粍鍐呮爣绛句竴鑷村垯鐩存帴浣跨敤锛?
    group_labels_list = [[] for _ in range(num_groups)]
    for sample_idx, g_idx in enumerate(group_inverse):
      group_labels_list[g_idx].append(labels[sample_idx])
    group_labels = np.array([
      int(np.round(np.mean(lst))) if len(set(lst)) > 1 else lst[0]
      for lst in group_labels_list
    ])

    # Step1: 缁勫眰闈㈠垝鍒?test vs 鍏朵綑锛堝垎灞傞殢鏈猴級
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    group_indices = np.arange(num_groups)
    train_val_group_idx, test_group_idx = next(sss_test.split(group_indices, group_labels))

    # Step2: 鍦╰rain_val鍐呴儴鍐嶅垝鍒?train vs val锛堢浉瀵规瘮渚嬶級
    relative_val_frac = val_frac / (train_frac + val_frac)  # 0.2/0.8 = 0.25
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed + 1)
    tr_group_idx, val_group_idx = next(sss_val.split(train_val_group_idx, group_labels[train_val_group_idx]))

    # 鏄犲皠鍒版牱鏈储寮?
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

    # 鏋勫缓 DataLoader
    training_set   = GraphDataset(self._hparams, train_graphs, self._subgnn_border_anchor_walk_features)
    validation_set = GraphDataset(self._hparams, val_graphs, self._subgnn_border_anchor_walk_features)
    test_set       = GraphDataset(self._hparams, test_graphs, self._subgnn_border_anchor_walk_features)

    training_loader   = DataLoader(training_set, batch_size=self._hparams.batch_size, shuffle=True, worker_init_fn=reproducibility.worker_init_fn)
    validation_loader = DataLoader(validation_set, batch_size=self._hparams.batch_size, shuffle=False, worker_init_fn=reproducibility.worker_init_fn)
    test_loader       = DataLoader(test_set, batch_size=self._hparams.batch_size, shuffle=False, worker_init_fn=reproducibility.worker_init_fn)

    return training_loader, validation_loader, test_loader

  def _build_cv_folds(self, labels, groups, num_folds, seed):
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    num_groups = len(unique_groups)
    if num_folds < 3:
      raise ValueError(f'cv_num_folds must be at least 3, got {num_folds}')
    if num_groups < num_folds:
      raise ValueError(f'Not enough groups for {num_folds}-fold CV: only {num_groups} groups available')

    group_labels_list = [[] for _ in range(num_groups)]
    for sample_idx, g_idx in enumerate(group_inverse):
      group_labels_list[g_idx].append(int(labels[sample_idx]))
    group_labels = np.array([
      int(np.round(np.mean(lst))) if len(set(lst)) > 1 else int(lst[0])
      for lst in group_labels_list
    ])

    group_indices = np.arange(num_groups)
    used_group_stratified = True
    try:
      splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
      group_fold_indices = [
        np.array(test_idx, dtype=np.int64)
        for _, test_idx in splitter.split(group_indices, group_labels)
      ]
    except ValueError as exc:
      logging.warning('Falling back to non-stratified KFold for CV folds: %s', exc)
      used_group_stratified = False
      splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
      group_fold_indices = [
        np.array(test_idx, dtype=np.int64)
        for _, test_idx in splitter.split(group_indices)
      ]

    sample_folds = []
    fold_infos = []
    for fold_id, fold_group_idx in enumerate(group_fold_indices):
      fold_group_set = set(int(i) for i in fold_group_idx.tolist())
      sample_idx = [sample_idx for sample_idx, g_idx in enumerate(group_inverse) if int(g_idx) in fold_group_set]
      sample_folds.append(sample_idx)
      label_hist = {}
      for sample_id in sample_idx:
        label_key = str(int(labels[sample_id]))
        label_hist[label_key] = label_hist.get(label_key, 0) + 1
      fold_infos.append({
        'fold_id': int(fold_id),
        'sample_positions': [int(i) for i in sample_idx],
        'group_ids': [self._serialize_group_value(unique_groups[int(i)]) for i in fold_group_idx.tolist()],
        'label_hist': label_hist,
      })
    build_info = {
      'used_group_stratified': bool(used_group_stratified),
      'num_groups': int(num_groups),
    }
    return sample_folds, build_info, fold_infos

  def _serialize_group_value(self, value):
    if isinstance(value, np.generic):
      return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
      return value
    return str(value)

  def get_cv_split_filename(self):
    data_name = self.data_name if self.data_name is not None else 'dataset'
    return f'{data_name}_cv{self.cv_num_folds}_seed{self.cv_seed}_{self.cv_val_policy}.json'

  def get_cv_split_path(self, ensure_dir=False):
    split_dir = getattr(self._hparams, 'cv_split_dir', '/data/yg/Subgraph-MIL/diffpool2/splits')
    if ensure_dir:
      os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, self.get_cv_split_filename())

  def build_cv_split_manifest(self):
    cv_folds, build_info, fold_infos = self._build_cv_folds(
      labels=self.cv_labels,
      groups=self.cv_groups,
      num_folds=self.cv_num_folds,
      seed=self.cv_seed
    )
    self.cv_folds = cv_folds
    self.cv_build_info = build_info

    folds = []
    for fold_info in fold_infos:
      sample_positions = fold_info['sample_positions']
      folds.append({
        'fold_id': int(fold_info['fold_id']),
        'sample_indices': [int(self.cv_orig_indices[i]) for i in sample_positions],
        'group_ids': [self._serialize_group_value(self.cv_groups[i]) for i in sample_positions],
        'label_hist': dict(fold_info['label_hist']),
      })

    return {
      'data_name': self.data_name,
      'cv_seed': int(self.cv_seed),
      'cv_num_folds': int(self.cv_num_folds),
      'cv_val_policy': self.cv_val_policy,
      'cv_use_all_samples': bool(self.cv_use_all_samples),
      'protocol': 'grouped_stratified_cv_8_1_1',
      'build_info': build_info,
      'folds': folds,
    }

  def _validate_cv_split_manifest(self, manifest):
    expected = {
      'data_name': self.data_name,
      'cv_seed': int(self.cv_seed),
      'cv_num_folds': int(self.cv_num_folds),
      'cv_val_policy': self.cv_val_policy,
      'cv_use_all_samples': bool(self.cv_use_all_samples),
    }
    for key, expected_value in expected.items():
      actual_value = manifest.get(key)
      if actual_value != expected_value:
        raise ValueError(f'Split manifest mismatch for {key}: expected {expected_value!r}, got {actual_value!r}')

    folds = manifest.get('folds', None)
    if not isinstance(folds, list) or len(folds) != self.cv_num_folds:
      raise ValueError(f'Split manifest must contain {self.cv_num_folds} folds')

    seen_indices = set()
    expected_indices = set(int(i) for i in self.cv_orig_indices)
    for fold_id, fold in enumerate(folds):
      if int(fold.get('fold_id', -1)) != fold_id:
        raise ValueError(f'Split manifest fold_id mismatch at fold {fold_id}')
      sample_indices = [int(i) for i in fold.get('sample_indices', [])]
      overlap = seen_indices.intersection(sample_indices)
      if overlap:
        raise ValueError(f'Split manifest has overlapping sample indices across folds: {sorted(overlap)[:5]}')
      seen_indices.update(sample_indices)
      missing_in_dataset = [idx for idx in sample_indices if idx not in self._cv_index_by_orig_idx]
      if missing_in_dataset:
        raise ValueError(f'Split manifest references unknown sample indices: {missing_in_dataset[:5]}')
    if seen_indices != expected_indices:
      missing = sorted(expected_indices - seen_indices)
      extra = sorted(seen_indices - expected_indices)
      raise ValueError(f'Split manifest coverage mismatch: missing={missing[:5]}, extra={extra[:5]}')

  def save_cv_split_manifest(self, split_path, overwrite=False):
    manifest = self.build_cv_split_manifest()
    if os.path.exists(split_path) and not overwrite:
      raise FileExistsError(f'Split manifest already exists: {split_path}')
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'w', encoding='utf-8') as f:
      json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest

  def load_cv_split_manifest(self, split_path):
    with open(split_path, 'r', encoding='utf-8') as f:
      manifest = json.load(f)
    self._validate_cv_split_manifest(manifest)
    return manifest

  def _build_loaders_from_indices(self, train_idx, val_idx, test_idx):
    train_graphs = [self.cv_graphs[i] for i in train_idx]
    val_graphs = [self.cv_graphs[i] for i in val_idx]
    test_graphs = [self.cv_graphs[i] for i in test_idx]

    training_set = GraphDataset(self._hparams, train_graphs, self._subgnn_border_anchor_walk_features)
    validation_set = GraphDataset(self._hparams, val_graphs, self._subgnn_border_anchor_walk_features)
    test_set = GraphDataset(self._hparams, test_graphs, self._subgnn_border_anchor_walk_features)

    training_loader = DataLoader(training_set, batch_size=self._hparams.batch_size, shuffle=True, worker_init_fn=reproducibility.worker_init_fn)
    validation_loader = DataLoader(validation_set, batch_size=self._hparams.batch_size, shuffle=False, worker_init_fn=reproducibility.worker_init_fn)
    test_loader = DataLoader(test_set, batch_size=self._hparams.batch_size, shuffle=False, worker_init_fn=reproducibility.worker_init_fn)
    return training_loader, validation_loader, test_loader

  def get_cv_loaders_from_manifest(self, manifest, fold_idx):
    self._validate_cv_split_manifest(manifest)
    if self.cv_val_policy != 'adjacent':
      raise ValueError(f'Unsupported cv_val_policy: {self.cv_val_policy}')
    if not (0 <= int(fold_idx) < self.cv_num_folds):
      raise IndexError(f'fold_idx out of range: {fold_idx}')

    test_fold = int(fold_idx)
    val_fold = (test_fold + 1) % self.cv_num_folds
    train_folds = [i for i in range(self.cv_num_folds) if i not in (test_fold, val_fold)]

    def _fold_positions(manifest_fold_id):
      sample_indices = [int(i) for i in manifest['folds'][manifest_fold_id]['sample_indices']]
      return sorted(self._cv_index_by_orig_idx[idx] for idx in sample_indices)

    train_idx = sorted(idx for fold_id in train_folds for idx in _fold_positions(fold_id))
    val_idx = _fold_positions(val_fold)
    test_idx = _fold_positions(test_fold)

    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
      raise RuntimeError('CV split overlap detected between train/val/test sets')

    training_loader, validation_loader, test_loader = self._build_loaders_from_indices(train_idx, val_idx, test_idx)
    split_meta = {
      'fold_idx': test_fold,
      'train_folds': train_folds,
      'val_fold': val_fold,
      'test_fold': test_fold,
      'train_size': len(train_idx),
      'val_size': len(val_idx),
      'test_size': len(test_idx),
      'train_indices': [int(self.cv_orig_indices[i]) for i in train_idx],
      'val_indices': [int(self.cv_orig_indices[i]) for i in val_idx],
      'test_indices': [int(self.cv_orig_indices[i]) for i in test_idx],
      'train_groups': [self._serialize_group_value(self.cv_groups[i]) for i in train_idx],
      'val_groups': [self._serialize_group_value(self.cv_groups[i]) for i in val_idx],
      'test_groups': [self._serialize_group_value(self.cv_groups[i]) for i in test_idx],
    }
    return training_loader, validation_loader, test_loader, split_meta
