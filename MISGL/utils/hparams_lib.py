# coding=utf-8

""" HParams handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from MISGL.utils import hparam


DEFAULT_HPARAMS = {
  'device': 'cuda',
  'cuda_visible_devices': None,
  'auto_select_gpu': False,
  'gpu_candidate_devices': None,
  'gpu_memory_used_max_mb': 1024,
  'gpu_utilization_max_pct': 10,
  'gpu_select_wait_seconds': 0,
  'gpu_select_poll_interval': 30,
  'gpu_lock_idle_card': True,
  'gpu_lock_dir': '/tmp/misgl_gpu_locks',
  'preload_data_to_gpu': True,
  'model_save_path': 'results',
  'processed_data_dir': '/data/yg/Subgraph-MIL/Data/processed_data',
  'enable_experiment_result_excel': True,
  'experiment_result_excel_path': '/data/yg/Subgraph-MIL/diffpool2/experiment_results.xlsx',
  'experiment_result_excel_strict': False,
  'weight_decay': 0.0,
  'label_smoothing': 0.0,
  'early_stop_min_delta': 0.0,
  'early_stop_loss_delta': 0.0001,
  'cv_split_dir': '/data/yg/Subgraph-MIL/diffpool2/splits',
  'cv_seed': 1024,
  'cv_num_folds': 10,
  'cv_val_policy': 'adjacent',
  'cv_use_all_samples': True,
  'position_head': {
    'use': False,
    'top_k': 16,
    'normalize': True,
    'include_self': False,
    'symmetrize': True,
    'type': 'residual_gcn',
    'num_layers': 1,
    'dropout': 0.1,
    'row_normalize': True,
    'residual_init': 0.1,
    'cache': True,
    'cache_dir': '.cache/position_head',
  },
  'enable_gat_export': False,
  'enable_tensorboard': False,
  'train_eval_interval': 10,
  'enable_train_eval_during_training': True,
  'train_eval_max_num_examples': 100,
  'final_eval_splits': ['train', 'val', 'test'],
  'export_attention': False,
  'analyze_attention': False,
  'attention_sample_frac': 0.1,
  'attention_train_sample_frac': 0.1,
  'attention_top_k': 20,
  'attention_export_train': True,
  'branch_b': {
    'use': False,
    'attn_hidden': 128,
    'gate_hidden': 64,
    'use_structural_features': False,
    'structural_hidden_dim': 32,
    'structural_embed_dim': 32,
    'structural_dropout': 0.1,
    'structural_fusion': 'gated_residual',
    'structural_gate_hidden_dim': 64,
    'structural_residual_init': 0.1,
    'structural_undirected': True,
    'attention_shape_loss_enabled': True,
    'attention_shape_loss_weight': 0.0,
    'attention_shape_loss_eps': 1e-8,
  },
}


def _merge_defaults(value, defaults):
  merged = copy.deepcopy(value)
  for key, default_value in defaults.items():
    if key not in merged:
      merged[key] = copy.deepcopy(default_value)
    elif isinstance(merged[key], dict) and isinstance(default_value, dict):
      merged[key] = _merge_defaults(merged[key], default_value)
  return merged


def apply_defaults(hparams):
  for name, value in DEFAULT_HPARAMS.items():
    if hasattr(hparams, name):
      current = getattr(hparams, name)
      if isinstance(current, dict) and isinstance(value, dict):
        setattr(hparams, name, _merge_defaults(current, value))
    else:
      hparams.add_hparam(name, copy.deepcopy(value))
  return hparams


def copy_hparams(hparams):
  hp_vals = hparams.values()
  new_hparams = hparam.HParams(**hp_vals)
  for name in ('data_name',):
    if hasattr(hparams, name):
      setattr(new_hparams, name, getattr(hparams, name))
  return new_hparams


def create_hparams(config_dir):
  hparams = hparam.HParams()
  hparams.from_yaml(config_dir)
  return apply_defaults(hparams)
